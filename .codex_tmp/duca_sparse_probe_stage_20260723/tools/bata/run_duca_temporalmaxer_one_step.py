from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_file(path: str | Path, expected_sha256: str, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is missing: {resolved}")
    observed = sha256_file(resolved)
    if observed != expected_sha256.lower():
        raise RuntimeError(f"{label} SHA256 mismatch")
    return resolved


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    return value


def _nonzero_grad_names(named_parameters) -> list[str]:
    import torch

    names = []
    for name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            continue
        if not bool(torch.isfinite(gradient).all().item()):
            raise RuntimeError(f"non-finite gradient: {name}")
        if bool((gradient.detach().abs() > 0).any().item()):
            names.append(name)
    return names


def run_one_step(
    *,
    config_path: str | Path,
    pretrain_path: str | Path,
    pretrain_sha256: str,
    output_path: str | Path,
    seed: int = 3407,
    device_name: str = "cuda:0",
) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn as nn
    from mmengine.config import Config

    from opentad.cores import build_optimizer, prepare_optimizer_parameter_freezing
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from tools.bata.duca_frontend_initialization import (
        initialize_frame_selector_from_checkpoint,
    )

    config = Path(config_path).expanduser().resolve()
    if not config.is_file():
        raise FileNotFoundError(f"TemporalMaxer config is missing: {config}")
    pretrain = _require_file(
        pretrain_path, pretrain_sha256, "AdaTAD VideoMAE pretrain"
    )
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the real TemporalMaxer one-step gate requires one CUDA GPU")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("the gate requires exactly one Slurm-visible GPU")
    torch.cuda.set_device(device)

    cfg = Config.fromfile(str(config))
    if cfg.model.get("type") != "TemporalMaxer":
        raise ValueError("gate config does not build TemporalMaxer")
    if cfg.model.get("frame_selector") is None:
        raise ValueError("gate config has no live DUCA frame_selector")
    if (
        cfg.model.backbone.get("type") != "mmaction.Recognizer3D"
        or cfg.model.backbone.backbone.get("type") != "VisionTransformerAdapter"
    ):
        raise ValueError("gate config is not the raw-RGB VideoMAE path")
    detector_gradient_mode = str(
        cfg.model.frame_selector.get("detector_gradient_mode", "none")
    )
    if detector_gradient_mode == "none":
        raise ValueError("learned gate config must expose detector-to-selector gradient")
    cfg.model.backbone.custom.pretrain = str(pretrain)

    logger = logging.getLogger("duca-r5-temporalmaxer-one-step")
    dataset = build_dataset(cfg.dataset.train, default_args={"logger": logger})
    loader_cfg = dict(cfg.solver.train)
    loader_cfg.update(batch_size=1, num_workers=0)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=False,
        **loader_cfg,
    )

    model = build_detector(copy.deepcopy(cfg.model))
    if model.__class__.__name__ != "TemporalMaxer" or not model.with_frame_selector:
        raise RuntimeError("real config did not build live-selector TemporalMaxer")
    initialization = initialize_frame_selector_from_checkpoint(
        model,
        cfg.workflow.get("selector_initialization"),
        logger=logger,
    )
    if initialization is None:
        raise RuntimeError("learned TemporalMaxer gate requires P0 selector initialization")

    optimizer_cfg = copy.deepcopy(dict(cfg.optimizer))
    prepare_optimizer_parameter_freezing(optimizer_cfg, model, logger)
    model = model.to(device).train()
    schedule_step = max(
        0,
        int(cfg.workflow.get("expected_successful_optimizer_updates", 6000)) - 1,
    )
    schedule_state = getattr(model.frame_selector, "_loss_weight_schedule_step", None)
    if torch.is_tensor(schedule_state):
        schedule_state.fill_(schedule_step)

    optimizer = build_optimizer(
        copy.deepcopy(optimizer_cfg), SimpleNamespace(module=model), logger
    )
    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    trainable_parameter_ids = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if optimizer_parameter_ids != trainable_parameter_ids:
        raise RuntimeError("production optimizer does not exactly cover trainable parameters")

    covered_module_types: dict[str, int] = {}
    for module_type in (nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.Embedding):
        parameters = {
            id(parameter)
            for module in model.modules()
            if isinstance(module, module_type)
            for parameter in module.parameters(recurse=False)
            if parameter.requires_grad
        }
        if not parameters.issubset(optimizer_parameter_ids):
            raise RuntimeError(f"optimizer missed {module_type.__name__} parameters")
        covered_module_types[module_type.__name__] = len(parameters)

    batch = _move_to_device(next(iter(loader)), device)
    optimizer.zero_grad(set_to_none=True)
    amp_enabled = bool(cfg.solver.get("amp", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    with torch.cuda.amp.autocast(enabled=amp_enabled):
        losses = model(**batch, return_loss=True)
    total = losses.get("cost")
    if not torch.is_tensor(total) or total.ndim != 0 or not bool(torch.isfinite(total).item()):
        raise RuntimeError("TemporalMaxer returned a non-finite one-step loss")
    detector_leaves = [
        value
        for key, value in losses.items()
        if key != "cost"
        and str(key).endswith("_loss")
        and not str(key).startswith("selector_")
    ]
    if not detector_leaves:
        raise RuntimeError("TemporalMaxer head returned no detector leaf losses")
    detector_loss = sum(detector_leaves)
    selector_named = [
        (name, parameter)
        for name, parameter in model.frame_selector.named_parameters()
        if parameter.requires_grad
    ]
    detector_selector_grads = torch.autograd.grad(
        detector_loss,
        [parameter for _, parameter in selector_named],
        retain_graph=True,
        allow_unused=True,
    )
    detector_selector_names = [
        name
        for (name, _), gradient in zip(selector_named, detector_selector_grads)
        if gradient is not None
        and bool(torch.isfinite(gradient).all().item())
        and bool((gradient.detach().abs() > 0).any().item())
    ]
    if not detector_selector_names:
        raise RuntimeError("detector loss did not reach the live DUCA selector")

    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    selector_grad_names = _nonzero_grad_names(model.frame_selector.named_parameters())
    head_grad_names = _nonzero_grad_names(model.rpn_head.named_parameters())
    if not selector_grad_names or not head_grad_names:
        raise RuntimeError("one-step backward missed selector or TemporalMaxer head")

    lr_by_parameter = {
        id(parameter): float(group["lr"])
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    changed_candidates = [
        (prefix + name, parameter)
        for prefix, named_parameters in (
            ("frame_selector.", model.frame_selector.named_parameters()),
            ("rpn_head.", model.rpn_head.named_parameters()),
        )
        for name, parameter in named_parameters
        if parameter.requires_grad
        and parameter.grad is not None
        and bool((parameter.grad.detach().abs() > 0).any().item())
        and lr_by_parameter.get(id(parameter), 0.0) > 0.0
    ]
    if not changed_candidates:
        raise RuntimeError("no selector/head parameter has a nonzero-gradient optimizer group")
    before = {name: parameter.detach().clone() for name, parameter in changed_candidates}
    scaler.step(optimizer)
    scaler.update()
    changed_names = [
        name
        for name, parameter in changed_candidates
        if not torch.equal(before[name], parameter.detach())
    ]
    if not changed_names:
        raise RuntimeError("optimizer step changed no selector/head parameter")

    selected_axis_summary = getattr(
        model, "_last_selected_axis_training_summary", None
    )
    if not isinstance(selected_axis_summary, dict) or (
        selected_axis_summary.get("inverse_map_present") is not True
    ):
        raise RuntimeError("TemporalMaxer did not validate selected-axis GT/remap metadata")

    result = {
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": git_commit,
        "config": str(config),
        "config_sha256": sha256_file(config),
        "pretrain_path": str(pretrain),
        "pretrain_sha256": sha256_file(pretrain),
        "seed": int(seed),
        "device": str(device),
        "dataset_type": dataset.__class__.__name__,
        "input_shape": [int(value) for value in batch["inputs"].shape],
        "detector_type": model.__class__.__name__,
        "backbone_type": model.backbone.__class__.__name__,
        "selector_type": model.frame_selector.__class__.__name__,
        "detector_gradient_mode": detector_gradient_mode,
        "schedule_step": schedule_step,
        "selected_axis_summary": selected_axis_summary,
        "loss_keys": sorted(str(key) for key in losses),
        "optimizer_trainable_parameter_count": len(trainable_parameter_ids),
        "optimizer_covered_module_types": covered_module_types,
        "detector_to_selector_gradient_parameters": detector_selector_names,
        "selector_gradient_parameters": selector_grad_names,
        "head_gradient_parameters": head_grad_names,
        "changed_selector_or_head_parameters": changed_names,
        "selector_initialization": initialization,
        "forward_backward_optimizer_step_completed": True,
    }
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real DUCA -> VideoMAE -> TemporalMaxer optimizer step."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--pretrain-sha256", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_one_step(
        config_path=args.config,
        pretrain_path=args.pretrain,
        pretrain_sha256=args.pretrain_sha256,
        output_path=args.output,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
