from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from tools.bata.diagnose_duca_stage2_nonfinite_loss import (
    _atomic_write_json,
    _finite_tensor_summary,
    _load_checkpoint,
    _move_to_device,
    _normalize_state_dict,
    sha256_file,
)


SCHEMA_VERSION = "duca_stage2_postupdate_nonfinite_diagnostic_v1"


def _optimizer_step_ran(scale_before: float, scale_after: float) -> bool:
    """Match train_one_epoch's AMP-step receipt rule exactly."""

    return bool(scale_after >= scale_before)


def _summarize_named_tensors(
    values: Iterable[tuple[str, Any]], *, limit: int = 20
) -> dict[str, Any]:
    """Summarize tensor health without serializing full parameter tensors."""

    import torch

    tensor_count = 0
    element_count = 0
    finite_count = 0
    nan_count = 0
    posinf_count = 0
    neginf_count = 0
    nonfinite_tensor_count = 0
    nonfinite_names: list[str] = []
    for name, value in values:
        if not torch.is_tensor(value):
            continue
        tensor_count += 1
        summary = _finite_tensor_summary(value)
        element_count += int(summary["element_count"])
        finite_count += int(summary["finite_count"])
        nan_count += int(summary["nan_count"])
        posinf_count += int(summary["posinf_count"])
        neginf_count += int(summary["neginf_count"])
        if not bool(summary["finite"]):
            nonfinite_tensor_count += 1
            if len(nonfinite_names) < limit:
                nonfinite_names.append(str(name))
    return {
        "tensor_count": tensor_count,
        "element_count": element_count,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "finite": bool(
            nan_count == 0 and posinf_count == 0 and neginf_count == 0
        ),
        "nonfinite_names": nonfinite_names,
        "nonfinite_tensor_count": nonfinite_tensor_count,
    }


def _model_parameter_health(model: Any) -> dict[str, Any]:
    return _summarize_named_tensors(model.named_parameters())


def _model_gradient_health(model: Any) -> dict[str, Any]:
    return _summarize_named_tensors(
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    )


def _optimizer_state_health(optimizer: Any, model: Any) -> dict[str, Any]:
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    tensors: list[tuple[str, Any]] = []
    for parameter, state in optimizer.state.items():
        parameter_name = parameter_names.get(id(parameter), "<unregistered>")
        for state_name, value in state.items():
            tensors.append((f"{parameter_name}.{state_name}", value))
    return _summarize_named_tensors(tensors)


def _loss_health(losses: Mapping[str, Any]) -> dict[str, Any]:
    summary = {
        str(name): _finite_tensor_summary(value) for name, value in losses.items()
    }
    return {
        "components": summary,
        "cost_finite": bool(summary.get("cost", {}).get("finite", False)),
    }


def _output_tensor_health(value: Any) -> dict[str, Any]:
    import torch

    tensors: list[tuple[str, Any]] = []

    def visit(item: Any, prefix: str) -> None:
        if torch.is_tensor(item):
            tensors.append((prefix, item))
        elif isinstance(item, Mapping):
            for key, nested in item.items():
                visit(nested, f"{prefix}.{key}")
        elif isinstance(item, (list, tuple)):
            for index, nested in enumerate(item):
                visit(nested, f"{prefix}[{index}]")

    visit(value, "output")
    return _summarize_named_tensors(tensors)


def _critical_forward_health(model: Any) -> tuple[dict[str, Any], list[Any]]:
    """Attach non-mutating hooks to the coarse-to-detector boundaries."""

    targets = {
        "backbone": getattr(model, "backbone", None),
        "frame_selector": getattr(model, "frame_selector", None),
        "projection": getattr(model, "projection", None),
        "neck": getattr(model, "neck", None),
        "rpn_head": getattr(model, "rpn_head", None),
    }
    recorded: dict[str, Any] = {}
    handles: list[Any] = []
    for name, module in targets.items():
        if module is None or not callable(getattr(module, "register_forward_hook", None)):
            continue

        def hook(_module: Any, _inputs: Any, output: Any, *, name: str = name) -> None:
            recorded[name] = _output_tensor_health(output)

        handles.append(module.register_forward_hook(hook))
    return recorded, handles


def _remove_hooks(handles: Iterable[Any]) -> None:
    for handle in handles:
        handle.remove()


def _load_exact_trial_state(
    *,
    model: Any,
    checkpoint: Mapping[str, Any],
    cfg: Any,
    train_loader_length: int,
    logger: logging.Logger,
) -> tuple[Any, Any, Any, Any]:
    """Restore the mutable post-epoch-9 state only in RAM."""

    import torch
    from torch.cuda.amp import GradScaler

    from opentad.cores import build_optimizer, build_scheduler
    from opentad.utils import ModelEma

    state_dict = checkpoint.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks state_dict")
    incompatible = model.load_state_dict(_normalize_state_dict(state_dict), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-2 model strict load did not close")

    selector = getattr(model, "frame_selector", None)
    if selector is None:
        raise RuntimeError("Stage-2 model lacks a DUCA frame selector")
    selector._pending_loss_schedule_advance = False

    # build_optimizer intentionally consumes cfg keys; use the same deep-copy
    # convention as tools/train.py and expose only the .module facade it needs.
    optimizer_model = SimpleNamespace(module=model)
    optimizer = build_optimizer(copy.deepcopy(cfg.optimizer), optimizer_model, logger)
    optimizer_state = checkpoint.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks optimizer state")
    optimizer.load_state_dict(optimizer_state)

    scheduler, _ = build_scheduler(
        copy.deepcopy(cfg.scheduler), optimizer, int(train_loader_length)
    )
    scheduler_state = checkpoint.get("scheduler")
    if not isinstance(scheduler_state, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks scheduler state")
    scheduler.load_state_dict(scheduler_state)

    if not bool(cfg.solver.get("amp", False)):
        raise RuntimeError("post-update diagnostic requires Stage-2 AMP")
    scaler_state = checkpoint.get("grad_scaler")
    if not isinstance(scaler_state, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks GradScaler state")
    scaler = GradScaler()
    scaler.load_state_dict(scaler_state)

    if not bool(cfg.solver.get("ema", False)):
        raise RuntimeError("post-update diagnostic requires Stage-2 EMA")
    ema_state = checkpoint.get("state_dict_ema")
    if not isinstance(ema_state, Mapping):
        raise RuntimeError("Stage-2 checkpoint lacks EMA state")
    model_ema = ModelEma(model)
    incompatible = model_ema.module.load_state_dict(
        _normalize_state_dict(ema_state), strict=True
    )
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage-2 EMA strict load did not close")
    return optimizer, scheduler, scaler, model_ema


def _call_after_optimizer_step(model: Any) -> Any:
    """Use the training engine's selector-step mutation without a train loop."""

    from opentad.cores.train_engine import _call_after_optimizer_step

    return _call_after_optimizer_step(model)


def _replay_one_trial(
    *,
    model: Any,
    checkpoint: Mapping[str, Any],
    cfg: Any,
    train_loader_length: int,
    logger: logging.Logger,
    batch_zero: Mapping[str, Any],
    batch_one: Mapping[str, Any],
    device: Any,
    seed: int,
) -> dict[str, Any]:
    import torch

    from opentad.utils import set_seed

    # The original checkpoint did not serialize the pre-epoch-10 RNG state.
    # Each controlled trial therefore starts from an explicit seed and shares
    # the same frozen pair of loader batches.
    set_seed(int(seed), False)
    optimizer, scheduler, scaler, model_ema = _load_exact_trial_state(
        model=model,
        checkpoint=checkpoint,
        cfg=cfg,
        train_loader_length=train_loader_length,
        logger=logger,
    )
    selector = model.frame_selector
    selector_step_before = int(selector._loss_weight_schedule_step.detach().item())
    scheduler_step_before = int(scheduler.last_epoch)
    scale_before = float(scaler.get_scale())
    result: dict[str, Any] = {
        "seed": int(seed),
        "selector_step_before": selector_step_before,
        "scheduler_step_before": scheduler_step_before,
        "grad_scaler_scale_before": scale_before,
        "initial_parameter_health": _model_parameter_health(model),
    }
    if not result["initial_parameter_health"]["finite"]:
        result["outcome"] = "checkpoint_state_nonfinite"
        return result

    optimizer.zero_grad()
    batch_zero_gpu = _move_to_device(batch_zero, device)
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            losses_zero = model(**batch_zero_gpu, return_loss=True)
    except Exception as exc:
        result.update(
            {
                "outcome": "batch_zero_forward_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return result
    if not isinstance(losses_zero, Mapping):
        result["outcome"] = "batch_zero_non_mapping_loss"
        return result
    result["batch_zero_losses"] = _loss_health(losses_zero)
    if not result["batch_zero_losses"]["cost_finite"]:
        result["outcome"] = "batch_zero_nonfinite_cost"
        return result

    scaler.scale(losses_zero["cost"]).backward()
    scaler.unscale_(optimizer)
    clip_norm = float(cfg.solver.get("clip_grad_norm", -1))
    if clip_norm > 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    result["batch_zero_gradient_health"] = _model_gradient_health(model)

    scale_before_step = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after_step = float(scaler.get_scale())
    optimizer_step_ran = _optimizer_step_ran(scale_before_step, scale_after_step)
    result["batch_zero_amp_step"] = {
        "scale_before": scale_before_step,
        "scale_after": scale_after_step,
        "optimizer_step_ran": optimizer_step_ran,
    }
    if not optimizer_step_ran:
        result["outcome"] = "batch_zero_amp_overflow"
        return result

    result["post_batch_zero_parameter_health"] = _model_parameter_health(model)
    result["post_batch_zero_optimizer_health"] = _optimizer_state_health(optimizer, model)
    schedule_update = _call_after_optimizer_step(model)
    scheduler.step()
    model_ema.update(model)
    result["post_batch_zero_state"] = {
        "selector_step_after": int(selector._loss_weight_schedule_step.detach().item()),
        "scheduler_step_after": int(scheduler.last_epoch),
        "schedule_update": schedule_update,
    }
    if not result["post_batch_zero_parameter_health"]["finite"]:
        result["outcome"] = "post_batch_zero_parameter_nonfinite"
        return result
    if not result["post_batch_zero_optimizer_health"]["finite"]:
        result["outcome"] = "post_batch_zero_optimizer_nonfinite"
        return result

    batch_one_gpu = _move_to_device(batch_one, device)
    critical_health, hooks = _critical_forward_health(model)
    try:
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            losses_one = model(**batch_one_gpu, return_loss=True)
    except Exception as exc:
        result.update(
            {
                "outcome": "batch_one_forward_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "batch_one_critical_forward_health": critical_health,
            }
        )
        return result
    finally:
        _remove_hooks(hooks)
    if not isinstance(losses_one, Mapping):
        result["outcome"] = "batch_one_non_mapping_loss"
        return result
    result["batch_one_critical_forward_health"] = critical_health
    result["batch_one_losses"] = _loss_health(losses_one)
    result["outcome"] = (
        "batch_one_finite" if result["batch_one_losses"]["cost_finite"] else "batch_one_nonfinite_cost"
    )
    return result


def run_diagnostic(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    expected_checkpoint_epoch: int,
    expected_commit: str,
    pretrain_path: str | Path,
    stage1_checkpoint: str | Path,
    stage1_checkpoint_sha256: str,
    stage1_checkpoint_epoch: int,
    epoch: int,
    seed: int,
    trials: int,
    device_name: str,
) -> dict[str, Any]:
    """Replay batch 0's exact update and inspect the next batch in memory."""

    import torch
    from mmengine.config import Config

    from opentad.cores import prepare_optimizer_parameter_freezing
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed
    from tools.bata.duca_frontend_initialization import initialize_model_from_checkpoint

    if int(trials) <= 0:
        raise ValueError("trials must be positive")
    repo_root = Path(__file__).resolve().parents[2]
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
    ).strip()
    if git_commit != expected_commit:
        raise RuntimeError("checkout commit differs from --expected-commit")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=repo_root,
        text=True,
    ).strip():
        raise RuntimeError("post-update diagnostic requires a clean exact checkout")

    config = Path(config_path).expanduser().resolve()
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    pretrain = Path(pretrain_path).expanduser().resolve()
    stage1 = Path(stage1_checkpoint).expanduser().resolve()
    if not config.is_file() or not pretrain.is_file() or not stage1.is_file():
        raise FileNotFoundError("config, pretrain, and Stage-1 checkpoint must exist")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Stage-2 checkpoint is missing: {checkpoint}")
    checkpoint_sha_before = sha256_file(checkpoint)
    if checkpoint_sha_before != checkpoint_sha256.lower():
        raise RuntimeError("Stage-2 checkpoint SHA256 mismatch")

    os.environ["DUCA_STAGE1_CHECKPOINT"] = str(stage1)
    os.environ["DUCA_STAGE1_CHECKPOINT_SHA256"] = stage1_checkpoint_sha256.lower()
    os.environ["DUCA_STAGE1_CHECKPOINT_EPOCH"] = str(int(stage1_checkpoint_epoch))
    set_seed(int(seed), False)
    device = torch.device(device_name)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("one CUDA device is required")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("diagnostic requires exactly one Slurm-visible GPU")
    torch.cuda.set_device(device)

    cfg = Config.fromfile(str(config))
    cfg.model.backbone.custom.pretrain = str(pretrain)
    logger = logging.getLogger("duca-stage2-postupdate-nonfinite-diagnostic")
    dataset = build_dataset(cfg.dataset.train, default_args={"logger": logger})
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        **cfg.solver.train,
    )
    set_dataset_epoch = getattr(dataset, "set_epoch", None)
    if bool(cfg.workflow.get("derive_train_loader_contract", False)):
        if not callable(set_dataset_epoch):
            raise RuntimeError("Stage-2 training dataset lacks set_epoch")
        set_dataset_epoch(int(epoch))
    loader.sampler.set_epoch(int(epoch))
    iterator = iter(loader)
    try:
        batch_zero = next(iterator)
        batch_one = next(iterator)
    except StopIteration as exc:
        raise RuntimeError("Stage-2 loader lacks the first two training batches") from exc

    model = build_detector(cfg.model)
    stage1_receipt = initialize_model_from_checkpoint(
        model, cfg.workflow.get("model_initialization"), logger=logger
    )
    if stage1_receipt is None:
        raise RuntimeError("diagnostic requires strict Stage-1 model initialization")
    prepare_optimizer_parameter_freezing(cfg.optimizer, model, logger)
    model = model.to(device).train()
    checkpoint_payload = _load_checkpoint(checkpoint)
    if int(checkpoint_payload.get("epoch", -1)) != int(expected_checkpoint_epoch):
        raise RuntimeError("Stage-2 checkpoint epoch mismatch")
    if "rng_state" in checkpoint_payload:
        rng_status = "checkpoint_rng_present_but_not_consumed_by_this_diagnostic"
    else:
        rng_status = "checkpoint_rng_absent_controlled_trial_seeds_required"

    trial_results = []
    for trial_index in range(int(trials)):
        trial_seed = int(seed) + trial_index
        result = _replay_one_trial(
            model=model,
            checkpoint=checkpoint_payload,
            cfg=cfg,
            train_loader_length=len(loader),
            logger=logger,
            batch_zero=batch_zero,
            batch_one=batch_one,
            device=device,
            seed=trial_seed,
        )
        result["trial_index"] = trial_index
        trial_results.append(result)
        torch.cuda.empty_cache()

    checkpoint_sha_after = sha256_file(checkpoint)
    if checkpoint_sha_after != checkpoint_sha_before:
        raise RuntimeError("in-memory diagnostic changed its input checkpoint")
    outcomes: dict[str, int] = {}
    for result in trial_results:
        outcome = str(result.get("outcome", "missing_outcome"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "ok": bool(outcomes.get("batch_one_finite", 0) == int(trials)),
        "task": "offline_temporal_action_detection",
        "mode": "in_memory_epoch10_batch0_optimizer_update_then_batch1_forward",
        "git_commit": git_commit,
        "config_path": str(config),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha_before,
        "checkpoint_sha256_after": checkpoint_sha_after,
        "checkpoint_epoch": int(checkpoint_payload["epoch"]),
        "checkpoint_state_key": "state_dict",
        "stage1_initialization": stage1_receipt,
        "replayed_epoch": int(epoch),
        "replayed_batch_indices": [0, 1],
        "trials": int(trials),
        "seed_base": int(seed),
        "rng_reconstruction": rng_status,
        "checkpoint_mutation": False,
        "optimizer_persistence": False,
        "scheduler_persistence": False,
        "ema_persistence": False,
        "terminal_evaluation": False,
        "outcomes": outcomes,
        "trial_results": trial_results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="In-memory Stage-2 post-update non-finite diagnosis."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-checkpoint-epoch", required=True, type=int)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--stage1-checkpoint-sha256", required=True)
    parser.add_argument("--stage1-checkpoint-epoch", required=True, type=int)
    parser.add_argument("--epoch", required=True, type=int)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_json).expanduser().resolve()
    try:
        report = run_diagnostic(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            expected_checkpoint_epoch=args.expected_checkpoint_epoch,
            expected_commit=args.expected_commit,
            pretrain_path=args.pretrain,
            stage1_checkpoint=args.stage1_checkpoint,
            stage1_checkpoint_sha256=args.stage1_checkpoint_sha256,
            stage1_checkpoint_epoch=args.stage1_checkpoint_epoch,
            epoch=args.epoch,
            seed=args.seed,
            trials=args.trials,
            device_name=args.device,
        )
    except Exception as exc:
        report = {
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "task": "offline_temporal_action_detection",
            "mode": "in_memory_epoch10_batch0_optimizer_update_then_batch1_forward",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        _atomic_write_json(output, report)
        raise
    _atomic_write_json(output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
