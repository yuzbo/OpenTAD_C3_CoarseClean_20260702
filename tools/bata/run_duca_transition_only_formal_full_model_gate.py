from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

from opentad.cores.optimizer import (
    assert_optimizer_exact_coverage,
    build_optimizer,
    prepare_optimizer_parameter_freezing,
)
from opentad.models import build_detector
from opentad.models.duca.structured_selection import global_structured_topk
from opentad.models.duca.transition_only import continuous_policy_logits


CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"formal DUCA full-model gate failed: {message}")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _grad_sum(module: torch.nn.Module) -> float:
    return float(
        sum(
            parameter.grad.detach().abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )
    )


def _named_grad_sum(module: torch.nn.Module, predicate: Callable[[str], bool]) -> float:
    return float(
        sum(
            parameter.grad.detach().abs().sum().item()
            for name, parameter in module.named_parameters()
            if predicate(name) and parameter.grad is not None
        )
    )


def _finite_scalar(value: torch.Tensor) -> float:
    scalar = float(value.detach().float().cpu().item())
    _require(torch.isfinite(value.detach()).all().item(), "encountered a non-finite scalar")
    return scalar


def _representative_parameters(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    groups = {
        "selector": lambda name: name.startswith("frame_selector."),
        "backbone": lambda name: name.startswith("backbone."),
        "detector_head": lambda name: name.startswith("rpn_head."),
    }
    named = list(model.named_parameters())
    selected: dict[str, torch.nn.Parameter] = {}
    for group, predicate in groups.items():
        candidates = [
            parameter
            for name, parameter in named
            if predicate(name)
            and parameter.requires_grad
            and parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            and float(parameter.grad.detach().abs().sum().item()) > 0.0
        ]
        if candidates:
            selected[group] = candidates[0]
    _require(selected, "no finite non-zero trainable gradient was available for optimizer-step proof")
    return selected


def _max_hole(positions: torch.Tensor, temporal_len: int) -> int:
    values = [int(value) for value in positions.detach().cpu().tolist()]
    holes = [values[0]]
    holes.extend(right - left - 1 for left, right in zip(values[:-1], values[1:]))
    holes.append(int(temporal_len) - values[-1] - 1)
    return max(holes)


def _verify_exact_uniform_reference(
    *,
    temporal_len: int,
    budget: int,
    max_unselected_hole: int,
    device: str,
) -> dict[str, Any]:
    learned_scores = torch.zeros((1, temporal_len), dtype=torch.float32, device=device)
    valid_mask = torch.ones_like(learned_scores, dtype=torch.bool)
    reference_logits = continuous_policy_logits(
        learned_scores,
        valid_mask,
        k=budget,
        alpha=0.0,
    )
    selection = global_structured_topk(
        reference_logits,
        k=budget,
        max_unselected_hole=max_unselected_hole,
        training=False,
    )
    actual = selection.selected_positions[0]
    expected = torch.linspace(
        0,
        temporal_len - 1,
        steps=budget,
        device=device,
        dtype=torch.float32,
    ).round().to(dtype=torch.long)
    max_rank_error = int((actual - expected).abs().max().item()) if budget else 0
    exact = bool(torch.equal(actual, expected))
    _require(exact, "uniform reference does not decode to exact round-linspace positions")
    return {
        "uniform_reference_definition": "round_linspace_endpoints",
        "uniform_reference_exact": exact,
        "uniform_reference_max_rank_error": max_rank_error,
    }


def _real_detector_losses(model, selector_outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
    _require(model.token_compressor is None, "formal AdaTAD gate does not expect a token compressor")
    _require(model.pc_ot_mras_reader is None, "formal AdaTAD gate does not expect a PC-OT reader")
    inputs = selector_outputs["inputs"]
    masks = selector_outputs["masks"]
    metas = selector_outputs["metas"]
    gt_segments = selector_outputs["gt_segments"]
    gt_labels = selector_outputs["gt_labels"]

    features = model.backbone(inputs) if model.with_backbone else inputs
    model._assert_feature_mask_temporal_match(features, masks, "formal gate before projection")
    features, masks = model.pad_data(features, masks)
    if model.with_projection:
        features, masks = model.projection(features, masks)
    if model.with_neck:
        features, masks, metas = model._call_neck_forward(features, masks, metas=metas)
    return model._call_rpn_head_forward_train(
        features,
        masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )


def run_formal_gate(
    config_path: str = CONFIG_DEFAULT,
    *,
    checkpoint_path: str,
    official_repos_root: str,
    device: str = "cuda",
) -> dict[str, Any]:
    _require(device.startswith("cuda"), "the formal full-model gate requires CUDA")
    _require(torch.cuda.is_available(), "CUDA is unavailable")
    checkpoint = Path(checkpoint_path).expanduser().resolve()
    source = Path(official_repos_root).expanduser().resolve() / "ASFormer" / "model.py"
    config_file = Path(config_path).expanduser().resolve()
    _require(checkpoint.is_file(), f"AdaTAD checkpoint is missing: {checkpoint}")
    _require(source.is_file(), f"official ASFormer source is missing: {source}")
    _require(config_file.is_file(), f"config is missing: {config_file}")
    os.environ["C3_OFFICIAL_ACTION_SEG_REPOS"] = str(Path(official_repos_root).expanduser().resolve())

    cfg = Config.fromfile(str(config_file))
    dense_window_size = int(cfg.dense_window_size)
    budget = int(cfg.window_size)
    spatial_size = 160
    _require(dense_window_size == 768, "formal selector input must be T=768")
    _require(budget == 384, "formal detector input must be K=384")
    _require(int(cfg.model.backbone.backbone.total_frames) == budget, "VideoMAE total_frames must be 384")
    _require(int(cfg.model.projection.max_seq_len) == budget, "projection max_seq_len must be 384")
    uniform_reference = _verify_exact_uniform_reference(
        temporal_len=dense_window_size,
        budget=budget,
        max_unselected_hole=int(cfg.model.frame_selector.max_unselected_hole),
        device=device,
    )

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = str(checkpoint)
    model = build_detector(model_cfg)
    _require(model.__class__.__name__ == "ActionFormer", "formal detector is not ActionFormer")
    _require(model.rpn_head.__class__.__name__ == "ActionFormerHead", "formal head is not ActionFormerHead")
    _require(
        model.backbone.__class__.__name__ != "_ProofTemporalMeanBackbone",
        "formal gate replaced the real VideoMAE backbone",
    )

    logger = logging.getLogger("duca-formal-gate")
    optimizer_cfg = copy.deepcopy(cfg.optimizer)
    prepare_optimizer_parameter_freezing(optimizer_cfg, model, logger)
    model.to(device)
    optimizer = build_optimizer(
        copy.deepcopy(cfg.optimizer), SimpleNamespace(module=model), logger
    )
    assert_optimizer_exact_coverage(model, optimizer)
    model.train()
    selector = model.frame_selector
    selector._loss_weight_schedule_step.fill_(7920)

    torch.manual_seed(20260711)
    inputs = torch.randint(
        0,
        256,
        (1, 1, 3, dense_window_size, spatial_size, spatial_size),
        dtype=torch.uint8,
        device=device,
    )
    masks = torch.ones(1, dense_window_size, dtype=torch.bool, device=device)
    metas = [{"video_name": "duca_transition_only_formal_full_model_gate"}]
    gt_segments = [torch.tensor([[192.0, 560.0]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([1], dtype=torch.long, device=device)]

    optimizer.zero_grad(set_to_none=True)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scale_before_backward = float(scaler.get_scale())
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        detector_losses = model(
            inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, return_loss=True
        )
        _require("cls_loss" in detector_losses, "real ActionFormerHead cls_loss is missing")
        _require("reg_loss" in detector_losses, "real ActionFormerHead reg_loss is missing")
        detector_only_loss = detector_losses["cls_loss"] + detector_losses["reg_loss"]
        counterfactual_loss = detector_losses["counterfactual_utility_distillation_loss"]
    scaler.scale(counterfactual_loss).backward(retain_graph=True)

    counterfactual_gradients = {
        "coarse_probe": _grad_sum(selector.raw_actionness_source),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
    }
    _require(counterfactual_gradients["transition_scorer"] > 0.0, "counterfactual teacher did not train scorer")
    _require(counterfactual_gradients["coarse_probe"] == 0.0, "counterfactual teacher leaked into coarse probe")
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(detector_only_loss).backward()
    scaler.unscale_(optimizer)

    positions = selector._last_selected_positions[0]
    selected_count = int(positions.numel())
    max_hole = _max_hole(positions, dense_window_size)
    gradients = {
        "coarse_probe": _grad_sum(selector.raw_actionness_source),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
        "backbone_adapter": _named_grad_sum(model.backbone, lambda name: "adapter" in name),
        "detector_head": _grad_sum(model.rpn_head),
    }
    _require(selected_count == budget, "selector did not emit exact K=384")
    _require(max_hole <= int(selector.max_unselected_hole), "selector violated the max-gap contract")
    _require(gradients["transition_scorer"] == 0.0, "detector loss leaked through removed direct bridge")
    _require(gradients["coarse_probe"] == 0.0, "real detector losses leaked into protected coarse probe")
    _require(gradients["backbone_adapter"] > 0.0, "real detector losses did not train VideoMAE adapters")
    _require(gradients["detector_head"] > 0.0, "real detector losses did not train ActionFormerHead")
    alignment = dict(getattr(selector, "last_counterfactual_summary", {}))
    _require(alignment.get("finite") is True, "counterfactual utility is non-finite")
    gradient_alignment = dict(alignment.get("distillation_gradient_alignment", {}))
    _require(
        all(
            math.isfinite(float(gradient_alignment.get(key, float("nan"))))
            for key in ("sign_agreement", "spearman")
        ),
        "counterfactual distillation gradient diagnostics are non-finite",
    )

    # Run the same aggregate loss used by the training loop, then prove that AMP
    # executes an optimizer update instead of silently skipping it.
    # The detector-only diagnostic above already called unscale_ on its scaler;
    # a real training step must start with a fresh scaler state machine.
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    scale_before_backward = float(scaler.get_scale())
    optimizer.zero_grad(set_to_none=True)
    normalizer_before = model.rpn_head.loss_normalizer.detach().clone()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        step_losses = model(
            inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, return_loss=True
        )
        _require("cost" in step_losses, "formal training aggregate loss 'cost' is missing")
        step_loss = step_losses["cost"]
    step_loss_value = _finite_scalar(step_loss)
    _require(step_loss_value >= 0.0, "formal training aggregate loss is negative")
    normalizer_after_forward = model.rpn_head.loss_normalizer.detach().clone()
    _require(torch.isfinite(normalizer_after_forward).all().item(), "loss normalizer became non-finite")
    _require(float(normalizer_after_forward.item()) > 0.0, "loss normalizer became non-positive")
    _require(
        not torch.equal(normalizer_before, normalizer_after_forward),
        "training forward did not update the ActionFormerHead EMA loss normalizer",
    )

    scaler.scale(step_loss).backward()
    scaler.unscale_(optimizer)
    named_trainable_gradients = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    trainable_gradients = [gradient for _, gradient in named_trainable_gradients]
    _require(trainable_gradients, "formal optimizer step has no gradients")
    nonfinite_gradient_names = [
        name for name, gradient in named_trainable_gradients
        if not bool(torch.isfinite(gradient).all().item())
    ]
    _require(
        not nonfinite_gradient_names,
        "formal optimizer step has non-finite gradients: " + ", ".join(nonfinite_gradient_names[:20]),
    )
    representatives = _representative_parameters(model)
    parameters_before_step = {
        name: parameter.detach().clone() for name, parameter in representatives.items()
    }
    scale_before_step = float(scaler.get_scale())
    scaler.step(optimizer)
    scaler.update()
    scale_after_step = float(scaler.get_scale())
    parameter_changes = {
        name: float((parameter.detach() - parameters_before_step[name]).abs().max().item())
        for name, parameter in representatives.items()
    }
    changed_groups = [name for name, delta in parameter_changes.items() if delta > 0.0]
    _require(changed_groups, "optimizer.step produced no trainable parameter change")
    _require(scale_after_step > 0.0, "GradScaler entered an invalid state")
    normalizer_after_step = model.rpn_head.loss_normalizer.detach().clone()
    _require(
        torch.equal(normalizer_after_forward, normalizer_after_step),
        "optimizer.step unexpectedly modified the non-parameter loss normalizer",
    )

    return {
        "ok": True,
        "formal_proof_ok": True,
        "git_commit": _git_commit(),
        "config_path": str(config_file),
        "reference_config_sha256": _sha256(config_file),
        "official_asformer_source_path": str(source),
        "official_asformer_source_sha256": _sha256(source),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "model_type": model.__class__.__name__,
        "backbone_type": model.backbone.__class__.__name__,
        "detector_head_type": model.rpn_head.__class__.__name__,
        "dense_window_size": dense_window_size,
        "selected_count": selected_count,
        "input_spatial_size": spatial_size,
        "max_unselected_hole_observed": max_hole,
        "detector_only_loss_keys": ["cls_loss", "reg_loss"],
        "detector_only_loss": float(detector_only_loss.detach().float().cpu().item()),
        "detector_only_gradients": gradients,
        "counterfactual_gradients": counterfactual_gradients,
        "counterfactual_alignment": alignment,
        "optimizer_exact_coverage": True,
        "amp": True,
        "grad_scaler_enabled": True,
        "grad_scale": scale_before_backward,
        "optimizer_step_ran": True,
        "optimizer_parameter_change_verified": True,
        "optimizer_changed_parameter_groups": changed_groups,
        "optimizer_parameter_max_abs_change": parameter_changes,
        "optimizer_step_loss": step_loss_value,
        "optimizer_step_loss_finite": True,
        "optimizer_step_gradients_finite": True,
        "grad_scale_before_step": scale_before_step,
        "grad_scale_after_step": scale_after_step,
        "loss_normalizer_contract": {
            "state_kind": "ActionFormerHead.loss_normalizer_ema_buffer",
            "finite": True,
            "positive": True,
            "updated_by_training_forward": True,
            "unchanged_by_optimizer_step": True,
            "before_forward": float(normalizer_before.float().cpu().item()),
            "after_forward": float(normalizer_after_forward.float().cpu().item()),
            "after_optimizer_step": float(normalizer_after_step.float().cpu().item()),
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        **uniform_reference,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--official-repos-root", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = run_formal_gate(
            args.config,
            checkpoint_path=args.checkpoint,
            official_repos_root=args.official_repos_root,
            device=args.device,
        )
    except Exception as exc:
        summary = {"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)}
        code = 1
    else:
        code = 0
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
