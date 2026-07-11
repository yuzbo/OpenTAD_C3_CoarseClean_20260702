from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

from opentad.cores.optimizer import assert_optimizer_exact_coverage
from opentad.models import build_detector
from tools.bata.run_duca_official_adatad_one_step_grad_proof import (
    _ProofTemporalMeanBackbone,
    _grad_sum,
    _plain,
    _scaled_model_cfg,
)


CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"transition-only official proof failed: {message}")


def _scaled_selector_cfg(cfg: Config, *, temporal_len: int, budget: int, hidden_dim: int, spatial_size: int) -> dict[str, Any]:
    selector = _plain(cfg.model.frame_selector)
    selector.update(
        {
            "dense_window_size": int(temporal_len),
            "budget": int(budget),
            "selector_hidden_channels": int(hidden_dim),
            "coarse_hidden_dim": int(hidden_dim),
            "max_unselected_hole": 1,
            "max_gap_loss_max_unselected_hole": 1,
            "soft_max_gap_loss_enabled": False,
            "profile_runtime": False,
            "profile_sync_cuda": False,
        }
    )
    source = selector["actionness_source_cfg"]
    source.update(
        {
            "spatial_size": int(spatial_size),
            "tcn_hidden_dim": int(hidden_dim),
            "official_num_layers": 1,
            "checkpoint_path": "",
            "require_checkpoint": False,
            "frozen": False,
            "trainable": True,
        }
    )
    selector["loss_weight_schedule"] = {
        "type": "progressive_joint",
        "shape": "linear",
        "warmup_steps": 0,
        "transition_steps": 1,
        "actionness": {"start": 1.0, "end": 1.0},
        "transition": {"start": 0.5, "end": 0.5},
        "transition_boundary": {"start": 0.25, "end": 0.25},
        "detector_gradient": {"start": 0.25, "end": 0.25},
        "policy_alpha": {"start": 1.0, "end": 1.0},
    }
    return selector


def _zero_grad(model) -> None:
    model.zero_grad(set_to_none=True)


def _max_hole(positions: torch.Tensor, temporal_len: int) -> int:
    values = positions.detach().cpu().tolist()
    holes = [int(values[0])]
    holes.extend(int(right - left - 1) for left, right in zip(values[:-1], values[1:]))
    holes.append(int(temporal_len - values[-1] - 1))
    return max(holes)


def run_proof(
    config_path: str = CONFIG_DEFAULT,
    *,
    temporal_len: int = 16,
    budget: int = 8,
    hidden_dim: int = 16,
    feature_dim: int = 16,
    spatial_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg = Config.fromfile(config_path)
    contract = cfg.duca_transition_only_contract
    _require(contract.official_adatad_backend is True, "official AdaTAD contract is missing")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "config does not use ActionFormerHead")
    selector_cfg = _scaled_selector_cfg(
        cfg,
        temporal_len=int(temporal_len),
        budget=int(budget),
        hidden_dim=int(hidden_dim),
        spatial_size=int(spatial_size),
    )
    model_cfg = _scaled_model_cfg(
        cfg,
        selector_cfg=selector_cfg,
        proof_budget=int(budget),
        proof_feature_dim=int(feature_dim),
    )
    model = build_detector(model_cfg)
    model.backbone = _ProofTemporalMeanBackbone(in_channels=3, out_channels=int(feature_dim))
    model.to(device)
    _require(model.__class__.__name__ == "ActionFormer", "built detector is not ActionFormer")
    _require(model.rpn_head.__class__.__name__ == "ActionFormerHead", "built head is not ActionFormerHead")

    selector = model.frame_selector
    _require(selector.adapter.transition_scorer is not None, "shared transition scorer is missing")
    for name in ("center_head", "start_head", "end_head", "context_head", "utility_head", "radius_head"):
        _require(getattr(selector.adapter, name) is None, f"legacy direct head {name} is still enabled")

    groups = model.get_optim_groups({"lr": 1.0e-4, "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(groups, lr=1.0e-4)
    assert_optimizer_exact_coverage(model, optimizer)
    lr_by_param = {id(param): float(group["lr"]) for group in groups for param in group["params"]}
    component_lrs = {"coarse_trunk": set(), "action_head": set(), "transition_scorer": set()}
    for name, param in model.named_parameters():
        if not name.startswith("frame_selector.") or not param.requires_grad:
            continue
        if name.startswith("frame_selector.adapter.transition_scorer."):
            component_lrs["transition_scorer"].add(lr_by_param[id(param)])
        elif name.startswith("frame_selector.raw_actionness_source.probe_module."):
            tail = name.split(".official_temporal.", 1)[1] if ".official_temporal." in name else ""
            parts = tail.split(".")
            action_head = tail.startswith("encoder.conv_out.") or (
                len(parts) >= 4 and parts[0] == "decoders" and parts[2] == "conv_out"
            )
            component_lrs["action_head" if action_head else "coarse_trunk"].add(lr_by_param[id(param)])
    expected_lrs = {
        "coarse_trunk": {2.5e-5},
        "action_head": {5.0e-5},
        "transition_scorer": {1.0e-4},
    }
    _require(component_lrs == expected_lrs, f"component LR contract mismatch: {component_lrs}")

    torch.manual_seed(20260711)
    inputs = torch.randn(1, 3, int(temporal_len), int(spatial_size), int(spatial_size), device=device)
    masks = torch.ones(1, int(temporal_len), dtype=torch.bool, device=device)
    metas = [{"video_name": "duca_transition_only_official_proof"}]
    gt_segments = [torch.tensor([[3.0, float(temporal_len - 4)]], dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor([1], dtype=torch.long, device=device)]

    model.train()
    selector._loss_weight_schedule_step.fill_(1)
    losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    for key in (
        "cost",
        "cls_loss",
        "reg_loss",
        "actionness_bce_loss",
        "transition_distribution_loss",
        "transition_boundary_coverage_loss",
    ):
        _require(key in losses, f"required loss {key!r} is missing")
    _zero_grad(model)
    losses["cost"].backward()
    cost_gradients = {
        "coarse_probe": _grad_sum(selector.raw_actionness_source),
        "asformer_encoder": _grad_sum(selector.raw_actionness_source.probe.official_temporal.encoder),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
        "detector_head": _grad_sum(model.rpn_head),
    }
    for key, value in cost_gradients.items():
        _require(value is not None and float(value) > 0.0, f"cost backward gave no gradient to {key}")

    _zero_grad(model)
    transition_outputs = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    transition_outputs["losses"]["transition_distribution_loss"].backward()
    transition_gradients = {
        "spatial_stem": _grad_sum(selector.raw_actionness_source.probe.spatial_stem),
        "asformer_encoder": _grad_sum(selector.raw_actionness_source.probe.official_temporal.encoder),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
        "asformer_encoder_action_head": _grad_sum(
            selector.raw_actionness_source.probe.official_temporal.encoder.conv_out
        ),
        "asformer_decoder": _grad_sum(selector.raw_actionness_source.probe.official_temporal.decoders),
    }
    for key in ("spatial_stem", "asformer_encoder", "transition_scorer"):
        _require(float(transition_gradients[key] or 0.0) > 0.0, f"transition loss gave no gradient to {key}")
    _require(float(transition_gradients["asformer_encoder_action_head"] or 0.0) == 0.0, "transition loss leaked into action head")
    _require(float(transition_gradients["asformer_decoder"] or 0.0) == 0.0, "transition loss leaked into ASFormer decoder")

    _zero_grad(model)
    coverage_outputs = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    coverage_outputs["losses"]["transition_boundary_coverage_loss"].backward()
    coverage_gradients = {
        "coarse_probe": _grad_sum(selector.raw_actionness_source),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
    }
    _require(float(coverage_gradients["transition_scorer"] or 0.0) > 0.0, "coverage loss did not train scorer")
    _require(float(coverage_gradients["coarse_probe"] or 0.0) == 0.0, "coverage loss leaked into coarse probe")

    _zero_grad(model)
    detector_route = selector.forward_train(
        inputs=inputs,
        masks=masks,
        metas=metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
    )
    detector_losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    detector_only_loss = detector_losses["cls_loss"] + detector_losses["reg_loss"]
    detector_only_loss.backward()
    detector_route_gradients = {
        "coarse_probe": _grad_sum(selector.raw_actionness_source),
        "transition_scorer": _grad_sum(selector.adapter.transition_scorer),
    }
    _require(float(detector_route_gradients["transition_scorer"] or 0.0) > 0.0, "detector route did not train scorer")
    _require(float(detector_route_gradients["coarse_probe"] or 0.0) == 0.0, "detector route leaked into coarse probe")

    grid = detector_route["selector_outputs"]["grid"]
    positions = grid.selected_positions[0]
    _require(int(positions.numel()) == int(budget), "hard policy did not select exact K")
    _require(_max_hole(positions, int(temporal_len)) <= 1, "hard policy violated max-gap")
    _require(
        detector_route["selector_outputs"]["coarse_hidden_kind"] == "official_asformer_encoder_hidden",
        "selector did not receive official ASFormer encoder hidden",
    )

    optimizer.zero_grad(set_to_none=True)
    losses = model(
        inputs,
        masks,
        metas,
        gt_segments=gt_segments,
        gt_labels=gt_labels,
        return_loss=True,
    )
    losses["cost"].backward()
    optimizer.step()
    schedule_update = model.after_optimizer_step()

    model.eval()
    with torch.no_grad():
        predictions = model.forward_test(inputs, masks, metas)
    _require(predictions is not None, "official test forward returned no predictions")

    return {
        "ok": True,
        "config_path": str(config_path),
        "model_type": model.__class__.__name__,
        "detector_head_type": model.rpn_head.__class__.__name__,
        "proof_backbone": "_ProofTemporalMeanBackbone",
        "selector_variant": selector.selector_variant,
        "selected_count": int(positions.numel()),
        "max_unselected_hole_observed": _max_hole(positions, int(temporal_len)),
        "cost": float(losses["cost"].detach().cpu().item()),
        "cost_gradients": cost_gradients,
        "transition_gradients": transition_gradients,
        "coverage_gradients": coverage_gradients,
        "detector_route_gradients": detector_route_gradients,
        "detector_route_loss_keys": ["cls_loss", "reg_loss"],
        "detector_route_loss": float(detector_only_loss.detach().cpu().item()),
        "optimizer_exact_coverage": True,
        "component_learning_rates": {key: sorted(values) for key, values in component_lrs.items()},
        "optimizer_step_ran": True,
        "schedule_update": schedule_update,
        "train_forward": True,
        "test_forward": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = run_proof(args.config, device=args.device)
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
