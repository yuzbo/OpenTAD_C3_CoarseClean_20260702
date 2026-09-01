from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from mmengine.config import Config

from opentad.cores.optimizer import assert_optimizer_exact_coverage
from opentad.models import build_detector
from opentad.models.selectors.duca_online_frame_selector import _gather_time
from tools.bata.run_duca_official_adatad_one_step_grad_proof import (
    _ProofTemporalMeanBackbone,
    _plain,
    _scaled_model_cfg,
)


DEFAULT_CONFIG = (
    "configs/adatad/thumos/duca_protected_e2e_fixed384_official60.py"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"DUCA protected-E2E gradient gate failed: {message}")


def _scaled_selector_cfg(
    cfg: Config,
    *,
    temporal_len: int,
    budget: int,
    hidden_dim: int,
    spatial_size: int,
) -> dict[str, Any]:
    selector = _plain(cfg.model.frame_selector)
    max_hole = 1
    selector.update(
        {
            "dense_window_size": int(temporal_len),
            "budget": int(budget),
            "selector_hidden_channels": int(hidden_dim),
            "coarse_hidden_dim": int(hidden_dim),
            "max_unselected_hole": max_hole,
            "max_gap_loss_max_unselected_hole": max_hole,
            "soft_max_gap_loss_enabled": False,
            "counterfactual_utility_distillation_weight": 0.0,
            "require_counterfactual_utility_teacher": False,
            "profile_runtime": False,
            "profile_sync_cuda": False,
            "temporal_sampling_contract": {
                "hard_budget": int(budget),
                "dense_window_size": int(temporal_len),
                "max_unselected_hole_dense_candidates": max_hole,
                "dataset_feature_stride_source_frames": 4,
                "dataset_sample_stride": 1,
                "requested_max_source_frame_interval": 15,
                "detector_axis": "selected_axis_index",
                "dense_axis_unit": "dense_candidate_index",
                "task": "offline_temporal_action_detection",
            },
        }
    )
    source = selector["actionness_source_cfg"]
    source.update(
        {
            "spatial_size": int(spatial_size),
            "tcn_hidden_dim": int(hidden_dim),
            "official_num_layers": 2,
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
        "detector_gradient": {
            "start": 1.0,
            "end": 1.0,
            "warmup_steps": 0,
            "transition_steps": 0,
        },
        "policy_alpha": {
            "start": 1.0,
            "end": 1.0,
            "warmup_steps": 0,
            "transition_steps": 0,
        },
    }
    return selector


def _is_action_head(name: str) -> bool:
    marker = ".official_temporal."
    if marker not in name:
        return False
    tail = name.split(marker, 1)[1]
    parts = tail.split(".")
    return tail.startswith("encoder.conv_out.") or (
        len(parts) >= 4 and parts[0] == "decoders" and parts[2] == "conv_out"
    )


def _parameter_grad_sum(model, predicate: Callable[[str], bool]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if not predicate(name) or parameter.grad is None:
            continue
        total += float(parameter.grad.detach().abs().sum().item())
    return total


def _gradient_partition(model) -> dict[str, float]:
    encoder_layers = model.frame_selector.raw_actionness_source.probe_module.official_temporal.encoder.layers
    last_layer_prefix = (
        "frame_selector.raw_actionness_source.probe_module."
        f"official_temporal.encoder.layers.{len(encoder_layers) - 1}."
    )
    return {
        "detector": _parameter_grad_sum(model, lambda name: not name.startswith("frame_selector.")),
        "selector_scorer": _parameter_grad_sum(
            model,
            lambda name: name.startswith("frame_selector.adapter.transition_scorer."),
        ),
        "asformer_trunk": _parameter_grad_sum(
            model,
            lambda name: name.startswith("frame_selector.raw_actionness_source.probe_module.")
            and not _is_action_head(name),
        ),
        "asformer_last_encoder_layer": _parameter_grad_sum(
            model,
            lambda name: name.startswith(last_layer_prefix),
        ),
        "asformer_earlier_or_spatial": _parameter_grad_sum(
            model,
            lambda name: name.startswith(
                "frame_selector.raw_actionness_source.probe_module."
            )
            and not _is_action_head(name)
            and not name.startswith(last_layer_prefix),
        ),
        "action_head": _parameter_grad_sum(model, _is_action_head),
    }


def _max_hole(positions: torch.Tensor, valid_len: int) -> int:
    active = positions[positions >= 0].detach().long()
    holes = torch.cat(
        (
            active[:1],
            active[1:] - active[:-1] - 1,
            active.new_tensor([int(valid_len) - int(active[-1].item()) - 1]),
        )
    )
    return int(holes.max().item())


def run_gate(
    config_path: str = DEFAULT_CONFIG,
    *,
    temporal_len: int = 16,
    budget: int = 8,
    hidden_dim: int = 16,
    feature_dim: int = 16,
    spatial_size: int = 16,
    device: str = "cpu",
) -> dict[str, Any]:
    cfg = Config.fromfile(config_path)
    _require(cfg.duca_transition_only_contract.task == "offline_temporal_action_detection", "task is not offline TAD")
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
    model.train()
    selector = model.frame_selector
    _require(selector.detector_gradient_mode == "protected_structured_transport", "protected bridge is not active")
    route = str(cfg.duca_transition_only_contract.route)
    rho_arm = route == "DUCA_PROTECTED_E2E_RHO_FIXED384_OFFICIAL60"
    if rho_arm:
        _require(
            0.0 < float(selector.policy_hidden_gradient_scale) <= 0.05,
            "rho arm must use a small fixed hidden gradient scale",
        )
    else:
        _require(float(selector.policy_hidden_gradient_scale) == 0.0, "main arm must fully protect ASFormer")

    groups = model.get_optim_groups({"lr": 1.0e-4, "weight_decay": 0.05})
    optimizer = torch.optim.AdamW(groups, lr=1.0e-4)
    assert_optimizer_exact_coverage(model, optimizer)
    covered = {id(parameter) for group in groups for parameter in group["params"]}
    missing_selector_parameters = [
        name
        for name, parameter in model.named_parameters()
        if name.startswith("frame_selector.") and parameter.requires_grad and id(parameter) not in covered
    ]
    _require(not missing_selector_parameters, f"optimizer misses selector parameters: {missing_selector_parameters}")

    torch.manual_seed(20260720)
    inputs = torch.randn(
        1,
        3,
        int(temporal_len),
        int(spatial_size),
        int(spatial_size),
        device=device,
    )
    masks = torch.ones(1, int(temporal_len), dtype=torch.bool, device=device)
    metas = [{"video_name": "duca_protected_e2e_gradient_gate", "fps": 30.0}]
    gt_segments = [
        torch.tensor([[3.0, float(temporal_len - 4)]], dtype=torch.float32, device=device)
    ]
    gt_labels = [torch.tensor([1], dtype=torch.long, device=device)]

    captured_backbone_inputs: list[torch.Tensor] = []

    def capture_backbone_input(_module, args):
        captured_backbone_inputs.append(args[0].detach().clone())

    hook = model.backbone.register_forward_pre_hook(capture_backbone_input)
    try:
        losses = model(
            inputs,
            masks,
            metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            return_loss=True,
        )
    finally:
        hook.remove()
    required_losses = {
        "cls_loss",
        "reg_loss",
        "actionness_bce_loss",
        "transition_distribution_loss",
        "transition_boundary_coverage_loss",
    }
    _require(required_losses.issubset(losses), f"missing losses: {sorted(required_losses - set(losses))}")
    _require(len(captured_backbone_inputs) == 1, "detector backbone must be called exactly once")
    positions = selector._last_selected_positions
    slot_mask = positions >= 0
    expected_hard = _gather_time(inputs, positions, slot_mask)
    actual_detector_input = captured_backbone_inputs[0]
    _require(
        torch.equal(actual_detector_input, expected_hard),
        "detector forward input differs elementwise from the exact hard gather",
    )
    active = positions[0][positions[0] >= 0]
    _require(int(active.numel()) == int(budget), "hard path is not exact-K")
    _require(bool(((active[1:] - active[:-1]) > 0).all()), "hard positions are not unique and ordered")
    _require(_max_hole(positions[0], int(temporal_len)) <= 1, "hard positions violate max-gap")

    def backward_partition(loss_names: tuple[str, ...]) -> dict[str, float]:
        model.zero_grad(set_to_none=True)
        route_losses = model(
            inputs,
            masks,
            metas,
            gt_segments=gt_segments,
            gt_labels=gt_labels,
            return_loss=True,
        )
        objective = sum(route_losses[name] for name in loss_names)
        objective.backward()
        return _gradient_partition(model)

    detector_gradients = backward_partition(("cls_loss", "reg_loss"))
    action_gradients = backward_partition(("actionness_bce_loss",))
    transition_gradients = backward_partition(
        ("transition_distribution_loss", "transition_boundary_coverage_loss")
    )

    _require(detector_gradients["detector"] > 0.0, "detector loss did not update official detector")
    _require(detector_gradients["selector_scorer"] > 0.0, "detector loss did not reach selector scorer")
    if rho_arm:
        _require(
            detector_gradients["asformer_last_encoder_layer"] > 0.0,
            "rho detector loss did not reach the last ASFormer encoder layer",
        )
        _require(
            detector_gradients["asformer_earlier_or_spatial"] == 0.0,
            "rho detector loss leaked before the last ASFormer encoder layer",
        )
    else:
        _require(detector_gradients["asformer_trunk"] == 0.0, "detector loss leaked into ASFormer trunk")
    _require(detector_gradients["action_head"] == 0.0, "detector loss leaked into action head")

    _require(action_gradients["detector"] == 0.0, "action BCE leaked into detector")
    _require(action_gradients["selector_scorer"] == 0.0, "action BCE leaked into selector scorer")
    _require(action_gradients["asformer_trunk"] > 0.0, "action BCE did not update ASFormer trunk")
    _require(action_gradients["action_head"] > 0.0, "action BCE did not update action head")

    _require(transition_gradients["detector"] == 0.0, "transition auxiliary leaked into detector")
    _require(transition_gradients["selector_scorer"] > 0.0, "transition auxiliary did not update selector scorer")
    _require(transition_gradients["asformer_trunk"] > 0.0, "transition auxiliary did not update ASFormer trunk")
    _require(transition_gradients["action_head"] == 0.0, "transition auxiliary leaked into action head")

    return {
        "ok": True,
        "status": "p1_p2_gradient_gate_passed",
        "config": str(config_path),
        "route": route,
        "model_type": model.__class__.__name__,
        "detector_head_type": model.rpn_head.__class__.__name__,
        "proof_backbone": "_ProofTemporalMeanBackbone",
        "hard_forward_equals_detector_input": True,
        "selected_count": int(active.numel()),
        "max_unselected_hole": _max_hole(positions[0], int(temporal_len)),
        "optimizer_exact_coverage": True,
        "missing_selector_parameters": [],
        "gradient_ownership": {
            "detector_only": detector_gradients,
            "action_bce_only": action_gradients,
            "transition_auxiliary_only": transition_gradients,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        payload = run_gate(args.config, device=args.device)
    except Exception as exc:
        payload = {
            "ok": False,
            "status": "p1_p2_gradient_gate_failed",
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
        exit_code = 1
    else:
        exit_code = 0
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        Path(args.output_json).write_text(text, encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
