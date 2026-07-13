from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from mmengine.config import Config


CONFIGS = {
    "uniform": "configs/adatad/thumos/duca_exact_uniform_fixed384_official_adatad_backend_full_train.py",
    "direct": "configs/adatad/thumos/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py",
    "transition_beta0": "configs/adatad/thumos/duca_transition_only_fixed384_no_detector_bridge_official_adatad_backend_full_train.py",
    "transition_counterfactual": "configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py",
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def validate_core_gate(
    path: str | Path,
    *,
    expected_commit: str,
    expected_config_sha256: str,
    expected_source_sha256: str,
    expected_checkpoint_sha256: str,
) -> dict[str, Any]:
    gate_path = Path(path)
    _require(gate_path.is_file(), f"DUCA core gate JSON missing: {gate_path}")
    payload = json.loads(gate_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "DUCA core gate JSON must contain an object")
    _require(payload.get("ok") is True, "DUCA core gate JSON must declare ok=true")
    gate_commit = payload.get("git_commit")
    _require(isinstance(gate_commit, str) and gate_commit, "DUCA core gate JSON missing git_commit")
    _require(gate_commit == str(expected_commit), "DUCA core gate git_commit does not match expected commit")
    _require(payload.get("formal_proof_ok") is True, "DUCA core gate missing formal_proof_ok=true")
    _require(payload.get("reference_config_sha256") == expected_config_sha256, "DUCA core gate config SHA256 mismatch")
    _require(payload.get("official_asformer_source_sha256") == expected_source_sha256, "DUCA core gate source SHA256 mismatch")
    _require(payload.get("checkpoint_sha256") == expected_checkpoint_sha256, "DUCA core gate checkpoint SHA256 mismatch")
    _require(int(payload.get("dense_window_size", -1)) == 768, "DUCA core gate did not test T=768")
    _require(int(payload.get("selected_count", -1)) == 384, "DUCA core gate did not test exact K=384")
    _require(payload.get("model_type") == "ActionFormer", "DUCA core gate did not test ActionFormer")
    _require(payload.get("detector_head_type") == "ActionFormerHead", "DUCA core gate did not test ActionFormerHead")
    _require(
        payload.get("uniform_reference_definition") == "round_linspace_endpoints",
        "DUCA core gate uniform reference definition is not exact endpoint linspace",
    )
    _require(
        payload.get("uniform_reference_exact") is True,
        "DUCA core gate uniform reference was not verified exactly",
    )
    _require(
        int(payload.get("uniform_reference_max_rank_error", -1)) == 0,
        "DUCA core gate uniform reference has nonzero rank-aligned position error",
    )
    return {
        "ok": True,
        "git_commit": gate_commit,
        "path": str(gate_path),
        "uniform_reference_exact": True,
    }


def validate_variant(variant: str, config_path: str | None = None) -> dict[str, Any]:
    variant = str(variant)
    _require(variant in CONFIGS, f"unknown P0 variant {variant!r}")
    path = str(config_path or CONFIGS[variant])
    cfg = Config.fromfile(path)
    reference = Config.fromfile(CONFIGS["transition_counterfactual"])
    selector = cfg.model.frame_selector

    _require(cfg.model.type == "ActionFormer", "detector must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "detector head must remain ActionFormerHead")
    _require(_plain(cfg.model.rpn_head) == _plain(reference.model.rpn_head), "detector head config mismatch")
    _require(_plain(cfg.dataset) == _plain(reference.dataset), "dataset/loader protocol mismatch")
    _require(_plain(cfg.optimizer) == _plain(reference.optimizer), "optimizer protocol mismatch")
    _require(int(cfg.window_size) == 384 and int(cfg.dense_window_size) == 768, "P0 geometry mismatch")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "detector physical input is not 384")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection length is not 384")
    _require(int(cfg.workflow.end_epoch) == 132, "workflow must run 132 epochs")
    _require(int(cfg.scheduler.max_epoch) == 132, "scheduler must run 132 epochs")
    for key in ("val_start_epoch", "val_eval_interval", "val_eval_interval_anchor_epoch"):
        _require(int(cfg.workflow[key]) == int(reference.workflow[key]), f"workflow {key} mismatch")
    component_lrs = {
        "coarse_trunk": float(selector.get("coarse_trunk_lr", 2.5e-5)),
        "action_head": float(selector.get("action_head_lr", 5.0e-5)),
        "transition_scorer": float(selector.get("transition_scorer_lr", 1.0e-4)),
    }
    reference_lrs = {
        "coarse_trunk": float(reference.model.frame_selector.coarse_trunk_lr),
        "action_head": float(reference.model.frame_selector.action_head_lr),
        "transition_scorer": float(reference.model.frame_selector.transition_scorer_lr),
    }
    _require(component_lrs == reference_lrs, "effective selector component LR protocol mismatch")

    details: dict[str, Any] = {}
    if variant == "uniform":
        _require(selector.selector_variant == "transition_only", "uniform baseline must use matched transition stack")
        _require(float(selector.inference_policy_alpha) == 0.0, "uniform eval alpha must be zero")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 0.0, "uniform train alpha must be zero")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "uniform beta must be zero")
        _require(float(selector.counterfactual_utility_distillation_weight) == 0.0, "uniform must disable counterfactual distillation")
        _require(not bool(selector.require_counterfactual_utility_teacher), "uniform must not build a counterfactual teacher")
        details["selection"] = "exact_uniform_reference"
    elif variant == "direct":
        _require(selector.get("selector_variant") == "direct_boundary", "direct baseline must explicitly declare its architecture")
        _require(int(cfg.duca_loss_schedule_total_steps) == 13200, "direct baseline horizon mismatch")
        source = selector.actionness_source_cfg
        _require(
            source.get("hidden_output_kind") == "pre_temporal_spatial_stem_hidden",
            "direct baseline must preserve legacy spatial-stem hidden",
        )
        _require(source.get("trained_with_thumos_labels") is True, "direct baseline training provenance is incomplete")
        _require(source.get("trained_with_gt_segments") is True, "direct baseline GT provenance is incomplete")
        details["selection"] = "a5_direct_boundary"
    elif variant == "transition_beta0":
        _require(selector.selector_variant == "transition_only", "beta0 must use transition-only selector")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 1.0, "beta0 must learn selection")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "beta0 bridge is not disabled")
        _require(float(selector.counterfactual_utility_distillation_weight) == 0.0, "beta0 must disable counterfactual distillation")
        _require(not bool(selector.require_counterfactual_utility_teacher), "beta0 must not build a counterfactual teacher")
        details["selection"] = "transition_only"
        details["detector_bridge_beta"] = 0.0
    else:
        _require(selector.selector_variant == "transition_only", "main candidate must use transition-only selector")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 1.0, "main candidate must learn selection")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "direct detector bridge must remain disabled")
        _require(float(selector.counterfactual_utility_distillation_weight) > 0.0, "counterfactual distillation must be enabled")
        _require(bool(selector.require_counterfactual_utility_teacher), "counterfactual teacher must fail closed")
        _require(int(selector.counterfactual_max_candidates) > 0, "counterfactual candidate bound is missing")
        details["selection"] = "transition_only"
        details["detector_bridge_beta"] = 0.0
        details["detector_utility_learning"] = "detached_hard_counterfactual_distillation"

    return {
        "ok": True,
        "status": "p0_variant_config_validated",
        "variant": variant,
        "config_path": path,
        "task": "offline_temporal_action_detection",
        "budget": 384,
        "dense_window_size": 768,
        "workflow_epochs": 132,
        "scheduler_epochs": 132,
        "detector": "ActionFormer/ActionFormerHead",
        "paper_claim_allowed": False,
        "component_learning_rates": component_lrs,
        **details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--config")
    parser.add_argument("--core-gate-json")
    parser.add_argument("--expected-commit")
    parser.add_argument("--expected-config-sha256")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_variant(args.variant, args.config)
        gate_args = (
            args.core_gate_json,
            args.expected_commit,
            args.expected_config_sha256,
            args.expected_source_sha256,
            args.expected_checkpoint_sha256,
        )
        if any(gate_args):
            _require(bool(args.core_gate_json), "--core-gate-json is required with --expected-commit")
            _require(bool(args.expected_commit), "--expected-commit is required with --core-gate-json")
            _require(all(gate_args), "all core-gate expected hashes are required")
            summary["core_gate"] = validate_core_gate(
                args.core_gate_json,
                expected_commit=args.expected_commit,
                expected_config_sha256=args.expected_config_sha256,
                expected_source_sha256=args.expected_source_sha256,
                expected_checkpoint_sha256=args.expected_checkpoint_sha256,
            )
    except Exception as exc:
        summary = {"ok": False, "variant": args.variant, "error_type": exc.__class__.__name__, "error": str(exc)}
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
