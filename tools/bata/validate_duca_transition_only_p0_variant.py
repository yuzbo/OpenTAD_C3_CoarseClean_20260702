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
    "transition_beta025": "configs/adatad/thumos/duca_transition_only_fixed384_official_adatad_backend_full_train.py",
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


def validate_variant(variant: str, config_path: str | None = None) -> dict[str, Any]:
    variant = str(variant)
    _require(variant in CONFIGS, f"unknown P0 variant {variant!r}")
    path = str(config_path or CONFIGS[variant])
    cfg = Config.fromfile(path)
    reference = Config.fromfile(CONFIGS["transition_beta025"])
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

    details: dict[str, Any] = {}
    if variant == "uniform":
        _require(selector.selector_variant == "transition_only", "uniform baseline must use matched transition stack")
        _require(float(selector.inference_policy_alpha) == 0.0, "uniform eval alpha must be zero")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 0.0, "uniform train alpha must be zero")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "uniform beta must be zero")
        details["selection"] = "exact_uniform_reference"
    elif variant == "direct":
        _require(selector.get("selector_variant", "direct_boundary") == "direct_boundary", "direct baseline changed architecture")
        _require(int(cfg.duca_loss_schedule_total_steps) == 13200, "direct baseline horizon mismatch")
        details["selection"] = "a5_direct_boundary"
    elif variant == "transition_beta0":
        _require(selector.selector_variant == "transition_only", "beta0 must use transition-only selector")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 1.0, "beta0 must learn selection")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "beta0 bridge is not disabled")
        details["selection"] = "transition_only"
        details["detector_bridge_beta"] = 0.0
    else:
        _require(selector.selector_variant == "transition_only", "main candidate must use transition-only selector")
        _require(float(selector.loss_weight_schedule.policy_alpha.end) == 1.0, "main candidate must learn selection")
        _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.25, "main beta endpoint must be 0.25")
        details["selection"] = "transition_only"
        details["detector_bridge_beta"] = 0.25

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
        **details,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_variant(args.variant, args.config)
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
