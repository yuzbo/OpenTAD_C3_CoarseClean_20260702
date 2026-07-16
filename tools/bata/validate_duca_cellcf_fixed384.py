from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_BASE = ROOT / "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"
VARIANTS = {
    "uniform": "configs/adatad/thumos/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py",
    "transition_beta0": "configs/adatad/thumos/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train.py",
    "cellcf": "configs/adatad/thumos/duca_cellcf_fixed384_official_adatad_backend_full_train.py",
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_config(variant: str, config_path: str | None = None) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ValueError(f"unknown CellCF variant {variant!r}")
    path = ROOT / (config_path or VARIANTS[variant])
    _require(path.is_file(), f"CellCF config is missing: {path}")
    cfg = Config.fromfile(str(path))
    official = Config.fromfile(str(OFFICIAL_BASE))
    selector = cfg.model.frame_selector
    contract = cfg.duca_transition_only_contract
    source = selector.actionness_source_cfg

    _require(contract.task == "offline_temporal_action_detection", "CellCF must be offline TAD")
    _require(contract.online_tad is False and contract.streaming is False, "online/streaming claims are forbidden")
    _require(contract.full_window_selector is True, "CellCF must observe the complete sampled window")
    _require(cfg.model.type == "ActionFormer", "official AdaTAD detector wrapper must remain ActionFormer")
    _require(cfg.model.rpn_head.type == "ActionFormerHead", "official ActionFormerHead is required")
    _require(_plain(cfg.model.rpn_head) == _plain(official.model.rpn_head), "detector head config drifted from official")
    _require(contract.detector_head_changed is False, "CellCF must not modify the detector head")
    _require(contract.detector_loss_changed is False, "CellCF must not modify detector losses")
    _require(contract.detector_nms_changed is False, "CellCF must not modify detector NMS")

    _require(selector.selector_variant == "transition_only", "CellCF must use transition-only indirect selection")
    _require(selector.acquisition_policy == "local_cell_deformation", "CellCF local policy is not active")
    _require(selector.counterfactual_objective == "local_cell_signed_logistic", "local signed utility is not active")
    _require(selector.budget_mode == "fixed" and int(selector.budget) == 384, "CellCF must use exact K=384")
    _require(int(selector.dense_window_size) == 768, "CellCF dense selector window must be T=768")
    _require(selector.max_unselected_hole is None, "CellCF must derive coverage from cells, not a configured G")
    _require(int(contract.derived_max_unselected_hole) == 3, "T=768/K=384 local-cell hole bound must be three")
    _require(selector.hard_max_gap_repair is False, "post-hoc hard repair is forbidden")
    _require(selector.soft_max_gap_loss_enabled is False, "local-cell geometry makes the old gap loss redundant")
    _require(selector.detector_gradient_mode == "none", "soft RGB detector bridge must be disabled")
    _require(float(selector.loss_weight_schedule.detector_gradient.start) == 0.0, "detector bridge must start at zero")
    _require(float(selector.loss_weight_schedule.detector_gradient.end) == 0.0, "detector bridge must remain zero")
    _require("policy_alpha" not in selector.loss_weight_schedule, "CellCF forbids policy homotopy fields")

    expected_uniform = variant == "uniform"
    expected_utility = variant == "cellcf"
    _require(bool(selector.local_cell_force_exact_uniform) is expected_uniform, "exact-uniform control flag mismatch")
    _require(
        (float(selector.counterfactual_utility_distillation_weight) > 0.0) is expected_utility,
        "counterfactual utility is enabled in the wrong variant",
    )
    _require(bool(selector.require_counterfactual_utility_teacher) is expected_utility, "teacher fail-closed flag mismatch")
    _require(int(selector.counterfactual_max_candidates) == 4, "counterfactual compute bound must be four")
    _require(contract.global_viterbi == "disabled", "global Viterbi must be disabled")
    _require(contract.global_gram_proximal == "disabled", "Gram proximal must be disabled")
    _require(contract.policy_homotopy == "disabled", "global policy homotopy must be disabled")
    _require(contract.detector_utility_is_direct_gradient is False, "detached utility must not be misnamed direct gradient")

    _require(source.probe_model == "official-action-seg", "coarse probe must use official action-seg code")
    _require(source.official_action_seg_backend == "official_asformer", "coarse temporal module must be official ASFormer")
    _require(source.trainable is True and source.frozen is False, "coarse probe must be jointly trainable")
    _require(source.require_hidden_features is True, "selector must receive official ASFormer encoder hidden states")
    _require(source.uses_labels_at_inference is False, "inference must not use labels")
    _require(source.uses_gt_at_inference is False, "inference must not use GT")
    _require(selector.forbid_external_actionness is True, "cached/external actionness is forbidden")
    _require(selector.forbid_ledger is True and selector.no_ledger_decision is True, "ledger decisions are forbidden")
    _require(selector.forbid_raw_prediction_cache is True, "prediction-cache decisions are forbidden")

    weights = selector.loss_weights
    _require(float(weights.actionness) == 1.0, "binary actionness supervision must remain active")
    _require(float(weights.transition) == 0.5, "transition distribution supervision mismatch")
    _require(float(weights.transition_boundary) == 0.25, "boundary coverage supervision mismatch")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE must consume exactly 384 selected frames")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection length must be 384")
    _require(cfg.model.backbone.backbone.with_cp is False, "audited dynamic graph requires with_cp=False")
    _require(cfg.solver.static_graph is False and cfg.solver.find_unused_parameters is True, "DDP protocol drift")
    _require(int(cfg.workflow.end_epoch) == 132 and int(cfg.scheduler.max_epoch) == 132, "132-epoch protocol drift")
    _require(int(cfg.workflow.checkpoint_interval) == 5, "checkpoint interval must remain five epochs")
    _require(int(cfg.workflow.expected_successful_optimizer_updates) == 13200, "successful-update contract drift")
    _require(int(cfg.workflow.primary_checkpoint_epoch) == 131, "terminal checkpoint must be epoch 131")
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "terminal evaluation must use EMA")
    _require(contract.paper_claim_allowed is False and contract.metric_claim_allowed is False, "untested method cannot claim results")

    official_source_checked = False
    repo_root = os.environ.get("C3_OFFICIAL_ACTION_SEG_REPOS")
    if repo_root:
        official_source = Path(repo_root).expanduser().resolve() / "ASFormer" / "model.py"
        _require(official_source.is_file(), f"official ASFormer source is missing: {official_source}")
        official_source_checked = True

    return {
        "ok": True,
        "status": "tested_config_contract",
        "variant": variant,
        "config": str(path),
        "config_sha256": _sha256(path),
        "task": str(contract.task),
        "detector": str(cfg.model.type),
        "detector_head": str(cfg.model.rpn_head.type),
        "acquisition_policy": str(selector.acquisition_policy),
        "counterfactual_objective": str(selector.counterfactual_objective),
        "force_exact_uniform": bool(selector.local_cell_force_exact_uniform),
        "counterfactual_weight": float(selector.counterfactual_utility_distillation_weight),
        "budget": int(selector.budget),
        "dense_window_size": int(selector.dense_window_size),
        "checkpoint_interval": int(cfg.workflow.checkpoint_interval),
        "end_epoch": int(cfg.workflow.end_epoch),
        "official_asformer_source_checked": official_source_checked,
        "paper_claim_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--config")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(args.variant, args.config)
    except Exception as exc:
        summary = {"ok": False, "variant": args.variant, "error_type": type(exc).__name__, "error": str(exc)}
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
