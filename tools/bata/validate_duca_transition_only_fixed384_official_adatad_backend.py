from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_DEFAULT = (
    "configs/adatad/thumos/"
    "duca_transition_only_fixed384_official_adatad_backend_full_train.py"
)
OFFICIAL_BASE_CONFIG = "configs/adatad/thumos/e2e_thumos_videomae_s_768x1_160_adapter.py"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _git_blob_sha1(path: Path) -> str:
    content = path.read_bytes()
    return hashlib.sha1(f"blob {len(content)}\0".encode("ascii") + content).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_config(config_path: str = CONFIG_DEFAULT) -> dict[str, Any]:
    cfg = Config.fromfile(config_path)
    official_path = ROOT / OFFICIAL_BASE_CONFIG
    official = Config.fromfile(str(official_path))
    contract = cfg.duca_transition_only_contract
    selector = cfg.model.frame_selector
    source = selector.actionness_source_cfg
    head = cfg.model.rpn_head
    schedule = selector.loss_weight_schedule
    loss_weights = selector.loss_weights

    _require(contract.task == "offline_temporal_action_detection", "method must be offline TAD")
    _require(contract.online_tad is False and contract.streaming is False, "method must not claim online/streaming TAD")
    _require(contract.full_window_selector is True, "selector must observe the full sampled window")
    _require(contract.official_adatad_backend is True, "official AdaTAD backend must be declared")
    _require(contract.official_detector_source_identical is False, "wrapper source extensions must be disclosed")
    _require(contract.detector_head_changed is False, "transition-only must not change the detector head")
    _require(contract.detector_loss_changed is False, "transition-only must not change detector losses")
    _require(contract.detector_nms_changed is False, "transition-only must not change detector NMS")
    _require(_git_blob_sha1(official_path) == contract.official_base_config_blob_sha1, "official base config blob mismatch")
    _require(_plain(head) == _plain(official.model.rpn_head), "ActionFormerHead config differs from official base")
    _require(cfg.model.type == "ActionFormer", "detector must remain ActionFormer/AdaTAD")
    _require(head.type == "ActionFormerHead", "detector head must remain ActionFormerHead")

    _require(selector.type == "DucaOnlineFrameSelector", "legacy registry class name changed unexpectedly")
    _require(selector.selector_variant == "transition_only", "selector_variant must be transition_only")
    _require(selector.budget_mode == "fixed" and int(selector.budget) == 384, "policy must be fixed exact-384")
    _require(int(selector.dense_window_size) == 768, "dense selector window must be 768")
    _require(selector.acquisition_policy == "global_structured_topk", "selector must use structured exact-K DP")
    _require(int(selector.max_unselected_hole) == 15, "audited maximum unselected hole must be 15")
    _require(selector.hard_max_gap_repair is False, "post-hoc hard repair is forbidden")
    _require(selector.soft_max_gap_loss_enabled is False, "exact-K/max-gap DP makes a separate soft gap loss redundant")
    _require(float(selector.transition_target_sigma) == 2.0, "transition target sigma must be two dense observations")
    _require(int(selector.transition_target_radius) == 4, "transition target must be truncated at radius four")
    _require(int(selector.transition_boundary_radius) == 4, "soft boundary coverage radius must be four")
    _require(float(selector.transition_distribution_temperature) == 0.7, "transition distribution temperature must be 0.7")
    _require(selector.detector_gradient_mode == "st_sparse_gather", "direct detector bridge must be disabled")
    _require(float(selector.counterfactual_utility_distillation_weight) > 0.0, "counterfactual distillation is disabled")
    _require(selector.require_counterfactual_utility_teacher is True, "counterfactual teacher must fail closed")
    _require(int(selector.counterfactual_max_candidates) == 4, "counterfactual compute bound must be four candidates")
    _require(contract.counterfactual_teacher_producer_integrated is True, "counterfactual producer is not integrated")
    _require(float(selector.inference_policy_alpha) == 1.0, "main transition-only inference must use the learned policy")
    _require(selector.use_coarse_hidden_features is True, "ASFormer encoder hidden must be used")
    _require(selector.require_coarse_hidden_features is True, "missing ASFormer hidden must fail closed")
    _require(selector.forbid_external_actionness is True, "external/cached actionness is forbidden")
    _require(selector.forbid_ledger is True and selector.no_ledger_decision is True, "ledger decisions are forbidden")
    _require(selector.forbid_raw_prediction_cache is True, "raw prediction cache is forbidden")
    for key in ("actionness_weight", "transition_weight", "uncertainty_weight", "utility_weight", "boundary_weight"):
        _require(float(getattr(selector, key)) == 0.0, f"legacy ranking coefficient {key} must be zero")

    _require(source.type == "C3CoarseProbeActionnessSource", "coarse source must run in graph")
    _require(source.probe_model == "official-action-seg", "coarse source must use official action segmentation code")
    _require(source.official_action_seg_backend == "official_asformer", "coarse temporal module must be official ASFormer")
    _require(source.trainable is True and source.frozen is False, "shared ASFormer must be trainable")
    _require(source.return_hidden_features is True and source.require_hidden_features is True, "true encoder hidden is mandatory")
    _require(
        source.hidden_output_kind == "official_asformer_encoder_hidden",
        "transition-only must explicitly opt into official ASFormer encoder hidden",
    )
    _require(source.thumos_trained is True, "task-adapted probe must disclose THUMOS training")
    _require(source.uses_labels is True, "task-adapted probe must disclose label supervision")
    _require(source.uses_gt is True, "task-adapted probe must disclose GT-segment supervision")
    _require(source.uses_teacher is False, "task-adapted probe must not use a teacher")
    _require(source.uses_prediction_cache is False, "task-adapted probe must not use prediction cache")
    _require(source.trained_with_thumos_labels is True, "THUMOS label training history must be explicit")
    _require(source.trained_with_gt_segments is True, "GT boundary training history must be explicit")
    _require(source.training_supervision_scope == "train_only", "supervision must be train-only")
    for key in (
        "uses_labels_at_inference",
        "uses_gt_at_inference",
        "uses_teacher_at_inference",
        "uses_prediction_cache_at_inference",
    ):
        _require(source.get(key) is False, f"inference provenance must set {key}=False")
    _require(contract.coarse_hidden_kind == "official_asformer_encoder_hidden", "hidden semantics are not locked")
    _require(contract.ranking_uses_absolute_hidden is False, "absolute hidden ranking is forbidden")
    _require(contract.ranking_uses_raw_rgb_mean is False, "RGB-mean ranking is forbidden")
    _require(contract.legacy_direct_heads_enabled is False, "legacy direct heads must be disabled")
    _require(contract.policy_homotopy_modulo_switching is False, "modulo duty-cycle switching is forbidden")
    _require(contract.protected_gradient_routing is True, "protected gradient routing must be declared")
    _require(float(selector.coarse_trunk_lr) == 2.5e-5, "coarse trunk LR must be 2.5e-5")
    _require(float(selector.action_head_lr) == 5.0e-5, "binary action head LR must be 5e-5")
    _require(float(selector.transition_scorer_lr) == 1.0e-4, "transition scorer LR must be 1e-4")

    _require(float(loss_weights.actionness) == 1.0, "binary actionness supervision must remain active")
    _require(float(loss_weights.transition) > 0.0, "transition distribution supervision must be active")
    _require(float(loss_weights.transition) == 0.5, "transition distribution coefficient must be 0.5")
    _require(float(loss_weights.transition_boundary) > 0.0, "soft boundary coverage must be active")
    for key in ("detector_utility", "start", "end", "context", "boundary", "hole", "redundancy", "radius", "entropy"):
        _require(float(loss_weights.get(key, 0.0)) == 0.0, f"legacy loss {key} must be disabled")
    _require(float(loss_weights.max_gap_hole) == 0.0, "fixed structured DP must disable redundant max-gap loss")
    _require(schedule.type == "progressive_joint" and schedule.shape == "cosine", "schedule must be continuous cosine")
    _require(float(schedule.actionness.start) == 1.0 and float(schedule.actionness.end) == 1.0, "actionness BCE must not decay")
    _require(float(schedule.policy_alpha.start) == 0.0 and float(schedule.policy_alpha.end) == 1.0, "policy alpha endpoints are wrong")
    _require(float(schedule.detector_gradient.start) == 0.0, "detector bridge must start at zero")
    _require(float(schedule.detector_gradient.end) == float(contract.detector_gradient_final_weight), "detector bridge endpoint mismatch")
    _require(int(schedule.policy_alpha.warmup_steps) == 660, "policy must remain uniform for the first 5%")
    _require(int(schedule.policy_alpha.transition_steps) == 3960, "policy alpha must ramp over steps 660-4620")
    _require(int(schedule.detector_gradient.warmup_steps) == 4620, "detector bridge must wait until policy alpha reaches one")
    _require(int(schedule.detector_gradient.transition_steps) == 3300, "detector bridge must ramp over steps 4620-7920")

    _require(int(cfg.duca_schedule_steps_per_epoch) == 100, "expected steps per epoch must be 100")
    _require(int(cfg.duca_loss_schedule_total_steps) == 13200, "expected schedule horizon must be 13200")
    _require(int(cfg.workflow.end_epoch) == 132, "workflow must run 132 epochs")
    _require(
        int(cfg.workflow.val_eval_interval) < 0,
        "formal workflow must forbid intermediate test-set evaluation",
    )
    _require(int(cfg.workflow.val_start_epoch) > int(cfg.workflow.end_epoch), "formal workflow must defer evaluation to terminal finalization")
    _require(bool(cfg.workflow.formal_successful_update_contract), "formal successful-update contract is missing")
    _require(int(cfg.workflow.expected_train_batches_per_epoch) == 100, "formal loader batch count is not frozen")
    _require(int(cfg.workflow.expected_successful_optimizer_updates) == 13200, "formal successful-update count is not frozen")
    _require(int(cfg.workflow.checkpoint_interval) == 5, "checkpoint interval must remain five epochs")
    _require(int(cfg.workflow.primary_checkpoint_epoch) == 131, "terminal primary checkpoint epoch is wrong")
    _require(cfg.workflow.primary_checkpoint_state_key == "state_dict_ema", "terminal primary state is not EMA")
    _require(int(cfg.scheduler.max_epoch) == 132, "cosine scheduler horizon must match workflow")
    _require("max_train_iters" not in cfg.workflow, "epoch runner must not claim an exact successful-step stop")
    _require(int(cfg.window_size) == 384 and int(cfg.chunk_num) == 24, "detector physical input must be 384 frames")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "VideoMAE must consume the selected 384 frames")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection length must match selected frames")
    _require(contract.paper_claim_allowed is False, "untested candidate cannot be paper-ready")
    _require(contract.metric_claim_allowed is False, "untested candidate cannot make metric claims")

    repo_root = os.environ.get("C3_OFFICIAL_ACTION_SEG_REPOS")
    source_summary: dict[str, Any] = {"checked": False}
    if repo_root:
        source_path = Path(repo_root).expanduser().resolve() / "ASFormer" / "model.py"
        _require(source_path.is_file(), f"official ASFormer source missing: {source_path}")
        digest = _sha256(source_path)
        _require(
            digest == contract.official_asformer_source_normalized_lf_sha256,
            "official ASFormer normalized-LF SHA256 mismatch",
        )
        source_summary = {"checked": True, "path": str(source_path), "normalized_lf_sha256": digest}

    return {
        "ok": True,
        "status": "tested_config_contract",
        "config_path": str(config_path),
        "task": str(contract.task),
        "selector_variant": str(selector.selector_variant),
        "budget": int(selector.budget),
        "dense_window_size": int(selector.dense_window_size),
        "max_unselected_hole": int(selector.max_unselected_hole),
        "detector_type": str(cfg.model.type),
        "detector_head": str(head.type),
        "official_head_config_match": True,
        "coarse_probe": str(source.official_action_seg_backend),
        "coarse_hidden_kind": str(contract.coarse_hidden_kind),
        "trained_with_thumos_labels": bool(source.trained_with_thumos_labels),
        "trained_with_gt_segments": bool(source.trained_with_gt_segments),
        "uses_labels_at_inference": False,
        "uses_gt_at_inference": False,
        "policy_uniform_steps": int(schedule.policy_alpha.warmup_steps),
        "policy_homotopy_steps": int(schedule.policy_alpha.transition_steps),
        "detector_bridge_delay_steps": int(schedule.detector_gradient.warmup_steps),
        "detector_bridge_ramp_steps": int(schedule.detector_gradient.transition_steps),
        "expected_steps_per_epoch": int(cfg.duca_schedule_steps_per_epoch),
        "expected_total_steps": int(cfg.duca_loss_schedule_total_steps),
        "workflow_epochs": int(cfg.workflow.end_epoch),
        "scheduler_epochs": int(cfg.scheduler.max_epoch),
        "val_start_epoch": int(cfg.workflow.val_start_epoch),
        "official_asformer_source": source_summary,
        "paper_claim_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = validate_config(args.config)
    except Exception as exc:
        summary = {
            "ok": False,
            "config_path": str(args.config),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
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
