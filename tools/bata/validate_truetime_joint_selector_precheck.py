from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config


CONFIG_DEFAULT = "configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py"
READY = "TRUETIME_JOINT_SELECTOR_PRECHECK_PASS"
ROUTE = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
PHASES = [
    "dense_teacher",
    "selector_pretrain",
    "frozen_detector",
    "sparse_detector",
    "joint_finetune",
]


def _require(condition, message):
    if not condition:
        raise AssertionError(message)


def _as_bool(value):
    return bool(value)


def _find_collect(pipeline):
    matches = [step for step in pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    _require(len(matches) == 1, f"expected exactly one Collect step, got {len(matches)}")
    return matches[0]


def _validate_gate(cfg, *, allow_launch_unlocked):
    gate = cfg.truetime_joint_selector_gate
    _require(gate.route_variant == ROUTE, "wrong route variant")
    _require(gate.stage == "stage3_4_experimental_smoke", "wrong stage")
    _require(_as_bool(gate.smoke_only), "route must remain smoke_only")
    _require(_as_bool(gate.requires_launch_gate), "launch gate must be required")
    _require(_as_bool(gate.requires_selector_grad_nonzero), "selector grad gate must be required")
    _require(_as_bool(gate.requires_geometry_roundtrip), "geometry gate must be required")
    _require(_as_bool(gate.end_to_end_claim_allowed) is False, "end-to-end claim must be locked")
    _require(_as_bool(gate.paper_claim_allowed) is False, "paper claim must be locked")
    _require(_as_bool(gate.runtime_flops_claim_allowed) is False, "runtime claim must be locked")
    _require(_as_bool(gate.deploy_claim_allowed) is False, "deploy claim must be locked")
    _require(list(gate.required_phases) == PHASES, "curriculum phases changed")
    if allow_launch_unlocked:
        _require(_as_bool(gate.launch_gate_passed), "execution config must pass launch gate")
        _require(_as_bool(gate.get("reviewed_execution_config", False)), "execution config must be reviewed")
    else:
        _require(not _as_bool(gate.launch_gate_passed), "default config must be launch-locked")
    _require(tuple(gate.allowed_entrypoints) == ("tools/train.py",), "only tools/train.py may be allowed")
    _require("tools/test.py" in tuple(gate.forbidden_entrypoints), "tools/test.py must be forbidden")


def _validate_model(cfg):
    selector = cfg.model.frame_selector
    _require(cfg.model.type == "ActionFormer", "route must use ActionFormer frame_selector slot")
    _require(selector.type == "TrueTimeRelaxedHardTopKSelector", "wrong selector type")
    _require(int(selector.selected_count) == 384, "selected_count must be 384")
    _require(int(selector.dense_len) == 768, "dense_len must be 768")
    _require(selector.coordinate_space == "selected_axis_index", "detector outputs must be selected-axis")
    _require(selector.true_time_source_axis == "true_time_dense_index", "true-time source axis mismatch")
    _require(selector.detector_gradient_mode == "st_sparse_gather", "detector gradient mode must be st_sparse_gather")
    _require(float(selector.slot_softmax_temperature) > 0.0, "slot_softmax_temperature must be positive")
    _require(float(selector.slot_distance_penalty) >= 0.0, "slot_distance_penalty must be non-negative")
    _require(_as_bool(selector.allow_gt_selection) is False, "GT selection must be disabled")
    _require(_as_bool(selector.allow_teacher_utility) is False, "teacher selection must be disabled")
    _require(int(cfg.model.backbone.backbone.total_frames) == 384, "backbone selected length must be 384")
    _require(int(cfg.model.projection.max_seq_len) == 384, "projection selected length must be 384")
    _require(_as_bool(cfg.inference.load_from_raw_predictions) is False, "raw prediction loading forbidden")
    _require(_as_bool(cfg.inference.save_raw_prediction) is False, "raw prediction saving forbidden")

    smoke_model = cfg.truetime_detector_path_smoke_model
    smoke_selector = smoke_model.frame_selector
    _require(smoke_model.type == "TrueTimeJointSelectorSmokeDetector", "wrong detector-path smoke model type")
    _require(smoke_selector.type == "TrueTimeRelaxedHardTopKSelector", "wrong smoke selector type")
    _require(smoke_selector.detector_gradient_mode == "st_sparse_gather", "smoke detector gradient mode mismatch")
    _require(int(smoke_selector.selected_count) < int(smoke_selector.dense_len), "smoke selector must be sparse")
    _require(_as_bool(smoke_selector.allow_gt_selection) is False, "smoke GT selection must be disabled")
    _require(_as_bool(smoke_selector.allow_teacher_utility) is False, "smoke teacher selection must be disabled")


def _validate_dataset_and_leakage(cfg):
    for split in ("train", "val", "test"):
        _require(int(cfg.dataset[split].window_size) == 768, f"{split}: dense dataset window must be 768")
        collect = _find_collect(cfg.dataset[split].pipeline)
        meta_keys = set(collect.get("meta_keys", []))
        for key in (
            "truetime_selected_positions",
            "truetime_dense_valid_len",
            "truetime_selected_count",
            "detector_output_coordinate_space",
            "selected_axis_to_true_time_dense_index",
            "irregular_selected_positions",
            "irregular_selected_count",
            "irregular_dense_valid_len",
            "irregular_selected_valid_len",
            "irregular_native_axis",
        ):
            _require(key in meta_keys, f"{split}: missing meta key {key}")

    test_keys = set(_find_collect(cfg.dataset.test.pipeline).get("keys", []))
    _require("gt_segments" not in test_keys, "test must not collect gt_segments")
    _require("gt_labels" not in test_keys, "test must not collect gt_labels")

    for split in ("val", "test"):
        contract = cfg.selection_contract[split]
        _require(contract.selection_uses_gt is False, f"{split}: selection must not use GT")
        _require(contract.selection_uses_teacher is False, f"{split}: selection must not use teacher")


def _validate_curriculum_and_scope(cfg):
    _require(cfg.experiment_scope.route_variant == ROUTE, "experiment route mismatch")
    _require(cfg.stage2_offline_detector_utility_selector_dependency is False, "must stay separate from Stage-2 offline selector")
    _require(cfg.experiment_scope.modifies_evaluator_map_semantics is False, "evaluator mAP semantics must stay unchanged")
    _require(cfg.workflow.end_epoch == 1, "smoke route must run one epoch max")
    _require(cfg.workflow.max_train_iters == 2, "smoke route must cap train iters at 2")
    _require(cfg.workflow.val_eval_interval <= 0, "smoke route must not unlock mAP loop")
    _require(list(cfg.truetime_joint_selector_gate.eight_week_questions_answered_by) == [
        "true_time_roundtrip_tests",
        "segment_inverse_map_tests",
        "selector_detector_loss_gradient_smoke",
        "fail_closed_curriculum_and_claim_gates",
    ], "8-week question evidence list changed")
    for metric in (
        "selector_grad_norm",
        "detector_loss_selector_grad_norm",
        "selected_input_selector_grad_norm",
        "selected_count_mean",
        "selected_count_std",
        "entropy",
        "loss_cls",
        "loss_reg",
        "geometry_roundtrip",
        "prediction_inverse_map",
        "claim_locks",
    ):
        _require(metric in cfg.truetime_metrics_to_log, f"missing metric {metric}")
    for forbidden_metric in ("mAP", "tIoU"):
        _require(forbidden_metric not in cfg.truetime_metrics_to_log, f"smoke metrics must not include {forbidden_metric}")


def _validate_grad_proof(path):
    proof_path = Path(path)
    _require(
        str(proof_path) and "REPLACE_WITH" not in str(proof_path),
        "selected-input selector gradient proof path unresolved",
    )
    _require(proof_path.is_file(), f"selected-input selector gradient proof file missing: {proof_path}")
    payload = json.loads(proof_path.read_text(encoding="utf-8"))
    _require(payload.get("route_variant") == ROUTE, "selector_grad_norm proof route mismatch")
    _require(payload.get("geometry_roundtrip_passed") is True, "geometry roundtrip proof missing")
    _require(payload.get("prediction_inverse_map_passed") is True, "prediction inverse-map proof missing")
    _require(
        payload.get("selected_input_st_gradient_passed") is True,
        "selected-input selector gradient proof missing",
    )
    _require(
        float(payload.get("selected_input_selector_grad_norm", 0.0)) > 0.0,
        "selected-input selector gradient proof must be > 0",
    )
    _require(
        payload.get("detector_loss_selector_grad_passed") is True,
        "detector-loss selector gradient proof missing",
    )
    _require(
        float(payload.get("detector_loss_selector_grad_norm", 0.0)) > 0.0,
        "detector-loss selector gradient proof must be > 0",
    )
    loss_keys = payload.get("loss_keys", [])
    _require("loss_cls" in loss_keys, "detector proof missing loss_cls")
    _require("loss_reg" in loss_keys, "detector proof missing loss_reg")
    _require(payload.get("proof_source") == "registered_detector_forward_train_cost_backward", "wrong proof source")
    _require(float(payload.get("selector_grad_norm", 0.0)) > 0.0, "selector_grad_norm proof must be > 0")
    _require(payload.get("selector_grad_nonzero") is True, "selector_grad_nonzero proof missing")


def validate_config(config_path=CONFIG_DEFAULT, *, require_grad_proof=False, allow_launch_unlocked=False, proof_json=None):
    cfg = Config.fromfile(str(config_path))
    _validate_gate(cfg, allow_launch_unlocked=allow_launch_unlocked)
    _validate_model(cfg)
    _validate_dataset_and_leakage(cfg)
    _validate_curriculum_and_scope(cfg)
    if require_grad_proof:
        _validate_grad_proof(proof_json or cfg.truetime_joint_selector_gate.selector_grad_proof_path)
    return cfg


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=CONFIG_DEFAULT)
    parser.add_argument("--require-grad-proof", action="store_true")
    parser.add_argument("--allow-launch-unlocked", action="store_true")
    parser.add_argument("--proof-json", default=None)
    args = parser.parse_args(argv)
    validate_config(
        args.config,
        require_grad_proof=args.require_grad_proof,
        allow_launch_unlocked=args.allow_launch_unlocked,
        proof_json=args.proof_json,
    )
    print(READY)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
