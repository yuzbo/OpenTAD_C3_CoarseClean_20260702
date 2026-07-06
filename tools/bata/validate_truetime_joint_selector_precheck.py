from __future__ import annotations

import argparse
import json
from pathlib import Path

from mmengine.config import Config


CONFIG_DEFAULT = "configs/adatad/thumos/c3_truetime_joint_selector_c3_adatad_smoke.py"
READY = "TRUETIME_JOINT_SELECTOR_PRECHECK_PASS"
ROUTE = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
SMOKE_STAGE = "stage3_4_experimental_smoke"
PRECHECK_STAGE = "stage3_true_time_e2e_adatad_selector_precheck"
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
    stage = str(gate.stage)
    smoke_only = _as_bool(gate.get("smoke_only", False))
    _require(gate.route_variant == ROUTE, "wrong route variant")
    _require(stage in (SMOKE_STAGE, PRECHECK_STAGE), "wrong stage")
    if stage == SMOKE_STAGE:
        _require(smoke_only, "smoke route must remain smoke_only")
    else:
        _require(smoke_only is False, "precheck route must not be smoke_only")
        _require(_as_bool(gate.get("precheck_only_default", False)), "precheck route must default to PRECHECK_ONLY")
        _require(_as_bool(gate.get("fulltrain_candidate", False)), "precheck route must declare fulltrain candidate")
        _require(
            _as_bool(gate.get("real_detector_gradient_proof_required", False)),
            "precheck route must require real detector gradient proof",
        )
    _require(_as_bool(gate.requires_launch_gate), "launch gate must be required")
    _require(_as_bool(gate.requires_selector_grad_nonzero), "selector grad gate must be required")
    _require(
        _as_bool(gate.get("requires_actionformer_detector_grad_nonzero", False)),
        "ActionFormer detector grad gate must be required",
    )
    _require(_as_bool(gate.requires_geometry_roundtrip), "geometry gate must be required")
    _require(
        _as_bool(gate.get("requires_physical_grid_actionformer", False)),
        "physical-grid ActionFormer gate must be required",
    )
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
    _require(
        cfg.experiment_scope.detector_stack == "truetime_physical_grid_actionformer_frame_selector_slot",
        "TrueTime route must use the physical-grid ActionFormer detector stack",
    )
    _require(
        _as_bool(cfg.experiment_scope.get("uses_physical_grid_actionformer", False)),
        "experiment scope must declare physical-grid ActionFormer usage",
    )
    _require(
        _as_bool(cfg.experiment_scope.get("changes_loss_assignment", False)),
        "TrueTime physical-grid route must declare changed loss-assignment geometry",
    )
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
    physical_grid = cfg.model.rpn_head.physical_grid_actionformer
    _require(_as_bool(physical_grid.enabled), "main ActionFormer rpn_head must enable physical_grid_actionformer")
    _require(_as_bool(physical_grid.required), "main ActionFormer physical grid must be required")
    _require(_as_bool(physical_grid.strict), "main ActionFormer physical grid must be strict")
    _require(physical_grid.coordinate_space == "true_time_dense_index", "main physical grid coordinate space mismatch")
    _require(physical_grid.selected_position_key == "irregular_selected_positions", "main physical grid selected key mismatch")
    _require(physical_grid.dense_valid_len_key == "irregular_dense_valid_len", "main physical grid dense-valid key mismatch")
    _require(
        _as_bool(physical_grid.get("requires_irregular_native_axis", False)),
        "main physical grid must require irregular_native_axis",
    )

    smoke_only = _as_bool(cfg.truetime_joint_selector_gate.get("smoke_only", False))
    if smoke_only:
        smoke_model = cfg.truetime_detector_path_smoke_model
        smoke_selector = smoke_model.frame_selector
        _require(smoke_model.type == "TrueTimeJointSelectorSmokeDetector", "wrong detector-path smoke model type")
        _require(smoke_selector.type == "TrueTimeRelaxedHardTopKSelector", "wrong smoke selector type")
        _require(smoke_selector.detector_gradient_mode == "st_sparse_gather", "smoke detector gradient mode mismatch")
        _require(int(smoke_selector.selected_count) < int(smoke_selector.dense_len), "smoke selector must be sparse")
        _require(_as_bool(smoke_selector.allow_gt_selection) is False, "smoke GT selection must be disabled")
        _require(_as_bool(smoke_selector.allow_teacher_utility) is False, "smoke teacher selection must be disabled")

    actionformer_smoke = cfg.truetime_actionformer_path_smoke_model
    actionformer_selector = actionformer_smoke.frame_selector
    _require(actionformer_smoke.type == "ActionFormer", "wrong ActionFormer detector-path smoke model type")
    _require(actionformer_selector.type == "TrueTimeRelaxedHardTopKSelector", "wrong ActionFormer smoke selector type")
    _require(
        actionformer_selector.detector_gradient_mode == "st_sparse_gather",
        "ActionFormer smoke detector gradient mode mismatch",
    )
    _require(
        int(actionformer_selector.selected_count) < int(actionformer_selector.dense_len),
        "ActionFormer smoke selector must be sparse",
    )
    _require(
        int(actionformer_smoke.projection.max_seq_len) == int(actionformer_selector.selected_count),
        "ActionFormer smoke projection length must equal selected_count",
    )
    _require(_as_bool(actionformer_selector.allow_gt_selection) is False, "ActionFormer smoke GT selection must be disabled")
    _require(
        _as_bool(actionformer_selector.allow_teacher_utility) is False,
        "ActionFormer smoke teacher selection must be disabled",
    )
    actionformer_physical_grid = actionformer_smoke.rpn_head.physical_grid_actionformer
    _require(
        _as_bool(actionformer_physical_grid.enabled),
        "ActionFormer smoke rpn_head must enable physical_grid_actionformer",
    )
    _require(_as_bool(actionformer_physical_grid.required), "ActionFormer smoke physical grid must be required")
    _require(_as_bool(actionformer_physical_grid.strict), "ActionFormer smoke physical grid must be strict")
    _require(
        actionformer_physical_grid.coordinate_space == "true_time_dense_index",
        "ActionFormer smoke physical grid coordinate space mismatch",
    )
    _require(
        _as_bool(actionformer_physical_grid.get("requires_irregular_native_axis", False)),
        "ActionFormer smoke physical grid must require irregular_native_axis",
    )
    if not smoke_only:
        actionformer_precheck = cfg.truetime_actionformer_path_precheck_model
        precheck_selector = actionformer_precheck.frame_selector
        _require(actionformer_precheck.type == "ActionFormer", "wrong ActionFormer precheck model type")
        _require(precheck_selector.type == "TrueTimeRelaxedHardTopKSelector", "wrong ActionFormer precheck selector type")
        _require(
            precheck_selector.detector_gradient_mode == "st_sparse_gather",
            "ActionFormer precheck detector gradient mode mismatch",
        )
        _require(
            int(actionformer_precheck.projection.max_seq_len) == int(precheck_selector.selected_count),
            "ActionFormer precheck projection length must equal selected_count",
        )
        _require(_as_bool(precheck_selector.allow_gt_selection) is False, "ActionFormer precheck GT selection disabled")
        _require(_as_bool(precheck_selector.allow_teacher_utility) is False, "ActionFormer precheck teacher selection disabled")
        precheck_physical_grid = actionformer_precheck.rpn_head.physical_grid_actionformer
        _require(_as_bool(precheck_physical_grid.enabled), "ActionFormer precheck must enable physical grid")
        _require(_as_bool(precheck_physical_grid.required), "ActionFormer precheck physical grid must be required")
        _require(_as_bool(precheck_physical_grid.strict), "ActionFormer precheck physical grid must be strict")
        _require(
            precheck_physical_grid.coordinate_space == "true_time_dense_index",
            "ActionFormer precheck physical grid coordinate space mismatch",
        )
        _require(
            _as_bool(precheck_physical_grid.get("requires_irregular_native_axis", False)),
            "ActionFormer precheck physical grid must require irregular_native_axis",
        )

    distill_gate = cfg.sparse_detector_distillation_gate
    _require(_as_bool(distill_gate.enabled) is False, "sparse distill gate must default disabled")
    _require(_as_bool(distill_gate.fail_closed), "sparse distill gate must fail closed")
    _require(_as_bool(distill_gate.required_before_full_detector_loss), "sparse distill must gate full detector loss")
    _require(_as_bool(distill_gate.map_claim_allowed) is False, "sparse distill mAP claim must be locked")
    adapter = cfg.sparse_detector_distillation.loss_adapter
    _require(adapter.type == "SparseDetectorDistillationLossAdapter", "wrong sparse distill loss adapter type")
    _require(_as_bool(adapter.fail_closed_without_teacher_targets), "sparse distill adapter must fail closed without targets")
    _require(_as_bool(adapter.map_claim_allowed) is False, "sparse distill adapter mAP claim must be locked")


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
    stage = str(cfg.truetime_joint_selector_gate.stage)
    smoke_only = _as_bool(cfg.truetime_joint_selector_gate.get("smoke_only", False))
    _require(cfg.experiment_scope.route_variant == ROUTE, "experiment route mismatch")
    _require(cfg.stage2_offline_detector_utility_selector_dependency is False, "must stay separate from Stage-2 offline selector")
    _require(cfg.experiment_scope.modifies_evaluator_map_semantics is False, "evaluator mAP semantics must stay unchanged")
    if smoke_only:
        _require(stage == SMOKE_STAGE, "smoke route stage mismatch")
        _require(cfg.workflow.end_epoch == 1, "smoke route must run one epoch max")
        _require(cfg.workflow.max_train_iters == 2, "smoke route must cap train iters at 2")
        _require(list(cfg.truetime_joint_selector_gate.eight_week_questions_answered_by) == [
            "true_time_roundtrip_tests",
            "segment_inverse_map_tests",
            "selector_detector_loss_gradient_smoke",
            "actionformer_forward_train_selector_gradient_smoke",
            "fail_closed_curriculum_and_claim_gates",
        ], "8-week question evidence list changed")
    else:
        _require(stage == PRECHECK_STAGE, "precheck route stage mismatch")
        _require(cfg.workflow.end_epoch >= 1, "precheck route must define a positive epoch budget")
        _require(cfg.workflow.max_train_iters <= 4, "default precheck route must cap train iters at 4")
        _require(cfg.experiment_scope.full_map_claim_required is True, "precheck must require full mAP before claims")
        _require(cfg.experiment_scope.end_to_end_claim_allowed is False, "end-to-end claim must be locked")
        _require(_as_bool(cfg.truetime_joint_selector_gate.allow_detector_map) is False, "precheck must not unlock mAP loop")
        _require(
            "real_actionformer_forward_train_selector_gradient_precheck"
            in list(cfg.truetime_joint_selector_gate.eight_week_questions_answered_by),
            "precheck evidence must include real ActionFormer detector gradient proof",
        )
    _require(cfg.workflow.val_eval_interval <= 0, "route must not unlock mAP loop")
    for metric in (
        "selector_grad_norm",
        "detector_loss_selector_grad_norm",
        "selected_input_selector_grad_norm",
        "selected_count_mean",
        "selected_count_std",
        "entropy",
        "loss_cls",
        "loss_reg",
        "actionformer_cls_loss",
        "actionformer_reg_loss",
        "actionformer_detector_loss_selector_grad_norm",
        "sparse_distill_loss",
        "geometry_roundtrip",
        "prediction_inverse_map",
        "claim_locks",
    ):
        _require(metric in cfg.truetime_metrics_to_log, f"missing metric {metric}")
    for forbidden_metric in ("mAP", "tIoU"):
        _require(forbidden_metric not in cfg.truetime_metrics_to_log, f"smoke metrics must not include {forbidden_metric}")


def _validate_grad_proof(path, *, require_real_detector=False):
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
    if not require_real_detector:
        loss_keys = payload.get("loss_keys", [])
        _require("loss_cls" in loss_keys, "detector proof missing loss_cls")
        _require("loss_reg" in loss_keys, "detector proof missing loss_reg")
        _require(payload.get("proof_source") == "registered_detector_forward_train_cost_backward", "wrong proof source")
    _require(float(payload.get("selector_grad_norm", 0.0)) > 0.0, "selector_grad_norm proof must be > 0")
    _require(payload.get("selector_grad_nonzero") is True, "selector_grad_nonzero proof missing")
    _require(
        payload.get("actionformer_proof_source") == "opentad_actionformer_forward_train_cost_backward",
        "wrong ActionFormer proof source",
    )
    _require(
        payload.get("actionformer_detector_loss_selector_grad_passed") is True,
        "ActionFormer detector-loss selector gradient proof missing",
    )
    _require(
        float(payload.get("actionformer_detector_loss_selector_grad_norm", 0.0)) > 0.0,
        "ActionFormer detector-loss selector gradient proof must be > 0",
    )
    actionformer_loss_keys = payload.get("actionformer_loss_keys", [])
    _require("cls_loss" in actionformer_loss_keys, "ActionFormer proof missing cls_loss")
    _require("reg_loss" in actionformer_loss_keys, "ActionFormer proof missing reg_loss")
    if not require_real_detector:
        _require(payload.get("actionformer_selected_axis_smoke") is True, "ActionFormer selected-axis smoke flag missing")
    if require_real_detector:
        _require(payload.get("stage") == PRECHECK_STAGE, "real detector proof stage mismatch")
        _require(
            payload.get("real_detector_proof_source") == "opentad_actionformer_forward_train_cost_backward",
            "wrong real detector proof source",
        )
        _require(
            payload.get("real_detector_loss_selector_grad_passed") is True,
            "real detector-loss selector gradient proof missing",
        )
        _require(
            float(payload.get("real_detector_loss_selector_grad_norm", 0.0)) > 0.0,
            "real detector-loss selector gradient proof must be > 0",
        )
        real_detector_loss_keys = payload.get("real_detector_loss_keys", [])
        _require("cls_loss" in real_detector_loss_keys, "real detector proof missing cls_loss")
        _require("reg_loss" in real_detector_loss_keys, "real detector proof missing reg_loss")
        _require(payload.get("actionformer_selected_axis_smoke") is False, "precheck proof must not be smoke-only")
        _require(payload.get("actionformer_physical_grid_precheck") is True, "physical-grid precheck flag missing")
    _require(payload.get("sparse_distill_adapter_ready") is True, "sparse distill adapter proof missing")
    _require(payload.get("sparse_distill_claim_allowed") is False, "sparse distill claim must remain locked")
    _require(payload.get("sparse_distill_map_claim_allowed") is False, "sparse distill mAP claim must remain locked")
    _require(
        payload.get("sparse_distill_proof_source") == "fail_closed_sparse_detector_distillation_adapter",
        "sparse distill proof source mismatch",
    )


def validate_config(config_path=CONFIG_DEFAULT, *, require_grad_proof=False, allow_launch_unlocked=False, proof_json=None):
    cfg = Config.fromfile(str(config_path))
    _validate_gate(cfg, allow_launch_unlocked=allow_launch_unlocked)
    _validate_model(cfg)
    _validate_dataset_and_leakage(cfg)
    _validate_curriculum_and_scope(cfg)
    if require_grad_proof:
        _validate_grad_proof(
            proof_json or cfg.truetime_joint_selector_gate.selector_grad_proof_path,
            require_real_detector=_as_bool(
                cfg.truetime_joint_selector_gate.get("real_detector_gradient_proof_required", False)
            ),
        )
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
