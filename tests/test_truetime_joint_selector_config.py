from __future__ import annotations

import importlib.util
from pathlib import Path
import json

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_c3_adatad_smoke.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_c3_adatad_smoke_exec.py"
PRECHECK_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_adatad_precheck.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_truetime_joint_selector_precheck.py"
DUCA_VALIDATOR = ROOT / "tools" / "bata" / "validate_duca_stage23_precheck.py"
PRECHECK_RUNNER = ROOT / "tools" / "bata" / "run_truetime_joint_selector_precheck.py"
LAUNCHER = ROOT / "scripts" / "run_c3_truetime_joint_selector_adatad_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_truetime_joint_selector_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_duca_validator():
    spec = importlib.util.spec_from_file_location("validate_duca_stage23_test", DUCA_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _collect(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    assert len(matches) == 1
    return matches[0]


def _selector_step_proof_fields() -> dict:
    return {
        "selector_param_delta_l2": 0.004,
        "selector_param_delta_passed": True,
        "selected_position_drift_mean": 0.5,
        "selected_position_drift_max": 1.0,
        "selected_position_drift_passed": True,
        "selector_logits_drift_l2": 0.03,
        "selector_logits_drift_max": 0.02,
        "selector_logits_drift_passed": True,
    }


def _real_actionformer_precheck_proof_payload(**overrides) -> dict:
    payload = {
        "route_variant": "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3",
        "stage": "stage3_true_time_e2e_adatad_selector_precheck",
        "geometry_roundtrip_passed": True,
        "prediction_inverse_map_passed": True,
        "selected_input_st_gradient_passed": True,
        "selected_input_selector_grad_norm": 0.25,
        "detector_loss_selector_grad_passed": True,
        "detector_loss_selector_grad_norm": 0.31,
        "selector_grad_norm": 0.31,
        "selector_grad_nonzero": True,
        "real_detector_loss_selector_grad_passed": True,
        "real_detector_loss_selector_grad_norm": 0.31,
        "real_detector_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "real_detector_loss_keys": ["cls_loss", "reg_loss"],
        "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "actionformer_detector_loss_selector_grad_passed": True,
        "actionformer_detector_loss_selector_grad_norm": 0.31,
        "actionformer_loss_keys": ["cls_loss", "reg_loss"],
        "actionformer_selected_axis_smoke": False,
        "actionformer_physical_grid_precheck": True,
        "sparse_distill_adapter_ready": True,
        "sparse_distill_claim_allowed": False,
        "sparse_distill_map_claim_allowed": False,
        "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
        **_selector_step_proof_fields(),
    }
    payload.update(overrides)
    return payload


def _smoke_detector_loss_proof_payload(**overrides) -> dict:
    payload = {
        "route_variant": "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3",
        "geometry_roundtrip_passed": True,
        "prediction_inverse_map_passed": True,
        "selected_input_st_gradient_passed": True,
        "selected_input_selector_grad_norm": 0.25,
        "detector_loss_selector_grad_passed": True,
        "detector_loss_selector_grad_norm": 0.25,
        "selector_grad_norm": 0.25,
        "selector_grad_nonzero": True,
        "loss_keys": ["loss_cls", "loss_reg"],
        "proof_source": "registered_detector_forward_train_cost_backward",
        "actionformer_proof_source": "opentad_actionformer_forward_train_cost_backward",
        "actionformer_detector_loss_selector_grad_passed": True,
        "actionformer_detector_loss_selector_grad_norm": 0.31,
        "actionformer_loss_keys": ["cls_loss", "reg_loss"],
        "actionformer_selected_axis_smoke": True,
        "sparse_distill_adapter_ready": True,
        "sparse_distill_claim_allowed": False,
        "sparse_distill_map_claim_allowed": False,
        "sparse_distill_proof_source": "fail_closed_sparse_detector_distillation_adapter",
        **_selector_step_proof_fields(),
    }
    payload.update(overrides)
    return payload


def test_truetime_joint_selector_config_is_stage34_locked_and_explicit() -> None:
    cfg = Config.fromfile(str(CONFIG))

    assert cfg.experiment_scope.route_variant == "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
    assert cfg.experiment_scope.stage == "stage3_4_experimental_smoke"
    assert cfg.experiment_scope.detector_stack == "truetime_physical_grid_actionformer_frame_selector_slot"
    assert cfg.experiment_scope.uses_physical_grid_actionformer is True
    assert cfg.experiment_scope.changes_loss_assignment is True
    assert cfg.experiment_scope.paper_claim_allowed is False
    assert cfg.experiment_scope.runtime_flops_claim_allowed is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.launch_gate_passed is False
    assert cfg.truetime_joint_selector_gate.requires_selector_grad_nonzero is True
    assert cfg.truetime_joint_selector_gate.requires_actionformer_detector_grad_nonzero is True
    assert cfg.truetime_joint_selector_gate.requires_geometry_roundtrip is True
    assert cfg.truetime_joint_selector_gate.requires_physical_grid_actionformer is True
    assert cfg.truetime_joint_selector_gate.end_to_end_claim_allowed is False
    assert cfg.sparse_detector_distillation_gate.enabled is False
    assert cfg.sparse_detector_distillation_gate.fail_closed is True
    assert cfg.sparse_detector_distillation_gate.map_claim_allowed is False
    assert cfg.sparse_detector_distillation_gate.required_before_full_detector_loss is True
    assert cfg.sparse_detector_distillation.loss_adapter.type == "SparseDetectorDistillationLossAdapter"
    assert cfg.sparse_detector_distillation.loss_adapter.fail_closed_without_teacher_targets is True
    assert cfg.sparse_detector_distillation.loss_adapter.map_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.eight_week_questions_answered_by == [
        "true_time_roundtrip_tests",
        "segment_inverse_map_tests",
        "selector_detector_loss_gradient_smoke",
        "actionformer_forward_train_selector_gradient_smoke",
        "fail_closed_curriculum_and_claim_gates",
    ]
    assert cfg.truetime_joint_selector_gate.required_phases == [
        "dense_teacher",
        "selector_pretrain",
        "frozen_detector",
        "sparse_detector",
        "joint_finetune",
    ]
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "TrueTimeRelaxedHardTopKSelector"
    assert cfg.model.frame_selector.selected_count == 384
    assert cfg.model.frame_selector.dense_len == 768
    assert cfg.model.frame_selector.allow_teacher_utility is False
    assert cfg.model.frame_selector.allow_gt_selection is False
    assert cfg.model.frame_selector.detector_gradient_mode == "st_sparse_gather"
    assert cfg.model.frame_selector.slot_softmax_temperature == 0.7
    assert cfg.model.frame_selector.slot_distance_penalty == 2.0
    assert cfg.model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.model.rpn_head.physical_grid_actionformer.strict is True
    assert cfg.model.rpn_head.physical_grid_actionformer.coordinate_space == "true_time_dense_index"
    assert cfg.model.rpn_head.physical_grid_actionformer.selected_position_key == "irregular_selected_positions"
    assert cfg.model.rpn_head.physical_grid_actionformer.dense_valid_len_key == "irregular_dense_valid_len"
    assert cfg.truetime_detector_path_smoke_model.type == "TrueTimeJointSelectorSmokeDetector"
    assert cfg.truetime_detector_path_smoke_model.frame_selector.detector_gradient_mode == "st_sparse_gather"
    assert cfg.truetime_detector_path_smoke_model.frame_selector.selected_count == 4
    assert cfg.truetime_detector_path_smoke_model.frame_selector.dense_len == 8
    assert cfg.truetime_actionformer_path_smoke_model.type == "ActionFormer"
    assert cfg.truetime_actionformer_path_smoke_model.frame_selector.detector_gradient_mode == "st_sparse_gather"
    assert cfg.truetime_actionformer_path_smoke_model.frame_selector.selected_count == 4
    assert cfg.truetime_actionformer_path_smoke_model.frame_selector.dense_len == 8
    assert cfg.truetime_actionformer_path_smoke_model.projection.max_seq_len == 4
    assert cfg.truetime_actionformer_path_smoke_model.rpn_head.physical_grid_actionformer.enabled is True
    assert cfg.truetime_actionformer_path_smoke_model.rpn_head.physical_grid_actionformer.required is True
    assert cfg.truetime_actionformer_path_smoke_model.rpn_head.physical_grid_actionformer.strict is True
    assert cfg.truetime_actionformer_path_smoke_model.rpn_head.physical_grid_actionformer.coordinate_space == "true_time_dense_index"
    assert cfg.truetime_actionformer_path_smoke_model.rpn_head.physical_grid_actionformer.requires_irregular_native_axis is True
    assert cfg.workflow.max_train_iters == 2
    assert cfg.truetime_metrics_to_log == [
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
    ]
    assert "mAP" not in cfg.truetime_metrics_to_log
    assert "tIoU" not in cfg.truetime_metrics_to_log
    assert "gt_segments" not in _collect(cfg.dataset.test).get("keys", [])
    assert "gt_labels" not in _collect(cfg.dataset.test).get("keys", [])
    assert cfg.selection_contract.val.selection_uses_gt is False
    assert cfg.selection_contract.val.selection_uses_teacher is False
    assert cfg.selection_contract.test.selection_uses_gt is False
    assert cfg.selection_contract.test.selection_uses_teacher is False
    assert cfg.stage2_offline_detector_utility_selector_dependency is False
    for split in ("train", "val", "test"):
        meta_keys = set(_collect(cfg.dataset[split]).get("meta_keys", []))
        assert "truetime_dense_valid_len" in meta_keys
        assert "irregular_selected_count" in meta_keys
        assert "irregular_dense_valid_len" in meta_keys
        load_steps = [step for step in cfg.dataset[split].pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
        assert len(load_steps) == 1
        if split == "train":
            assert "window_size" not in cfg.dataset[split]
            assert load_steps[0].method == "random_trunc"
            assert load_steps[0].trunc_len == cfg.dense_window_size
        else:
            assert cfg.dataset[split].window_size == cfg.dense_window_size
            assert load_steps[0].method == "sliding_window"


def test_truetime_joint_selector_precheck_config_is_not_smoke_only_and_claim_locked() -> None:
    cfg = Config.fromfile(str(PRECHECK_CONFIG))

    assert cfg.experiment_scope.route_variant == "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
    assert cfg.experiment_scope.stage == "stage3_true_time_e2e_adatad_selector_precheck"
    assert cfg.experiment_scope.minimum_useful_deliverable == "tiny_synthetic_actionformer_gradient_precheck_ready_fulltrain_candidate"
    assert cfg.experiment_scope.paper_claim_allowed is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert cfg.experiment_scope.end_to_end_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.smoke_only is False
    assert cfg.truetime_joint_selector_gate.precheck_only_default is True
    assert any(
        "tiny synthetic real ActionFormer loss-path precheck only" in item
        for item in cfg.truetime_joint_selector_gate.limitations
    )
    assert cfg.truetime_joint_selector_gate.real_detector_gradient_proof_required is True
    assert cfg.truetime_joint_selector_gate.launch_gate_passed is False
    assert cfg.truetime_joint_selector_gate.end_to_end_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.paper_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.deploy_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.metric_claim_allowed is False
    assert cfg.model.type == "ActionFormer"
    assert cfg.model.frame_selector.type == "TrueTimeRelaxedHardTopKSelector"
    assert cfg.model.frame_selector.selected_count == 384
    assert cfg.model.frame_selector.dense_len == 768
    assert cfg.workflow.max_train_iters <= 4
    assert cfg.workflow.val_eval_interval <= 0
    assert "mAP" not in cfg.truetime_metrics_to_log
    assert "tIoU" not in cfg.truetime_metrics_to_log
    train_load_steps = [
        step for step in cfg.dataset.train.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"
    ]
    assert "window_size" not in cfg.dataset.train
    assert len(train_load_steps) == 1
    assert train_load_steps[0].method == "random_trunc"
    assert train_load_steps[0].trunc_len == cfg.dense_window_size
    assert cfg.dataset.val.window_size == cfg.dense_window_size
    assert cfg.dataset.test.window_size == cfg.dense_window_size


def test_truetime_joint_selector_precheck_validator_accepts_real_actionformer_proof_schema(tmp_path: Path) -> None:
    proof = tmp_path / "precheck_proof.json"
    proof.write_text(
        json.dumps(_real_actionformer_precheck_proof_payload()),
        encoding="utf-8",
    )
    validator = _load_validator()

    cfg = validator.validate_config(
        str(PRECHECK_CONFIG),
        require_grad_proof=True,
        allow_launch_unlocked=False,
        proof_json=str(proof),
    )

    assert cfg.truetime_joint_selector_gate.smoke_only is False
    assert cfg.truetime_joint_selector_gate.launch_gate_passed is False


def test_truetime_joint_selector_precheck_runner_defaults_to_precheck_config_and_real_detector() -> None:
    text = PRECHECK_RUNNER.read_text(encoding="utf-8")

    assert "c3_truetime_joint_selector_adatad_precheck.py" in text
    assert "truetime_actionformer_path_precheck_model" in text
    assert "opentad_actionformer_forward_train_cost_backward" in text
    assert "real_detector_loss_selector_grad_norm" in text
    assert "selector_param_delta_l2" in text
    assert "selected_position_drift_mean" in text
    assert "TrueTimeJointSelectorSmokeDetector" not in text


def test_truetime_selector_marks_physical_grid_metas_as_native_axis() -> None:
    selector_source = (ROOT / "opentad" / "models" / "selectors" / "truetime_joint_selector.py").read_text(encoding="utf-8")

    assert 'meta["detector_output_coordinate_space"] = self.coordinate_space' in selector_source
    assert 'meta["selected_axis_to_true_time_dense_index"] = positions' in selector_source
    assert 'meta["irregular_selected_positions"] = positions' in selector_source
    assert 'meta["irregular_native_axis"] = True' in selector_source


def test_truetime_joint_selector_validator_blocks_end_to_end_without_grad_proof(monkeypatch) -> None:
    monkeypatch.delenv("TRUETIME_SELECTOR_GRAD_PROOF_JSON", raising=False)
    validator = _load_validator()

    cfg = validator.validate_config(str(CONFIG), require_grad_proof=False)
    assert cfg.truetime_joint_selector_gate.launch_gate_passed is False

    try:
        validator.validate_config(str(EXEC_CONFIG), require_grad_proof=True, allow_launch_unlocked=True)
    except AssertionError as exc:
        assert "selected-input selector gradient proof" in str(exc)
    else:
        raise AssertionError("exec config must fail closed without a selected-input selector gradient proof")


def test_truetime_joint_selector_validator_accepts_detector_loss_proof_schema(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(_smoke_detector_loss_proof_payload()),
        encoding="utf-8",
    )
    validator = _load_validator()

    cfg = validator.validate_config(
        str(EXEC_CONFIG),
        require_grad_proof=True,
        allow_launch_unlocked=True,
        proof_json=str(proof),
    )

    assert cfg.truetime_joint_selector_gate.launch_gate_passed is True


def test_truetime_joint_selector_validator_rejects_zero_selector_step_delta(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(
            _real_actionformer_precheck_proof_payload(
                selector_param_delta_l2=0.0,
                selector_param_delta_passed=False,
            )
        ),
        encoding="utf-8",
    )
    validator = _load_validator()

    try:
        validator.validate_config(
            str(PRECHECK_CONFIG),
            require_grad_proof=True,
            allow_launch_unlocked=False,
            proof_json=str(proof),
        )
    except AssertionError as exc:
        assert "selector parameter delta" in str(exc)
    else:
        raise AssertionError("validator must fail closed without a positive selector parameter step delta")


def test_truetime_joint_selector_validator_rejects_missing_position_drift(tmp_path: Path) -> None:
    payload = _real_actionformer_precheck_proof_payload()
    payload.pop("selected_position_drift_mean")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps(payload), encoding="utf-8")
    validator = _load_validator()

    try:
        validator.validate_config(
            str(PRECHECK_CONFIG),
            require_grad_proof=True,
            allow_launch_unlocked=False,
            proof_json=str(proof),
        )
    except AssertionError as exc:
        assert "selected-position drift" in str(exc)
    else:
        raise AssertionError("validator must fail closed without selected-position drift evidence")


def test_duca_stage23_validator_requires_selector_step_proof(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(_real_actionformer_precheck_proof_payload(selector_param_delta_l2=0.0)),
        encoding="utf-8",
    )
    validator = _load_duca_validator()

    try:
        validator._validate_stage3_proof(proof)
    except AssertionError as exc:
        assert "selector parameter delta" in str(exc)
    else:
        raise AssertionError("DUCA Stage2/3 validator must require a positive selector parameter step delta")


def test_truetime_joint_selector_validator_rejects_unlocked_distill_claim(tmp_path: Path) -> None:
    proof = tmp_path / "proof.json"
    proof.write_text(
        json.dumps(_smoke_detector_loss_proof_payload(sparse_distill_claim_allowed=True)),
        encoding="utf-8",
    )
    validator = _load_validator()

    try:
        validator.validate_config(
            str(EXEC_CONFIG),
            require_grad_proof=True,
            allow_launch_unlocked=True,
            proof_json=str(proof),
        )
    except AssertionError as exc:
        assert "sparse distill claim" in str(exc)
    else:
        raise AssertionError("distill gate must fail closed when a sparse distill claim is unlocked")


def test_truetime_joint_selector_launcher_is_gpu1_precheck_default_and_slurm_gated() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert 'ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN="${ALLOW_TRUETIME_JOINT_SELECTOR_FULLTRAIN:-0}"' in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert 'OUTPUT_ROOT="${C3_TRUETIME_OUTPUT_ROOT:-${BASE}/projects/c3_lowres_action_probe/truetime_joint_selector}"' in text
    assert "--require-grad-proof" in text
    assert "--proof-json" in text
    assert "validate_truetime_joint_selector_precheck.py" in text
    assert "tests/test_truetime" in text
    assert "tests/test_truetime_joint_selector_config.py" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text
