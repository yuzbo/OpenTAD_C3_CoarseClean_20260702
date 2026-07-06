from __future__ import annotations

import importlib.util
from pathlib import Path
import json

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_c3_adatad_smoke.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_truetime_joint_selector_c3_adatad_smoke_exec.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_truetime_joint_selector_precheck.py"
LAUNCHER = ROOT / "scripts" / "run_c3_truetime_joint_selector_adatad_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_truetime_joint_selector_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _collect(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    assert len(matches) == 1
    return matches[0]


def test_truetime_joint_selector_config_is_stage34_locked_and_explicit() -> None:
    cfg = Config.fromfile(str(CONFIG))

    assert cfg.experiment_scope.route_variant == "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
    assert cfg.experiment_scope.stage == "stage3_4_experimental_smoke"
    assert cfg.experiment_scope.paper_claim_allowed is False
    assert cfg.experiment_scope.runtime_flops_claim_allowed is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.launch_gate_passed is False
    assert cfg.truetime_joint_selector_gate.requires_selector_grad_nonzero is True
    assert cfg.truetime_joint_selector_gate.requires_geometry_roundtrip is True
    assert cfg.truetime_joint_selector_gate.end_to_end_claim_allowed is False
    assert cfg.truetime_joint_selector_gate.eight_week_questions_answered_by == [
        "true_time_roundtrip_tests",
        "segment_inverse_map_tests",
        "selector_detector_loss_gradient_smoke",
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
    assert cfg.truetime_detector_path_smoke_model.type == "TrueTimeJointSelectorSmokeDetector"
    assert cfg.truetime_detector_path_smoke_model.frame_selector.detector_gradient_mode == "st_sparse_gather"
    assert cfg.truetime_detector_path_smoke_model.frame_selector.selected_count == 4
    assert cfg.truetime_detector_path_smoke_model.frame_selector.dense_len == 8
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
        json.dumps(
            {
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
            }
        ),
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
