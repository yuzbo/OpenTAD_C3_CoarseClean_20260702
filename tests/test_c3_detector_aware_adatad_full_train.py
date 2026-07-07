from __future__ import annotations

import importlib.util
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_detector_aware_ledger_adatad_full_train.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_detector_aware_ledger_adatad_full_train_exec.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_c3_detector_aware_adatad_full_train.py"
LAUNCHER = ROOT / "scripts" / "run_c3_detector_aware_selector_adatad_full_train_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_c3_detector_aware_adatad_full_train_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _loadframes(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    assert len(matches) == 1
    return matches[0]


def test_detector_aware_adatad_config_supports_fixed384_fixed768_and_dynamic(monkeypatch) -> None:
    expected = {
        "detector_aware_fixed_384": (384, 384, "detector_aware_fixed_384"),
        "detector_aware_fixed_768": (768, 768, "detector_aware_fixed_768"),
        "detector_aware_dynamic": (768, None, "detector_aware_dynamic"),
    }
    for variant, (target_len, required_count, strategy) in expected.items():
        monkeypatch.setenv("C3_DETECTOR_AWARE_LEDGER_VARIANT", variant)
        monkeypatch.setenv("C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH", f"/tmp/{variant}.train.jsonl")
        monkeypatch.setenv("C3_DETECTOR_AWARE_VAL_LEDGER_PATH", f"/tmp/{variant}.val.jsonl")
        monkeypatch.setenv("C3_DETECTOR_AWARE_TEST_LEDGER_PATH", f"/tmp/{variant}.test.jsonl")

        cfg = Config.fromfile(str(CONFIG))

        assert cfg.experiment_scope.route_variant == "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3"
        assert cfg.experiment_scope.stage == "Stage-2 detector-aware offline selector"
        assert cfg.experiment_scope.end_to_end is False
        assert cfg.experiment_scope.deploy_claim_allowed is False
        assert cfg.detector_aware_ledger_variant == variant
        assert int(cfg.window_size) == target_len
        assert cfg.detector_aware_ledger_strategy == strategy
        assert cfg.c3_value_transport_source == "learned_detector_aware_policy_checkpoint"
        assert int(cfg.model.backbone.backbone.total_frames) == target_len
        assert int(cfg.model.projection.max_seq_len) == target_len
        assert cfg.solver.ema is True
        assert "val_eval_epochs" not in cfg.workflow
        assert int(cfg.workflow.val_eval_interval) == 10
        assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
        assert int(cfg.workflow.val_start_epoch) == 9
        assert cfg.baseline_comparison.matched_budget_baselines == ["p_action_only", "GAS-VT"]
        for split in ("train", "val", "test"):
            loader = _loadframes(cfg.dataset[split])
            assert loader.method == "bata_value_transport_ledger_subsample"
            assert int(loader.target_len) == target_len
            assert loader.bata_value_transport_require_selected_count == required_count
            assert loader.bata_value_transport_require_deployable is True
            assert loader.bata_value_transport_allow_short_valid_ratio_count is False
            assert loader.bata_value_transport_allow_missing_fallback is False
            assert loader.bata_value_transport_source == "learned_detector_aware_policy_checkpoint"


def test_detector_aware_validator_passes_locked_and_exec_configs(monkeypatch) -> None:
    monkeypatch.setenv("C3_DETECTOR_AWARE_LEDGER_VARIANT", "detector_aware_dynamic")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TRAIN_LEDGER_PATH", "/tmp/train.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_VAL_LEDGER_PATH", "/tmp/val.jsonl")
    monkeypatch.setenv("C3_DETECTOR_AWARE_TEST_LEDGER_PATH", "/tmp/test.jsonl")
    validator = _load_validator()

    cfg = validator.validate_config(str(CONFIG), require_ledger_files=False)
    assert cfg.c3_detector_aware_full_train_gate.launch_gate_passed is False
    assert cfg.detector_aware_ledger_strategy == "detector_aware_dynamic"

    exec_cfg = validator.validate_config(str(EXEC_CONFIG), require_ledger_files=False, allow_launch_unlocked=True)
    assert exec_cfg.c3_detector_aware_full_train_gate.launch_gate_passed is True
    assert exec_cfg.c3_detector_aware_full_train_gate.reviewed_execution_config is True


def test_detector_aware_launcher_is_gpu1_precheck_fail_closed_and_uses_detector_tools() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert 'ALLOW_C3_DETECTOR_AWARE_GPU0="${ALLOW_C3_DETECTOR_AWARE_GPU0:-0}"' in text
    assert 'REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY="${REQUIRE_C3_DETECTOR_AWARE_POINT_RESPONSIBILITY:-1}"' in text
    assert 'ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC="${ALLOW_C3_DETECTOR_AWARE_SURROGATE_STAGE2_DIAGNOSTIC:-0}"' in text
    assert "Stage-2 paper-main route requires C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL" in text
    assert "proposal-score surrogate Stage-2 is diagnostic-only" in text
    assert "--require-point-responsibility-utility" in text
    assert "explicit GPU0 override accepted for Stage-2" in text
    assert "Set ALLOW_C3_DETECTOR_AWARE_GPU0=1 only when GPU0 is explicitly assigned to this route." in text
    assert "detector_teacher_utility.py" in text
    assert "export_dense_adatad_teacher_points.py" in text
    assert "export_adatad_responsibility_utility.py" in text
    assert "validate_adatad_responsibility_utility.py" in text
    assert "C3_DETECTOR_AWARE_RESPONSIBILITY_POINTS_JSONL" in text
    assert "C3_DETECTOR_AWARE_RESPONSIBILITY_MANIFEST_JSON" in text
    assert "C3_DETECTOR_AWARE_DENSE_TEACHER_POINTS_JSONL" in text
    assert "C3_DETECTOR_AWARE_TEACHER_GENERATOR_MANIFEST_JSON" in text
    assert "--generator-manifest-json" in text
    assert "C3_DETECTOR_AWARE_TEACHER_UTILITY_EXPORT_SUMMARY_JSON" in text
    assert "teacher_utility_export.summary.json" in text
    assert "responsibility_utility_export.summary.json" in text
    assert "validate_teacher_utility_export_evidence" in text
    assert "validate_responsibility_utility_export" in text
    assert "train_detector_aware_acquisition_policy.py" in text
    assert "run_detector_aware_ledger_pipeline.py" in text
    assert "validate_detector_aware_policy_ledger.py" in text
    assert "validate_c3_detector_aware_adatad_full_train.py" in text
    assert "--allow-short-valid-ratio-count" not in text
    assert 'DETECTOR_AWARE_ADATAD_VARIANTS="${DETECTOR_AWARE_ADATAD_VARIANTS:-detector_aware_fixed_384}"' in text
    assert 'ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384="${ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384:-0}"' in text
    assert "exceeds the <=384 main-claim budget" in text
    assert "ALLOW_C3_DETECTOR_AWARE_DIAGNOSTIC_GT384=1" in text
    assert "learned_detector_aware_policy_checkpoint" in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert "ALLOW_C3_DETECTOR_AWARE_ADATAD_FULLTRAIN=1" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text
