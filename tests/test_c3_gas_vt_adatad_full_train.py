from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_gas_vt_ledger_adatad_full_train.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_gas_vt_ledger_adatad_full_train_exec.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_c3_paction_learned_adatad_full_train.py"
LAUNCHER = ROOT / "scripts" / "run_c3_gas_vt_policy_adatad_full_train_gpu1.sh"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_c3_gas_vt_adatad_full_train_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _loadframes(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    assert len(matches) == 1
    return matches[0]


def test_gas_vt_adatad_config_supports_fixed384_fixed768_and_dynamic(monkeypatch) -> None:
    expected = {
        "gas_vt_fixed_384": (384, 384, "gas_vt_fixed_384"),
        "gas_vt_fixed_768": (768, 768, "gas_vt_fixed_768"),
        "gas_vt_dynamic": (768, None, "gas_vt_dynamic"),
    }
    for variant, (target_len, required_count, strategy) in expected.items():
        monkeypatch.setenv("C3_GAS_VT_LEDGER_VARIANT", variant)
        monkeypatch.setenv("C3_GAS_VT_TRAIN_LEDGER_PATH", f"/tmp/{variant}.train.jsonl")
        monkeypatch.setenv("C3_GAS_VT_VAL_LEDGER_PATH", f"/tmp/{variant}.val.jsonl")
        monkeypatch.setenv("C3_GAS_VT_TEST_LEDGER_PATH", f"/tmp/{variant}.test.jsonl")

        cfg = Config.fromfile(str(CONFIG))

        assert cfg.experiment_scope.route_variant == "C3_GAS_VT_STRICT_LEDGER"
        assert cfg.experiment_scope.stage == "gas_vt_ledger_original_adatad_full_train"
        assert cfg.experiment_scope.selection_strategy == strategy
        assert cfg.gas_vt_ledger_variant == variant
        assert int(cfg.window_size) == target_len
        assert int(cfg.dense_window_size) == 768
        assert cfg.c3_value_transport_source == "learned_paction_gas_vt_policy_checkpoint"
        assert int(cfg.model.backbone.backbone.total_frames) == target_len
        assert int(cfg.model.projection.max_seq_len) == target_len
        assert cfg.solver.ema is True
        assert "val_eval_epochs" not in cfg.workflow
        assert int(cfg.workflow.val_eval_interval) == 10
        assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
        assert int(cfg.workflow.val_start_epoch) == 9
        assert "frame_selector" not in repr(cfg.model)
        for split in ("train", "val", "test"):
            loader = _loadframes(cfg.dataset[split])
            assert loader.method == "bata_value_transport_ledger_subsample"
            assert int(loader.target_len) == target_len
            assert loader.bata_value_transport_require_selected_count == required_count
            assert loader.bata_value_transport_require_deployable is True
            assert loader.bata_value_transport_allow_missing_fallback is False
            assert loader.bata_value_transport_allow_short_valid_ratio_count is True
            assert loader.bata_value_transport_source == "learned_paction_gas_vt_policy_checkpoint"


def test_gas_vt_adatad_config_uses_absolute_pretrain_env(monkeypatch) -> None:
    pretrain = "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    monkeypatch.setenv("C3_GAS_VT_LEDGER_VARIANT", "gas_vt_fixed_384")
    monkeypatch.setenv("C3_GAS_VT_TRAIN_LEDGER_PATH", "/tmp/train.jsonl")
    monkeypatch.setenv("C3_GAS_VT_VAL_LEDGER_PATH", "/tmp/val.jsonl")
    monkeypatch.setenv("C3_GAS_VT_TEST_LEDGER_PATH", "/tmp/test.jsonl")
    monkeypatch.setenv("C3_GAS_VT_ADATAD_PRETRAIN_PATH", pretrain)

    cfg = Config.fromfile(str(CONFIG))

    assert cfg.model.backbone.custom.pretrain == pretrain
    assert cfg.c3_gas_vt_adatad_pretrain_path == pretrain


def test_gas_vt_validator_passes_locked_and_exec_configs(monkeypatch) -> None:
    monkeypatch.setenv("C3_GAS_VT_LEDGER_VARIANT", "gas_vt_dynamic")
    monkeypatch.setenv("C3_GAS_VT_TRAIN_LEDGER_PATH", "/tmp/gas_train.jsonl")
    monkeypatch.setenv("C3_GAS_VT_VAL_LEDGER_PATH", "/tmp/gas_val.jsonl")
    monkeypatch.setenv("C3_GAS_VT_TEST_LEDGER_PATH", "/tmp/gas_test.jsonl")
    validator = _load_validator()

    cfg = validator.validate_config(str(CONFIG), require_ledger_files=False)
    assert cfg.c3_gas_vt_ledger_full_train_gate.launch_gate_passed is False
    assert cfg.gas_vt_ledger_strategy == "gas_vt_dynamic"

    exec_cfg = validator.validate_config(str(EXEC_CONFIG), require_ledger_files=False, allow_launch_unlocked=True)
    assert exec_cfg.c3_gas_vt_ledger_full_train_gate.launch_gate_passed is True
    assert exec_cfg.c3_gas_vt_ledger_full_train_gate.reviewed_execution_config is True


def test_gas_vt_full_train_gate_requires_gas_vt_source_and_provenance(tmp_path: Path, monkeypatch) -> None:
    checkpoint_sha = "c" * 64
    ledger = tmp_path / "value_transport_ledger_gas_vt_fixed_384.jsonl"
    row = {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": list(range(384)),
        "target_len": 384,
        "selected_count": 384,
        "valid_len": 768,
        "dense_len": 768,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "learned_paction_gas_vt_policy_checkpoint",
        "policy_checkpoint_path": str(tmp_path / "policy.pth"),
        "policy_checkpoint_sha256": checkpoint_sha,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
        "diagnostics": {
            "uniform_visible_fill_count": 0,
            "source_strategy": "gas_vt_fixed_384",
            "policy_source": "learned_paction_gas_vt_policy_checkpoint",
            "policy_checkpoint_sha256": checkpoint_sha,
        },
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv("C3_GAS_VT_LEDGER_VARIANT", "gas_vt_fixed_384")
    monkeypatch.setenv("C3_GAS_VT_TRAIN_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_GAS_VT_VAL_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_GAS_VT_TEST_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_GAS_VT_LEDGER_CONFIG_HASH", checkpoint_sha)
    validator = _load_validator()
    cfg = validator.validate_config(str(CONFIG), require_ledger_files=False)

    try:
        validator._validate_ledger_file(ledger, cfg=cfg, require_exists=True)
    except AssertionError as exc:
        assert "p_action provenance" in str(exc)
    else:
        raise AssertionError("missing provenance should fail the GAS-VT full-train ledger gate")

    row["diagnostics"]["p_action_provenance"] = {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    validator._validate_ledger_file(ledger, cfg=cfg, require_exists=True)


def test_gas_vt_launcher_is_gpu1_precheck_fail_closed_and_uses_gas_vt_tools() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert 'ALLOW_C3_GAS_VT_GPU0="${ALLOW_C3_GAS_VT_GPU0:-0}"' in text
    assert "explicit GPU0 override accepted for Stage-0/1" in text
    assert "Set ALLOW_C3_GAS_VT_GPU0=1 only after explicitly stopping the GPU0 model zoo." in text
    assert "train_gap_aware_acquisition_policy.py" in text
    assert "--allow-missing-split-from-source-path" in text
    assert "--allow-gt-diagnostics-in-training-source" in text
    assert 'C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD="${C3_GAS_VT_REUSE_EXISTING_LEDGER_BUILD:-0}"' in text
    assert "reusing existing GAS-VT ledgers" in text
    assert 'GAS_VT_MAX_P95_UNSELECTED_HOLE="${GAS_VT_MAX_P95_UNSELECTED_HOLE:-${GAS_VT_MAX_UNSELECTED_HOLE}}"' in text
    assert 'GAS_VT_MAX_UNIFORM_SIMILARITY="${GAS_VT_MAX_UNIFORM_SIMILARITY:-0.60}"' in text
    assert "run_gap_aware_ledger_pipeline.py" in text
    assert "apply_gap_aware_acquisition_policy.py" in text
    assert "validate_paction_learned_policy_ledger.py" in text
    assert "validate_c3_paction_learned_adatad_full_train.py" in text
    assert "learned_paction_gas_vt_policy_checkpoint" in text
    assert "gas_vt_fixed_384 gas_vt_fixed_768 gas_vt_dynamic" in text
    assert "C3_GAS_VT_ADATAD_PRETRAIN_PATH" in text
    assert "required file missing: ${ADATAD_PRETRAIN_PATH}" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text
    assert "formal full train must run inside a Slurm allocation/step" in text
