from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_uniform_sparse_384_ledger_adatad_full_train.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_uniform_sparse_384_ledger_adatad_full_train_exec.py"
LAUNCHER = ROOT / "scripts" / "run_uniform_sparse_384_adatad_full_train_gpu0.sh"
GUARD = ROOT / "opentad" / "utils" / "training_guard.py"


def _loadframes(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    assert len(matches) == 1
    return matches[0]


def _collect(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    assert len(matches) == 1
    return matches[0]


def test_uniform_sparse_384_config_is_matched_prebackbone_adatad_baseline(monkeypatch) -> None:
    pretrain = "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH", "/tmp/uniform_train.jsonl")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_VAL_LEDGER_PATH", "/tmp/uniform_val.jsonl")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TEST_LEDGER_PATH", "/tmp/uniform_test.jsonl")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH", "uniform_exact_384_test_hash")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_ADATAD_PRETRAIN_PATH", pretrain)

    cfg = Config.fromfile(str(CONFIG))

    assert cfg.experiment_scope.route == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.experiment_scope.route_variant == "C3_UNIFORM_SPARSE_EXACT_384_BASELINE"
    assert cfg.experiment_scope.stage == "uniform_sparse_384_ledger_original_adatad_full_train"
    assert cfg.experiment_scope.selection_strategy == "uniform_exact_384"
    assert cfg.experiment_scope.selector_source == "exact_uniform_sparse_ledger_generator"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.uses_offline_deploy_selection_ledger is True
    assert cfg.experiment_scope.uses_uniform_scaffold is True
    assert cfg.experiment_scope.uses_uniform_fill is False
    assert cfg.experiment_scope.uses_gt is False
    assert cfg.experiment_scope.uses_teacher is False
    assert cfg.experiment_scope.uses_oracle is False
    assert cfg.experiment_scope.deploy_claim_allowed is False
    assert cfg.experiment_scope.paper_claim_allowed is False

    assert int(cfg.window_size) == 384
    assert int(cfg.dense_window_size) == 768
    assert int(cfg.model.backbone.backbone.total_frames) == 384
    assert int(cfg.model.projection.max_seq_len) == 384
    assert cfg.model.backbone.custom.pretrain == pretrain
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False
    assert "frame_selector" not in repr(cfg.model)
    assert cfg.solver.ema is True

    assert int(cfg.workflow.end_epoch) == 60
    assert cfg.workflow.max_train_iters is None
    assert int(cfg.workflow.val_eval_interval) == 10
    assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
    assert int(cfg.workflow.val_start_epoch) == 9
    assert "val_eval_epochs" not in cfg.workflow

    for split in ("train", "val", "test"):
        loader = _loadframes(cfg.dataset[split])
        assert loader.method == "bata_value_transport_ledger_subsample"
        assert loader.method_base == "sliding_window"
        assert int(loader.target_len) == 384
        assert int(loader.bata_value_transport_require_selected_count) == 384
        assert loader.bata_value_transport_require_deployable is True
        assert loader.bata_value_transport_allow_missing_fallback is False
        assert loader.bata_value_transport_allow_short_valid_ratio_count is True
        assert loader.remap_gt_to_selected_axis is True
        assert loader.bata_value_transport_source == "uniform_exact_sparse_384"
        assert loader.bata_value_transport_config_hash == "uniform_exact_384_test_hash"

    assert "gt_segments" not in _collect(cfg.dataset.test).get("keys", [])
    assert "gt_labels" not in _collect(cfg.dataset.test).get("keys", [])


def test_uniform_sparse_384_exec_config_passes_training_guard(monkeypatch) -> None:
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH", "/tmp/uniform_train.jsonl")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_VAL_LEDGER_PATH", "/tmp/uniform_val.jsonl")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TEST_LEDGER_PATH", "/tmp/uniform_test.jsonl")

    cfg = Config.fromfile(str(EXEC_CONFIG))

    assert cfg.c3_uniform_sparse_384_full_train_gate.launch_gate_passed is True
    assert cfg.c3_uniform_sparse_384_full_train_gate.reviewed_execution_config is True
    guard_spec = importlib.util.spec_from_file_location("training_guard_uniform_sparse_test", GUARD)
    guard = importlib.util.module_from_spec(guard_spec)
    assert guard_spec.loader is not None
    guard_spec.loader.exec_module(guard)
    guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")


def test_uniform_sparse_384_launcher_is_gpu0_precheck_first_and_fail_closed() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN="${ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN:-0}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "0" ]]' in text
    assert "generate_uniform_sparse_ledger.py" in text
    assert "--target-len \"${UNIFORM_SPARSE_TARGET_LEN}\"" in text
    assert "UNIFORM_SPARSE_TARGET_LEN=\"${UNIFORM_SPARSE_TARGET_LEN:-384}\"" in text
    assert "C3_UNIFORM_SPARSE_LEDGER_SOURCE=\"uniform_exact_sparse_384\"" in text
    assert "C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH" in text
    assert "PRECHECK_ONLY variant=uniform_sparse_384 target_len=${UNIFORM_SPARSE_TARGET_LEN} complete" in text
    assert "ALLOW_C3_UNIFORM_SPARSE_ADATAD_FULLTRAIN=1 is required for formal full train" in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert "c3_uniform_sparse_384_ledger_adatad_full_train_exec.py" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text


def test_uniform_sparse_384_ledger_rows_match_loader_contract(tmp_path: Path, monkeypatch) -> None:
    ledger = tmp_path / "value_transport_ledger_uniform_sparse_384.jsonl"
    row = {
        "schema_version": "c3_uniform_sparse_ledger_v1",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": list(range(384)),
        "target_len": 384,
        "selected_count": 384,
        "valid_len": 768,
        "dense_len": 768,
        "selection_family": "uniform_exact",
        "uses_uniform_scaffold": True,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "uniform_exact_sparse_384",
        "policy_checkpoint_sha256": "uniform_exact_384_test_hash",
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TRAIN_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_UNIFORM_SPARSE_VAL_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_UNIFORM_SPARSE_TEST_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_UNIFORM_SPARSE_LEDGER_CONFIG_HASH", "uniform_exact_384_test_hash")

    cfg = Config.fromfile(str(CONFIG))

    for split in ("train", "val", "test"):
        loader = _loadframes(cfg.dataset[split])
        assert loader.bata_value_transport_ledger_path == str(ledger)
        assert loader.bata_value_transport_source == row["policy_source"]
        assert loader.bata_value_transport_config_hash == row["policy_checkpoint_sha256"]
