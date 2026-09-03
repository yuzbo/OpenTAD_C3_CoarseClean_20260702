"""Matrix integrity and configuration suite tests for DUCA Evidence Recovery 8-arm matrix."""
from __future__ import annotations

import os
from pathlib import Path

import pytest
from mmengine.config import Config


ARM_CONFIGS = {
    "C0": "configs/adatad/thumos/duca_evidence_recovery_matched_h65_60.py",
    "F": "configs/adatad/thumos/duca_evidence_recovery_full.py",
    "A1": "configs/adatad/thumos/duca_evidence_recovery_no_coverage.py",
    "A2": "configs/adatad/thumos/duca_evidence_recovery_no_time.py",
    "A3": "configs/adatad/thumos/duca_evidence_recovery_no_robust.py",
    "A4": "configs/adatad/thumos/duca_evidence_recovery_no_merge.py",
    "A5": "configs/adatad/thumos/duca_evidence_recovery_no_recovery.py",
    "A6": "configs/adatad/thumos/duca_evidence_recovery_h65_selection.py",
}

FORMAL_SEEDS = [8261, 19237, 31153]


def test_eval_uses_runtime_workdir_and_preserves_training_identity():
    eval_source = Path(
        "scripts/run_duca_evidence_recovery_eval_array_n16r4.sbatch"
    ).read_text(encoding="utf-8")
    test_source = Path("tools/test.py").read_text(encoding="utf-8")

    assert 'CHECKPOINT="${TRAIN_WORK_DIR}/gpu1_id0/checkpoint/epoch_59.pth"' in eval_source
    assert (
        'TRAINING_COMMIT="${DUCA_EVIDENCE_TRAINING_COMMIT:-$EXPECTED_COMMIT}"'
        in eval_source
    )
    assert 'export DUCA_TRAINING_COMMIT="$TRAINING_COMMIT"' in eval_source
    assert "duca_evidence_eval_${EXPECTED_COMMIT:0:8}" in eval_source
    assert 'work_dir="${EVAL_WORK_DIR}"' in eval_source
    assert "THUMOS14_TRAIN_DATA_PATH" in eval_source
    assert "DUCA_H65_TRAIN_LEDGER_PATH" in eval_source
    assert "DUCA_H65_VAL_LEDGER_PATH" in eval_source
    assert "c3_official_asformer_delta_ledgers_fullgrid_ec0bb6a" in eval_source
    assert "duca_p0_training.canonical_sha256" in test_source
    assert '"DUCA_TRAINING_COMMIT"' in test_source


def test_24_matrix_cell_uniqueness():
    """Verify 24 task IDs map 1-to-1 to (arm_idx, seed_idx) without collision."""
    task_map = {}
    arms = list(ARM_CONFIGS.keys())
    for arm_idx, arm in enumerate(arms):
        for seed_idx, seed in enumerate(FORMAL_SEEDS):
            task_id = arm_idx * len(FORMAL_SEEDS) + seed_idx
            assert task_id not in task_map, f"Duplicate task_id {task_id}"
            task_map[task_id] = (arm, seed)

    assert len(task_map) == 24
    assert set(task_map.keys()) == set(range(24))


def test_all_arm_configs_exist_and_load():
    """Verify all 8 arm configs can be loaded and have expected ablation flags."""
    for arm_id, cfg_path in ARM_CONFIGS.items():
        assert os.path.exists(cfg_path), f"Config path {cfg_path} must exist"
        cfg = Config.fromfile(cfg_path)
        assert cfg.total_epochs == 60
        assert cfg.max_updates == 6000
        assert cfg.model.frame_selector.budget == 384
        assert cfg.model.frame_selector.window_size == 768
        for split in ("train", "val", "test"):
            pipeline = cfg.dataset[split].pipeline
            ledger_step = next(
                step
                for step in pipeline
                if step.get("type") == "DucaH65PositionsFromLedger"
            )
            assert ledger_step.source == "c3_lowres_probe_delta_p_action"
            assert ledger_step.config_hash == ""
            assert ledger_step.allow_missing is False
            collect = next(step for step in pipeline if step.get("type") == "Collect")
            meta_keys = set(collect.get("meta_keys", []))
            assert "irregular_selected_positions" in meta_keys
            assert "bata_selected_dense_indices" in meta_keys
            assert "irregular_dense_valid_len" in meta_keys

        fs = cfg.model.frame_selector
        bb = cfg.model.backbone.backbone

        if arm_id == "C0":
            assert not fs.use_coverage
            assert not fs.use_time_conditioning
            assert not fs.use_temporal_merge
            assert not fs.use_dense_recovery
            assert not fs.use_robust_training
            assert fs.use_h65_selection
            assert not bb.bounded_interval_adapter.enabled
            assert not bb.continuous_timestamp_conditioner.enabled
            assert not bb.temporal_token_merge.enabled

        elif arm_id == "F":
            assert fs.use_coverage
            assert fs.use_time_conditioning
            assert fs.use_temporal_merge
            assert fs.use_dense_recovery
            assert fs.use_robust_training
            assert not fs.use_h65_selection
            assert bb.bounded_interval_adapter.enabled
            assert bb.continuous_timestamp_conditioner.enabled
            assert bb.temporal_token_merge.enabled

        elif arm_id == "A1":
            assert not fs.use_coverage
            assert fs.use_time_conditioning
            assert fs.use_temporal_merge
            assert fs.use_dense_recovery

        elif arm_id == "A2":
            assert not fs.use_time_conditioning
            assert not bb.bounded_interval_adapter.enabled
            assert not bb.continuous_timestamp_conditioner.enabled

        elif arm_id == "A3":
            assert not fs.use_robust_training

        elif arm_id == "A4":
            assert not fs.use_temporal_merge
            assert not bb.temporal_token_merge.enabled

        elif arm_id == "A5":
            assert not fs.use_dense_recovery

        elif arm_id == "A6":
            assert fs.use_h65_selection


def test_build_detector_for_full_and_c0():
    """Verify build_detector successfully instantiates detector for FULL and C0 configs."""
    try:
        import opentad.datasets  # noqa: F401 - register config-referenced transforms
        from opentad.models.builder import build_detector
    except OSError as exc:
        pytest.skip(f"torch runtime unavailable in this environment: {exc}")

    cfg_full = Config.fromfile(ARM_CONFIGS["F"])
    model_full = build_detector(cfg_full.model)
    assert model_full is not None
    assert model_full.with_frame_selector

    cfg_c0 = Config.fromfile(ARM_CONFIGS["C0"])
    model_c0 = build_detector(cfg_c0.model)
    assert model_c0 is not None
    assert model_c0.with_frame_selector


def test_legacy_c3_ledger_policy_field_is_accepted_as_source():
    try:
        from opentad.datasets.transforms.end_to_end import DucaH65PositionsFromLedger
    except OSError as exc:
        pytest.skip(f"torch runtime unavailable in this environment: {exc}")

    assert (
        DucaH65PositionsFromLedger._row_metadata(
            {"policy": "c3_lowres_probe_delta_p_action"},
            "policy_source",
        )
        == "c3_lowres_probe_delta_p_action"
    )
