from __future__ import annotations

from pathlib import Path

from mmengine.config import Config

from tools.bata.d2s_tad_full200_compute import (
    D2S_ARMS,
    D2S_SEEDS,
    config_path,
    validate_d2s_matrix,
    validate_d2s_parameter_fairness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_d2s_matrix_is_complete_and_full_population():
    receipt = validate_d2s_matrix(ROOT)
    assert receipt["cell_count"] == 9
    assert [(row["arm"], row["seed"]) for row in receipt["cells"]] == [
        (arm, seed) for arm in D2S_ARMS for seed in D2S_SEEDS
    ]
    assert receipt["training_identities"] == 200
    assert receipt["evaluation_videos"] == 211
    assert receipt["evaluation_ordered_windows"] == 792
    assert receipt["successful_updates_per_cell"] == 6000


def test_d2s_candidate_uses_raw_source_and_declares_added_parameters():
    cfg = Config.fromfile(config_path(ROOT, "D2S-U128-B128", 4407))
    custom = cfg.model.backbone.custom
    assert custom.source_key == "source"
    assert custom.return_feature_bundle is False
    assert custom.burst_chunks == 16
    assert custom.total_chunks == 48
    for split in ("train", "val", "test"):
        types = [step.type for step in cfg.dataset[split].pipeline]
        assert "ContinuousRoiSourceViews" in types
        assert "NativeCropSourceViews" not in types
    disclosure = validate_d2s_parameter_fairness(ROOT)
    assert disclosure["candidate_parameter_parity_claimed"] is False
    assert disclosure["candidate_added_trainable_modules"] == [
        "proj_local",
        "proj_global",
        "gamma",
    ]

