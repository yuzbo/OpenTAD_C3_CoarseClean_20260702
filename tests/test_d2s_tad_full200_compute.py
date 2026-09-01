from __future__ import annotations

from pathlib import Path
from mmengine.config import Config
import pytest

from tools.bata.d2s_tad_full200_compute import (
    D2S_ARMS,
    D2S_SEEDS,
    config_path,
    validate_d2s_matrix,
    validate_d2s_parameter_fairness,
)


ROOT = Path(__file__).resolve().parents[1]


def test_d2s_matrix_and_parameter_surface_are_frozen():
    receipt = validate_d2s_matrix(ROOT)
    assert receipt["cell_count"] == 9
    assert [cell["arm"] for cell in receipt["cells"]] == [
        arm for arm in D2S_ARMS for _ in D2S_SEEDS
    ]
    assert [cell["seed"] for cell in receipt["cells"]] == [
        seed for _ in D2S_ARMS for seed in D2S_SEEDS
    ]
    assert receipt["training_identities"] == 200
    assert receipt["evaluation_videos"] == 211
    assert receipt["evaluation_ordered_windows"] == 792
    assert receipt["successful_updates_per_cell"] == 6000
    validate_d2s_parameter_fairness(ROOT)


def test_d2s_burst_is_exact_16_of_48_chunks_and_zero_parameters():
    cfg = Config.fromfile(config_path(ROOT, "D2S-U128-B128", 4407))
    custom = cfg.model.backbone.custom
    assert custom.wrapper_type == "d2s_temporal_zoom_shared_videomae"
    assert custom.total_chunks == 48
    assert custom.burst_chunks == 16
    assert custom.global_size == 96
    assert custom.local_size == 128
    assert custom.burst_chunks / custom.total_chunks == pytest.approx(1.0 / 3.0)
