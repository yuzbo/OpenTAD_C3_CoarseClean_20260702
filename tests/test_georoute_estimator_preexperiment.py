from __future__ import annotations

import torch

from tools.bata.deploy_georoute_estimator_preexperiment import (
    PHASE_M_SPECS,
)
from tools.bata.run_georoute_decode_census import (
    _validate_sample,
    _window_descriptor,
)
from tools.bata.run_georoute_estimator_kat import (
    _amp_horizon_kat,
    _pl_probability_kats,
    _representation_kats,
    _risk_sign_kat,
    _st_vs_pl_reachability_kat,
)


def test_frozen_phase_m_uses_only_prediction_bearing_old_arms():
    assert tuple(PHASE_M_SPECS) == (
        "dense",
        "fixed",
        "fixed_geometry",
        "random",
        "free",
        "hybrid",
    )
    assert "roi" not in PHASE_M_SPECS
    assert len(
        {spec["config"] for spec in PHASE_M_SPECS.values()}
    ) == len(PHASE_M_SPECS)
    assert len(
        {spec["cell"] for spec in PHASE_M_SPECS.values()}
    ) == len(PHASE_M_SPECS)


def test_estimator_and_representation_known_answers_pass():
    checks = (
        _pl_probability_kats(),
        _risk_sign_kat(),
        _st_vs_pl_reachability_kat(),
        _amp_horizon_kat(device=torch.device("cpu")),
        _representation_kats(),
    )
    assert all(check["passed"] for check in checks)


def test_decode_census_validates_per_item_uint8_source_and_scout():
    sample = {
        "inputs": {
            "source": torch.zeros(
                1, 3, 4, 180, 320, dtype=torch.uint8
            ),
            "scout": torch.zeros(
                1, 3, 4, 96, 96, dtype=torch.uint8
            ),
        }
    }
    schema = _validate_sample(sample, expected_frames=4)
    assert schema["source_shape"] == [1, 3, 4, 180, 320]
    assert schema["scout_shape"] == [1, 3, 4, 96, 96]


def test_decode_window_descriptor_excludes_annotations_and_video_metadata():
    row = [
        "video_validation_0000001",
        {"duration": 12.0, "annotations": [{"label": "secret"}]},
        {"gt_segments": [[1.0, 2.0]]},
        [0, 4, 8],
    ]
    descriptor = _window_descriptor(3, row)
    assert descriptor["dataset_index"] == 3
    assert descriptor["video_id"] == row[0]
    assert descriptor["window_center_first"] == 0.0
    assert descriptor["window_center_last"] == 8.0
    assert "annotations" not in descriptor
    assert "gt_segments" not in descriptor
