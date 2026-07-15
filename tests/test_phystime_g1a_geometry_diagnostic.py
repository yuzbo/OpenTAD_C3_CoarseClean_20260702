import importlib.util
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "analyze_phystime_g1a_geometry.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phystime_g1a_geometry_diagnostic", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rank_to_axis_matches_half_cell_anchorfree_contract():
    module = load_module()

    mapped = module._map_rank_to_axis(
        np.asarray([-0.5, 0.0, 0.5, 1.0, 1.5]),
        np.asarray([10.0, 14.0]),
        domain_start=8.0,
        domain_end=20.0,
    )

    assert np.allclose(mapped, [8.0, 10.0, 12.0, 14.0, 20.0])


def test_native_axis_positions_compare_uniform_and_physical_seconds():
    module = load_module()

    axis = module._native_axis_positions(
        selected_positions=np.asarray([0, 1, 8, 9]),
        fps=2.0,
        snippet_stride=1.0,
        dense_origin_frame=0.0,
        dense_valid_len=10,
        duration_sec=5.0,
        tubelet_size=2,
    )

    assert axis["native_valid_count"] == 2
    assert np.allclose(axis["uniform_rank_seconds"], [1.25, 3.75])
    assert np.allclose(axis["physical_time_seconds"], [0.25, 4.25])


def test_g1a_window_reports_assignment_delta_between_axes():
    module = load_module()

    report = module.summarize_g1a_window(
        dense_valid_len=10,
        selected_positions=np.asarray([0, 1, 8, 9]),
        fps=2.0,
        snippet_stride=1.0,
        dense_origin_frame=0.0,
        duration_sec=5.0,
        gt_segments_sec=np.asarray([[0.1, 0.8], [4.0, 4.8]]),
        sequence_length=2,
        strides=(1, 2),
        regression_ranges=((0, 4), (4, 8)),
        center_sample_radius=1.5,
    )

    assert report["uniform_rank_seconds"]["candidate_count"] == 3
    assert report["physical_time_seconds"]["candidate_count"] == 3
    assert report["axis_abs_delta_sec"]["mean"] > 0.0
    assert (
        report["physical_time_seconds"]["assignment"]["positive_location_count"]
        >= report["uniform_rank_seconds"]["assignment"]["positive_location_count"]
    )
    assert "physical_time_rank_assignment" in report
    assert report["physical_time_rank_assignment"]["candidate_count"] == 3
    assert report["physical_time_rank_assignment"]["assignment"]["physical_inside_required"] is True


def test_aggregate_g1a_reports_keeps_high_level_assignment_fields():
    module = load_module()
    report = module.summarize_g1a_window(
        dense_valid_len=8,
        selected_positions=np.asarray([0, 2, 4, 6]),
        fps=4.0,
        snippet_stride=1.0,
        dense_origin_frame=0.0,
        duration_sec=2.0,
        gt_segments_sec=np.asarray([[0.25, 0.75]]),
        sequence_length=2,
        strides=(1, 2),
        regression_ranges=((0, 4), (4, 8)),
        center_sample_radius=1.5,
    )

    summary = module.aggregate_g1a_reports([report])

    assert summary["window_count"] == 1
    assert summary["gt_count"] == 1
    assert "uniform_rank_seconds" in summary
    assert "physical_time_seconds" in summary
    assert "physical_time_rank_assignment" in summary
    assert summary["uniform_rank_seconds"]["assignment"]["gt_count"] == 1
