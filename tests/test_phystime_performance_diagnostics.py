import importlib.util
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "analyze_phystime_performance_drop.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phystime_performance_diagnostics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_script_import_registers_repository_root():
    original = list(sys.path)
    try:
        sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != ROOT]
        load_module()
        assert str(ROOT) in sys.path
    finally:
        sys.path[:] = original


def test_physical_query_summary_preserves_holes_and_counts_only_covered_cells():
    module = load_module()
    supports = np.asarray([[0.0, 0.25], [0.5, 0.75]], dtype=np.float64)

    levels = module.build_physical_query_levels(
        duration_sec=1.0,
        domain_start_sec=0.0,
        domain_end_sec=1.0,
        support_intervals_sec=supports,
        base_spacing_sec=0.5,
        num_levels=2,
    )

    assert [level["query_count"] for level in levels] == [2, 1]
    assert [level["covered_query_count"] for level in levels] == [2, 1]
    assert np.allclose(levels[0]["coverage_sec"], [0.25, 0.25])
    assert np.allclose(levels[1]["coverage_sec"], [0.5])
    assert levels[0]["coverage_fraction"] == 0.5
    assert levels[1]["coverage_fraction"] == 0.5


def test_actionformer_physical_grid_uses_local_physical_stride():
    module = load_module()
    levels = module.build_actionformer_levels(
        sequence_length=4,
        strides=(1, 2),
        regression_ranges=((0, 4), (4, 8)),
    )
    physical = module.map_actionformer_levels_to_physical_grid(
        levels,
        selected_positions=np.asarray([0.0, 2.0, 4.0, 6.0]),
        dense_valid_len=8.0,
    )

    assert np.allclose(physical[0]["centers"], [0.0, 2.0, 4.0, 6.0])
    assert np.allclose(physical[0]["strides"], [1.0, 2.0, 2.0, 2.0])
    assert np.allclose(physical[1]["centers"], [0.0, 4.0])
    assert np.allclose(physical[1]["strides"], [2.0, 4.0])


def test_assignment_reports_gt_without_any_eligible_location():
    module = load_module()
    points = np.asarray(
        [
            [1.0, 0.0, 2.0, 1.0],
            [3.0, 0.0, 2.0, 1.0],
        ],
        dtype=np.float64,
    )
    segments = np.asarray([[0.5, 1.5], [10.0, 12.0]], dtype=np.float64)

    report = module.summarize_target_assignment(
        points=points,
        valid_mask=np.asarray([True, True]),
        gt_segments=segments,
        center_sample_radius=1.5,
    )

    assert report["positive_location_count"] == 1
    assert report["gt_count"] == 2
    assert report["gt_without_eligible_location_count"] == 1
    assert report["eligible_location_count_per_gt"] == [1, 0]


def test_assignment_reports_equal_duration_multi_gt_conflicts():
    module = load_module()
    points = np.asarray([[1.0, 0.0, 2.0, 1.0]], dtype=np.float64)
    segments = np.asarray([[0.5, 1.5], [0.5, 1.5]], dtype=np.float64)

    report = module.summarize_target_assignment(
        points=points,
        valid_mask=np.asarray([True]),
        gt_segments=segments,
        center_sample_radius=1.5,
    )

    assert report["positive_location_count"] == 1
    assert report["multi_min_gt_location_count"] == 1
    assert report["max_min_gt_multiplicity"] == 2


def test_selected_axis_warp_matches_loader_piecewise_linear_contract():
    module = load_module()
    segments = np.asarray([[1.0, 5.0]], dtype=np.float64)
    selected = np.asarray([0.0, 2.0, 6.0], dtype=np.float64)

    warped = module.warp_segments_to_selected_axis(segments, selected, dense_valid_len=8.0)

    assert np.allclose(warped, [[0.5, 1.75]])


def test_window_report_compares_all_three_heads_on_identical_observations():
    module = load_module()
    report = module.summarize_window_geometry(
        dense_valid_len=8,
        selected_positions=np.asarray([0, 2, 4, 6]),
        fps=4.0,
        snippet_stride=1.0,
        window_start_frame=1.0,
        duration_sec=3.0,
        gt_segments_dense=np.asarray([[1.0, 3.0], [6.0, 8.0]]),
        actionformer_sequence_length=4,
        actionformer_strides=(1, 2),
        actionformer_ranges=((0, 4), (4, 8)),
        phystime_base_spacing_sec=0.5,
        phystime_ranges=((0.0, 2.0), (2.0, 4.0)),
    )

    assert report["selected_count"] == 4
    assert report["sampling"]["max_gap_dense_with_edges"] == 2.0
    assert report["sampling"]["max_gap_sec_with_edges"] == 0.5
    assert report["sampling"]["support_fraction"] == 0.5
    assert set(report["assignment"]) == {"selected_axis", "physical_grid", "phystime"}
    assert report["candidate_count"]["selected_axis"] == 6
    assert report["candidate_count"]["physical_grid"] == 6
    assert report["candidate_count"]["phystime"] == 6


def test_aggregate_reports_exposes_candidate_and_gt_coverage_deficits():
    module = load_module()
    report = module.summarize_window_geometry(
        dense_valid_len=8,
        selected_positions=np.asarray([0, 2, 4, 6]),
        fps=4.0,
        snippet_stride=1.0,
        window_start_frame=1.0,
        duration_sec=3.0,
        gt_segments_dense=np.asarray([[1.0, 3.0], [6.0, 8.0]]),
        actionformer_sequence_length=4,
        actionformer_strides=(1, 2),
        actionformer_ranges=((0, 4), (4, 8)),
        phystime_base_spacing_sec=0.5,
        phystime_ranges=((0.0, 2.0), (2.0, 4.0)),
    )

    summary = module.aggregate_window_reports([report])

    assert summary["window_count"] == 1
    assert summary["candidate_count"]["selected_axis"]["mean"] == 6.0
    assert summary["candidate_count"]["phystime_to_selected_ratio"] == 1.0
    assert summary["assignment"]["phystime"]["gt_count"] == 2
    assert summary["phystime_levels"][0]["query_count"] == 5
    assert summary["phystime_levels"][0]["covered_query_fraction"] == 0.8


def test_dataset_row_analysis_reuses_the_loader_selection_contract():
    module = load_module()

    class FakeLoadFrames:
        def __init__(self):
            self.keys = []

        def _select_random_fixed_positions(self, valid_len, target_len, sample_key):
            self.keys.append((valid_len, target_len, sample_key))
            return np.asarray([0, 2, 4, 6])

    selector = FakeLoadFrames()
    rows = [
        [
            "video_test",
            {"frame": 12, "duration": 3.0},
            {"gt_segments": np.asarray([[2.0, 4.0]]), "gt_labels": np.asarray([0])},
            np.arange(1, 9),
        ]
    ]

    result = module.analyze_dataset_rows(
        rows,
        load_frames=selector,
        snippet_stride=1,
        target_len=4,
        actionformer_sequence_length=4,
        actionformer_strides=(1, 2),
        actionformer_ranges=((0, 4), (4, 8)),
        phystime_base_spacing_sec=0.5,
        phystime_ranges=((0.0, 2.0), (2.0, 4.0)),
    )

    assert selector.keys == [(8, 4, "video_test|random_fixed|1|8|8|4")]
    assert result["summary"]["window_count"] == 1
    assert result["summary"]["selected_count"]["mean"] == 4.0


def test_training_row_analysis_uses_the_real_random_trunc_contract_without_decoding():
    module = load_module()

    class FakeLoadFrames:
        crop_ratio = None
        trunc_thresh = 0.75

        def random_trunc(self, frames, trunc_len, gt_segments, gt_labels):
            return frames[:trunc_len], gt_segments, gt_labels

        def _select_random_fixed_positions(self, valid_len, target_len, sample_key):
            return np.asarray([0, 2, 4, 6])

    rows = [
        [
            "video_train",
            {"frame": 12, "duration": 3.0},
            {"gt_segments": np.asarray([[2.0, 4.0]]), "gt_labels": np.asarray([0])},
        ]
    ]
    result = module.analyze_training_rows(
        rows,
        load_frames=FakeLoadFrames(),
        snippet_stride=1,
        target_len=4,
        source_len=8,
        samples_per_video=1,
        actionformer_sequence_length=4,
        actionformer_strides=(1, 2),
        actionformer_ranges=((0, 4), (4, 8)),
        phystime_base_spacing_sec=0.5,
        phystime_ranges=((0.0, 2.0), (2.0, 4.0)),
    )

    assert result["summary"]["window_count"] == 1
    assert result["summary"]["gt_count"] == 1
    assert result["summary"]["selected_count"]["mean"] == 4.0


def test_query_embedding_features_match_the_model_coordinate_order():
    module = load_module()
    features = module.build_query_embedding_features(
        centers_sec=np.asarray([1.0]),
        widths_sec=np.asarray([0.5]),
        duration_sec=np.asarray([2.0]),
        num_fourier_bands=1,
    )

    assert features.shape == (1, 7)
    assert np.allclose(
        features[0],
        [1.0, 0.5, np.log1p(2.0), 0.5, 0.25, 0.0, -1.0],
        atol=1.0e-7,
    )


def test_linear_input_scale_audit_identifies_absolute_time_dominance():
    module = load_module()
    report = module.summarize_linear_input_scale(
        features=np.asarray([[100.0, 1.0], [200.0, 1.0]]),
        weight=np.eye(2),
        bias=np.zeros(2),
        feature_names=("absolute_center_sec", "normalized_center"),
    )

    assert report["dominant_feature"] == "absolute_center_sec"
    assert report["mean_abs_contribution"]["absolute_center_sec"] == 150.0
    assert report["mean_abs_contribution"]["normalized_center"] == 1.0
    assert np.isclose(report["contribution_share"]["absolute_center_sec"], 150.0 / 151.0)
    assert report["preactivation_abs"]["max"] == 200.0


def test_attention_summary_distinguishes_uniform_from_one_hot_selection():
    module = load_module()
    weights = np.asarray([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]])
    mass = np.asarray([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    logits = np.asarray([[0.0, 0.0, -99.0], [20.0, 0.0, -99.0]])

    report = module.summarize_attention_rows(weights=weights, mass=mass, logits=logits)

    assert report["query_count"] == 2
    assert report["effective_observation_count"]["min"] == 1.0
    assert report["effective_observation_count"]["max"] == 2.0
    assert report["normalized_entropy"]["min"] < 1.0e-6
    assert report["normalized_entropy"]["max"] == 1.0
    assert report["covered_logit_span"]["max"] == 20.0
