import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "validate_phystime_independent_recompute.py"
SPEC = importlib.util.spec_from_file_location("independent_recompute", MODULE_PATH)
independent = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(independent)

PINNED_RUNTIME_SORT = {
    "operation": "sort_descending",
    "provider": "pytorch_cpu",
    "stable": False,
    "torch_git_version": "e9ebda29d87ce0916ab08c06ab26fd3766a870e5",
    "torch_version": "2.0.1",
}


def build_axis_capture(axis_values, count=2):
    axis_values = np.asarray([axis_values], dtype=np.float32)
    return {
        "base_points": np.asarray(
            [[0.0, 0.0, 10.0, 1.0], [1.0, 0.0, 10.0, 1.0]],
            dtype=np.float32,
        ),
        "reg_distances": np.asarray(
            [[[1.0, 1.0], [1.0, 1.0]]],
            dtype=np.float32,
        ),
        "base_mask": np.asarray([[True, True]], dtype=np.bool_),
        "native_valid_count": np.asarray([count], dtype=np.int32),
        "domain_sec": np.asarray([[-0.5, 1.5]], dtype=np.float64),
        "uniform_axis_sec": axis_values.copy(),
        "physical_axis_sec": axis_values.copy(),
    }


def test_module_does_not_import_production_decode_nms_or_evaluator():
    source = MODULE_PATH.read_text(encoding="utf-8")
    forbidden = (
        "from opentad",
        "import opentad",
        "replay_phystime_decode_cross",
        "validate_phystime_decode_cross_replay",
        "apply_sliding_window_nms",
        "build_evaluator",
    )
    assert not any(token in source for token in forbidden)


def test_rank_to_seconds_identity_and_boundary_interpolation():
    positions = np.asarray([0.0, 1.0, 2.0], dtype=np.float64)
    coordinates = np.asarray([-0.5, 0.0, 0.5, 2.0, 2.5])
    mapped = independent.map_rank_to_seconds(
        coordinates,
        positions,
        domain_start=-0.5,
        domain_end=2.5,
    )
    np.testing.assert_allclose(mapped, coordinates, atol=0.0, rtol=0.0)


def test_dense_decode_uses_same_mask_but_changes_physical_geometry():
    capture = {
        "base_points": np.asarray(
            [[0.0, 0.0, 10.0, 1.0], [1.0, 0.0, 10.0, 1.0]],
            dtype=np.float32,
        ),
        "reg_distances": np.asarray([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32),
        "base_mask": np.asarray([[True, True]], dtype=np.bool_),
        "native_valid_count": np.asarray([2], dtype=np.int32),
        "domain_sec": np.asarray([[-0.5, 3.5]], dtype=np.float64),
        "uniform_axis_sec": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "physical_axis_sec": np.asarray([[0.0, 3.0]], dtype=np.float32),
    }
    uniform, uniform_mask, _ = independent.recompute_dense_decode(
        capture,
        "uniform_rank_seconds",
    )
    physical, physical_mask, _ = independent.recompute_dense_decode(
        capture,
        "physical_time_seconds",
    )
    assert np.array_equal(uniform_mask, physical_mask)
    assert not np.allclose(uniform, physical)
    assert physical[0, 1, 0] > uniform[0, 1, 0]


def test_dense_decode_accepts_contractual_nan_axis_padding():
    capture = build_axis_capture([0.0, 1.0, np.nan])
    dense, mask, points = independent.recompute_dense_decode(
        capture,
        "physical_time_seconds",
    )
    assert np.isfinite(dense).all()
    assert np.isfinite(points).all()
    assert mask.tolist() == [[True, True]]


def test_dense_decode_preserves_sealed_source_float32_geometry_semantics():
    capture = build_axis_capture(
        [0.06684980541467667, 0.33424901962280273],
    )
    capture["domain_sec"] = np.asarray([[0.0, 33.826]], dtype=np.float64)
    dense, _, points = independent.recompute_dense_decode(
        capture,
        "physical_time_seconds",
    )
    assert dense.dtype == np.float32
    assert points.dtype == np.float32
    mapped = independent.map_rank_to_seconds(
        capture["base_points"][:, 0],
        capture["physical_axis_sec"][0],
        capture["domain_sec"][0, 0],
        capture["domain_sec"][0, 1],
        compute_dtype=np.float32,
    )
    assert mapped.dtype == np.float32
    np.testing.assert_array_equal(points[0, :, 0], mapped)


def test_dense_decode_rejects_widened_geometry_artifact():
    capture = build_axis_capture([0.0, 1.0])
    capture["base_points"] = capture["base_points"].astype(np.float64)
    with pytest.raises(
        independent.IndependentClosureError,
        match="base_points must retain sealed source float32 semantics",
    ):
        independent.recompute_dense_decode(capture, "physical_time_seconds")


@pytest.mark.parametrize(
    "axis_values",
    (
        [0.0, np.nan, np.nan],
        [0.0, np.inf, np.nan],
    ),
)
def test_dense_decode_rejects_non_finite_axis_valid_prefix(axis_values):
    capture = build_axis_capture(axis_values)
    with pytest.raises(
        independent.IndependentClosureError,
        match="axis valid prefix contains non-finite values",
    ):
        independent.recompute_dense_decode(capture, "physical_time_seconds")


def test_dense_decode_rejects_finite_axis_padding():
    capture = build_axis_capture([0.0, 1.0, 2.0])
    with pytest.raises(
        independent.IndependentClosureError,
        match="axis padding must contain only NaN",
    ):
        independent.recompute_dense_decode(capture, "physical_time_seconds")


@pytest.mark.parametrize(
    "axis_values",
    (
        [1.0, 1.0, np.nan],
        [1.0, 0.0, np.nan],
    ),
)
def test_dense_decode_rejects_non_increasing_axis_valid_prefix(axis_values):
    capture = build_axis_capture(axis_values)
    with pytest.raises(
        independent.IndependentClosureError,
        match="axis valid prefix must be strictly increasing",
    ):
        independent.recompute_dense_decode(capture, "physical_time_seconds")


def test_stable_gaussian_soft_nms_preserves_equal_score_input_order():
    segments = np.asarray([[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]])
    scores = np.asarray([0.8, 0.8, 0.8])
    indices = np.asarray([11, 7, 5])
    _, output_scores, output_indices = independent.gaussian_soft_nms(
        segments,
        scores,
        indices,
        sigma=0.7,
        min_score=0.0,
    )
    np.testing.assert_allclose(output_scores, scores)
    assert output_indices.tolist() == indices.tolist()


def test_pinned_torch_sort_reproduces_sealed_unstable_tie_order():
    values = np.asarray(
        [
            0.8,
            0.7,
            0.8,
            0.6,
            0.8,
            0.7,
            0.8,
            0.6,
            0.8,
            0.7,
            0.8,
            0.6,
            0.8,
            0.7,
            0.8,
            0.6,
            0.8,
            0.7,
            0.8,
            0.6,
        ],
        dtype=np.float32,
    )
    order = independent.pinned_torch_descending_order(
        values,
        PINNED_RUNTIME_SORT,
    )
    assert order.tolist() == [
        0,
        2,
        18,
        4,
        6,
        16,
        8,
        14,
        10,
        12,
        1,
        17,
        13,
        9,
        5,
        11,
        15,
        7,
        3,
        19,
    ]


def test_float32_gaussian_soft_nms_matches_compiled_source_near_ties():
    try:
        import nms_1d_cpu
        import torch
    except (ImportError, OSError) as error:
        pytest.skip(f"compiled production NMS is unavailable: {error}")

    segments = torch.tensor(
        [
            [0.0, 2.0],
            [0.00003, 2.00004],
            [1.0, 3.0],
            [4.0, 5.0],
            [4.00002, 5.00003],
        ],
        dtype=torch.float32,
    )
    scores = torch.tensor(
        [0.8000001, 0.8, 0.8000001, 0.7, 0.7],
        dtype=torch.float32,
    )
    dets = segments.new_empty((segments.size(0), 3), device="cpu")
    expected_indices = nms_1d_cpu.softnms(
        segments,
        scores,
        dets,
        iou_threshold=0.0,
        sigma=0.7,
        min_score=0.0,
        method=2,
        t1=0.0,
        t2=0.0,
    )
    observed_segments, observed_scores, observed_indices = (
        independent.gaussian_soft_nms(
            segments.numpy(),
            scores.numpy(),
            np.arange(segments.size(0), dtype=np.int64),
            sigma=0.7,
            min_score=0.0,
        )
    )
    count = int(expected_indices.numel())
    np.testing.assert_array_equal(observed_segments, dets[:count, :2].numpy())
    np.testing.assert_array_equal(observed_scores, dets[:count, 2].numpy())
    np.testing.assert_array_equal(observed_indices, expected_indices.numpy())


def test_independent_thumos_ap_perfect_prediction():
    annotation = {
        "database": {
            "video_test_1": {
                "subset": "test",
                "annotations": [
                    {"label": "Diving", "segment": [1.0, 2.0]},
                    # Duplicate must be removed just as in the ActionFormer evaluator.
                    {"label": "Diving", "segment": [1.0, 2.0]},
                ],
            }
        }
    }
    predictions = {
        "video_test_1": [
            {"label": "Diving", "segment": [1.0, 2.0], "score": 0.9}
        ]
    }
    metrics, per_class = independent.independent_thumos_evaluate(
        annotation,
        predictions,
    )
    assert metrics["average_mAP"] == pytest.approx(1.0)
    assert all(value == pytest.approx(1.0) for value in per_class["Diving"].values())


def test_cross_window_nms_is_multiclass_and_caps_global_output():
    policy = {
        "proposal_min_duration": 0.0,
        "filter_invalid_proposals": True,
        "round_before_cross_window_nms": False,
        "round_after_cross_window_nms": False,
        "segment_round_digits": 2,
        "score_round_digits": 4,
        "nms": {
            "use_soft_nms": True,
            "multiclass": True,
            "method": 2,
            "sigma": 0.7,
            "min_score": 0.0,
            "max_seg_num": 2,
            "numeric_dtype": "float32",
        },
        "runtime_sort": PINNED_RUNTIME_SORT,
    }
    pre_cross = {
        "video": [
            {"label": "A", "segment": [0.0, 2.0], "score": 0.9},
            {"label": "A", "segment": [0.1, 2.1], "score": 0.8},
            {"label": "B", "segment": [0.0, 2.0], "score": 0.85},
        ]
    }
    merged, audit = independent.independent_cross_window_nms(pre_cross, policy)
    assert len(merged["video"]) == 2
    assert [item["label"] for item in merged["video"]] == ["A", "B"]
    assert audit["post_nms_detections"] == 2


def test_gaussian_soft_nms_decays_overlap_and_applies_min_score():
    segments = np.asarray([[0.0, 2.0], [0.1, 2.1], [4.0, 5.0]])
    scores = np.asarray([0.9, 0.8, 0.7])
    indices = np.asarray([0, 1, 2])
    output_segments, output_scores, output_indices = independent.gaussian_soft_nms(
        segments,
        scores,
        indices,
        sigma=0.7,
        min_score=0.5,
    )
    assert output_indices.tolist() == [0, 2]
    np.testing.assert_allclose(output_segments, [[0.0, 2.0], [4.0, 5.0]])
    np.testing.assert_allclose(output_scores, [0.9, 0.7])


def test_non_finite_and_invalid_geometry_fail_closed(tmp_path):
    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text('{"value": NaN}', encoding="utf-8")
    with pytest.raises(independent.IndependentClosureError):
        independent.load_json(invalid_json)
    with pytest.raises(independent.IndependentClosureError):
        independent.map_rank_to_seconds(
            np.asarray([0.0]),
            np.asarray([0.0]),
            domain_start=1.0,
            domain_end=1.0,
        )
    with pytest.raises(independent.IndependentClosureError):
        independent.gaussian_soft_nms(
            np.asarray([[0.0, np.inf]]),
            np.asarray([0.9]),
            np.asarray([0]),
            sigma=0.7,
            min_score=0.0,
        )


def test_detection_map_comparison_is_permutation_invariant_and_tolerance_bounded():
    expected = {
        "video": [
            {"label": "A", "segment": [0.0, 1.0], "score": 0.9},
            {"label": "B", "segment": [2.0, 3.0], "score": 0.8},
        ]
    }
    within = {
        "video": [
            {"label": "A", "segment": [0.0, 1.00001], "score": 0.90001},
            {"label": "B", "segment": [2.0, 3.0], "score": 0.8},
        ]
    }
    comparison = independent.compare_detection_maps(
        expected,
        within,
        segment_atol=1.0e-4,
        score_atol=1.0e-4,
    )
    assert comparison["match"] is True
    assert comparison["canonical_exact_match"] is False

    reordered = {"video": list(reversed(expected["video"]))}
    comparison = independent.compare_detection_maps(
        expected,
        reordered,
        segment_atol=1.0e-4,
        score_atol=1.0e-4,
    )
    assert comparison["match"] is True
    assert comparison["canonical_exact_match"] is False
    assert comparison["sequence_order_exact_match"] is False
    assert comparison["ordering_only_difference"] is True
    assert comparison["matched_detection_count"] == 2
    assert comparison["unmatched_expected_count"] == 0
    assert comparison["unmatched_observed_count"] == 0


def test_detection_map_comparison_fails_true_out_of_tolerance_difference():
    expected = {
        "video": [
            {"label": "A", "segment": [0.0, 1.0], "score": 0.9},
        ]
    }
    observed = {
        "video": [
            {"label": "A", "segment": [0.0, 1.001], "score": 0.9},
        ]
    }
    comparison = independent.compare_detection_maps(
        expected,
        observed,
        segment_atol=1.0e-4,
        score_atol=1.0e-4,
    )
    assert comparison["match"] is False
    assert comparison["matched_detection_count"] == 0
    assert comparison["unmatched_expected_count"] == 1
    assert comparison["unmatched_observed_count"] == 1


def test_detection_map_comparison_uses_one_to_one_matching_for_near_duplicates():
    expected = {
        "video": [
            {"label": "A", "segment": [0.0, 1.0], "score": 0.9},
            {"label": "A", "segment": [0.00008, 1.00008], "score": 0.9},
        ]
    }
    observed = {
        "video": [
            {"label": "A", "segment": [0.00009, 1.00009], "score": 0.9},
            {"label": "A", "segment": [0.0, 1.0], "score": 0.9},
        ]
    }
    comparison = independent.compare_detection_maps(
        expected,
        observed,
        segment_atol=1.0e-4,
        score_atol=1.0e-4,
    )
    assert comparison["match"] is True
    assert comparison["matched_detection_count"] == 2


def test_detection_map_comparison_bounds_issue_payload():
    expected = {
        f"expected_{index}": [
            {"label": "A", "segment": [0.0, 1.0], "score": 0.9},
        ]
        for index in range(5)
    }
    observed = {
        f"observed_{index}": [
            {"label": "B", "segment": [0.0, 1.0], "score": 0.9},
        ]
        for index in range(5)
    }
    comparison = independent.compare_detection_maps(
        expected,
        observed,
        segment_atol=1.0e-4,
        score_atol=1.0e-4,
        max_issues=2,
    )
    assert comparison["match"] is False
    assert len(comparison["issues"]) == 2
    assert comparison["issue_count"] > len(comparison["issues"])
    assert comparison["suppressed_issue_count"] > 0
    assert comparison["issues_truncated"] is True


def test_independent_thumos_ap_counts_duplicate_predictions_as_false_positive():
    annotation = {
        "database": {
            "video_test_1": {
                "subset": "test",
                "annotations": [
                    {"label": "Diving", "segment": [1.0, 2.0]},
                    {"label": "Diving", "segment": [3.0, 4.0]},
                ],
            }
        }
    }
    predictions = {
        "video_test_1": [
            {"label": "Diving", "segment": [1.0, 2.0], "score": 0.9},
            {"label": "Diving", "segment": [1.0, 2.0], "score": 0.8},
            {"label": "Diving", "segment": [3.0, 4.0], "score": 0.7},
        ]
    }
    metrics, _ = independent.independent_thumos_evaluate(annotation, predictions)
    assert 0.0 < metrics["average_mAP"] < 1.0


def test_annotation_contract_explicitly_binds_logical_test_to_validation():
    annotation = {
        "database": {
            "video_test_1": {
                "subset": "validation",
                "annotations": [
                    {"label": "Diving", "segment": [1.0, 2.0]},
                ],
            },
            "video_train_1": {
                "subset": "training",
                "annotations": [],
            },
        }
    }
    policy = {
        "subset": "test",
        "annotation_subset": "validation",
        "expected_annotation_video_count": 1,
        "expected_annotation_gt_count": 1,
        "expected_annotation_class_count": 1,
    }
    contract = independent.validate_annotation_contract(annotation, policy)
    assert contract == {
        "logical_evaluation_subset": "test",
        "annotation_subset": "validation",
        "video_count": 1,
        "ground_truth_count": 1,
        "class_count": 1,
        "video_subset_histogram": {"validation": 1, "training": 1},
    }


def test_annotation_contract_rejects_wrong_validation_video_count():
    annotation = {
        "database": {
            "video_test_1": {
                "subset": "validation",
                "annotations": [
                    {"label": "Diving", "segment": [1.0, 2.0]},
                ],
            },
        }
    }
    policy = {
        "subset": "test",
        "annotation_subset": "validation",
        "expected_annotation_video_count": 2,
        "expected_annotation_gt_count": 1,
        "expected_annotation_class_count": 1,
    }
    with pytest.raises(
        independent.IndependentClosureError,
        match="annotation subset validation video count mismatch",
    ):
        independent.validate_annotation_contract(annotation, policy)
