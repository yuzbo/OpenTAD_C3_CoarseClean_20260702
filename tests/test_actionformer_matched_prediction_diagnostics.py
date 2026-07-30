import numpy as np
import pytest

from tools.bata import analyze_actionformer_matched_predictions as diagnostics


def _raw(rows):
    return {
        "video-id": [row[0] for row in rows],
        "t-start": np.asarray([row[1] for row in rows], dtype=np.float64),
        "t-end": np.asarray([row[2] for row in rows], dtype=np.float64),
        "label": np.asarray([row[3] for row in rows], dtype=np.int64),
        "score": np.asarray([row[4] for row in rows], dtype=np.float64),
    }


def test_retained_prediction_summary_exposes_boundary_duration_and_topk():
    ground_truth = {
        "v1": [
            {
                "video_id": "v1",
                "label_id": 0,
                "label_name": "A",
                "segment": np.asarray([0.0, 1.0]),
                "duration": 1.0,
            },
            {
                "video_id": "v1",
                "label_id": 1,
                "label_name": "B",
                "segment": np.asarray([4.0, 8.0]),
                "duration": 4.0,
            },
        ],
        "v2": [
            {
                "video_id": "v2",
                "label_id": 0,
                "label_name": "A",
                "segment": np.asarray([10.0, 11.0]),
                "duration": 1.0,
            }
        ],
    }
    raw = _raw(
        [
            ("v1", 0.0, 1.0, 0, 0.9),
            ("v1", 0.0, 1.0, 1, 0.8),
            ("v1", 4.5, 7.5, 1, 0.7),
            ("v2", 10.5, 11.5, 0, 0.6),
        ]
    )
    summary = diagnostics.summarize_arm(raw, ground_truth, {0: "A", 1: "B"})
    assert summary["overall"]["gt_count"] == 3
    assert summary["overall"]["class_aware_recall"]["0.3"] == 1.0
    assert summary["overall"]["class_aware_recall"]["0.7"] == 2 / 3
    assert summary["per_class"]["A"]["gt_count"] == 2
    assert summary["per_class"]["B"]["gt_count"] == 1
    assert summary["duration_bins"]["1_2s"]["gt_count"] == 2
    assert summary["duration_bins"]["4_8s"]["gt_count"] == 1
    assert summary["fixed_topk_class_aware_recall"]["1"]["0.3"] == 2 / 3
    assert summary["fixed_topk_class_aware_recall"]["5"]["0.3"] == 1.0
    assert (
        summary["retained_output_diagnostics"]["prediction_count"]
        == 4
    )
    assert len(summary["worst_videos"]) == 2


def test_segment_iou_handles_nonoverlap_and_exact_match():
    values = diagnostics.segment_iou(
        np.asarray([1.0, 2.0]),
        np.asarray([[1.0, 2.0], [2.0, 3.0], [1.5, 2.5]]),
    )
    assert values.tolist() == [1.0, 0.0, 1.0 / 3.0]


def test_contrast_is_sparse_minus_dense():
    dense = {
        "official": {"average_mAP": 0.6, "mAP": {"0.3": 0.7}},
        "overall": {
            "class_aware_recall": {"0.3": 0.8},
            "class_agnostic_recall": {"0.3": 0.9},
        },
        "per_class": {"A": {"official_ap": {"0.3": 0.7}}},
        "duration_bins": {
            "lt_1s": {"class_aware_recall": {"0.3": 0.8}}
        },
        "fixed_topk_class_aware_recall": {"1": {"0.3": 0.5}},
    }
    sparse = {
        "official": {"average_mAP": 0.4, "mAP": {"0.3": 0.5}},
        "overall": {
            "class_aware_recall": {"0.3": 0.6},
            "class_agnostic_recall": {"0.3": 0.7},
        },
        "per_class": {"A": {"official_ap": {"0.3": 0.4}}},
        "duration_bins": {
            "lt_1s": {"class_aware_recall": {"0.3": 0.5}}
        },
        "fixed_topk_class_aware_recall": {"1": {"0.3": 0.3}},
    }
    contrast = diagnostics.build_contrasts(dense, sparse)
    assert contrast["average_mAP"] == pytest.approx(-0.2)
    assert contrast["official_mAP"]["0.3"] == pytest.approx(-0.2)
    assert contrast["per_class_official_ap"]["A"]["0.3"] == pytest.approx(-0.3)
