import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.analyze_dcsr_checkpoint_dynamics import (
    _difference,
    _stats,
    parameter_group,
)
from tools.analyze_dcsr_internal_predictions import (
    _load_ground_truth,
    segment_iou,
    summarize_arm,
)
from tools.run_dcsr_counterfactual_eval import apply_counterfactual


class _FakeDcsrModel:
    def __init__(self):
        self.dcsr_mode = "cheap_dense_scaffold"
        self.dcsr_scaffold_num_layers = 1
        self.dcsr_query_selector = SimpleNamespace(budget=384)
        self.dcsr_residual_enabled = True


@pytest.mark.parametrize(
    "arm,expected_budget,expected_enabled",
    [
        ("scaffold_only", 384, False),
        ("k384_reference", 384, True),
        ("all_query_residual", 2 ** 31 - 1, True),
    ],
)
def test_counterfactual_arms_only_change_residual_execution(
    arm, expected_budget, expected_enabled
):
    model = _FakeDcsrModel()
    receipt = apply_counterfactual(model, arm)
    assert model.dcsr_query_selector.budget == expected_budget
    assert model.dcsr_residual_enabled is expected_enabled
    assert receipt["training_changed"] is False
    assert receipt["checkpoint_changed"] is False
    assert receipt["decoder_or_nms_changed"] is False


def test_counterfactual_rejects_non_g1_source():
    model = _FakeDcsrModel()
    model.dcsr_query_selector.budget = 768
    with pytest.raises(ValueError, match="frozen K384"):
        apply_counterfactual(model, "all_query_residual")


def test_segment_iou_and_holdout_summary_are_validation_only(tmp_path):
    annotation_path = tmp_path / "annotation.json"
    annotation_path.write_text(
        json.dumps(
            {
                "database": {
                    "v0": {
                        "subset": "validation",
                        "annotations": [
                            {
                                "segment": [0.0, 2.0],
                                "label": "a",
                                "label_id": 0,
                            }
                        ],
                    },
                    "v1": {
                        "subset": "validation",
                        "annotations": [
                            {
                                "segment": [4.0, 8.0],
                                "label": "b",
                                "label_id": 1,
                            }
                        ],
                    },
                    "test_ignored": {
                        "subset": "test",
                        "annotations": [
                            {
                                "segment": [0.0, 1.0],
                                "label": "a",
                                "label_id": 0,
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    manifest = {
        "holdout_video_ids": ["v0", "v1"],
        "all_class_ids": [0, 1],
    }
    ground_truth, class_names = _load_ground_truth(
        str(annotation_path), manifest
    )
    assert len(ground_truth) == 2
    assert class_names == {0: "a", 1: "b"}
    assert segment_iou(
        (0.0, 2.0), [(0.0, 2.0), (1.0, 3.0)]
    ).tolist() == pytest.approx([1.0, 1.0 / 3.0])

    predictions = [
        {
            "video_id": "v0",
            "start": 0.0,
            "end": 2.0,
            "label": 0,
            "score": 0.9,
        },
        {
            "video_id": "v1",
            "start": 4.0,
            "end": 8.0,
            "label": 1,
            "score": 0.8,
        },
    ]
    summary = summarize_arm(
        str(annotation_path),
        frozenset(("v0", "v1")),
        ground_truth,
        predictions,
    )
    assert summary["official_holdout_evaluator"]["average_mAP"] == pytest.approx(
        1.0
    )
    assert summary["post_nms_recall"]["class_aware"]["200"] == pytest.approx(
        [1.0] * 5
    )
    assert summary["duration"]["2_4s"]["gt_count"] == 1
    assert summary["duration"]["4_8s"]["gt_count"] == 1


def test_checkpoint_parameter_groups_and_norm_differences():
    assert (
        parameter_group(
            "module.dcsr_scaffold_cls_head.cls_head.conv.weight"
        )
        == "scaffold_classification"
    )
    assert (
        parameter_group("module.cls_head.cls_head.conv.weight")
        == "residual_classification_final"
    )
    assert (
        parameter_group("module.cls_head.head.0.conv.weight")
        == "residual_classification_hidden"
    )
    assert parameter_group("module.backbone.foo.cls_head.weight") is None
    left = {"weight": torch.tensor([3.0, 4.0])}
    right = {"weight": torch.tensor([0.0, 0.0])}
    assert _stats(left)["l2_norm"] == pytest.approx(5.0)
    assert _stats({"scalar": torch.tensor(0.0)})[
        "exact_zero_fraction"
    ] == pytest.approx(1.0)
    assert _stats({"scalar": torch.tensor(2.0)})[
        "exact_zero_fraction"
    ] == pytest.approx(0.0)
    difference = _difference(left, right)
    assert difference["l2_norm"] == pytest.approx(5.0)
    assert difference["relative_l2_to_reference"] is None


def test_launchers_are_fail_closed_and_preserve_slurm_cuda():
    repository = Path(__file__).resolve().parents[1]
    counterfactual = (
        repository
        / "scripts"
        / "run_dcsr_counterfactual_diagnostics_n16r4.sbatch"
    ).read_text(encoding="utf-8")
    analysis = (
        repository
        / "scripts"
        / "run_dcsr_negative_analysis_n16r4.sbatch"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=" not in counterfactual
    assert "python -m tools.run_dcsr_counterfactual_eval" in counterfactual
    assert "test_gt_used" in analysis
    assert "paper_performance_row_allowed" in analysis
    assert "python -m tools.analyze_dcsr_internal_predictions" in analysis
    assert "python -m tools.analyze_dcsr_checkpoint_dynamics" in analysis
