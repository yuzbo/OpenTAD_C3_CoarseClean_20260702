import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "bata" / "evaluate_zoomtoken_boundary.py"
SPEC = importlib.util.spec_from_file_location("zoomtoken_boundary", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _gt(annotations):
    return {"database": {"video_1": {"subset": "validation", "annotations": annotations}}}


def _pred(predictions):
    return {"results": {"video_1": predictions}}


def test_score_ordered_greedy_matching_and_normalized_errors():
    gt = _gt([
        {"id": "g1", "label": "A", "segment": [0.0, 10.0]},
        {"id": "g2", "label": "A", "segment": [5.0, 15.0]},
    ])
    predictions = _pred([
        {"id": "late", "label": "A", "segment": [4.0, 14.0], "score": 0.8},
        {"id": "early", "label": "A", "segment": [0.0, 10.0], "score": 0.9},
    ])
    result = MODULE.evaluate_frozen_diagnostics(gt, predictions)
    assert result["boundary"]["matched_count"] == 2
    assert result["boundary"]["unmatched_gt_count"] == 0
    assert result["boundary"]["mean_abs_start_norm"] == pytest.approx(0.05)
    assert result["boundary"]["mean_abs_end_norm"] == pytest.approx(0.05)
    assert result["high_iou_gt_bins"]["HIT_070"] == 2


def test_inclusive_half_iou_and_endpoint_decomposition():
    gt = _gt([
        {"id": "g1", "label": "A", "segment": [0.0, 10.0]},
        {"id": "g2", "label": "A", "segment": [20.0, 30.0]},
    ])
    predictions = _pred([
        {"id": "p1", "label": "A", "segment": [0.0, 5.0], "score": 0.9},
        {"id": "p2", "label": "A", "segment": [25.0, 30.0], "score": 0.8},
    ])
    result = MODULE.evaluate_frozen_diagnostics(gt, predictions)
    assert result["boundary"]["matched_count"] == 2
    assert result["high_iou_gt_bins"]["END_LIMITED"] == 1
    assert result["high_iou_gt_bins"]["START_LIMITED"] == 1


def test_short_action_boundary_and_zero_match_na():
    gt = _gt([
        {"id": "short", "label": "A", "segment": [0.0, 5.0]},
        {"id": "long", "label": "A", "segment": [10.0, 15.0001]},
    ])
    result = MODULE.evaluate_frozen_diagnostics(gt, _pred([]))
    assert result["short_action_report_only"]["short_gt_count"] == 1
    assert result["short_action_report_only"]["non_short_gt_count"] == 1
    assert result["short_action_report_only"]["tp_at_070_recall"] == 0.0
    assert result["boundary"]["mean_abs_start_norm"] == "NA"
    assert result["boundary"]["mean_abs_end_norm"] == "NA"


def test_class_confusion_and_unmatched_prediction_precedence():
    gt = _gt([{ "id": "g1", "label": "A", "segment": [0.0, 10.0]}])
    predictions = _pred([
        {"id": "wrong", "label": "B", "segment": [0.0, 10.0], "score": 0.9},
        {"id": "other", "label": "C", "segment": [20.0, 30.0], "score": 0.8},
    ])
    result = MODULE.evaluate_frozen_diagnostics(gt, predictions)
    assert result["high_iou_gt_bins"]["CLASS_CONFUSION"] == 1
    assert result["unmatched_prediction_bins"]["CLASS_CONFUSION_FP"] == 1
    assert result["unmatched_prediction_bins"]["OTHER_FP"] == 1


@pytest.mark.parametrize(
    "annotation",
    [
        {"id": "bad", "label": "A", "segment": [1.0, 1.0]},
        {"id": "bad", "label": "A", "segment": [0.0, float("nan")]},
        {"id": [], "label": "A", "segment": [0.0, 1.0]},
    ],
)
def test_invalid_ground_truth_fails_closed(annotation):
    with pytest.raises(ValueError):
        MODULE.evaluate_frozen_diagnostics(_gt([annotation]), _pred([]))


def test_final_evaluation_entry_is_configurable_and_final_only():
    test_source = (ROOT / "tools" / "test.py").read_text(encoding="utf-8")
    launcher = (
        ROOT / "scripts" / "run_zoomtoken_rc32_final_eval_n16r4.sh"
    ).read_text(encoding="utf-8")
    assert '"--cfg-options"' in test_source
    assert "cfg.merge_from_dict(args.cfg_options)" in test_source
    assert 'model.module.load_state_dict(checkpoint["state_dict_ema"])' in test_source
    assert 'CHECKPOINT="${WORK_DIR}/checkpoint/epoch_59.pth"' in launcher
    assert "recovery_epoch_" not in launcher
    assert '"post_processing.save_dict=True"' in launcher
    assert '"dataset.test.subset_name=validation"' in launcher
