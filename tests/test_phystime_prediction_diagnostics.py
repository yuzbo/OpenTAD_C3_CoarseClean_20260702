import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "analyze_phystime_prediction_diagnostics.py"


def load_module():
    spec = importlib.util.spec_from_file_location("phystime_prediction_diagnostics", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prediction_diagnostics_separate_localization_from_classification():
    module = load_module()
    ground_truth = {
        "video_1": [
            {"segment": [0.0, 10.0], "label": "A"},
            {"segment": [20.0, 30.0], "label": "B"},
        ]
    }
    predictions = {
        "video_1": [
            {"segment": [0.0, 10.0], "label": "A", "score": 0.9},
            {"segment": [20.0, 30.0], "label": "C", "score": 0.8},
            {"segment": [22.0, 28.0], "label": "B", "score": 0.7},
        ]
    }

    report = module.analyze_prediction_dict(
        predictions,
        ground_truth,
        tiou_thresholds=(0.5, 0.7),
        topk_values=(1, 3),
    )

    assert report["gt_count"] == 2
    assert report["prediction_count"] == 3
    assert report["all_predictions"]["class_agnostic_recall"]["0.70"] == 1.0
    assert report["all_predictions"]["class_aware_recall"]["0.70"] == 0.5
    assert report["all_predictions"]["class_aware_recall"]["0.50"] == 1.0
    assert report["best_localization_label_accuracy"] == 0.5
    assert report["best_class_aware_boundary_error_sec"]["start_mae"] == 1.0
    assert report["best_class_aware_boundary_error_sec"]["end_mae"] == 1.0
    assert report["best_class_aware_boundary_error_sec"]["by_min_iou"]["0.50"]["start_mae"] == 1.0
    assert report["best_class_aware_boundary_error_sec"]["by_min_iou"]["0.70"]["start_mae"] == 0.0
    assert report["topk"]["1"]["class_aware_recall"]["0.50"] == 0.5
    assert report["topk"]["3"]["class_aware_recall"]["0.50"] == 1.0
