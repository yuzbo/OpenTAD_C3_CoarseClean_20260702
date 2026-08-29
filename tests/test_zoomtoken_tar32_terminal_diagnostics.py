import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "tools"
    / "bata"
    / "evaluate_zoomtoken_tar32_terminal_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("zoomtoken_tar32_terminal_diagnostics", MODULE_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(diagnostics)


def _annotation(rows):
    return {
        "database": {
            "video_validation_0000001": {
                "subset": "validation",
                "annotations": rows,
            }
        }
    }


def _prediction(rows):
    return {"results": {"video_validation_0000001": rows}}


def _ann(start, end, label="A"):
    return {"segment": [start, end], "label": label}


def _pred(start, end, score=1.0, label="A"):
    return {"segment": [start, end], "score": score, "label": label}


def test_short_action_definition_is_inclusive_at_five_seconds():
    annotation = _annotation([_ann(0, 5), _ann(10, 15.01)])
    filtered = diagnostics.short_action_annotation(annotation, subset="validation")
    assert filtered["database"]["video_validation_0000001"]["annotations"] == [
        _ann(0, 5)
    ]


def test_serial_short_action_map_uses_official_ap_primitive():
    annotation = _annotation([_ann(0, 5)])
    prediction = _prediction([_pred(0, 5)])
    result = diagnostics.serial_short_action_map(
        annotation, prediction, subset="validation"
    )
    assert result["average_mAP"] == pytest.approx(1.0)
    assert set(result["mAP_by_tiou"]) == {"0.3", "0.4", "0.5", "0.6", "0.7"}
    assert all(value == pytest.approx(1.0) for value in result["mAP_by_tiou"].values())


def test_boundary_match_includes_exact_tiou_half():
    annotation = _annotation([_ann(0, 2)])
    prediction = _prediction([_pred(0, 1)])
    result = diagnostics.boundary_diagnostics(
        annotation, prediction, subset="validation"
    )
    assert result["matched_count"] == 1
    assert result["median_abs_start_error_normalized"] == pytest.approx(0.0)
    assert result["median_abs_end_error_normalized"] == pytest.approx(0.5)


def test_boundary_matching_is_score_ordered_and_one_to_one():
    annotation = _annotation([_ann(0, 10)])
    prediction = _prediction(
        [
            _pred(0, 10, score=0.1),
            _pred(0, 8, score=0.9),
        ]
    )
    result = diagnostics.boundary_diagnostics(
        annotation, prediction, subset="validation"
    )
    assert result["matched_count"] == 1
    assert result["median_abs_end_error_normalized"] == pytest.approx(0.2)


def test_boundary_gate_uses_median_absolute_not_mean_or_signed_error():
    annotation = _annotation(
        [
            _ann(0, 10, "A"),
            _ann(20, 30, "B"),
            _ann(40, 50, "C"),
        ]
    )
    prediction = _prediction(
        [
            _pred(1, 9, label="A"),
            _pred(19, 31, label="B"),
            _pred(44, 50, label="C"),
        ]
    )
    result = diagnostics.boundary_diagnostics(
        annotation, prediction, subset="validation"
    )
    assert result["median_abs_start_error_normalized"] == pytest.approx(0.1)
    assert result["mean_abs_start_error_normalized"] == pytest.approx(0.2)
    assert result["median_abs_end_error_normalized"] == pytest.approx(0.1)


def test_frozen_guard_boundaries_are_inclusive():
    reference = {
        "short_action_mAP": {"average_mAP": 0.50},
        "boundary": {
            "median_abs_start_error_normalized": 0.10,
            "median_abs_end_error_normalized": 0.20,
        },
    }
    candidate = {
        "short_action_mAP": {"average_mAP": 0.485},
        "boundary": {
            "median_abs_start_error_normalized": 0.11,
            "median_abs_end_error_normalized": 0.22,
        },
    }
    comparison = diagnostics.compare_frozen_guards(reference, candidate)
    assert comparison["short_action_mAP_decrease_pp"] == pytest.approx(1.5)
    assert comparison["boundary_start_worsening_ratio"] == pytest.approx(1.1)
    assert comparison["boundary_end_worsening_ratio"] == pytest.approx(1.1)
    assert comparison["reconstructed_guards_passed"] is True


def test_zero_reference_boundary_fails_closed():
    reference = {
        "short_action_mAP": {"average_mAP": 0.50},
        "boundary": {
            "median_abs_start_error_normalized": 0.0,
            "median_abs_end_error_normalized": 0.20,
        },
    }
    candidate = {
        "short_action_mAP": {"average_mAP": 0.50},
        "boundary": {
            "median_abs_start_error_normalized": 0.0,
            "median_abs_end_error_normalized": 0.20,
        },
    }
    with pytest.raises(ValueError, match="must be positive"):
        diagnostics.compare_frozen_guards(reference, candidate)


def test_no_boundary_match_fails_closed():
    annotation = _annotation([_ann(0, 1)])
    prediction = _prediction([_pred(10, 11)])
    with pytest.raises(ValueError, match="no boundary match"):
        diagnostics.boundary_diagnostics(
            annotation, prediction, subset="validation"
        )


def test_nonfinite_prediction_score_fails_closed():
    annotation = _annotation([_ann(0, 1)])
    reference = _prediction([_pred(0, 1)])
    candidate = _prediction([_pred(0, 1, score=float("nan"))])
    with pytest.raises(ValueError, match="must be finite"):
        diagnostics.evaluate_pair(annotation, reference, candidate)


def test_zero_length_prediction_remains_an_official_false_positive():
    annotation = _annotation([_ann(0, 1)])
    prediction = _prediction(
        [
            _pred(2, 2, score=2.0),
            _pred(0, 1, score=1.0),
        ]
    )
    result = diagnostics.serial_short_action_map(
        annotation, prediction, subset="validation"
    )
    assert 0.0 < result["average_mAP"] < 1.0
