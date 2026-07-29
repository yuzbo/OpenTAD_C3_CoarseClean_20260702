from __future__ import annotations

import math

import pytest

from tools.bata.duca_gate_diagnostics import (
    detector_loss_parity_report,
    uniform_axis_geometry_report,
)


def _exact_uniform_anchor_list(temporal_len: int, k: int) -> list[int]:
    anchors = []
    denominator = k - 1
    for index in range(k):
        quotient, remainder = divmod(index * (temporal_len - 1), denominator)
        if 2 * remainder > denominator or (
            2 * remainder == denominator and quotient % 2 == 1
        ):
            quotient += 1
        anchors.append(quotient)
    return anchors


def test_uniform_axis_geometry_reports_affine_integer_grid() -> None:
    report = uniform_axis_geometry_report(
        valid_len=7,
        positions=[0, 2, 4, 6],
    )

    assert report["divisibility_precondition_satisfied"] is True
    assert report["global_affine_coordinate_precondition_satisfied"] is True
    assert report["physical_step_histogram"] == {"2": 3}


def test_uniform_axis_geometry_rejects_768_to_384_affine_assumption() -> None:
    report = uniform_axis_geometry_report(
        valid_len=768,
        positions=_exact_uniform_anchor_list(768, 384),
    )

    assert report["divisibility_precondition_satisfied"] is False
    assert report["global_affine_coordinate_precondition_satisfied"] is False
    assert report["physical_step_histogram"] == {"2": 382, "3": 1}
    assert "not implied" in report["interpretation"]


def test_uniform_axis_geometry_handles_singleton_identity_axis() -> None:
    report = uniform_axis_geometry_report(valid_len=1, positions=[0])

    assert report["effective_k"] == 1
    assert report["endpoint_anchored"] is True
    assert report["divisibility_precondition_satisfied"] is True
    assert report["global_affine_coordinate_precondition_satisfied"] is True
    assert report["nominal_physical_step"] is None


@pytest.mark.parametrize(
    ("valid_len", "positions", "error_type"),
    [
        (0, [0], ValueError),
        (4, [], ValueError),
        (4, [0, 4], ValueError),
        (4, [0, 2, 2, 3], ValueError),
        (4.0, [0, 3], TypeError),
        (4, [0, 1.5, 3], TypeError),
        (True, [0], TypeError),
        (1, [False], TypeError),
    ],
)
def test_uniform_axis_geometry_rejects_invalid_inputs(
    valid_len: object,
    positions: list[object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        uniform_axis_geometry_report(valid_len=valid_len, positions=positions)


def test_detector_loss_parity_report_identifies_worst_key_and_threshold() -> None:
    report = detector_loss_parity_report(
        {"cls_loss": 0.5, "reg_loss": 2.0},
        {"cls_loss": 0.50001, "reg_loss": 2.001},
        relative_tolerance=1.0e-4,
    )

    assert report["all_equal_within_registered_tolerance"] is False
    assert report["worst_key"] == "reg_loss"
    assert report["per_key"]["cls_loss"]["equal_within_registered_tolerance"] is True
    assert report["per_key"]["reg_loss"]["threshold"] == 2.001e-4


@pytest.mark.parametrize("relative_tolerance", [-1.0, math.inf, math.nan])
def test_detector_loss_parity_report_rejects_invalid_tolerance(
    relative_tolerance: float,
) -> None:
    with pytest.raises(ValueError, match="relative_tolerance"):
        detector_loss_parity_report(
            {"loss": 1.0},
            {"loss": 1.0},
            relative_tolerance=relative_tolerance,
        )


def test_detector_loss_parity_report_rejects_empty_or_mismatched_losses() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        detector_loss_parity_report({}, {}, relative_tolerance=1.0e-4)
    with pytest.raises(ValueError, match="keys differ"):
        detector_loss_parity_report(
            {"cls_loss": 1.0},
            {"reg_loss": 1.0},
            relative_tolerance=1.0e-4,
        )


@pytest.mark.parametrize("bad_value", [math.inf, -math.inf, math.nan])
def test_detector_loss_parity_report_rejects_non_finite_losses(
    bad_value: float,
) -> None:
    with pytest.raises(ValueError, match="non-finite"):
        detector_loss_parity_report(
            {"loss": bad_value},
            {"loss": 1.0},
            relative_tolerance=1.0e-4,
        )
