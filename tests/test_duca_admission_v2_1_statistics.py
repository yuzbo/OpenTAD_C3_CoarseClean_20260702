from __future__ import annotations

import hashlib
import math
from fractions import Fraction

import pytest

from tools.bata.duca_admission_v2_1_metrics import (
    METRIC_IDS,
    fit_scale_normalizers,
    summarize_cell_metric,
)
from tools.bata.duca_admission_v2_1_statistics import (
    MULTIPLIER_KAPPA,
    MULTIPLIER_PROBABILITIES,
    MULTIPLIER_VALUES,
    estimate_role_contrast,
    factor_moments,
    finalize_max_t,
    is_binary64_positive_zero,
    multiplier_replicates,
    type1_max_t_order_index,
)


def make_role_cells(role_id: str, value_fn):
    cells = []
    for rank in range(32):
        bases = (rank % 8, (rank + rank // 8 + 1) % 8)
        for slot, process in enumerate(bases):
            row = {
                "cell_id": hashlib.sha256(
                    f"{role_id}:{rank}:{slot}".encode("utf-8")
                ).hexdigest(),
                "role_id": role_id,
                "video_id": f"{role_id}_video_{rank:02d}",
                "canonical_video_rank": rank,
                "slot": slot,
                "logical_process_index": process,
                "process_id": f"{role_id}:p{process:02d}",
            }
            for metric_index, metric_id in enumerate(METRIC_IDS):
                row[metric_id] = float(value_fn(rank, slot, metric_index))
            cells.append(row)
    return cells


def test_type1_scale_fit_uses_lower_even_sample_median():
    cells = make_role_cells("scale_fit", lambda rank, slot, _metric: 2 * rank + slot)
    scales = fit_scale_normalizers(cells)
    assert scales == {metric_id: 31.0 for metric_id in METRIC_IDS}


def test_count_aware_tail_summary_uses_largest_ceil_five_percent():
    rows = [
        {"observation_id": f"o{index:02d}", "value": float(index)}
        for index in range(21)
    ]
    assert summarize_cell_metric(rows) == (20.0 + 19.0) / 2.0


def test_exact_zero_requires_binary64_positive_zero():
    assert is_binary64_positive_zero(0.0)
    assert not is_binary64_positive_zero(-0.0)
    assert not is_binary64_positive_zero(0)
    scales = {metric_id: 1.0 for metric_id in METRIC_IDS}
    positive = make_role_cells("calibration", lambda *_: 0.0) + make_role_cells(
        "admission_holdout", lambda *_: 0.0
    )
    contrast = estimate_role_contrast(cells=positive, scale_normalizers=scales)
    assert all(contrast["exact_zero"].values())
    negative = make_role_cells("calibration", lambda *_: -0.0) + make_role_cells(
        "admission_holdout", lambda *_: 0.0
    )
    contrast = estimate_role_contrast(cells=negative, scale_normalizers=scales)
    assert not any(contrast["exact_zero"].values())


def test_two_point_factor_moments_and_frozen_type1_indices():
    assert MULTIPLIER_VALUES == (0.5, 3.0)
    assert MULTIPLIER_PROBABILITIES == (0.8, 0.2)
    assert MULTIPLIER_KAPPA == 1.0
    assert factor_moments() == {"mean": 1.0, "variance": 1.0, "second_moment": 2.0}
    assert type1_max_t_order_index(100_000, 0.05) == 95_001
    assert type1_max_t_order_index(200_000, 0.05) == 190_001
    with pytest.raises(ValueError, match="INVALID_TYPE1_ORDER_INDEX"):
        type1_max_t_order_index(10, 0.05)


def test_multiplier_extension_preserves_exact_prefix():
    role_cells = {
        role: make_role_cells(role, lambda *_: 0.0)
        for role in ("calibration", "admission_holdout")
    }
    residuals = {
        role: {
            metric_id: [float(index - 31.5) for index in range(64)]
            for metric_id in METRIC_IDS
        }
        for role in role_cells
    }
    kwargs = {
        "role_cells": role_cells,
        "residuals": residuals,
        "registry_hashes": ["a" * 64, "b" * 64, "c" * 64, "d" * 64],
        "stream_id": 0,
        "replicate_start": 0,
    }
    first = multiplier_replicates(**kwargs, replicate_stop=5)
    extended = multiplier_replicates(**kwargs, replicate_stop=10)
    assert first == extended[:5]


def test_max_t_exact_zero_and_degenerate_scale_fail_closed():
    exact = finalize_max_t(
        delta_hat={metric_id: 0.0 for metric_id in METRIC_IDS},
        replicates=[],
        exact_zero={metric_id: True for metric_id in METRIC_IDS},
    )
    assert exact["numeric_tail_status"] == "PASSED_EXACT_ZERO"
    assert exact["mc_required"] is False
    replicates = [{metric_id: 1.0 for metric_id in METRIC_IDS} for _ in range(20)]
    with pytest.raises(ValueError, match="DEGENERATE_SCALE"):
        finalize_max_t(
            delta_hat={metric_id: 0.0 for metric_id in METRIC_IDS},
            replicates=replicates,
            exact_zero={metric_id: False for metric_id in METRIC_IDS},
            alpha=0.1,
        )


def test_statistics_reject_identity_grid_drift_and_fake_exact_zero():
    scales = {metric_id: 1.0 for metric_id in METRIC_IDS}
    cells = make_role_cells("calibration", lambda *_: 1.0) + make_role_cells(
        "admission_holdout", lambda *_: 1.0
    )
    cells[1]["canonical_video_rank"] = 0
    cells[1]["slot"] = 0
    cells[1]["cell_id"] = hashlib.sha256(b"unique-tampered-cell").hexdigest()
    with pytest.raises(ValueError, match="complete 32x2 role grid"):
        estimate_role_contrast(cells=cells, scale_normalizers=scales)
    delta = {metric_id: 0.0 for metric_id in METRIC_IDS}
    delta["M00"] = -0.0
    with pytest.raises(ValueError, match="positive-zero contrast"):
        finalize_max_t(
            delta_hat=delta,
            replicates=[],
            exact_zero={metric_id: True for metric_id in METRIC_IDS},
        )


def _trace_for_component(cells, component):
    count = len(cells)
    sigma = []
    for left in cells:
        row = []
        for right in cells:
            if component == "row":
                value = int(left[0] == right[0])
            elif component == "process":
                value = int(left[1] == right[1])
            else:
                value = int(left == right)
            row.append(Fraction(value, 1))
        sigma.append(row)
    row_means = [sum(row, Fraction()) / count for row in sigma]
    column_means = [
        sum(sigma[i][j] for i in range(count)) / count for j in range(count)
    ]
    grand = sum(row_means, Fraction()) / count
    trace = Fraction()
    for i, left in enumerate(cells):
        for j, right in enumerate(cells):
            omega = (
                3
                if left == right
                else 1
                if left[0] == right[0] or left[1] == right[1]
                else 0
            )
            centered = sigma[j][i] - row_means[j] - column_means[i] + grand
            trace += omega * centered
    return trace


def test_exact_frozen_variance_gain_traces():
    cells = []
    for rank in range(32):
        cells.extend(
            (
                (rank, rank % 8),
                (rank, (rank + rank // 8 + 1) % 8),
            )
        )
    assert _trace_for_component(cells, "row") == 234
    assert _trace_for_component(cells, "process") == 552
    assert _trace_for_component(cells, "cell") == 181
    assert Fraction(234, 4096) / Fraction(1, 32) == Fraction(117, 64)
    assert Fraction(552, 4096) / Fraction(1, 8) == Fraction(69, 64)
    assert Fraction(181, 4096) / Fraction(1, 64) == Fraction(181, 64)
    rejected_kappa_squared = Fraction(32, 31) * Fraction(8, 7)
    assert Fraction(117, 64) * rejected_kappa_squared == Fraction(468, 217)
    assert Fraction(69, 64) * rejected_kappa_squared == Fraction(276, 217)
    assert Fraction(181, 64) * rejected_kappa_squared == Fraction(724, 217)
