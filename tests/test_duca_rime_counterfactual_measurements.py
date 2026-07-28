from __future__ import annotations

import math

import pytest

from tools.bata.produce_duca_rime_counterfactual_measurements import (
    budget_features,
    legal_local_swaps,
    observed_pair_failures,
)
from tools.bata.produce_duca_rime_o2_panel import _observed_max_gap


def test_budget_features_are_label_free_finite_and_budget_sensitive():
    frame_features = [
        [index / 8.0, (-1.0) ** index, math.sin(index)]
        for index in range(8)
    ]
    low = budget_features(frame_features, budget=2, maximum_budget=8)
    high = budget_features(frame_features, budget=6, maximum_budget=8)
    assert len(low) == len(high) == 9
    assert all(math.isfinite(value) for value in low + high)
    assert low[:3] != high[:3]
    assert low[3:] == high[3:]


def test_local_swaps_preserve_exact_k_order_uniqueness_and_gap():
    seconds = [index * 0.25 for index in range(12)]
    baseline = [0, 3, 6, 9, 11]
    swaps = legal_local_swaps(
        baseline,
        seconds,
        max_gap_seconds=1.0,
        max_candidates=6,
    )
    assert len(swaps) >= 3
    for row in swaps:
        selected = row["selected_positions"]
        assert selected == sorted(set(selected))
        assert len(selected) == len(baseline)
        assert row["added_position"] in selected
        assert row["removed_position"] not in selected
        observed = _observed_max_gap(selected, seconds)
        assert observed == pytest.approx(row["observed_max_gap_seconds"])
        assert observed <= 1.0 + 1.0e-8


def test_pair_failure_is_directional_and_kmax_is_safe_anchor():
    failures = observed_pair_failures(
        [1.20, 1.00, 1.01, 0.75],
        relative_tolerance=0.05,
        absolute_tolerance=0.01,
    )
    assert failures == [1, 0, 1, 0]


def test_o2_observed_gap_rejects_duplicate_positions():
    with pytest.raises(ValueError, match="invalid"):
        _observed_max_gap([0, 1, 1], [0.0, 0.5, 1.0])
