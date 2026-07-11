import pytest

from opentad.models.chronotransport.adjudication import (
    gate1_oracle_headroom,
    gate2_matched_transport,
)


HOLD = (
    "periodic2_hold",
    "periodic4_hold",
    "periodic8_hold",
    "hold_only",
    "layer_only_early_recompute_hold",
    "layer_only_late_recompute_hold",
    "joint_progressive_hold",
    "joint_reverse_hold",
)
CONTROLS = ("motion_topk_p2", "motion_topk_p4", "motion_topk_p8", "random_p2", "random_p4", "random_p8")


def _gate1_records(count, joint_gain):
    records = {}
    for index in range(count):
        row = {name: 1.0 + 0.05 * position for position, name in enumerate(HOLD)}
        row["joint_progressive_hold"] = 1.0 - joint_gain if index % 2 == 0 else 1.2
        row["joint_reverse_hold"] = 1.2 if index % 2 == 0 else 1.0 - joint_gain
        row.update({name: 1.05 for name in CONTROLS})
        records[f"w{index:02d}"] = row
    return records


def test_gate1_passes_only_with_equal_cost_oracle_headroom_and_saving():
    costs = {name: 7.0 for name in (*HOLD, *CONTROLS)}
    result = gate1_oracle_headroom(
        calibration=_gate1_records(30, 0.2),
        evaluation=_gate1_records(30, 0.2),
        candidate_cost_p50=costs,
        dense_cost_p50=10.0,
        budget=7.5,
        bootstrap_samples=300,
    )
    assert result["status"] == "PASS"
    assert result["oracle_headroom"] is True
    assert result["relative_reduction"] >= 0.10
    assert result["bootstrap_ci95"][0] > 0
    assert result["budget_saving"] >= 0.20


def test_gate1_fails_when_strongest_comparator_mean_is_zero():
    records = _gate1_records(30, 0.0)
    for row in records.values():
        row["periodic2_hold"] = 0.0
    costs = {name: 7.0 for name in (*HOLD, *CONTROLS)}
    result = gate1_oracle_headroom(
        calibration=records,
        evaluation=records,
        candidate_cost_p50=costs,
        dense_cost_p50=10.0,
        budget=7.5,
        bootstrap_samples=50,
    )
    assert result["status"] == "FAIL"
    assert result["hard_conditions"]["relative_reduction_ge_10pct"] is False


def test_gate1_fails_closed_on_missing_hard_comparator():
    costs = {name: 7.0 for name in (*HOLD, *CONTROLS) if name != "random_p8"}
    records = _gate1_records(30, 0.2)
    with pytest.raises(ValueError, match="requires cost-feasible comparators"):
        gate1_oracle_headroom(
            calibration=records,
            evaluation=records,
            candidate_cost_p50=costs,
            dense_cost_p50=10.0,
            budget=7.5,
            bootstrap_samples=10,
        )


def test_gate2_hierarchical_matched_pass_and_per_seed_condition():
    rows = []
    for window in range(8):
        for seed in (3407, 3408, 3409):
            for period in (2, 4, 8):
                rows.append(
                    {
                        "window_id": f"w{window}",
                        "seed": seed,
                        "period": period,
                        "hold_regret": 1.0,
                        "transport_regret": 0.8,
                        "hold_mse": 2.0,
                        "transport_mse": 1.5,
                    }
                )
    result = gate2_matched_transport(rows, bootstrap_samples=200)
    assert result["status"] == "PASS"
    assert result["mechanism"] is True
    assert result["detector_ci95"][0] > 0
