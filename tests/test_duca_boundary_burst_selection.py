from __future__ import annotations

import json
from pathlib import Path

from tools.bata.select_duca_boundary_burst_candidates import (
    _effective_budget_contract_verified,
    _read_candidate,
    _ranking_key,
)
from tools.bata.select_duca_frontend_checkpoint import sha256_file


def _contract_summary() -> dict:
    return {
        "sample_count": 3,
        "protocol": {
            "budget_matched": True,
            "valid_length_matched": True,
            "max_hole_matched": True,
            "sampling_contract_evidence": {
                "sample_count": 3,
                "requested_budget_min": 384,
                "requested_budget_max": 384,
                "effective_budget_min": 300,
                "effective_budget_max": 384,
                "selected_count_min": 300,
                "selected_count_max": 384,
                "budget_violation_count": 0,
                "requested_max_unselected_hole_min": 2,
                "requested_max_unselected_hole_max": 2,
                "observed_max_unselected_hole_min": 0,
                "observed_max_unselected_hole_max": 2,
                "max_hole_violation_count": 0,
            },
        },
    }


def test_effective_budget_contract_accepts_short_valid_windows() -> None:
    summary = _contract_summary()
    assert _effective_budget_contract_verified(summary)
    summary["protocol"]["sampling_contract_evidence"]["budget_violation_count"] = 1
    assert not _effective_budget_contract_verified(summary)
    assert not _effective_budget_contract_verified({})


def test_effective_budget_contract_rejects_self_asserted_boole_without_rows() -> None:
    summary = {
        "sample_count": 3,
        "protocol": {
            "budget_matched": True,
            "valid_length_matched": True,
            "max_hole_matched": True,
        },
    }
    assert not _effective_budget_contract_verified(summary)


def _candidate(variant: str, quota_gain: float, epoch: int) -> dict:
    return {
        "variant": variant,
        "epoch_one_based": epoch,
        "metrics": {
            "both_endpoints_quota_recall_gain": quota_gain,
            "endpoint_quota_recall_gain": quota_gain,
            "endpoint_bilateral_recall_gain": quota_gain,
            "boundary_recall_r0_gain": 0.0,
            "uniform_minus_learned_endpoint_distance": 0.0,
            "policy_transition_auroc_r0": 0.5,
        },
    }


def test_checkpoint_selection_uses_earliest_passing_epoch_not_holdout_best() -> None:
    weak_early = _candidate("burst_r2q3", 0.05, 5)
    strong_late = _candidate("burst_r2q3", 0.20, 20)

    assert sorted([strong_late, weak_early], key=_ranking_key)[0] is weak_early


def test_checkpoint_selection_uses_epoch_for_all_mechanisms() -> None:
    early = _candidate("burst_r4q5", 0.20, 10)
    late = _candidate("burst_r4q5", 0.20, 20)

    assert sorted([late, early], key=_ranking_key)[0] is early


def _write_candidate(
    tmp_path: Path,
    *,
    variant: str,
    policy_auroc: float,
    distance_gain: float,
    burst_gain: float = 0.0,
    simple_delta_recall: float = 0.65,
    simple_delta_distance: float = 1.10,
) -> dict:
    checkpoint = tmp_path / f"{variant}.pth"
    records = tmp_path / f"{variant}.jsonl"
    summary_path = tmp_path / f"{variant}.summary.json"
    checkpoint.write_bytes(b"checkpoint")
    records.write_text("{}\n", encoding="utf-8")
    summary = _contract_summary()
    summary.update(
        {
            "schema_version": "duca_selection_quality_summary_v2",
            "coarse": {"pooled": {"auroc": 0.70, "auprc_lift": 1.20}},
            "transition": {
                "r0": {
                    "policy": {"auroc": policy_auroc},
                    "pure_abs_delta_p_action": {"auroc": 0.60},
                }
            },
            "selection": {
                "learned": {
                    "boundary_recall": {"r0": {"mean": 0.7}},
                    "mean_endpoint_distance": {"mean": 1.0 - distance_gain},
                    "max_unselected_hole": {"mean": 2.0},
                    "selected_count": {"mean": 360.0},
                    "boundary_burst": {},
                },
                "uniform": {
                    "boundary_recall": {"r0": {"mean": 0.6}},
                    "mean_endpoint_distance": {"mean": 1.0},
                    "boundary_burst": {},
                },
                "pure_delta_same_feasible_dp": {
                    "boundary_recall": {"r0": {"mean": simple_delta_recall}},
                    "mean_endpoint_distance": {"mean": simple_delta_distance},
                    "boundary_burst": {},
                },
            },
        }
    )
    if variant.startswith("burst_"):
        key = "r2q3" if variant == "burst_r2q3" else "r4q5"
        for method, base in (
            ("learned", burst_gain),
            ("uniform", 0.0),
            ("pure_delta_same_feasible_dp", 0.0),
        ):
            summary["selection"][method]["boundary_burst"][key] = {
                "endpoint_quota_recall": {"mean": base},
                "endpoint_bilateral_recall": {"mean": base},
                "both_endpoints_quota_recall": {"mean": base},
            }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return {
        "epoch_one_based": 5,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "records_path": str(records),
        "records_sha256": sha256_file(records),
    }


def test_candidate_gate_requires_scorer_centering_and_burst_mechanism_gains(tmp_path: Path) -> None:
    passing = _read_candidate(
        _write_candidate(
            tmp_path,
            variant="burst_r2q3",
            policy_auroc=0.61,
            distance_gain=0.01,
            burst_gain=0.01,
        ),
        "burst_r2q3",
    )
    assert passing["all_sanity_gates_pass"] is True

    weak = _read_candidate(
        _write_candidate(
            tmp_path,
            variant="burst_r4q5",
            policy_auroc=0.59,
            distance_gain=-0.01,
            burst_gain=0.0,
        ),
        "burst_r4q5",
    )
    assert weak["gates"]["transition_scorer_not_worse_than_pure_delta_r0"] is False
    assert weak["gates"]["endpoint_centering_not_worse_than_uniform"] is False
    assert weak["gates"]["burst_bilateral_gain_positive"] is False
    assert weak["all_sanity_gates_pass"] is False


def test_candidate_stop_rule_rejects_simple_delta_dominance(tmp_path: Path) -> None:
    candidate = _read_candidate(
        _write_candidate(
            tmp_path,
            variant="burst_r2q3",
            policy_auroc=0.65,
            distance_gain=0.05,
            burst_gain=0.05,
            simple_delta_recall=0.80,
            simple_delta_distance=0.50,
        ),
        "burst_r2q3",
    )
    assert candidate["gates"][
        "learned_selector_strictly_pareto_beats_same_feasible_simple_delta"
    ] is False
    assert candidate["all_sanity_gates_pass"] is False


def test_gaussian_candidate_does_not_require_burst_metrics(tmp_path: Path) -> None:
    candidate = _read_candidate(
        _write_candidate(
            tmp_path,
            variant="gaussian_matched",
            policy_auroc=0.60,
            distance_gain=0.0,
        ),
        "gaussian_matched",
    )
    assert candidate["all_sanity_gates_pass"] is True
    assert not any(key.startswith("burst_") for key in candidate["gates"])
