from __future__ import annotations

from tools.bata.select_duca_boundary_burst_candidates import _ranking_key


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


def test_boundary_burst_checkpoint_selection_prioritizes_oracle_like_microclusters() -> None:
    weak_early = _candidate("burst_r2q3", 0.05, 5)
    strong_late = _candidate("burst_r2q3", 0.20, 20)

    assert sorted([weak_early, strong_late], key=_ranking_key)[0] is strong_late


def test_boundary_burst_checkpoint_selection_uses_earlier_epoch_only_as_tie_break() -> None:
    early = _candidate("burst_r4q5", 0.20, 10)
    late = _candidate("burst_r4q5", 0.20, 20)

    assert sorted([late, early], key=_ranking_key)[0] is early
