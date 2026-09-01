from __future__ import annotations

import pytest

from tools.bata import paction_lattice_replacement_policy as policy


def _scores_with_local_peaks(length: int, centers: list[int], *, peak: float = 10.0) -> list[float]:
    scores = [0.0 for _ in range(length)]
    for offset, center in enumerate(centers):
        scores[int(center)] = float(peak - offset * 0.01)
    return scores


def test_move50_protects_exactly_192_anchors_and_keeps_384_selected_on_long_valid_sequence() -> None:
    scores = _scores_with_local_peaks(1024, list(range(2, 1024, 3))[:220])

    result = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.MOVE50_STRATEGY,
        local_radius=2,
    )

    assert result.diagnostics["protected_uniform_count"] == 192
    assert result.diagnostics["replaceable_uniform_count"] == 192
    assert result.diagnostics["selected_count"] == 384
    assert len(result.selected_positions) == 384


def test_move25_protects_288_anchors_and_replaces_only_one_quarter_budget() -> None:
    scores = _scores_with_local_peaks(1024, list(range(2, 1024, 3))[:220])

    result = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.MOVE25_STRATEGY,
        local_radius=2,
    )

    assert result.diagnostics["protected_uniform_count"] == 288
    assert result.diagnostics["replaceable_uniform_count"] == 96
    assert result.diagnostics["selected_count"] == 384
    assert len(result.selected_positions) == 384


def test_move75_allows_more_replacements_than_move50_on_same_score_vector() -> None:
    scores = _scores_with_local_peaks(1024, list(range(2, 1024, 3))[:320])

    move25 = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.MOVE25_STRATEGY,
        local_radius=2,
    )
    move50 = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.MOVE50_STRATEGY,
        local_radius=2,
    )
    move75 = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.MOVE75_STRATEGY,
        local_radius=2,
    )

    assert move75.diagnostics["protected_uniform_count"] == 96
    assert move75.diagnostics["replaceable_uniform_count"] == 288
    assert move50.diagnostics["replaced_uniform_count"] > move25.diagnostics["replaced_uniform_count"]
    assert move75.diagnostics["replaced_uniform_count"] > move50.diagnostics["replaced_uniform_count"]


def test_selection_requires_no_manual_role_slots_and_diagnostics_have_no_gt_or_leak_keys() -> None:
    scores = _scores_with_local_peaks(768, list(range(1, 768, 2))[:250])

    result = policy.decode_paction_lattice_replacement(
        score=scores,
        variant=policy.MOVE50_STRATEGY,
        local_radius=1,
    )

    forbidden_fragments = ("gt", "teacher", "oracle", "boundary", "transition", "uncertainty", "context", "role")
    diagnostic_keys = set(result.diagnostics)
    assert result.diagnostics["strategy_name"] == policy.MOVE50_STRATEGY
    assert not any(fragment in key.lower() for key in diagnostic_keys for fragment in forbidden_fragments)


def test_local_radius_prevents_replacing_far_anchors_even_for_very_high_score_candidates() -> None:
    scores = [0.0 for _ in range(1024)]
    scores[511] = 1_000_000.0

    result = policy.decode_paction_lattice_replacement(
        paction_score=scores,
        variant=policy.NO_PROTECT_STRATEGY,
        local_radius=0,
    )

    assert 511 not in result.selected_positions
    assert result.diagnostics["inserted_candidate_count"] == 0
    assert result.diagnostics["replaced_uniform_count"] == 0


def test_output_is_sorted_unique_and_within_valid_positions() -> None:
    valid = [idx % 5 != 0 for idx in range(900)]
    scores = _scores_with_local_peaks(900, [idx for idx in range(900) if valid[idx] and idx % 7 == 1][:260])

    result = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        valid=valid,
        variant=policy.MOVE75_STRATEGY,
        local_radius=3,
    )

    assert result.selected_positions == sorted(result.selected_positions)
    assert len(result.selected_positions) == len(set(result.selected_positions))
    assert all(valid[idx] for idx in result.selected_positions)
    assert len(result.selected_positions) <= 384


def test_no_protect_is_score_driven_but_obeys_local_replacement_and_budget_constraints() -> None:
    scores = _scores_with_local_peaks(1024, list(range(2, 1024, 3))[:360])

    result = policy.decode_paction_lattice_replacement(
        frame_values=scores,
        variant=policy.NO_PROTECT_STRATEGY,
        local_radius=2,
    )

    topk = set(sorted(range(len(scores)), key=lambda idx: (scores[idx], -idx), reverse=True)[:384])
    assert result.diagnostics["protected_uniform_count"] == 0
    assert result.diagnostics["replaceable_uniform_count"] == 384
    assert result.diagnostics["selected_count"] == 384
    assert result.diagnostics["inserted_candidate_count"] <= 384
    assert set(result.selected_positions) != topk
    assert result.diagnostics["paction_topk_overlap"] > 0.70
