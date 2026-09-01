from __future__ import annotations

import torch

from opentad.models.duca.hard_soft_alignment import (
    enumerate_legal_local_hard_swaps,
    hard_soft_alignment_report,
    preregistered_hard_soft_gate,
    surrogate_hard_swap_descent,
)


def _max_hole(positions: torch.Tensor, valid_len: int) -> int:
    sentinels = torch.cat(
        (positions.new_tensor([-1]), positions, positions.new_tensor([valid_len]))
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def test_local_hard_swap_enumerator_preserves_exact_k_order_and_gap() -> None:
    output = enumerate_legal_local_hard_swaps(
        torch.tensor([[0, 2, 5, 7]]),
        torch.ones(1, 8, dtype=torch.bool),
        max_unselected_hole=2,
        max_displacement=2,
        max_candidates_per_sample=8,
    )
    valid = output["candidate_valid"][0]

    assert int(valid.sum()) > 0
    for candidate in output["candidate_selections"][0, valid]:
        assert candidate.numel() == 4
        assert torch.unique(candidate).numel() == 4
        assert bool(((candidate[1:] - candidate[:-1]) > 0).all())
        assert _max_hole(candidate, 8) <= 2


def test_surrogate_swap_descent_uses_add_minus_remove_gradient_direction() -> None:
    gradient = torch.tensor([[0.0, 3.0, 0.0, -2.0]])
    predicted = surrogate_hard_swap_descent(
        gradient,
        add_positions=torch.tensor([[3, 1]]),
        remove_positions=torch.tensor([[1, 3]]),
        candidate_valid=torch.tensor([[True, True]]),
    )

    assert torch.equal(predicted, torch.tensor([[5.0, -5.0]]))


def test_preregistered_alignment_gate_passes_only_strong_multi_batch_evidence() -> None:
    predicted = [float(index - 16) for index in range(32)]
    observed = [2.0 * value for value in predicted]
    report = hard_soft_alignment_report(
        predicted,
        observed,
        batch_ids=[index // 8 for index in range(32)],
        bootstrap_samples=200,
    )
    gate = preregistered_hard_soft_gate(report)

    assert report["informative_nonzero_count"] == 31
    assert report["informative_batch_count"] == 4
    assert report["sign_agreement"] == 1.0
    assert report["spearman"] == 1.0
    assert gate["passed"] is True


def test_preregistered_alignment_gate_rejects_anti_aligned_surrogate() -> None:
    predicted = [float(index + 1) for index in range(32)]
    observed = [-value for value in predicted]
    report = hard_soft_alignment_report(
        predicted,
        observed,
        batch_ids=[index // 8 for index in range(32)],
        bootstrap_samples=200,
    )
    gate = preregistered_hard_soft_gate(report)

    assert report["sign_agreement"] == 0.0
    assert report["spearman"] == -1.0
    assert gate["passed"] is False
