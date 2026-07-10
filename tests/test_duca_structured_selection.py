from __future__ import annotations

from itertools import combinations
import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

import torch

from opentad.models.duca.structured_selection import global_structured_topk


def _max_hole(indices: tuple[int, ...], temporal_len: int) -> int:
    if not indices:
        return temporal_len
    return max(
        indices[0],
        *(right - left - 1 for left, right in zip(indices, indices[1:])),
        temporal_len - indices[-1] - 1,
    )


def test_structured_hard_map_matches_bruteforce_optimum() -> None:
    logits = torch.tensor([[0.2, 1.4, -0.7, 0.8, 1.1, -0.2, 0.5]])
    k = 3
    max_hole = 2
    feasible = [
        choice
        for choice in combinations(range(logits.shape[1]), k)
        if _max_hole(choice, logits.shape[1]) <= max_hole
    ]
    expected = max(feasible, key=lambda choice: sum(float(logits[0, idx]) for idx in choice))

    out = global_structured_topk(logits, k=k, max_unselected_hole=max_hole, training=False)

    assert tuple(out.selected_positions[0].tolist()) == expected
    assert out.hard_occupancy.sum().item() == k
    assert _max_hole(expected, logits.shape[1]) <= max_hole


def test_structured_soft_slots_obey_exact_budget_and_st_forward_identity() -> None:
    logits = torch.randn(2, 9, requires_grad=True)
    out = global_structured_topk(
        logits,
        k=4,
        max_unselected_hole=2,
        temperature=0.7,
        training=True,
    )

    assert torch.equal(out.selection_st.detach(), out.hard_occupancy)
    assert torch.allclose(out.soft_slot_assignment.sum(dim=2), torch.ones(2, 4), atol=1e-5)
    assert torch.allclose(out.soft_occupancy.sum(dim=1), torch.full((2,), 4.0), atol=1e-5)
    assert torch.allclose(out.soft_occupancy, out.soft_slot_assignment.sum(dim=1), atol=1e-6)

    time_cost = torch.arange(9, dtype=logits.dtype)[None, :]
    (out.selection_st * time_cost).sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum().item() > 0.0


def test_structured_selection_fails_closed_when_budget_gap_contract_is_infeasible() -> None:
    with pytest.raises(ValueError, match="infeasible"):
        global_structured_topk(
            torch.randn(1, 8),
            k=1,
            max_unselected_hole=2,
            training=False,
        )


def test_structured_soft_budget_remains_exact_on_longer_sequences() -> None:
    logits = torch.randn(1, 128, requires_grad=True)

    out = global_structured_topk(
        logits,
        k=64,
        max_unselected_hole=15,
        temperature=0.7,
        training=True,
    )

    assert torch.allclose(out.soft_slot_assignment.sum(dim=2), torch.ones(1, 64), atol=1e-6)
    assert torch.allclose(out.soft_occupancy.sum(dim=1), torch.tensor([64.0]), atol=1e-5)


def test_structured_selection_is_global_not_prefix_invariant() -> None:
    prefix = torch.tensor([[2.0, 1.0, 0.5, 0.2]])
    first = torch.cat((prefix, torch.full((1, 4), -5.0)), dim=1)
    second = torch.cat((prefix, torch.full((1, 4), 5.0)), dim=1)

    first_out = global_structured_topk(first, k=4, max_unselected_hole=4, training=False)
    second_out = global_structured_topk(second, k=4, max_unselected_hole=4, training=False)

    assert not torch.equal(first_out.hard_occupancy[:, :4], second_out.hard_occupancy[:, :4])
    assert first_out.selection_scope == "full_window_non_streaming"
    assert second_out.selection_scope == "full_window_non_streaming"
