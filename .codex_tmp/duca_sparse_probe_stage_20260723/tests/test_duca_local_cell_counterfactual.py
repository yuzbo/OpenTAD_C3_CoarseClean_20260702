from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch

from opentad.models.duca.counterfactual_utility import (
    build_local_cell_hard_flip_candidates,
    local_cell_signed_logistic_loss,
)
from opentad.models.duca.structured_selection import exact_uniform_cell_bounds


def test_hard_flip_candidates_use_distinct_cells_and_replace_within_the_same_cell() -> None:
    temporal_len, budget = 12, 4
    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, budget)
    selected = anchors.reshape(1, -1)
    cell_starts = starts.reshape(1, -1)
    cell_ends = ends.reshape(1, -1)
    scores = torch.arange(temporal_len, dtype=torch.float32).reshape(1, -1)
    valid_mask = torch.ones_like(scores, dtype=torch.bool)

    result = build_local_cell_hard_flip_candidates(
        selected,
        scores,
        valid_mask,
        cell_starts,
        cell_ends,
        max_candidates=budget,
    )

    active = result["candidate_valid"][0]
    candidate_cells = result["candidate_cell_indices"][0, active]
    replaced_slots = result["replaced_slots"][0, active]
    assert active.sum().item() == budget
    assert torch.equal(candidate_cells, replaced_slots)
    assert candidate_cells.unique().numel() == candidate_cells.numel()
    for candidate_index in torch.nonzero(active, as_tuple=False).flatten().tolist():
        proposal = result["candidate_selections"][0, candidate_index]
        replaced_slot = int(result["replaced_slots"][0, candidate_index])
        added_position = int(result["candidate_positions"][0, candidate_index])
        changed_slots = torch.nonzero(proposal != selected[0], as_tuple=False).flatten()

        assert changed_slots.tolist() == [replaced_slot]
        assert int(proposal[replaced_slot]) == added_position
        assert int(starts[replaced_slot]) <= int(selected[0, replaced_slot]) < int(ends[replaced_slot])
        assert int(starts[replaced_slot]) <= added_position < int(ends[replaced_slot])
        assert added_position != int(selected[0, replaced_slot])


def test_signed_local_utility_sets_positive_and_negative_pair_gradient_directions() -> None:
    scores = torch.zeros((1, 6), requires_grad=True)
    baseline = torch.tensor([[0, 3]])
    candidate_positions = torch.tensor([[1, 4]])
    replaced_slots = torch.tensor([[0, 1]])
    utility = torch.tensor([[2.0, -3.0]], requires_grad=True)
    valid = torch.ones_like(utility, dtype=torch.bool)

    loss = local_cell_signed_logistic_loss(
        scores,
        candidate_positions,
        replaced_slots,
        baseline,
        utility,
        valid,
        temperature=0.7,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None and torch.isfinite(scores.grad).all()
    assert scores.grad[0, 1] < 0 and scores.grad[0, 0] > 0
    assert scores.grad[0, 4] > 0 and scores.grad[0, 3] < 0
    assert utility.grad is None


@pytest.mark.parametrize(
    ("utility", "valid"),
    (
        (torch.tensor([[0.0]]), torch.tensor([[True]])),
        (torch.tensor([[7.0]]), torch.tensor([[False]])),
    ),
    ids=("zero-utility", "no-candidate"),
)
def test_zero_utility_or_no_candidate_has_finite_zero_loss_and_gradient(
    utility: torch.Tensor,
    valid: torch.Tensor,
) -> None:
    scores = torch.tensor([[0.2, -0.4, 0.7]], requires_grad=True)
    loss = local_cell_signed_logistic_loss(
        scores,
        torch.tensor([[1]]),
        torch.tensor([[0]]),
        torch.tensor([[0]]),
        utility,
        valid,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss.item() == 0.0
    assert scores.grad is not None
    assert torch.equal(scores.grad, torch.zeros_like(scores.grad))


def test_repeated_local_cell_candidates_fail_closed() -> None:
    with pytest.raises(ValueError, match="distinct cells"):
        local_cell_signed_logistic_loss(
            torch.zeros((1, 5), requires_grad=True),
            torch.tensor([[1, 2]]),
            torch.tensor([[0, 0]]),
            torch.tensor([[0, 3]]),
            torch.tensor([[1.0, -1.0]]),
            torch.tensor([[True, True]]),
        )
