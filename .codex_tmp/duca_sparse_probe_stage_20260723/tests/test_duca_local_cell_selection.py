from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch

from opentad.models.duca.structured_selection import (
    exact_uniform_cell_bounds,
    exact_uniform_positions,
    local_cell_deformation,
)


def _observed_max_hole(positions: torch.Tensor, temporal_len: int) -> int:
    sentinels = torch.cat(
        (positions.new_tensor([-1]), positions, positions.new_tensor([temporal_len]))
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _theoretical_local_cell_max_hole(
    starts: torch.Tensor,
    ends: torch.Tensor,
    temporal_len: int,
) -> int:
    candidates = [int(ends[0]) - 1, temporal_len - int(starts[-1]) - 1]
    candidates.extend(
        int(ends[right]) - int(starts[right - 1]) - 2
        for right in range(1, starts.numel())
    )
    return max(candidates)


def test_zero_initialized_768_to_384_is_exact_uniform_one_frame_per_cell() -> None:
    temporal_len, budget = 768, 384
    logits = torch.zeros(2, temporal_len)
    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, budget)

    output = local_cell_deformation(logits, k=budget, training=True)

    assert torch.equal(anchors, exact_uniform_positions(temporal_len, budget))
    assert torch.equal(output.anchor_positions, anchors)
    assert torch.equal(output.cell_starts, starts)
    assert torch.equal(output.cell_ends, ends)
    assert torch.equal(output.selected_positions, anchors.expand(2, -1))
    assert torch.equal(output.hard_occupancy.sum(dim=1), torch.full((2,), float(budget)))
    assert torch.all(output.selected_positions >= starts)
    assert torch.all(output.selected_positions < ends)


def test_hard_and_soft_paths_are_members_of_the_same_local_cell_family() -> None:
    temporal_len, budget = 23, 7
    logits = torch.linspace(-1.2, 1.7, temporal_len).repeat(2, 1)
    logits[1] = logits[1].flip(0)

    output = local_cell_deformation(logits, k=budget, temperature=0.6, training=True)

    assert torch.allclose(output.selection_st.detach(), output.hard_occupancy, atol=1.0e-7, rtol=0.0)
    assert torch.allclose(
        output.soft_slot_assignment.sum(dim=2),
        torch.ones(2, budget),
        atol=1.0e-6,
    )
    assert torch.allclose(output.soft_occupancy.sum(dim=1), torch.full((2,), float(budget)))
    for cell_index, (start, end) in enumerate(zip(output.cell_starts, output.cell_ends)):
        start_index, end_index = int(start), int(end)
        selected = output.selected_positions[:, cell_index]
        assert torch.all((selected >= start_index) & (selected < end_index))
        assert torch.equal(
            output.soft_slot_assignment[:, cell_index, :start_index],
            torch.zeros_like(output.soft_slot_assignment[:, cell_index, :start_index]),
        )
        assert torch.equal(
            output.soft_slot_assignment[:, cell_index, end_index:],
            torch.zeros_like(output.soft_slot_assignment[:, cell_index, end_index:]),
        )


def test_768_to_384_max_hole_bound_is_theoretical_and_attainable() -> None:
    temporal_len, budget = 768, 384
    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, budget)
    theoretical_bound = _theoretical_local_cell_max_hole(starts, ends, temporal_len)
    worst_right = max(
        range(1, budget),
        key=lambda right: int(ends[right]) - int(starts[right - 1]) - 2,
    )
    desired = anchors.clone()
    desired[worst_right - 1] = starts[worst_right - 1]
    desired[worst_right] = ends[worst_right] - 1
    logits = torch.zeros(1, temporal_len)
    logits[0, desired] = 1.0

    output = local_cell_deformation(logits, k=budget, training=False)

    assert theoretical_bound == 3
    assert output.max_unselected_hole == theoretical_bound
    assert torch.equal(output.selected_positions[0], desired)
    assert _observed_max_hole(output.selected_positions[0], temporal_len) == theoretical_bound


def test_nonzero_local_logits_receive_finite_nonzero_gradient() -> None:
    logits = torch.linspace(-0.8, 1.1, 31).reshape(1, -1).requires_grad_()
    output = local_cell_deformation(logits, k=11, temperature=0.7, training=True)
    time_cost = torch.arange(logits.shape[1], dtype=logits.dtype).reshape(1, -1)

    loss = (output.selection_st * time_cost).sum()
    loss.backward()

    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert float(logits.grad.abs().sum().item()) > 0.0
