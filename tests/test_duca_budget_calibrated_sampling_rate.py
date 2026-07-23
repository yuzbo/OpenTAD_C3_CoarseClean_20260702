from __future__ import annotations

import torch

from opentad.models.duca.structured_selection import (
    budget_calibrated_sampling_rate,
    exact_uniform_positions,
)
from opentad.models.duca.transition_only import DucaTransitionUtilityScorer


def test_budget_calibrated_rate_is_capped_exact_and_strictly_ordered():
    logits = torch.tensor(
        [[-2.0, -1.0, 3.0, 2.0, 1.0, -3.0, -4.0, 0.5]],
        requires_grad=True,
    )
    valid = torch.ones_like(logits, dtype=torch.bool)
    output = budget_calibrated_sampling_rate(
        logits,
        valid,
        k=4,
        temperature=0.6,
        coverage_floor=0.0,
        smoothing_kernel=1,
        training=True,
    )

    rates = output.sampling_rates[0]
    assert torch.all(rates >= 0.0)
    assert torch.all(rates <= 1.0)
    assert torch.allclose(rates.sum(), torch.tensor(4.0), atol=2.0e-4)
    assert rates[2] > rates[0]
    positions = output.selected_positions[0]
    assert torch.all(positions[1:] > positions[:-1])
    assert int(output.hard_occupancy.sum().item()) == 4
    assert torch.equal(
        torch.nonzero(output.hard_occupancy[0], as_tuple=False).flatten(),
        positions,
    )
    expected = torch.einsum(
        "kt,t->k",
        output.soft_slot_assignment[0],
        torch.arange(8, dtype=output.soft_slot_assignment.dtype),
    )
    assert torch.allclose(expected.detach(), positions.to(expected.dtype), atol=1.0e-6)

    time = torch.arange(8, dtype=output.selection_st.dtype)
    loss = (output.selection_st[0] * time.square()).sum()
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum().item()) > 0.0


def test_budget_calibrated_rate_backward_is_invariant_to_a_common_logit_shift():
    base = torch.tensor([[0.2, -1.1, 0.7, 2.5, -0.4, 1.3]], requires_grad=True)
    shifted = (base.detach() + 7.0).requires_grad_(True)
    valid = torch.ones_like(base, dtype=torch.bool)
    weights = torch.tensor([[0.3, -0.4, 1.7, -0.8, 0.6, 1.1]])

    first = budget_calibrated_sampling_rate(
        base, valid, k=3, temperature=0.8, coverage_floor=0.0, smoothing_kernel=1
    )
    second = budget_calibrated_sampling_rate(
        shifted, valid, k=3, temperature=0.8, coverage_floor=0.0, smoothing_kernel=1
    )
    assert torch.allclose(first.sampling_rates, second.sampling_rates, atol=1.0e-6)
    (first.sampling_rates * weights).sum().backward()
    (second.sampling_rates * weights).sum().backward()
    assert torch.allclose(base.grad, shifted.grad, atol=1.0e-6)
    assert abs(float(base.grad.sum().item())) < 1.0e-6


def test_budget_calibrated_rate_has_exact_uniform_warmup_without_duplicate_slots():
    logits = torch.randn(2, 12, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    output = budget_calibrated_sampling_rate(
        logits,
        valid,
        k=5,
        force_exact_uniform=True,
        training=True,
    )

    expected = exact_uniform_positions(12, 5)
    assert torch.equal(output.selected_positions[0], expected)
    assert torch.equal(output.selected_positions[1], expected)
    assert torch.allclose(
        output.soft_slot_assignment.sum(dim=-1),
        torch.ones(2, 5),
    )
    assert torch.equal(
        output.soft_slot_assignment.argmax(dim=-1),
        output.selected_positions,
    )


def test_transition_scorer_exposes_two_train_only_contribution_channels():
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=6)
    descriptors = torch.randn(2, 7, scorer.input_dim, requires_grad=True)
    logits = scorer.detector_utility_logits(descriptors)

    assert logits.shape == (2, 7, 2)
    logits.square().mean().backward()
    assert descriptors.grad is not None
    assert float(descriptors.grad.abs().sum().item()) > 0.0
