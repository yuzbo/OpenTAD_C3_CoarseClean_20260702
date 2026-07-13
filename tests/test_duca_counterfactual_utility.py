from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch

from opentad.models.duca.counterfactual_utility import (
    build_finite_hard_one_swap_candidates,
    counterfactual_utility_distillation_loss,
    detached_hard_one_swap_utilities,
    gradient_utility_alignment,
)
from opentad.models.selectors.duca_online_frame_selector import _add_structured_zero_forward_gradient_path


def test_finite_candidates_are_exact_k_unique_bounded_and_max_gap_feasible() -> None:
    selected = torch.tensor([[0, 2, 4, 6]])
    scores = torch.tensor([[0.0, 9.0, 0.0, 8.0, 0.0, 7.0, 0.0, 6.0]])
    valid = torch.ones_like(scores, dtype=torch.bool)
    result = build_finite_hard_one_swap_candidates(
        selected, scores, valid, max_candidates=3, max_unselected_hole=2
    )
    assert int(result["candidate_valid"].sum()) <= 3
    for row in result["candidate_selections"][0, result["candidate_valid"][0]]:
        assert row.numel() == 4
        assert torch.unique(row).numel() == 4
        sentinels = torch.cat((row.new_tensor([-1]), row, row.new_tensor([8])))
        assert int((sentinels[1:] - sentinels[:-1] - 1).max()) <= 2


def test_zero_forward_bridge_fails_nonlinear_hard_swap_alignment_gate() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
    dense = torch.tensor([[[1.0, 2.0, 4.0]]])
    hard = dense[:, :, :1]
    assignment = torch.softmax(logits, dim=-1)
    bridged = _add_structured_zero_forward_gradient_path(
        hard, dense, soft_slot_assignment=assignment,
        slot_mask=torch.ones(1, 1, dtype=torch.bool), bridge_weight=1.0,
    )
    surrogate_loss = bridged.square().mean()
    hard_losses = dense.square().reshape(1, 3)
    hard_utility = hard_losses[:, :1] - hard_losses
    metrics = gradient_utility_alignment(logits, surrogate_loss, hard_utility, torch.ones_like(hard_utility, dtype=torch.bool))
    assert metrics["sign_agreement"] < 1.0 or metrics["spearman"] < 0.9


def test_hard_one_swap_teacher_uses_actual_discrete_detector_losses() -> None:
    selected = torch.tensor([[0, 2]])
    candidates = torch.tensor([[1, 3]])

    def detector_loss(positions: torch.Tensor) -> torch.Tensor:
        target = torch.tensor([0.0, 3.0])
        return (positions.float() - target).square().sum(dim=1)

    result = detached_hard_one_swap_utilities(selected, candidates, detector_loss)
    assert result["direct_detector_gradient"] is False
    assert result["candidate_utility"].requires_grad is False
    assert result["candidate_utility"][0, 3 - 2] > result["candidate_utility"][0, 0]


def test_counterfactual_distillation_improves_teacher_ranking_and_detaches_teacher() -> None:
    scores = torch.zeros(1, 4, requires_grad=True)
    teacher = torch.tensor([[0.0, 1.0, -1.0, 3.0]], requires_grad=True)
    valid = torch.ones(1, 4, dtype=torch.bool)
    loss = counterfactual_utility_distillation_loss(scores, teacher, valid, temperature=0.7)
    loss.backward()
    assert scores.grad is not None
    assert scores.grad[0, 3] < 0
    assert scores.grad[0, 2] > 0
    assert teacher.grad is None
