from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch

from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.duca.counterfactual_utility import (
    build_finite_hard_one_swap_candidates,
    counterfactual_pair_scores,
    counterfactual_utility_distillation_loss,
    detached_hard_one_swap_utilities,
    gradient_utility_alignment,
)
from opentad.models.selectors.duca_online_frame_selector import _add_structured_zero_forward_gradient_path


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_counterfactual_teacher_does_not_poison_outer_autocast_parameter_cache() -> None:
    layer = torch.nn.Linear(4, 4).cuda()
    inputs = torch.randn(2, 4, device="cuda")

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        with torch.no_grad(), ActionFormer._duca_counterfactual_teacher_autocast(inputs):
            layer(inputs)
        main_loss = layer(inputs).square().mean()
    main_loss.backward()

    assert layer.weight.grad is not None
    assert torch.isfinite(layer.weight.grad).all()
    assert float(layer.weight.grad.abs().sum().item()) > 0.0


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


def test_short_window_with_all_valid_frames_selected_has_no_swap_candidate() -> None:
    selected = torch.tensor([[0, 1, 2, -1]])
    scores = torch.tensor([[3.0, 2.0, 1.0, -9.0, -9.0]])
    valid = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)

    result = build_finite_hard_one_swap_candidates(
        selected, scores, valid, max_candidates=3, max_unselected_hole=2
    )

    assert not result["candidate_valid"].any()
    zero = counterfactual_utility_distillation_loss(
        torch.zeros(1, 3, requires_grad=True),
        torch.zeros(1, 3),
        torch.zeros(1, 3, dtype=torch.bool),
    )
    assert zero.dtype == torch.float32
    assert zero.item() == 0.0


def test_mixed_batch_distills_only_samples_with_feasible_swaps() -> None:
    scores = torch.zeros(2, 3, requires_grad=True)
    teacher = torch.tensor([[0.0, 0.0, 0.0], [2.0, -1.0, 0.0]])
    valid = torch.tensor([[0, 0, 0], [1, 1, 0]], dtype=torch.bool)

    loss = counterfactual_utility_distillation_loss(scores, teacher, valid)
    loss.backward()

    assert torch.equal(scores.grad[0], torch.zeros_like(scores.grad[0]))
    assert scores.grad[1, 0] < 0


def test_zero_forward_bridge_fails_nonlinear_hard_swap_alignment_gate() -> None:
    logits = torch.tensor([[[2.0, 0.0, -1.0]]], requires_grad=True)
    # The local surrogate is attracted toward -2 although its hard squared
    # detector loss is worse than the selected value 1.
    dense = torch.tensor([[[1.0, -2.0, 4.0]]])
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


def test_pair_score_is_add_minus_removed_and_updates_both_positions() -> None:
    scores = torch.tensor([[1.0, 0.0, 3.0, 2.0]], requires_grad=True)
    baseline = torch.tensor([[0, 2]])
    additions = torch.tensor([[1, 3]])
    removed_slots = torch.tensor([[0, 1]])
    valid = torch.ones_like(additions, dtype=torch.bool)
    pair = counterfactual_pair_scores(scores, additions, removed_slots, baseline, valid)
    assert torch.equal(pair.detach(), torch.tensor([[-1.0, -1.0]]))
    teacher = torch.tensor([[2.0, -2.0]])
    loss = counterfactual_utility_distillation_loss(pair, teacher, valid)
    loss.backward()
    assert scores.grad[0, 1] < 0
    assert scores.grad[0, 0] > 0
    assert scores.grad[0, 3] > 0
    assert scores.grad[0, 2] < 0


def test_pair_score_ranking_is_independent_of_teacher_construction() -> None:
    scores = torch.tensor([[4.0, 1.0, 0.0, 3.0]])
    baseline = torch.tensor([[0, 2]])
    additions = torch.tensor([[1, 3]])
    removed_slots = torch.tensor([[0, 1]])
    valid = torch.ones_like(additions, dtype=torch.bool)
    pair = counterfactual_pair_scores(scores, additions, removed_slots, baseline, valid)
    detector_gain = torch.tensor([[-3.0, 2.0]])
    assert pair[0, 1] > pair[0, 0]
    assert detector_gain[0, 1] > detector_gain[0, 0]
