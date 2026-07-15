from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll is unstable; Linux runs this gate", allow_module_level=True)

import torch

from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.duca.counterfactual_utility import (
    build_finite_hard_one_swap_candidates,
    build_swap_incidence_matrix,
    counterfactual_pair_scores,
    counterfactual_utility_distillation_loss,
    detached_hard_one_swap_utilities,
    gradient_utility_alignment,
    score_space_utility_alignment,
    signed_one_swap_proximal_loss,
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


def test_noop_anchor_suppresses_all_harmful_swaps() -> None:
    scores = torch.zeros((1, 2), requires_grad=True)
    teacher = torch.tensor([[-2.0, -1.0]])
    valid = torch.ones_like(teacher, dtype=torch.bool)

    loss = counterfactual_utility_distillation_loss(scores, teacher, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert torch.all(scores.grad > 0)


def test_noop_softmax_anchor_is_not_used_as_formal_directional_surrogate() -> None:
    scores = torch.zeros((1, 2), requires_grad=True)
    teacher = torch.tensor([[2.0, 1.0]])
    valid = torch.ones_like(teacher, dtype=torch.bool)

    loss = counterfactual_utility_distillation_loss(scores, teacher, valid)
    loss.backward()

    assert torch.isfinite(loss)
    assert scores.grad is not None
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 1] > 0


def test_noop_anchor_preserves_mixed_signed_utility() -> None:
    scores = torch.zeros((1, 3), requires_grad=True)
    teacher = torch.tensor([[2.0, 0.0, -1.0]])
    valid = torch.ones_like(teacher, dtype=torch.bool)

    loss = counterfactual_utility_distillation_loss(scores, teacher, valid, temperature=0.7)
    loss.backward()

    assert scores.grad is not None
    assert scores.grad[0, 0] < 0
    assert scores.grad[0, 2] > 0


def test_noop_anchor_handles_one_valid_candidate_and_extreme_finite_temperatures() -> None:
    for temperature in (1.0e-2, 100.0):
        scores = torch.zeros((1, 2), requires_grad=True)
        teacher = torch.tensor([[1.0, 0.0]])
        valid = torch.tensor([[True, False]])

        loss = counterfactual_utility_distillation_loss(
            scores,
            teacher,
            valid,
            temperature=temperature,
        )
        loss.backward()

        assert torch.isfinite(loss)
        assert scores.grad is not None
        assert scores.grad[0, 0] < 0
        assert scores.grad[0, 1] == 0


def _proximal_case(utility: torch.Tensor, valid: torch.Tensor):
    scores = torch.zeros((utility.shape[0], 6), requires_grad=True)
    baseline = torch.tensor([[0, 2]]).expand(utility.shape[0], -1).clone()
    additions = torch.tensor([[1, 3]]).expand(utility.shape[0], -1).clone()
    removed_slots = torch.tensor([[0, 0]]).expand(utility.shape[0], -1).clone()
    incidence = build_swap_incidence_matrix(scores, additions, removed_slots, baseline, valid)
    loss = signed_one_swap_proximal_loss(
        scores,
        incidence,
        utility,
        valid,
        temperature=0.7,
    )
    return scores, incidence, loss


@pytest.mark.parametrize(
    "utility",
    (
        torch.tensor([[2.0, 1.0]]),
        torch.tensor([[-2.0, -1.0]]),
        torch.tensor([[1.0, -3.0]]),
    ),
)
def test_signed_proximal_score_space_direction_matches_utility(utility: torch.Tensor) -> None:
    valid = torch.ones_like(utility, dtype=torch.bool)
    scores, incidence, loss = _proximal_case(utility, valid)

    alignment = score_space_utility_alignment(
        scores,
        loss,
        incidence,
        utility,
        valid,
        temperature=0.7,
    )

    assert torch.isfinite(loss)
    assert alignment["sign_agreement"] == pytest.approx(1.0)
    assert alignment["spearman"] == pytest.approx(1.0)
    assert alignment["normalized_direction_max_abs_error"] < 1.0e-6
    assert max(alignment["swap_gram_condition_numbers"]) <= 5.0


def test_signed_proximal_shared_remove_resolves_conflicting_pair_gradients() -> None:
    utility = torch.tensor([[1.0, -3.0]], requires_grad=True)
    valid = torch.ones_like(utility, dtype=torch.bool)
    scores, incidence, loss = _proximal_case(utility, valid)

    score_gradient = torch.autograd.grad(loss, scores, retain_graph=True)[0]
    pair_direction = incidence[0] @ (-score_gradient[0])

    assert pair_direction[0] > 0
    assert pair_direction[1] < 0
    loss.backward()
    assert utility.grad is None


def test_signed_proximal_zero_and_masked_candidates_have_zero_gradient() -> None:
    utility = torch.zeros((1, 2))
    for valid in (
        torch.tensor([[True, False]]),
        torch.tensor([[False, False]]),
    ):
        scores, _, loss = _proximal_case(utility, valid)
        loss.backward()
        assert torch.isfinite(loss)
        assert torch.equal(scores.grad, torch.zeros_like(scores.grad))


def test_signed_proximal_is_invariant_to_candidate_permutation() -> None:
    utility = torch.tensor([[1.0, -3.0]])
    valid = torch.ones_like(utility, dtype=torch.bool)
    scores_a, incidence_a, loss_a = _proximal_case(utility, valid)
    grad_a = torch.autograd.grad(loss_a, scores_a)[0]

    scores_b = torch.zeros_like(scores_a, requires_grad=True)
    incidence_b = incidence_a[:, [1, 0]]
    loss_b = signed_one_swap_proximal_loss(
        scores_b,
        incidence_b,
        utility[:, [1, 0]],
        valid[:, [1, 0]],
        temperature=0.7,
    )
    grad_b = torch.autograd.grad(loss_b, scores_b)[0]

    assert loss_a.item() == pytest.approx(loss_b.item())
    assert torch.allclose(grad_a, grad_b, atol=1.0e-7, rtol=1.0e-7)


def test_signed_proximal_keeps_fp32_loss_for_half_precision_scores() -> None:
    scores = torch.zeros((1, 6), dtype=torch.float16, requires_grad=True)
    baseline = torch.tensor([[0, 2]])
    additions = torch.tensor([[1, 3]])
    removed_slots = torch.tensor([[0, 0]])
    valid = torch.ones((1, 2), dtype=torch.bool)
    utility = torch.tensor([[1.0, -2.0]])
    incidence = build_swap_incidence_matrix(scores, additions, removed_slots, baseline, valid)

    loss = signed_one_swap_proximal_loss(scores, incidence, utility, valid)
    loss.backward()

    assert loss.dtype == torch.float32
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_signed_proximal_disables_outer_cuda_autocast() -> None:
    scores = torch.zeros((1, 6), device="cuda", dtype=torch.float16, requires_grad=True)
    baseline = torch.tensor([[0, 2]], device="cuda")
    additions = torch.tensor([[1, 3]], device="cuda")
    removed_slots = torch.tensor([[0, 0]], device="cuda")
    valid = torch.ones((1, 2), device="cuda", dtype=torch.bool)
    utility = torch.tensor([[1.0, -2.0]], device="cuda")
    incidence = build_swap_incidence_matrix(scores, additions, removed_slots, baseline, valid)

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        loss = signed_one_swap_proximal_loss(scores, incidence, utility, valid)
    loss.backward()

    assert loss.dtype == torch.float32
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


@pytest.mark.parametrize(
    ("additions", "removed_slots", "baseline"),
    (
        (torch.tensor([[-1]]), torch.tensor([[0]]), torch.tensor([[0, 2]])),
        (torch.tensor([[1]]), torch.tensor([[-1]]), torch.tensor([[0, 2]])),
        (torch.tensor([[1]]), torch.tensor([[0]]), torch.tensor([[-1, 2]])),
    ),
)
def test_valid_swap_indices_fail_closed(additions, removed_slots, baseline) -> None:
    scores = torch.zeros((1, 4), requires_grad=True)
    valid = torch.ones((1, 1), dtype=torch.bool)

    with pytest.raises(ValueError):
        counterfactual_pair_scores(scores, additions, removed_slots, baseline, valid)
    with pytest.raises(ValueError):
        build_swap_incidence_matrix(scores, additions, removed_slots, baseline, valid)


def test_score_space_alignment_is_computed_per_sample_for_mixed_candidate_counts() -> None:
    scores = torch.zeros((2, 6), requires_grad=True)
    baseline = torch.tensor([[0, 2], [0, 2]])
    additions = torch.tensor([[1, 3], [1, -1]])
    removed_slots = torch.tensor([[0, 0], [0, -1]])
    valid = torch.tensor([[True, True], [True, False]])
    utility = torch.tensor([[1.0, -3.0], [100.0, 0.0]])
    incidence = build_swap_incidence_matrix(scores, additions, removed_slots, baseline, valid)
    loss = signed_one_swap_proximal_loss(scores, incidence, utility, valid)

    alignment = score_space_utility_alignment(
        scores,
        loss,
        incidence,
        utility,
        valid,
        temperature=1.0,
    )

    assert alignment["sign_agreement"] == pytest.approx(1.0)
    assert alignment["spearman"] == pytest.approx(1.0)
    assert alignment["normalized_direction_max_abs_error"] < 1.0e-6
    assert [item["candidate_count"] for item in alignment["per_sample_alignment"]] == [2, 1]


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
