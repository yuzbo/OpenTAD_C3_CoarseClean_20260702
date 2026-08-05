from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

import torch
import torch.nn as nn

from opentad.models.duca.acquisition import DucaAcquisitionAdapter, ZeroShotActionnessSource
from opentad.models.duca.transition_only import (
    ASFORMER_ENCODER_HIDDEN_KIND,
    DucaTransitionUtilityScorer,
    balanced_binary_actionness_loss,
    boundary_burst_coverage_loss,
    build_boundary_burst_utility,
    build_mandatory_bilateral_set,
    build_transition_descriptors,
    calibrated_actionness_probability,
    continuous_policy_logits,
    coverage_floor_distribution,
    local_boundary_coverage_loss,
    local_boundary_mass_coverage_loss,
    transition_utility_paths,
)
from opentad.models.duca.structured_selection import global_structured_topk
from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector


def _grad_sum(tensor: torch.Tensor) -> float:
    if tensor.grad is None:
        return 0.0
    return float(tensor.grad.detach().abs().sum().item())


def test_transition_descriptors_are_temporal_changes_not_absolute_state() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0, 0.5]], dtype=torch.float32)
    hidden = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]],
        dtype=torch.float32,
    )
    valid = torch.tensor([[True, True, True, False]])

    descriptors = build_transition_descriptors(logits, hidden, valid)

    assert descriptors.shape == (1, 4, 2 * hidden.shape[-1] + 5)
    assert torch.equal(descriptors[:, 0], torch.zeros_like(descriptors[:, 0]))
    assert torch.equal(descriptors[:, 3], torch.zeros_like(descriptors[:, 3]))
    assert torch.allclose(descriptors[0, 1, 0], logits[0, 1] - logits[0, 0])
    assert torch.allclose(descriptors[0, 1, 4:6], hidden[0, 1] - hidden[0, 0])
    assert torch.allclose(descriptors[0, 1, 6:8], (hidden[0, 1] - hidden[0, 0]).abs())


def test_balanced_actionness_loss_defaults_to_unweighted_posterior_bce() -> None:
    logits = torch.zeros(1, 10, requires_grad=True)
    target = torch.tensor([[1.0] + [0.0] * 9])
    valid = torch.ones(1, 10, dtype=torch.bool)

    loss, positive_weight = balanced_binary_actionness_loss(logits, target, valid)

    assert positive_weight.item() == pytest.approx(1.0)
    assert loss.item() > 0.0
    loss.backward()
    assert _grad_sum(logits) > 0.0


def test_actionness_weight_uses_fixed_prior_not_batch_prevalence() -> None:
    logits = torch.zeros(2, 10)
    valid = torch.ones_like(logits, dtype=torch.bool)
    sparse = torch.tensor([[1.0] + [0.0] * 9, [1.0] + [0.0] * 9])
    dense = torch.tensor([[1.0] * 5 + [0.0] * 5, [1.0] * 5 + [0.0] * 5])

    _, sparse_weight = balanced_binary_actionness_loss(logits, sparse, valid, positive_prior=0.2)
    _, dense_weight = balanced_binary_actionness_loss(logits, dense, valid, positive_prior=0.2)

    assert sparse_weight.item() == pytest.approx(4.0)
    assert dense_weight.item() == pytest.approx(4.0)


def test_class_balanced_actionness_is_mean_per_class_not_element_prevalence() -> None:
    logits = torch.tensor([[0.0, 2.0, 2.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    valid = torch.ones_like(target, dtype=torch.bool)

    loss, _ = balanced_binary_actionness_loss(
        logits,
        target,
        valid,
        reduction_mode="class_balanced_mean",
    )
    positive = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1], target[:, :1]
    )
    negative = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, 1:], target[:, 1:]
    )

    assert torch.allclose(loss, 0.5 * (positive + negative))
    loss.backward()
    assert _grad_sum(logits) > 0.0


def test_actionness_probability_supports_frozen_temperature_bias_calibration() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    probability = calibrated_actionness_probability(logits, temperature=2.0, bias=1.0)

    assert torch.allclose(probability, torch.sigmoid((logits + 1.0) / 2.0))
    assert bool(((probability >= 0.0) & (probability <= 1.0)).all())
    probability.sum().backward()
    assert _grad_sum(logits) > 0.0


def test_coverage_floor_keeps_amp_inputs_on_an_fp32_solver_boundary() -> None:
    scores = torch.tensor(
        [[-7.0, -0.5, 0.25, 3.0]],
        dtype=torch.float16,
        requires_grad=True,
    )
    valid = torch.tensor([[True, True, True, False]])

    probabilities, log_probabilities = coverage_floor_distribution(
        scores,
        valid,
        floor_weight=0.1,
        score_temperature=0.7,
    )

    assert probabilities.dtype == torch.float32
    assert log_probabilities.dtype == torch.float32
    assert torch.isfinite(log_probabilities[valid]).all()
    assert torch.isneginf(log_probabilities[~valid]).all()
    log_probabilities[valid].sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()

    oracle_probabilities, oracle_log_probabilities = coverage_floor_distribution(
        scores.detach().double(),
        valid,
        floor_weight=0.1,
        score_temperature=0.7,
    )
    assert oracle_probabilities.dtype == torch.float64
    assert oracle_log_probabilities.dtype == torch.float64


def test_transition_entropy_uses_the_frozen_actionness_calibration() -> None:
    logits = torch.tensor([[0.0, 1.0, -1.0]])
    hidden = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    valid = torch.ones_like(logits, dtype=torch.bool)

    uncalibrated = build_transition_descriptors(logits, hidden, valid)
    calibrated = build_transition_descriptors(
        logits,
        hidden,
        valid,
        calibration_temperature=2.0,
        calibration_bias=1.0,
    )

    assert not torch.equal(uncalibrated[:, 1:, 2:4], calibrated[:, 1:, 2:4])


def test_transition_target_uses_fixed_truncated_equal_mass_endpoint_gaussians() -> None:
    target = DucaOnlineFrameSelector._transition_target_from_gt_segments(
        [torch.tensor([[6.0, 12.0], [20.0, 27.0]])],
        torch.ones(1, 32, dtype=torch.bool),
        sigma=2.0,
        truncate_radius=4,
    )

    assert target is not None
    assert target.sum().item() == pytest.approx(1.0, abs=1e-6)
    assert target[0, 0].item() == pytest.approx(0.0)
    assert target[0, 1].item() == pytest.approx(0.0)
    assert target[0, 6].item() > target[0, 2].item()
    assert target[0, 12].item() > target[0, 16].item()


def test_shared_transition_scorer_has_protected_auxiliary_and_policy_routes() -> None:
    torch.manual_seed(3)
    logits = torch.randn(1, 7, requires_grad=True)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    valid = torch.ones(1, 7, dtype=torch.bool)
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=8)

    paths = transition_utility_paths(scorer, logits, hidden, valid)

    assert paths["hidden_kind"] == ASFORMER_ENCODER_HIDDEN_KIND
    assert torch.equal(paths["auxiliary_scores"], paths["policy_scores"])

    paths["policy_scores"].square().mean().backward(retain_graph=True)
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert sum(float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None) > 0.0

    scorer.zero_grad(set_to_none=True)
    hidden.grad = None
    logits.grad = None
    paths["auxiliary_scores"].square().mean().backward()
    assert _grad_sum(hidden) > 0.0
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert sum(float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None) > 0.0


def test_p0_auxiliary_transition_supervision_does_not_rewrite_coarse_hidden() -> None:
    torch.manual_seed(13)
    logits = torch.randn(1, 7, requires_grad=True)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    valid = torch.ones(1, 7, dtype=torch.bool)
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=8)

    paths = transition_utility_paths(
        scorer,
        logits,
        hidden,
        valid,
        auxiliary_hidden_gradient_scale=0.0,
    )
    paths["auxiliary_scores"].square().mean().backward()

    assert torch.equal(
        paths["auxiliary_descriptors"], paths["transition_descriptors"].detach()
    )
    assert paths["auxiliary_hidden_gradient_scale"] == pytest.approx(0.0)
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert sum(float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None) > 0.0


def test_p0_adaptive_transition_supervision_updates_hidden_but_not_action_logits() -> None:
    torch.manual_seed(17)
    logits = torch.randn(1, 7, requires_grad=True)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    valid = torch.ones(1, 7, dtype=torch.bool)
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=8)

    paths = transition_utility_paths(
        scorer,
        logits,
        hidden,
        valid,
        auxiliary_hidden_gradient_scale=0.25,
    )
    paths["auxiliary_scores"].square().mean().backward()

    assert paths["auxiliary_hidden_gradient_scale"] == pytest.approx(0.25)
    assert _grad_sum(hidden) > 0.0
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert sum(float(p.grad.abs().sum()) for p in scorer.parameters() if p.grad is not None) > 0.0


def test_p0_adaptive_transition_supervision_uses_only_restricted_hidden_route() -> None:
    torch.manual_seed(19)
    logits = torch.randn(1, 7, requires_grad=True)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    restricted_hidden = hidden.detach().clone().requires_grad_(True)
    valid = torch.ones(1, 7, dtype=torch.bool)
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=8)

    paths = transition_utility_paths(
        scorer,
        logits,
        hidden,
        valid,
        policy_hidden=restricted_hidden,
        policy_hidden_gradient_scale=0.05,
        auxiliary_hidden_gradient_scale=0.25,
    )
    paths["auxiliary_scores"].square().mean().backward()

    assert paths["auxiliary_hidden_uses_restricted_policy_route"] is True
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(restricted_hidden) > 0.0
    assert _grad_sum(logits) == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_auxiliary_hidden_gradient_scale_fails_closed(scale: float) -> None:
    scorer = DucaTransitionUtilityScorer(hidden_dim=2, scorer_hidden_dim=4)
    with pytest.raises(ValueError, match="auxiliary_hidden_gradient_scale"):
        transition_utility_paths(
            scorer,
            torch.zeros(1, 3),
            torch.zeros(1, 3, 2),
            torch.ones(1, 3, dtype=torch.bool),
            auxiliary_hidden_gradient_scale=scale,
        )


def test_policy_hidden_gradient_scale_only_opens_the_declared_hidden_route() -> None:
    torch.manual_seed(31)
    logits = torch.randn(1, 7, requires_grad=True)
    hidden = torch.randn(1, 7, 4, requires_grad=True)
    restricted_policy_hidden = hidden.detach().clone().requires_grad_(True)
    valid = torch.ones(1, 7, dtype=torch.bool)
    scorer = DucaTransitionUtilityScorer(hidden_dim=4, scorer_hidden_dim=8)

    protected = transition_utility_paths(
        scorer,
        logits,
        hidden,
        valid,
        policy_hidden_gradient_scale=0.0,
    )
    collaborative = transition_utility_paths(
        scorer,
        logits,
        hidden,
        valid,
        policy_hidden=restricted_policy_hidden,
        policy_hidden_gradient_scale=0.05,
    )
    assert torch.equal(protected["policy_scores"], collaborative["policy_scores"])

    collaborative["policy_scores"].square().mean().backward()
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(restricted_policy_hidden) > 0.0
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert collaborative["policy_hidden_gradient_scale"] == pytest.approx(0.05)


def test_positive_policy_hidden_scale_requires_a_restricted_route() -> None:
    scorer = DucaTransitionUtilityScorer(hidden_dim=2, scorer_hidden_dim=4)
    with pytest.raises(ValueError, match="restricted policy_hidden"):
        transition_utility_paths(
            scorer,
            torch.zeros(1, 3),
            torch.zeros(1, 3, 2),
            torch.ones(1, 3, dtype=torch.bool),
            policy_hidden_gradient_scale=0.05,
        )


@pytest.mark.parametrize("scale", [-0.1, 1.1, float("nan")])
def test_policy_hidden_gradient_scale_fails_closed(scale: float) -> None:
    scorer = DucaTransitionUtilityScorer(hidden_dim=2, scorer_hidden_dim=4)
    with pytest.raises(ValueError, match="policy_hidden_gradient_scale"):
        transition_utility_paths(
            scorer,
            torch.zeros(1, 3),
            torch.zeros(1, 3, 2),
            torch.ones(1, 3, dtype=torch.bool),
            policy_hidden_gradient_scale=scale,
        )


def test_continuous_policy_homotopy_has_exact_endpoints_and_smooth_midpoint() -> None:
    learned = torch.tensor([[2.0, -1.0, 0.5, 3.0]], dtype=torch.float32)
    valid = torch.ones_like(learned, dtype=torch.bool)

    reference = continuous_policy_logits(learned, valid, k=2, alpha=0.0)
    midpoint = continuous_policy_logits(learned, valid, k=2, alpha=0.5)
    final = continuous_policy_logits(learned, valid, k=2, alpha=1.0)

    assert not torch.equal(reference, final)
    assert torch.allclose(midpoint, 0.5 * (reference + final), atol=1e-6)
    assert torch.isfinite(reference).all()
    assert torch.isfinite(final).all()


@pytest.mark.parametrize(("temporal_len", "budget"), [(768, 384), (401, 384), (17, 8)])
def test_uniform_homotopy_endpoint_matches_exact_round_linspace(temporal_len: int, budget: int) -> None:
    learned = torch.randn(1, temporal_len, dtype=torch.float32)
    valid = torch.ones_like(learned, dtype=torch.bool)

    reference = continuous_policy_logits(learned, valid, k=budget, alpha=0.0)
    selection = global_structured_topk(
        reference,
        k=budget,
        max_unselected_hole=15,
        training=False,
    )
    expected = torch.linspace(0, temporal_len - 1, steps=budget).round().long()

    assert torch.equal(selection.selected_positions[0].cpu(), expected)


def test_transition_schedule_separates_policy_and_detector_bridge_ramps() -> None:
    selector = DucaOnlineFrameSelector.__new__(DucaOnlineFrameSelector)
    nn.Module.__init__(selector)
    selector.selector_variant = "transition_only"
    selector.acquisition_policy = "global_structured_topk"
    selector.loss_weights = {"actionness": 1.0}
    selector.loss_weight_schedule = selector._normalize_loss_weight_schedule(
        {
            "type": "progressive_joint",
            "shape": "cosine",
            "warmup_steps": 0,
            "transition_steps": 1,
            "policy_alpha": {
                "start": 0.0,
                "end": 1.0,
                "warmup_steps": 660,
                "transition_steps": 3960,
            },
            "detector_gradient": {
                "start": 0.0,
                "end": 0.25,
                "warmup_steps": 4620,
                "transition_steps": 3300,
            },
        }
    )
    selector.register_buffer("_loss_weight_schedule_step", torch.zeros((), dtype=torch.long))
    selector.train()

    def state(step: int):
        selector._loss_weight_schedule_step.fill_(step)
        return selector._loss_schedule_state()

    assert state(0)["weights"]["policy_alpha"] == pytest.approx(0.0)
    assert state(660)["weights"]["policy_alpha"] == pytest.approx(0.0)
    assert 0.0 < state(2640)["weights"]["policy_alpha"] < 1.0
    assert state(4620)["weights"]["policy_alpha"] == pytest.approx(1.0)
    assert state(4620)["detector_gradient_weight"] == pytest.approx(0.0)
    assert 0.0 < state(6270)["detector_gradient_weight"] < 0.25
    assert state(7920)["detector_gradient_weight"] == pytest.approx(0.25)
    assert selector._use_stable_structured_selection(state(2640)) is False


def test_local_boundary_coverage_rewards_mass_inside_boundary_neighborhood() -> None:
    boundary = torch.zeros(1, 9)
    boundary[0, 4] = 1.0
    valid = torch.ones(1, 9, dtype=torch.bool)
    near = torch.full((1, 9), -2.0)
    near[0, 3:6] = torch.tensor([1.0, 3.0, 1.0])
    far = torch.full((1, 9), -2.0)
    far[0, :3] = torch.tensor([1.0, 3.0, 1.0])

    near_loss = local_boundary_coverage_loss(
        near, boundary, valid, radius=1, k=3, max_unselected_hole=8
    )
    far_loss = local_boundary_coverage_loss(
        far, boundary, valid, radius=1, k=3, max_unselected_hole=8
    )

    assert near_loss < far_loss
    assert torch.isfinite(near_loss)
    assert torch.isfinite(far_loss)


def test_local_boundary_mass_coverage_has_bounded_finite_long_window_gradients() -> None:
    logits = torch.randn(1, 768, requires_grad=True)
    occupancy = torch.softmax(logits, dim=1) * 384.0
    boundary = torch.zeros(1, 768)
    boundary[0, [100, 390, 700]] = 1.0
    valid = torch.ones(1, 768, dtype=torch.bool)

    loss = local_boundary_mass_coverage_loss(occupancy, boundary, valid, radius=4)
    loss.backward()

    assert torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().max() < 1.0


def test_boundary_burst_utility_forms_a_symmetric_five_frame_microcluster() -> None:
    center = torch.full((1, 11), -20.0)
    center[0, 5] = 20.0
    offsets = torch.zeros(1, 11, 5)
    valid = torch.ones(1, 11, dtype=torch.bool)

    output = build_boundary_burst_utility(
        center,
        offsets,
        valid,
        k=6,
        radius=2,
        quota=5.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
    )

    burst = output["burst_utility"][0]
    assert torch.allclose(burst[3:8], burst[3].expand(5), atol=1e-6)
    assert float(burst[3]) > float(burst[2])
    assert float(burst.max()) < 1.0
    offset_probabilities = output["offset_probabilities"][0, 5]
    assert torch.allclose(
        offset_probabilities.sum(),
        offset_probabilities.new_tensor(1.0),
    )
    assert torch.allclose(offset_probabilities, offset_probabilities.flip(0))
    assert float(offset_probabilities[2]) >= float(offset_probabilities[1])
    assert float(offset_probabilities[1]) >= float(offset_probabilities[0])
    assert torch.allclose(
        output["offset_inclusion"][0, 5],
        output["offset_inclusion"].new_ones(5),
        atol=1.0e-6,
        rtol=0.0,
    )


def test_boundary_burst_quota_limits_each_predicted_center_support() -> None:
    center = torch.full((1, 11), -20.0)
    center[0, 5] = 20.0
    offsets = torch.zeros(1, 11, 5, requires_grad=True)
    output = build_boundary_burst_utility(
        center,
        offsets,
        torch.ones(1, 11, dtype=torch.bool),
        k=6,
        radius=2,
        quota=3.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
    )

    inclusion = output["offset_inclusion"][0, 5]
    assert torch.allclose(
        inclusion,
        inclusion.new_tensor([0.0, 1.0, 1.0, 1.0, 0.0]),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert inclusion.sum().item() == 3.0
    output["burst_utility"].sum().backward()
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()


def test_boundary_burst_hard_support_is_bilateral_when_offset_logits_are_one_sided() -> None:
    center = torch.full((1, 11), -20.0)
    center[0, 5] = 20.0
    offsets = torch.full((1, 11, 5), -10.0, requires_grad=True)
    with torch.no_grad():
        offsets[0, 5] = torch.tensor([-9.0, -8.0, 0.0, 9.0, 10.0])

    output = build_boundary_burst_utility(
        center,
        offsets,
        torch.ones(1, 11, dtype=torch.bool),
        k=6,
        radius=2,
        quota=3.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
        require_bilateral_offsets=True,
    )

    inclusion = output["offset_inclusion"][0, 5]
    assert torch.equal(inclusion.detach().bool(), torch.tensor([False, True, True, False, True]))
    assert bool(output["bilateral_offset_feasible"][0, 5])
    assert bool(output["bilateral_offset_satisfied"][0, 5])
    output["burst_utility"].sum().backward()
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()
    assert float(offsets.grad.abs().sum()) > 0.0


def test_boundary_burst_hard_support_falls_back_at_a_temporal_edge() -> None:
    center = torch.full((1, 7), -20.0)
    center[0, 0] = 20.0
    offsets = torch.zeros(1, 7, 5)
    output = build_boundary_burst_utility(
        center,
        offsets,
        torch.ones(1, 7, dtype=torch.bool),
        k=4,
        radius=2,
        quota=3.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
        require_bilateral_offsets=True,
    )

    assert not bool(output["bilateral_offset_feasible"][0, 0])
    assert output["offset_inclusion"][0, 0].sum().item() == 3.0


def test_mandatory_bilateral_set_survives_exact_k_max_hole_decode() -> None:
    temporal_len = 24
    center = torch.full((1, temporal_len), -20.0)
    center[0, [6, 17]] = torch.tensor([20.0, 19.0])
    valid = torch.ones_like(center, dtype=torch.bool)
    burst = build_boundary_burst_utility(
        center,
        torch.zeros(1, temporal_len, 5),
        valid,
        k=12,
        radius=2,
        quota=3.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
        require_bilateral_offsets=True,
    )
    mandatory = build_mandatory_bilateral_set(
        center,
        burst["offset_inclusion"],
        valid,
        radius=2,
        quota=3,
        max_mandatory=6,
    )
    decoded = global_structured_topk(
        burst["policy_utility"],
        k=12,
        max_unselected_hole=2,
        required_mask=mandatory["mandatory_mask"],
        training=True,
    )

    assert decoded.hard_occupancy.sum().item() == 12
    assert torch.all(
        decoded.hard_occupancy.bool() | ~mandatory["mandatory_mask"]
    )
    assert mandatory["retained_group_count"].item() == 2
    assert mandatory["mandatory_count"].item() == 6
    assert torch.all(decoded.soft_occupancy[mandatory["mandatory_mask"]] > 0.999)


def test_required_structured_mask_fails_when_exact_k_cannot_contain_it() -> None:
    logits = torch.zeros(1, 8)
    required = torch.zeros(1, 8, dtype=torch.bool)
    required[0, :5] = True

    with pytest.raises(ValueError, match="exceed"):
        global_structured_topk(
            logits,
            k=4,
            max_unselected_hole=4,
            required_mask=required,
        )


def test_required_structured_mask_fails_when_mandatory_geometry_breaks_max_hole() -> None:
    logits = torch.zeros(1, 8)
    required = torch.zeros(1, 8, dtype=torch.bool)
    required[0, [0, 7]] = True

    with pytest.raises(RuntimeError, match="terminal state"):
        global_structured_topk(
            logits,
            k=3,
            max_unselected_hole=2,
            required_mask=required,
            training=True,
        )


def test_boundary_burst_saturating_union_stays_bounded_for_overlapping_centers() -> None:
    center = torch.full((1, 12), -20.0)
    center[0, 5:7] = 20.0
    output = build_boundary_burst_utility(
        center,
        torch.zeros(1, 12, 5),
        torch.ones(1, 12, dtype=torch.bool),
        k=6,
        radius=2,
        quota=5.0,
        boundary_budget_fraction=0.5,
        context_weight=0.0,
    )

    assert bool((output["burst_utility"] >= 0.0).all())
    assert bool((output["burst_utility"] < 1.0).all())
    assert output["burst_mass"][0, 5] > output["burst_mass"][0, 3]


def test_boundary_burst_loss_prefers_complete_bilateral_quota() -> None:
    target = torch.zeros(1, 9)
    target[0, 4] = 1.0
    valid = torch.ones(1, 9, dtype=torch.bool)
    complete = torch.zeros(1, 9)
    complete[0, 2:7] = 1.0
    one_sided = torch.zeros(1, 9)
    one_sided[0, 4:7] = 1.0
    missing = torch.zeros(1, 9)

    complete_loss, _ = boundary_burst_coverage_loss(
        complete, target, valid, radius=2, quota=5.0
    )
    one_sided_loss, _ = boundary_burst_coverage_loss(
        one_sided, target, valid, radius=2, quota=5.0
    )
    missing_loss, _ = boundary_burst_coverage_loss(
        missing, target, valid, radius=2, quota=5.0
    )

    assert complete_loss < one_sided_loss < missing_loss


def test_boundary_burst_fairness_penalizes_an_unserved_endpoint() -> None:
    target = torch.zeros(1, 17)
    target[0, [4, 12]] = 1.0
    valid = torch.ones(1, 17, dtype=torch.bool)
    balanced = torch.zeros(1, 17)
    balanced[0, 2:7] = 1.0
    balanced[0, 10:15] = 1.0
    collapsed = torch.zeros(1, 17)
    collapsed[0, 2:7] = 1.0

    balanced_loss, balanced_parts = boundary_burst_coverage_loss(
        balanced, target, valid, radius=2, quota=5.0
    )
    collapsed_loss, collapsed_parts = boundary_burst_coverage_loss(
        collapsed, target, valid, radius=2, quota=5.0
    )

    assert balanced_loss < collapsed_loss
    assert balanced_parts["fairness"] < collapsed_parts["fairness"]


def test_boundary_burst_event_target_filters_cropped_boundaries_and_deduplicates() -> None:
    target = DucaOnlineFrameSelector._transition_event_target_from_gt_segments(
        [torch.tensor([[2.2, 7.2], [2.4, 7.1]])],
        torch.ones(1, 10, dtype=torch.bool),
        boundary_validity=[torch.tensor([[True, False], [True, True]])],
    )

    assert target is not None
    assert torch.equal(torch.nonzero(target[0]).flatten(), torch.tensor([2, 7]))
    assert target.sum().item() == pytest.approx(2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_local_boundary_mass_coverage_stays_fp32_under_autocast_and_scaled_backward() -> None:
    # GradScaler scales FP32 model parameters; using an FP16 leaf here makes any
    # nonzero gradient near scale 65536 overflow while being cast into the leaf.
    logits = torch.randn(1, 768, device="cuda", dtype=torch.float32, requires_grad=True)
    occupancy = torch.softmax(logits, dim=1) * 384.0
    boundary = torch.zeros(1, 768, device="cuda")
    boundary[0, [100, 390, 700]] = 1.0
    valid = torch.ones(1, 768, device="cuda", dtype=torch.bool)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = local_boundary_mass_coverage_loss(occupancy, boundary, valid, radius=4)
    assert loss.dtype == torch.float32
    (loss * 65536.0).backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_local_boundary_coverage_is_exact_under_structured_dependence() -> None:
    boundary = torch.zeros(1, 3)
    boundary[0, 1] = 1.0
    valid = torch.ones_like(boundary, dtype=torch.bool)
    logits = torch.zeros(1, 3, requires_grad=True)

    loss = local_boundary_coverage_loss(
        logits, boundary, valid, radius=1, k=1, max_unselected_hole=2
    )

    assert loss.item() == pytest.approx(0.0, abs=2e-7)
    assert loss.item() >= 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_local_boundary_coverage_is_finite_for_padded_windows_under_fp16_autocast() -> None:
    soft = torch.zeros(1, 453, device="cuda", dtype=torch.float16, requires_grad=True)
    target = torch.zeros(1, 768, device="cuda", dtype=torch.float32)
    target[:, 390] = 1.0
    target = target[:, :453]
    valid = torch.ones(1, 453, device="cuda", dtype=torch.bool)

    with torch.cuda.amp.autocast(dtype=torch.float16):
        loss = local_boundary_coverage_loss(
            soft, target, valid, radius=4, k=226, max_unselected_hole=15, temperature=0.7
        )

    assert torch.isfinite(loss)


def _transition_adapter() -> DucaAcquisitionAdapter:
    return DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        actionness_source=ZeroShotActionnessSource(mode="motion"),
        budget=4,
        budget_mode="fixed",
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        selector_variant="transition_only",
        coarse_hidden_dim=4,
        require_coarse_hidden_features=True,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )


def _boundary_burst_adapter() -> DucaAcquisitionAdapter:
    return DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        actionness_source=ZeroShotActionnessSource(mode="motion"),
        budget=6,
        budget_mode="fixed",
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        selector_variant="transition_only",
        transition_objective="boundary_burst",
        boundary_burst_radius=2,
        boundary_burst_quota=5.0,
        boundary_burst_budget_fraction=0.5,
        boundary_burst_context_weight=0.05,
        coarse_hidden_dim=4,
        require_coarse_hidden_features=True,
        max_unselected_hole=2,
        hard_max_gap_repair=False,
    )


def test_transition_adapter_removes_all_legacy_direct_heads() -> None:
    adapter = _transition_adapter()

    assert adapter.encoder is None
    assert adapter.center_head is None
    assert adapter.start_head is None
    assert adapter.end_head is None
    assert adapter.context_head is None
    assert adapter.utility_head is None
    assert adapter.radius_head is None
    assert adapter.transition_scorer is not None


def test_transition_adapter_exact_k_max_gap_and_protected_policy_gradients() -> None:
    torch.manual_seed(9)
    adapter = _transition_adapter()
    adapter.train()
    dense_placeholder = torch.zeros(1, 9, 3)
    logits = torch.randn(1, 9, requires_grad=True)
    hidden = torch.randn(1, 9, 4, requires_grad=True)
    valid = torch.ones(1, 9, dtype=torch.bool)

    grid, scores = adapter.acquire(
        dense_placeholder,
        valid_mask=valid,
        actionness_logits=logits,
        coarse_hidden_features=hidden,
        coarse_hidden_kind=ASFORMER_ENCODER_HIDDEN_KIND,
        policy_mix_alpha=0.5,
    )
    positions = grid.selected_positions[0]
    holes = torch.tensor(
        [
            int(positions[0]),
            *[int(right - left - 1) for left, right in zip(positions[:-1], positions[1:])],
            9 - int(positions[-1]) - 1,
        ]
    )

    assert positions.numel() == 4
    assert int(holes.max()) <= 2
    assert scores["selection_path"] == "transition_continuous_homotopy"
    assert scores["legacy_direct_heads_enabled"] is False
    scores["soft_coverage"].square().mean().backward()
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(logits) == pytest.approx(0.0)
    assert sum(
        float(param.grad.abs().sum())
        for param in adapter.transition_scorer.parameters()
        if param.grad is not None
    ) > 0.0


def test_boundary_burst_adapter_keeps_exact_k_gap_and_trains_offset_head() -> None:
    torch.manual_seed(23)
    adapter = _boundary_burst_adapter()
    adapter.train()
    hidden = torch.randn(1, 12, 4, requires_grad=True)
    logits = torch.randn(1, 12, requires_grad=True)
    grid, scores = adapter.acquire(
        torch.zeros(1, 12, 3),
        valid_mask=torch.ones(1, 12, dtype=torch.bool),
        actionness_logits=logits,
        coarse_hidden_features=hidden,
        coarse_hidden_kind=ASFORMER_ENCODER_HIDDEN_KIND,
        policy_mix_alpha=1.0,
    )

    positions = grid.selected_positions[0]
    holes = torch.tensor(
        [
            int(positions[0]),
            *[int(right - left - 1) for left, right in zip(positions[:-1], positions[1:])],
            12 - int(positions[-1]) - 1,
        ]
    )
    assert positions.numel() == 6
    assert int(holes.max()) <= 2
    assert scores["transition_objective"] == "boundary_burst"
    assert scores["burst_offset_logits"].shape == (1, 12, 5)
    assert scores["transition_center_scores"].shape == (1, 12)
    assert scores["center_scores"].shape == (1, 12)

    scores["soft_coverage"].square().mean().backward()
    assert _grad_sum(hidden) == pytest.approx(0.0)
    assert _grad_sum(logits) == pytest.approx(0.0)
    offset_head = adapter.transition_scorer.burst_offset_head
    assert offset_head is not None
    assert offset_head.weight.grad is not None
    assert float(offset_head.weight.grad.abs().sum()) > 0.0


def test_transition_adapter_fails_closed_on_spatial_stem_hidden() -> None:
    adapter = _transition_adapter()
    with pytest.raises(ValueError, match="official_asformer_encoder_hidden"):
        adapter.forward_scores(
            torch.zeros(1, 9, 3),
            valid_mask=torch.ones(1, 9, dtype=torch.bool),
            actionness_logits=torch.zeros(1, 9),
            coarse_hidden_features=torch.zeros(1, 9, 4),
            coarse_hidden_kind="pre_temporal_spatial_stem_hidden",
        )
