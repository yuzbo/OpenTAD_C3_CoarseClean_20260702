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
    build_transition_descriptors,
    calibrated_actionness_probability,
    continuous_policy_logits,
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


def test_actionness_probability_supports_frozen_temperature_bias_calibration() -> None:
    logits = torch.tensor([-2.0, 0.0, 2.0], requires_grad=True)
    probability = calibrated_actionness_probability(logits, temperature=2.0, bias=1.0)

    assert torch.allclose(probability, torch.sigmoid((logits + 1.0) / 2.0))
    assert bool(((probability >= 0.0) & (probability <= 1.0)).all())
    probability.sum().backward()
    assert _grad_sum(logits) > 0.0


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_local_boundary_mass_coverage_stays_fp32_under_autocast_and_scaled_backward() -> None:
    logits = torch.randn(1, 768, device="cuda", dtype=torch.float16, requires_grad=True)
    occupancy = torch.softmax(logits.float(), dim=1).to(dtype=logits.dtype) * 384.0
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
