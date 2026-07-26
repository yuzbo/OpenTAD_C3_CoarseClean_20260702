from __future__ import annotations

import torch

from opentad.models.duca import (
    DucaAcquisitionAdapter,
    ZeroShotActionnessSource,
    temporal_max_gap_hole_loss,
)
from opentad.models.duca.structured_selection import (
    continuous_density_transport,
    exact_uniform_positions,
)
from opentad.models.duca.transition_only import ASFORMER_ENCODER_HIDDEN_KIND
from opentad.models.selectors.duca_online_frame_selector import (
    _add_density_transport_gradient_path,
    _gather_time,
)


def _max_hole(positions: torch.Tensor, temporal_len: int) -> int:
    sentinels = torch.cat(
        (positions.new_tensor([-1]), positions, positions.new_tensor([temporal_len]))
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _grad_sum(module: torch.nn.Module) -> float:
    return sum(
        float(parameter.grad.abs().sum())
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_uniform_density_is_exact_uniform_without_a_hard_gap_contract() -> None:
    logits = torch.zeros(2, 32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    output = continuous_density_transport(logits, valid, k=8, training=True)
    expected = exact_uniform_positions(32, 8)
    assert torch.equal(output.selected_positions[0], expected)
    assert torch.equal(output.selected_positions[1], expected)
    assert torch.equal(output.hard_occupancy.sum(dim=1), torch.tensor([8.0, 8.0]))
    assert torch.allclose(output.soft_slot_assignment.sum(dim=-1), torch.ones(2, 8))
    assert output.max_unselected_hole is None
    assert torch.equal(
        output.observed_max_unselected_hole,
        torch.tensor([_max_hole(expected, 32), _max_hole(expected, 32)]),
    )


def test_unconstrained_density_forms_boundary_clusters_and_remains_exact_k() -> None:
    logits = torch.full((2, 64), -6.0)
    logits[0, 28:35] = 10.0
    logits[1, 13:18] = 10.0
    logits[1, 46:51] = 10.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    output = continuous_density_transport(
        logits,
        valid,
        k=16,
        max_unselected_hole=None,
        coverage_floor=0.05,
        smoothing_kernel=5,
    )
    uniform = exact_uniform_positions(64, 16)
    assert int(((output.selected_positions[0] - 31).abs() <= 4).sum()) > int(
        ((uniform - 31).abs() <= 4).sum()
    )
    assert int(((output.selected_positions[1] - 15).abs() <= 4).sum()) >= 2
    assert int(((output.selected_positions[1] - 48).abs() <= 4).sum()) >= 2
    for row in output.selected_positions:
        assert torch.all(row[1:] > row[:-1])
        assert int(row.unique().numel()) == 16


def test_no_soft_and_hard_max_gap_paths_are_behaviorally_distinct() -> None:
    logits = torch.full((1, 64), -12.0, requires_grad=True)
    logits.data[:, 29:36] = 12.0
    valid = torch.ones_like(logits, dtype=torch.bool)
    no_max = continuous_density_transport(
        logits,
        valid,
        k=8,
        max_unselected_hole=None,
        coverage_floor=0.0,
        smoothing_kernel=1,
        training=True,
    )
    hard_max = continuous_density_transport(
        logits,
        valid,
        k=8,
        max_unselected_hole=7,
        coverage_floor=0.0,
        smoothing_kernel=1,
        training=True,
    )
    assert _max_hole(no_max.selected_positions[0], 64) > 7
    assert _max_hole(hard_max.selected_positions[0], 64) <= 7
    soft_loss = temporal_max_gap_hole_loss(
        no_max.soft_occupancy,
        valid,
        max_unselected_hole=7,
        min_window_mass=1.0,
    )
    assert float(soft_loss.detach()) > 0.0
    soft_loss.backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_density_bridge_is_hard_forward_and_soft_backward() -> None:
    logits = torch.randn(1, 24, requires_grad=True)
    valid = torch.ones_like(logits, dtype=torch.bool)
    output = continuous_density_transport(
        logits,
        valid,
        k=8,
        max_unselected_hole=None,
        training=True,
    )
    dense = torch.linspace(-1.0, 2.0, 24).reshape(1, 1, 24)
    hard = _gather_time(dense, output.selected_positions, output.slot_mask)
    bridged = _add_density_transport_gradient_path(
        hard,
        dense,
        soft_slot_assignment=output.soft_slot_assignment,
        slot_mask=output.slot_mask,
        bridge_weight=1.0,
    )
    assert torch.equal(bridged.detach(), hard.detach())
    weights = torch.linspace(0.5, 1.5, 8).reshape(1, 1, 8)
    (bridged * weights).sum().backward()
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def _build_adapter(policy: str) -> DucaAcquisitionAdapter:
    return DucaAcquisitionAdapter(
        feature_dim=3,
        hidden_dim=8,
        actionness_source=ZeroShotActionnessSource(mode="motion"),
        budget=8,
        budget_mode="fixed",
        acquisition_policy=policy,
        density_temperature=0.7,
        density_coverage_floor=0.05,
        density_smoothing_kernel=5,
        selector_variant="transition_only",
        transition_objective="gaussian_mass",
        coarse_hidden_dim=4,
        require_coarse_hidden_features=True,
        policy_hidden_gradient_scale=0.0,
        max_unselected_hole=None,
        hard_max_gap_repair=False,
    )


def test_detector_style_gradient_reaches_base_density_scorer_only() -> None:
    torch.manual_seed(11)
    adapter = _build_adapter("continuous_density_transport")
    adapter.train()
    dense = torch.randn(1, 24, 3)
    action_logits = torch.randn(1, 24, requires_grad=True)
    coarse_hidden = torch.randn(1, 24, 4, requires_grad=True)
    grid, scores = adapter.acquire(
        dense,
        valid_mask=torch.ones(1, 24, dtype=torch.bool),
        actionness_logits=action_logits,
        coarse_hidden_features=coarse_hidden,
        coarse_hidden_kind=ASFORMER_ENCODER_HIDDEN_KIND,
        policy_mix_alpha=1.0,
    )
    soft_selected = torch.einsum(
        "bkt,btc->bkc", scores["structured_soft_slot_assignment"], dense
    )
    soft_selected.square().mean().backward()
    assert grid.selected_positions.shape == (1, 8)
    assert adapter.transition_scorer is not None
    assert _grad_sum(adapter.transition_scorer) > 0.0
    assert action_logits.grad is None or float(action_logits.grad.abs().sum()) == 0.0
    assert coarse_hidden.grad is None or float(coarse_hidden.grad.abs().sum()) == 0.0


def test_mixture_density_components_and_protected_detector_gradient() -> None:
    torch.manual_seed(17)
    adapter = _build_adapter("continuous_mixture_density_transport")
    adapter.train()
    dense = torch.randn(1, 24, 3)
    action_logits = torch.randn(1, 24, requires_grad=True)
    coarse_hidden = torch.randn(1, 24, 4, requires_grad=True)
    grid, scores = adapter.acquire(
        dense,
        valid_mask=torch.ones(1, 24, dtype=torch.bool),
        actionness_logits=action_logits,
        coarse_hidden_features=coarse_hidden,
        coarse_hidden_kind=ASFORMER_ENCODER_HIDDEN_KIND,
        policy_mix_alpha=1.0,
    )
    component_probabilities = scores["density_component_probabilities"]
    mixture_weights = scores["density_mixture_weights"]
    assert component_probabilities.shape == (1, 3, 24)
    assert torch.allclose(
        component_probabilities.sum(dim=-1), torch.ones(1, 3), atol=1.0e-6
    )
    assert torch.allclose(mixture_weights.sum(dim=-1), torch.ones(1), atol=1.0e-6)
    assert int(mixture_weights.argmax(dim=-1).item()) == 0
    assert tuple(scores["density_component_names"]) == (
        "boundary",
        "uncertainty",
        "context",
    )

    hard_selected = _gather_time(
        dense.transpose(1, 2),
        grid.selected_positions,
        scores["density_slot_mask"],
    )
    bridged = _add_density_transport_gradient_path(
        hard_selected,
        dense.transpose(1, 2),
        soft_slot_assignment=scores["structured_soft_slot_assignment"],
        slot_mask=scores["density_slot_mask"],
        bridge_weight=1.0,
    )
    weights = torch.linspace(0.2, 1.4, 8).reshape(1, 1, 8)
    (bridged * weights).sum().backward()
    assert adapter.transition_scorer is not None
    assert adapter.density_mixture_head is not None
    assert _grad_sum(adapter.transition_scorer) > 0.0
    assert _grad_sum(adapter.density_mixture_head) > 0.0
    assert action_logits.grad is None or float(action_logits.grad.abs().sum()) == 0.0
    assert coarse_hidden.grad is None or float(coarse_hidden.grad.abs().sum()) == 0.0
