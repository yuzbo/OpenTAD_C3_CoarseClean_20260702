from __future__ import annotations

import itertools

import torch

from opentad.models.backbones.georoute_routing import (
    GEOROUTE_STRUCTURED_ROUTING_SCHEMA,
    ordered_plackett_luce_log_prob,
    select_fixed_quota_structured_exact_k,
)


def _route(
    roi: torch.Tensor,
    residual: torch.Tensor,
    *,
    context: int,
    roi_count: int,
    residual_count: int,
    update: int = 4,
    training: bool = True,
    estimator: str = "score_function",
):
    return select_fixed_quota_structured_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="structured_hybrid",
        context_tokens=context,
        roi_tokens=roi_count,
        residual_tokens=residual_count,
        training=training,
        estimator=estimator,
        temperature=0.7,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
        study_seed=5227,
        successful_update_index=update if training else None,
        distributed_rank=0,
    )


def _mask(indices: torch.Tensor, count: int) -> torch.Tensor:
    return torch.zeros((*indices.shape[:2], count), dtype=torch.bool).scatter(
        -1,
        indices,
        True,
    )


def test_structured_pl_exact_k_exclusion_and_joint_log_probability():
    roi = torch.linspace(-1.2, 1.3, 16).reshape(1, 2, 8).requires_grad_()
    residual = torch.linspace(1.1, -1.4, 16).reshape(1, 2, 8).requires_grad_()
    route = _route(roi, residual, context=2, roi_count=2, residual_count=2)

    assert route["schema_version"] == GEOROUTE_STRUCTURED_ROUTING_SCHEMA
    assert route["indices"].shape == (1, 2, 6)
    assert route["role_counts"]["context"] == 2
    assert route["role_counts"]["roi"] == 2
    assert route["role_counts"]["residual"] == 2
    assert torch.all(route["selected_mask"].sum(dim=-1) == 6)
    assert route["role_ids"].shape == route["indices"].shape
    assert route["role_id_values"] == {"context": 0, "roi": 1, "residual": 2}
    for role, expected_count in (("context", 2), ("roi", 2), ("residual", 2)):
        role_id = route["role_id_values"][role]
        assert torch.all((route["role_ids"] == role_id).sum(dim=-1) == expected_count)

    context = route["role_indices"]["context"]
    roi_indices = route["role_indices"]["roi"]
    residual_indices = route["role_indices"]["residual"]
    context_mask = _mask(context, 8)
    roi_mask = _mask(roi_indices, 8)
    residual_mask = _mask(residual_indices, 8)
    assert not bool((context_mask & roi_mask).any())
    assert not bool((context_mask & residual_mask).any())
    assert not bool((roi_mask & residual_mask).any())

    valid = torch.ones_like(context_mask)
    manual_roi = ordered_plackett_luce_log_prob(
        roi,
        roi_indices,
        temperature=0.7,
        valid_mask=valid & ~context_mask,
    )
    manual_residual = ordered_plackett_luce_log_prob(
        residual,
        residual_indices,
        temperature=0.7,
        valid_mask=valid & ~context_mask & ~roi_mask,
    )
    assert torch.allclose(route["branch_log_probabilities"]["roi"], manual_roi)
    assert torch.allclose(
        route["branch_log_probabilities"]["residual"],
        manual_residual,
    )
    assert torch.allclose(
        route["ordered_log_prob"],
        manual_roi + manual_residual,
    )


def test_structured_joint_likelihood_enumerates_to_one():
    roi = torch.tensor([[[0.3, -0.2, 0.8, 0.1, -0.5]]], dtype=torch.float64)
    residual = torch.tensor([[[-0.4, 0.6, 0.2, -0.1, 0.9]]], dtype=torch.float64)
    valid = torch.ones_like(roi, dtype=torch.bool)
    # One deterministic context token from N=5 is index 2.
    context_mask = torch.zeros_like(valid).scatter(-1, torch.tensor([[[2]]]), True)
    total = roi.new_zeros(())
    for roi_index in (0, 1, 3, 4):
        roi_order = torch.tensor([[[roi_index]]])
        roi_logp = ordered_plackett_luce_log_prob(
            roi,
            roi_order,
            temperature=0.7,
            valid_mask=valid & ~context_mask,
        )
        roi_mask = torch.zeros_like(valid).scatter(-1, roi_order, True)
        for residual_index in (0, 1, 3, 4):
            if residual_index == roi_index:
                continue
            residual_order = torch.tensor([[[residual_index]]])
            residual_logp = ordered_plackett_luce_log_prob(
                residual,
                residual_order,
                temperature=0.7,
                valid_mask=valid & ~context_mask & ~roi_mask,
            )
            total = total + torch.exp(roi_logp + residual_logp).squeeze()
    assert torch.allclose(total, torch.ones_like(total), atol=1e-12, rtol=1e-12)


def test_structured_score_identity_matches_exact_joint_risk_gradient():
    roi = torch.tensor([[[0.4, -0.1, 0.2, 0.7]]], dtype=torch.float64, requires_grad=True)
    residual = torch.tensor([[[-0.3, 0.5, 0.1, -0.2]]], dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(roi, dtype=torch.bool)
    context_mask = torch.zeros_like(valid).scatter(-1, torch.tensor([[[2]]]), True)
    terms = []
    for roi_index, residual_index in itertools.permutations((0, 1, 3), 2):
        roi_order = torch.tensor([[[roi_index]]])
        roi_mask = torch.zeros_like(valid).scatter(-1, roi_order, True)
        residual_order = torch.tensor([[[residual_index]]])
        logp = ordered_plackett_luce_log_prob(
            roi,
            roi_order,
            temperature=0.7,
            valid_mask=valid & ~context_mask,
        ) + ordered_plackett_luce_log_prob(
            residual,
            residual_order,
            temperature=0.7,
            valid_mask=valid & ~context_mask & ~roi_mask,
        )
        risk = roi.new_tensor(0.2 + 0.13 * roi_index + 0.07 * residual_index)
        terms.append((logp.squeeze(), risk))

    exact_expected_risk = sum(torch.exp(logp) * risk for logp, risk in terms)
    exact_gradients = torch.autograd.grad(
        exact_expected_risk,
        (roi, residual),
        retain_graph=True,
    )
    score_objective = sum(
        torch.exp(logp).detach() * risk * logp for logp, risk in terms
    )
    score_gradients = torch.autograd.grad(score_objective, (roi, residual))
    assert torch.allclose(score_gradients[0], exact_gradients[0], atol=1e-12, rtol=1e-12)
    assert torch.allclose(score_gradients[1], exact_gradients[1], atol=1e-12, rtol=1e-12)


def test_structured_selected_and_unselected_logits_receive_expected_gradients():
    torch.manual_seed(9)
    roi = torch.randn(1, 2, 10, requires_grad=True)
    residual = torch.randn(1, 2, 10, requires_grad=True)
    route = _route(roi, residual, context=2, roi_count=3, residual_count=2)
    route["ordered_log_prob"].sum().backward()

    context_mask = _mask(route["role_indices"]["context"], 10)
    roi_mask = _mask(route["role_indices"]["roi"], 10)
    residual_mask = _mask(route["role_indices"]["residual"], 10)
    assert torch.count_nonzero(roi.grad.masked_select(context_mask)) == 0
    assert torch.count_nonzero(roi.grad.masked_select(~context_mask & roi_mask)) > 0
    assert torch.count_nonzero(roi.grad.masked_select(~context_mask & ~roi_mask)) > 0
    residual_excluded = context_mask | roi_mask
    assert torch.count_nonzero(residual.grad.masked_select(residual_excluded)) == 0
    assert torch.count_nonzero(residual.grad.masked_select(residual_mask)) > 0
    assert torch.count_nonzero(
        residual.grad.masked_select(~residual_excluded & ~residual_mask)
    ) > 0


def test_structured_inference_is_deterministic_and_emits_no_policy_likelihood():
    torch.manual_seed(3)
    roi = torch.randn(1, 3, 12)
    residual = torch.randn(1, 3, 12)
    first = _route(
        roi,
        residual,
        context=2,
        roi_count=3,
        residual_count=3,
        training=False,
    )
    second = _route(
        roi,
        residual,
        context=2,
        roi_count=3,
        residual_count=3,
        training=False,
    )
    assert torch.equal(first["indices"], second["indices"])
    assert first["ordered_log_prob"] is None
    assert first["route_rng"]["enabled"] is False
