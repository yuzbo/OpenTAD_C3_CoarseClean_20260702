from __future__ import annotations

import pytest
import torch

from opentad.models.backbones.georoute_routing import (
    GEOROUTE_ROUTING_SCHEMA,
    decode_continuous_geometry,
    native_patch_centers,
    ordered_plackett_luce_log_prob,
    score_function_policy_loss,
    select_exact_k,
)


def _logits(*, batch: int = 2, tubelets: int = 3, patches: int = 20):
    torch.manual_seed(11)
    roi = torch.randn(batch, tubelets, patches, requires_grad=True)
    residual = torch.randn(batch, tubelets, patches, requires_grad=True)
    return roi, residual


def test_continuous_geometry_is_in_bounds_and_differentiable():
    logits = torch.tensor(
        [[[0.4, -0.5, 1.2, -0.8], [-0.3, 0.9, -1.0, 0.6]]],
        requires_grad=True,
    )

    geometry = decode_continuous_geometry(logits, min_extent=0.15, max_extent=0.8)

    assert geometry.shape == (1, 2, 4)
    assert torch.all(geometry[..., 2:] >= 0.15)
    assert torch.all(geometry[..., 2:] <= 0.8)
    assert torch.all(geometry[..., :2] - geometry[..., 2:] / 2 >= 0.0)
    assert torch.all(geometry[..., :2] + geometry[..., 2:] / 2 <= 1.0)
    geometry.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_native_patch_centers_are_row_major_and_normalized():
    centers = native_patch_centers(2, 3, device=torch.device("cpu"), dtype=torch.float32)

    assert centers.shape == (6, 2)
    assert torch.equal(centers[0], torch.tensor([1.0 / 6.0, 0.25]))
    assert torch.equal(centers[-1], torch.tensor([5.0 / 6.0, 0.75]))
    assert torch.all((centers > 0.0) & (centers < 1.0))


@pytest.mark.parametrize(
    ("mode", "estimator", "context_tokens"),
    [
        ("dense", "none", 0),
        ("uniform", "none", 2),
        ("random", "none", 2),
        ("roi", "straight_through", 2),
        ("free", "straight_through", 2),
        ("hybrid", "straight_through", 2),
    ],
)
def test_exact_k_routes_have_no_duplicates_and_a_complete_schema(mode, estimator, context_tokens):
    roi, residual = _logits()
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode=mode,
        tokens_per_tubelet=6,
        context_tokens=context_tokens,
        roi_fraction=0.5,
        training=True,
        estimator=estimator,
        temperature=0.7,
    )

    expected_k = roi.shape[-1] if mode == "dense" else 6
    assert route["schema_version"] == GEOROUTE_ROUTING_SCHEMA
    assert route["indices"].shape == (2, 3, expected_k)
    assert torch.equal(route["indices"], torch.sort(route["indices"], dim=-1).values)
    assert torch.all(route["selected_mask"].sum(dim=-1) == expected_k)
    assert torch.all(route["indices"] >= 0)
    assert torch.all(route["indices"] < roi.shape[-1])
    assert torch.allclose(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    assert sum(route["role_counts"].values()) == expected_k


def test_st_gate_is_hard_in_forward_but_has_a_biased_surrogate_gradient():
    roi, residual = _logits(batch=1, tubelets=2, patches=12)
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="hybrid",
        tokens_per_tubelet=6,
        context_tokens=1,
        roi_fraction=0.5,
        training=True,
        estimator="straight_through",
        temperature=0.6,
    )

    assert torch.equal(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    route["st_gate"].sum().backward()
    assert roi.grad is not None and residual.grad is not None
    assert torch.isfinite(roi.grad).all() and torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(roi.grad) > 0
    assert torch.count_nonzero(residual.grad) > 0


def test_score_function_route_has_plackett_luce_gradient_and_loss():
    roi, residual = _logits(batch=1, tubelets=2, patches=9)
    torch.manual_seed(9)
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="roi",
        tokens_per_tubelet=4,
        context_tokens=0,
        roi_fraction=1.0,
        training=True,
        estimator="score_function",
        temperature=0.8,
    )
    assert route["ordered_log_prob"] is not None
    recomputed = ordered_plackett_luce_log_prob(
        roi,
        route["ordered_indices"],
        temperature=0.8,
    )
    assert torch.allclose(route["ordered_log_prob"], recomputed)

    policy_loss = score_function_policy_loss(
        detector_cost=torch.tensor(2.0, requires_grad=True),
        ordered_log_prob=route["ordered_log_prob"],
        baseline=torch.tensor(0.5),
        weight=0.75,
    )
    policy_loss.backward()
    assert roi.grad is not None
    assert torch.isfinite(roi.grad).all()
    assert torch.count_nonzero(roi.grad) > 0
    assert residual.grad is None


def test_score_function_and_training_contracts_fail_closed_when_semantics_are_invalid():
    roi, residual = _logits(batch=1, tubelets=1, patches=8)
    kwargs = dict(
        roi_logits=roi,
        residual_logits=residual,
        tokens_per_tubelet=4,
        context_tokens=1,
        roi_fraction=0.5,
        training=True,
        temperature=0.5,
    )
    with pytest.raises(ValueError, match="single-family"):
        select_exact_k(mode="hybrid", estimator="score_function", **kwargs)
    with pytest.raises(ValueError, match="explicit gradient estimator"):
        select_exact_k(mode="roi", estimator="none", **kwargs)
    with pytest.raises(ValueError, match="duplicate"):
        ordered_plackett_luce_log_prob(
            roi,
            torch.tensor([[[1, 1]]]),
            temperature=1.0,
        )
