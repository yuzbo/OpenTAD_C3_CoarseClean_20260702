import pytest

torch = pytest.importorskip("torch")

from opentad.models.duca.acquisition import duca_losses
from opentad.models.duca.structured_selection import global_structured_topk
from opentad.models.duca.transition_only import local_boundary_mass_coverage_loss


def _scores(policy_logits, valid, budgets, *, max_hole, temperature):
    return {
        "center_scores": policy_logits * 0.0,
        "selected_mask_st": torch.zeros_like(policy_logits),
        "valid_mask": valid,
        "selector_variant": "transition_only",
        "transition_auxiliary_scores": policy_logits,
        "decode_policy_logits": policy_logits,
        "effective_budget": torch.tensor(budgets, device=policy_logits.device),
        "max_unselected_hole": max_hole,
        "structured_temperature": temperature,
        "soft_coverage": torch.sigmoid(policy_logits),
    }


def test_duca_losses_uses_structured_soft_mass_with_padding():
    policy = torch.tensor(
        [[0.2, -0.1, 0.6, 0.4, -0.3], [0.8, -0.2, 0.1, 99.0, 99.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    valid = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.bool)
    target = torch.tensor([[0, 0, 1, 0, 0], [0, 1, 0, 0, 0]], dtype=torch.float64)
    temperature = 0.8
    scores = _scores(policy, valid, [2, 1], max_hole=None, temperature=temperature)

    losses = duca_losses(
        scores,
        budget=torch.tensor([2, 1]),
        transition_target=target,
        transition_boundary_radius=0,
        loss_weights={"transition_boundary": 1.0},
    )
    expected = local_boundary_mass_coverage_loss(
        torch.sigmoid(policy), target, valid, radius=0
    )
    actual = losses["transition_boundary_coverage_loss"]
    assert torch.isfinite(actual)
    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)
    actual.backward()
    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()
    assert policy.grad[0, :5].abs().sum() > 0
    assert policy.grad[1, :3].abs().sum() > 0
    assert torch.equal(policy.grad[1, 3:], torch.zeros_like(policy.grad[1, 3:]))


def test_duca_losses_fails_closed_without_structured_soft_coverage():
    policy = torch.zeros(1, 4)
    valid = torch.ones_like(policy, dtype=torch.bool)
    target = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    scores = _scores(policy, valid, [2], max_hole=2, temperature=0.7)
    del scores["soft_coverage"]
    with pytest.raises(ValueError, match="soft_coverage"):
        duca_losses(scores, budget=2, transition_target=target)


def test_boundary_mass_loss_backpropagates_through_structured_dp_occupancy():
    policy = torch.randn(1, 32, requires_grad=True)
    structured = global_structured_topk(
        policy, k=16, max_unselected_hole=15, temperature=0.7, training=True
    )
    valid = torch.ones_like(policy, dtype=torch.bool)
    target = torch.zeros_like(policy)
    target[0, [8, 24]] = 1.0
    scores = _scores(policy, valid, [16], max_hole=15, temperature=0.7)
    scores["soft_coverage"] = structured.soft_occupancy

    loss = duca_losses(
        scores,
        budget=16,
        transition_target=target,
        transition_boundary_radius=2,
        loss_weights={"transition_boundary": 1.0},
    )["transition_boundary_coverage_loss"]
    loss.backward()

    assert torch.isfinite(loss)
    assert policy.grad is not None and torch.isfinite(policy.grad).all()
    assert policy.grad.abs().sum() > 0
