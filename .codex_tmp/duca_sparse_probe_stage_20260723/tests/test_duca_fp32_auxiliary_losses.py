from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from opentad.models.duca.acquisition import _target_distribution_loss, temporal_max_gap_hole_loss
from opentad.models.duca.counterfactual_utility import counterfactual_utility_distillation_loss
from opentad.models.duca.transition_only import (
    balanced_binary_actionness_loss,
    transition_distribution_loss,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA autocast regression")
def test_all_active_duca_auxiliary_losses_are_fp32_and_scale_safely() -> None:
    device = "cuda"
    valid = torch.ones(1, 32, device=device, dtype=torch.bool)
    target = torch.zeros(1, 32, device=device)
    target[0, [8, 24]] = 1.0

    def half_logits() -> torch.Tensor:
        return torch.randn(1, 32, device=device, dtype=torch.float16, requires_grad=True)

    cases = []
    policy = half_logits()
    cases.append((policy, counterfactual_utility_distillation_loss(policy, target, valid)))
    action = half_logits()
    cases.append((action, balanced_binary_actionness_loss(action, target, valid)[0]))
    transition = half_logits()
    cases.append((transition, transition_distribution_loss(transition, target, valid)))
    direct = half_logits()
    cases.append((direct, _target_distribution_loss(direct, target, valid)))
    gap_logits = half_logits()
    gap = torch.sigmoid(gap_logits)
    cases.append((gap_logits, temporal_max_gap_hole_loss(gap, valid, max_unselected_hole=15)))

    for leaf, loss in cases:
        assert loss.dtype == torch.float32
        loss.mul(65536.0).backward()
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
