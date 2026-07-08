from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("local Windows torch/c10.dll import is unstable; Linux remote runs this suite", allow_module_level=True)

try:
    import torch
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.duca import (  # noqa: E402
    DucaAcquisitionAdapter,
    DynamicBudgetDecision,
    PrefixMarginalUtilityBudgetController,
    duca_losses,
)
from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector  # noqa: E402


def test_prefix_marginal_controller_outputs_st_dynamic_budget() -> None:
    torch.manual_seed(11)
    controller = PrefixMarginalUtilityBudgetController(
        hidden_dim=8,
        budget_min=4,
        budget_max=8,
        budget_multiple=2,
        target_budget=6.0,
    )
    features = torch.randn(2, 12, 8, requires_grad=True)
    scores = torch.randn(2, 12)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    decision = controller(features, scores, valid)

    assert isinstance(decision, DynamicBudgetDecision)
    assert decision.policy_name == "prefix_marginal_utility_stop"
    assert decision.budget_hard.shape == (2,)
    assert torch.all(decision.budget_hard >= 4)
    assert torch.all(decision.budget_hard <= 8)
    assert torch.all((decision.budget_hard - 4) % 2 == 0)
    assert torch.all(decision.budget_soft >= 4)
    assert torch.all(decision.budget_soft <= 8)

    decision.budget_soft.sum().backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum().item() > 0.0


def test_dynamic_adapter_predicts_budget_and_forbids_external_override() -> None:
    torch.manual_seed(23)
    dense = torch.randn(2, 12, 3)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        budget_mode="dynamic_must",
        budget_min=4,
        budget_max=8,
        budget_multiple=2,
        target_budget=6.0,
        max_radius=2,
    )

    out = adapter.forward_acquire(dense, valid_mask=valid)
    grid = out["grid"]
    decision = out["budget_decision"]

    assert isinstance(decision, DynamicBudgetDecision)
    assert grid.metadata["budget_is_dynamic"] is True
    assert grid.metadata["budget_policy"] == "prefix_marginal_utility_stop"
    assert grid.budget == 8
    assert torch.equal(grid.requested_budget, decision.budget_hard.to(grid.requested_budget.device))
    assert torch.all(grid.selected_count <= 8)
    assert torch.all(grid.selected_count <= grid.effective_budget)
    assert out["detector_input"].shape[1] == int(grid.selected_positions.shape[1])
    assert out["dynamic_budget"] is True

    with pytest.raises(ValueError, match="external budget override"):
        adapter.acquire(dense, budget=6)


def test_duca_losses_include_must_lagrangian_terms_and_backpropagate_to_budget_policy() -> None:
    torch.manual_seed(31)
    dense = torch.randn(2, 12, 3)
    adapter = DucaAcquisitionAdapter(
        feature_dim=3,
        budget_mode="dynamic_must",
        budget_min=4,
        budget_max=8,
        budget_multiple=2,
        target_budget=6.0,
        max_radius=2,
    )
    output = adapter.forward_acquire(dense)
    losses = duca_losses(
        output,
        detector_loss=torch.tensor(0.5, requires_grad=True),
        loss_weights={
            "budget": 0.0,
            "lagrangian_budget": 1.0,
            "marginal_monotonic": 0.1,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
            "teacher": 0.0,
        },
    )

    assert "lagrangian_budget_loss" in losses
    assert "marginal_monotonic_loss" in losses
    assert "hard_budget_cap_loss" in losses
    assert "dynamic_budget_mean_lossless_metric" in losses
    assert losses["total_loss"].requires_grad

    losses["total_loss"].backward()
    grads = [
        param.grad.detach().abs().sum().item()
        for name, param in adapter.named_parameters()
        if "budget_controller" in name and param.grad is not None
    ]
    assert grads
    assert sum(grads) > 0.0


def test_dynamic_selector_updates_dual_after_optimizer_step_from_pending_budget() -> None:
    torch.manual_seed(37)
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=None,
        budget_mode="dynamic_must",
        budget_min=4,
        budget_max=8,
        budget_multiple=2,
        target_budget=4,
        max_radius=2,
        selector_hidden_channels=8,
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "source_name": "zero_shot_motion_actionness",
            "mode": "motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
            "checkpoint_hash": "no_checkpoint_motion_energy",
        },
        loss_weights={
            "actionness": 0.0,
            "detector": 0.0,
            "lagrangian_budget": 1.0,
            "marginal_monotonic": 0.0,
            "budget": 0.0,
            "teacher": 0.0,
            "boundary": 0.0,
            "hole": 0.0,
            "redundancy": 0.0,
            "radius": 0.0,
            "entropy": 0.0,
        },
    )
    selector.train()
    controller = selector.adapter.budget_controller
    assert controller is not None
    before = controller.lambda_dual.detach().clone()

    selector.forward_train(
        inputs=torch.randn(2, 3, 12),
        masks=torch.ones(2, 12, dtype=torch.bool),
        metas=[{"video_name": "v0"}, {"video_name": "v1"}],
        gt_segments=[torch.empty(0, 2), torch.empty(0, 2)],
        gt_labels=[torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)],
    )
    summary = selector.after_optimizer_step()

    after = controller.lambda_dual.detach()
    assert summary["updated"] is True
    assert summary["source"] == "dynamic_must_expected_cost"
    assert summary["observed_mean_budget"] > 4.0
    assert float(after.item()) > float(before.item())
    assert summary["lambda_after"] == pytest.approx(float(after.item()))
    assert selector.last_dual_update_summary == summary


def test_dynamic_budget_contract_rejects_invalid_multiple() -> None:
    with pytest.raises(ValueError, match="budget_multiple"):
        PrefixMarginalUtilityBudgetController(hidden_dim=8, budget_min=4, budget_max=9, budget_multiple=2)

    with pytest.raises(ValueError, match="budget_min"):
        PrefixMarginalUtilityBudgetController(hidden_dim=8, budget_min=10, budget_max=8, budget_multiple=2)
