from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip(
        "local Windows torch/c10.dll import is unstable; Linux remote runs this suite",
        allow_module_level=True,
    )

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"torch is unavailable in this environment: {exc}", allow_module_level=True)

from opentad.models.duca import (  # noqa: E402
    SignedTwoSidedMarginalUtilityHead,
    allocate_equal_budget_marginal_reallocation,
    allocate_video_budgets_exact,
    build_frozen_scout_marginal_features,
    detached_three_budget_prefix_utilities,
    interpolate_acquisition_time_to_detector_grid,
    marginal_budget_accounting,
    nested_h65_budget_prefixes,
    validate_real_heavy_observation_tensor,
)


def test_nested_h65_prefixes_preserve_the_sealed_baseline_and_are_nested() -> None:
    valid_lengths = torch.tensor([67, 300, 401, 600])
    baseline = torch.full((4, 384), -1, dtype=torch.long)
    valid = torch.zeros(4, 600, dtype=torch.bool)
    for row, valid_length in enumerate(valid_lengths.tolist()):
        baseline_count = min(valid_length, 384)
        baseline[row, :baseline_count] = torch.arange(baseline_count)
        valid[row, :valid_length] = True
    priority = -torch.arange(600, dtype=torch.float32)[None, :].expand(4, -1)

    result = nested_h65_budget_prefixes(
        baseline,
        priority,
        valid,
        budgets=(256, 384, 512),
        baseline_budget=384,
    )

    assert torch.equal(result.positions_by_budget[384], baseline)
    assert torch.equal(result.actual_count_by_budget[256], torch.tensor([67, 256, 256, 256]))
    assert torch.equal(result.actual_count_by_budget[384], torch.tensor([67, 300, 384, 384]))
    assert torch.equal(result.actual_count_by_budget[512], torch.tensor([67, 300, 401, 512]))
    assert torch.equal(result.effective_budget_by_requested[256], torch.tensor([384, 256, 256, 256]))
    assert torch.equal(result.effective_budget_by_requested[512], torch.tensor([384, 384, 512, 512]))
    assert torch.equal(result.execution_slots_by_budget[256], torch.tensor([384, 256, 256, 256]))
    assert torch.equal(result.execution_slots_by_budget[384], torch.tensor([384, 384, 384, 384]))
    assert torch.equal(result.execution_slots_by_budget[512], torch.tensor([384, 384, 416, 512]))
    assert torch.equal(result.positions_by_budget[256][0, :67], baseline[0, :67])
    assert torch.equal(result.positions_by_budget[512][1, :300], baseline[1, :300])
    assert torch.equal(result.execution_positions_by_budget[512][2][:401], torch.arange(401))
    assert torch.equal(result.execution_positions_by_budget[512][2][401:], torch.full((15,), -1))


def test_three_budget_prefix_utility_targets_are_signed_and_detached() -> None:
    valid_lengths = torch.tensor([67, 300, 401])
    baseline = torch.full((3, 384), -1, dtype=torch.long)
    valid = torch.zeros(3, 401, dtype=torch.bool)
    for row, valid_length in enumerate(valid_lengths.tolist()):
        baseline[row, : min(valid_length, 384)] = torch.arange(min(valid_length, 384))
        valid[row, :valid_length] = True
    priority = -torch.arange(401, dtype=torch.float32)[None, :].expand(3, -1)
    nested = nested_h65_budget_prefixes(baseline, priority, valid)
    calls = []

    def evaluate(_positions, budget, counts):
        calls.append((budget, counts.clone()))
        value = {256: 5.0, 384: 3.0, 512: 2.0}[budget]
        return torch.full((counts.numel(),), value, requires_grad=True)

    output = detached_three_budget_prefix_utilities(
        nested.positions_by_budget,
        evaluate,
        actual_count_by_budget=nested.actual_count_by_budget,
    )

    assert torch.equal(output["downgrade_penalty"], torch.tensor([0.0, 2.0, 2.0]))
    assert torch.equal(output["upgrade_gain"], torch.tensor([0.0, 0.0, 1.0]))
    assert torch.equal(output["downgrade_target_valid"], torch.tensor([False, True, True]))
    assert torch.equal(output["upgrade_target_valid"], torch.tensor([False, False, True]))
    assert [(budget, counts.numel()) for budget, counts in calls] == [(384, 3), (256, 2), (512, 1)]
    assert output["downgrade_penalty"].requires_grad is False
    assert output["upgrade_gain"].requires_grad is False
    assert output["direct_detector_gradient"] is False


def test_variable_acquisition_time_mapping_is_monotone_and_k384_exact() -> None:
    k384 = torch.arange(384, dtype=torch.long)[None]
    exact = interpolate_acquisition_time_to_detector_grid(
        k384,
        torch.tensor([384]),
        detector_length=384,
    )
    assert torch.equal(exact, k384.float())

    k256 = torch.arange(0, 512, 2, dtype=torch.long)[None]
    expanded = interpolate_acquisition_time_to_detector_grid(
        k256,
        torch.tensor([256]),
        detector_length=384,
    )
    assert expanded.shape == (1, 384)
    assert torch.all(expanded[:, 1:] > expanded[:, :-1])
    assert float(expanded.min().item()) >= 0.0
    assert float(expanded.max().item()) <= 511.0


def test_signed_marginal_head_does_not_clamp_negative_utilities() -> None:
    head = SignedTwoSidedMarginalUtilityHead(input_dim=3, hidden_dim=4)
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
        head.net[-1].bias.copy_(torch.tensor([-2.0, 3.0]))

    output = head(torch.zeros(2, 3))

    assert torch.equal(output["downgrade_penalty"], torch.tensor([-2.0, -2.0]))
    assert torch.equal(output["upgrade_gain"], torch.tensor([3.0, 3.0]))


def test_frozen_scout_features_are_detached_and_ignore_train_only_targets() -> None:
    hidden = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3).requires_grad_(True)
    state = {
        "valid_mask": torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool),
        "coarse_hidden_features": hidden,
        "p_action": torch.tensor([[0.1, 0.4, 0.8, 0.2], [0.6, 0.5, 0.2, 0.0]]),
        "transition_score": torch.tensor([[0.2, 0.9, 0.3, 0.1], [0.1, 0.3, 0.7, 0.0]]),
        "uncertainty": torch.tensor([[0.7, 0.5, 0.2, 0.9], [0.4, 0.5, 0.8, 0.0]]),
        "action_target": torch.ones(2, 4),
    }
    baseline = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1]])

    first = build_frozen_scout_marginal_features(state, baseline)
    state["action_target"] = torch.zeros(2, 4)
    state["gt_segments"] = torch.full((2, 4), 99.0)
    second = build_frozen_scout_marginal_features(state, baseline)

    assert first.shape == (2, 3 + 3 * 3 + 4)
    assert first.requires_grad is False
    assert torch.equal(first, second)


def test_equal_budget_allocator_transfers_cost_only_for_positive_total_utility() -> None:
    result = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.tensor([0.1, 1.0, 1.0, 1.0]),
        upgrade_gain=torch.tensor([0.1, 2.0, 0.5, 0.4]),
        valid_observations=torch.full((4,), 512),
    )

    assert result.feasible is True
    assert torch.equal(result.budget, torch.tensor([256, 512, 384, 384]))
    assert int(result.actual_cost.sum().item()) == 4 * 384
    assert int(result.changed_mask.sum().item()) == 2
    assert float(result.predicted_total_utility.item()) == pytest.approx(1.9)

    no_transfer = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.ones(4),
        upgrade_gain=torch.full((4,), 0.5),
        valid_observations=torch.full((4,), 512),
    )
    assert no_transfer.feasible is True
    assert torch.equal(no_transfer.budget, torch.full((4,), 384))
    assert int(no_transfer.changed_mask.sum().item()) == 0


def test_allocator_accounts_for_real_short_window_cost_and_reports_infeasibility() -> None:
    feasible = allocate_video_budgets_exact(
        relative_utility=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 2.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.4]]),
        valid_observations=torch.tensor([300, 428, 512, 512]),
        budget_levels=(256, 384, 512),
        baseline_budget=384,
        max_changed_fraction=0.5,
    )
    assert feasible.feasible is True
    assert torch.equal(feasible.effective_budget, torch.tensor([256, 512, 384, 384]))
    assert torch.equal(feasible.actual_cost, torch.tensor([256, 428, 384, 384]))
    assert feasible.target_actual_cost == 1452

    baseline = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.zeros(4),
        upgrade_gain=torch.ones(4),
        valid_observations=torch.full((4,), 300),
    )
    assert baseline.feasible is True
    assert torch.equal(baseline.effective_budget, torch.full((4,), 384))
    assert int(baseline.actual_cost.sum().item()) == 1200
    assert "all-K384 fallback" in baseline.reason


def test_budget_grouped_heavy_execution_supports_partial_final_packet() -> None:
    observations = torch.zeros(1, 1, 3, 416, 2, 2)
    acquisition_mask = torch.arange(416)[None, :] < 401
    assert validate_real_heavy_observation_tensor(
        observations,
        actual_observations=401,
        execution_slots=416,
        acquisition_mask=acquisition_mask,
    ) is observations

    historical = torch.zeros(1, 1, 3, 384, 2, 2)
    historical_mask = torch.arange(384)[None, :] < 67
    assert validate_real_heavy_observation_tensor(
        historical,
        actual_observations=67,
        execution_slots=384,
        acquisition_mask=historical_mask,
        baseline_execution=True,
    ) is historical


def test_nominal_k256_rejects_a_tensor_padded_to_512_observations() -> None:
    with pytest.raises(ValueError, match="does not match execution_slots"):
        validate_real_heavy_observation_tensor(
            torch.zeros(1, 1, 3, 512, 2, 2),
            actual_observations=401,
            execution_slots=416,
            acquisition_mask=torch.arange(416)[None, :] < 401,
        )


def test_short_window_accounting_contract_examples() -> None:
    valid = torch.tensor([67, 300, 401, 600])
    lower = marginal_budget_accounting(valid, 256)
    baseline = marginal_budget_accounting(valid, 384)
    upper = marginal_budget_accounting(valid, 512)
    assert torch.equal(lower["actual_cost"], torch.tensor([67, 256, 256, 256]))
    assert torch.equal(baseline["actual_cost"], torch.tensor([67, 300, 384, 384]))
    assert torch.equal(upper["actual_cost"], torch.tensor([67, 300, 401, 512]))
    assert torch.equal(upper["execution_slots"], torch.tensor([384, 384, 416, 512]))
