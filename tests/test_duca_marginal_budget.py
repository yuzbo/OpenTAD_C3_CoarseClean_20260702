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
    nested_h65_budget_prefixes,
    validate_real_heavy_observation_tensor,
)


def test_nested_h65_prefixes_preserve_the_sealed_baseline_and_are_nested() -> None:
    baseline = torch.tensor(
        [
            [0, 2, 4, 6],
            [0, 1, 2, -1],
        ],
        dtype=torch.long,
    )
    priority = torch.tensor(
        [
            [0.1, 0.9, 0.4, 0.8, 0.3, 0.7, 0.2, 0.6],
            [0.3, 0.8, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )

    result = nested_h65_budget_prefixes(
        baseline,
        priority,
        valid,
        budgets=(2, 4, 6),
        baseline_budget=4,
    )

    assert torch.equal(result.positions_by_budget[4], baseline)
    assert torch.equal(result.positions_by_budget[2][0], torch.tensor([2, 4]))
    assert torch.equal(
        result.positions_by_budget[6][0],
        torch.tensor([0, 1, 2, 3, 4, 6]),
    )
    assert torch.equal(result.actual_count_by_budget[2], torch.tensor([2, 2]))
    assert torch.equal(result.actual_count_by_budget[4], torch.tensor([4, 3]))
    assert torch.equal(result.actual_count_by_budget[6], torch.tensor([6, 3]))
    assert torch.equal(result.positions_by_budget[6][1], torch.tensor([0, 1, 2, -1, -1, -1]))


def test_three_budget_prefix_utility_targets_are_signed_and_detached() -> None:
    selections = {
        2: torch.tensor([[0, 2], [1, 3]]),
        4: torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]]),
        6: torch.tensor([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]),
    }
    losses = {
        2: torch.tensor([5.0, 2.0], requires_grad=True),
        4: torch.tensor([3.0, 3.0], requires_grad=True),
        6: torch.tensor([2.0, 4.0], requires_grad=True),
    }

    output = detached_three_budget_prefix_utilities(
        selections,
        lambda _positions, budget: losses[budget],
        lower_budget=2,
        baseline_budget=4,
        upper_budget=6,
    )

    assert torch.equal(output["downgrade_penalty"], torch.tensor([2.0, -1.0]))
    assert torch.equal(output["upgrade_gain"], torch.tensor([1.0, -1.0]))
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
        relative_utility=torch.tensor([[1.0, 0.0, -1.0], [-1.0, 0.0, 2.0]]),
        valid_observations=torch.tensor([300, 512]),
        budget_levels=(256, 384, 512),
        baseline_budget=384,
        target_actual_cost=768,
        max_changed_fraction=1.0,
    )
    assert feasible.feasible is True
    assert torch.equal(feasible.budget, torch.tensor([256, 512]))
    assert torch.equal(feasible.actual_cost, torch.tensor([256, 512]))

    impossible = allocate_equal_budget_marginal_reallocation(
        downgrade_penalty=torch.zeros(4),
        upgrade_gain=torch.ones(4),
        valid_observations=torch.full((4,), 300),
    )
    assert impossible.feasible is False
    assert "infeasible" in impossible.reason
    assert int(impossible.actual_cost.sum().item()) == 1200


def test_budget_grouped_heavy_execution_uses_three_real_temporal_shapes() -> None:
    for budget in (256, 384, 512):
        observations = torch.zeros(1, 1, 3, budget, 2, 2)
        assert (
            validate_real_heavy_observation_tensor(
                observations,
                expected_budget=budget,
            )
            is observations
        )


def test_nominal_k256_rejects_a_tensor_padded_to_512_observations() -> None:
    with pytest.raises(ValueError, match="does not match requested budget"):
        validate_real_heavy_observation_tensor(
            torch.zeros(1, 1, 3, 512, 2, 2),
            expected_budget=256,
        )
