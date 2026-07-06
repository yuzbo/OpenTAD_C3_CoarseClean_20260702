from __future__ import annotations

import os

import pytest

if os.name == "nt":
    pytest.skip("torch policy tests run in the Linux/remote OpenTAD environment", allow_module_level=True)

torch = pytest.importorskip("torch")

from tools.bata import paction_acquisition_policy as policy
from tools.bata import paction_budget_contract as budget_contract


def test_dynamic_acquisition_policy_is_standard_torch_module() -> None:
    model = policy.PActionDynamicAcquisitionPolicy(hidden_dim=8, num_layers=1)

    assert isinstance(model, torch.nn.Module)


def test_feature_matrix_exposes_paction_dynamics_uncertainty_and_time() -> None:
    features = policy.build_paction_feature_matrix(
        [0.10, 0.90, 0.60, 0.20],
        valid=[True, True, True, False],
    )

    assert policy.PACTION_FEATURE_NAMES == (
        "p_action",
        "delta_p_action",
        "abs_delta_p_action",
        "entropy",
        "uncertainty",
        "time",
        "valid",
    )
    assert len(features) == 4
    assert features[0][policy.feature_index("p_action")] == 0.10
    assert features[1][policy.feature_index("delta_p_action")] == 0.80
    assert features[2][policy.feature_index("abs_delta_p_action")] == 0.30
    assert features[3][policy.feature_index("valid")] == 0.0
    assert features[0][policy.feature_index("time")] == 0.0
    assert features[2][policy.feature_index("time")] == 1.0
    assert 0.0 < features[1][policy.feature_index("entropy")] < 1.0
    assert features[1][policy.feature_index("uncertainty")] < features[2][policy.feature_index("uncertainty")]


def test_constrained_topk_selects_sorted_unique_valid_positions() -> None:
    selected = policy.constrained_topk(
        [0.1, 0.9, 0.8, 0.7, 0.95],
        valid=[True, True, False, True, True],
        budget=3,
    )

    assert selected == [1, 3, 4]


def test_gap_hole_loss_penalizes_large_empty_windows_without_uniform_fill() -> None:
    no_hole = policy.temporal_gap_hole_loss_from_probabilities(
        [1.0, 0.0, 0.0, 1.0],
        max_gap=3,
    )
    large_hole = policy.temporal_gap_hole_loss_from_probabilities(
        [1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        max_gap=3,
    )

    assert no_hole == 0.0
    assert large_hole > 1.0


def test_boundary_miss_loss_rewards_learned_boundary_coverage() -> None:
    missed = policy.boundary_miss_loss_from_probabilities([0.0, 0.0, 0.0, 0.0], [2], radius=1)
    covered = policy.boundary_miss_loss_from_probabilities([0.0, 0.0, 1.0, 0.0], [2], radius=1)

    assert covered == 0.0
    assert missed > covered


def test_dynamic_budget_decoder_clamps_to_valid_len_and_bounds() -> None:
    assert policy.decode_budget_from_scores([0.1, 0.2, 0.9], [2, 4, 8], valid_len=5) == 5
    assert policy.decode_budget_from_scores([0.1, 0.2, 0.9], [2, 4, 8], valid_len=5, max_budget=4) == 4
    assert policy.decode_budget_from_scores([0.9, 0.2, 0.1], [2, 4, 8], valid_len=5, min_budget=3) == 3


def test_short_valid_budget_contract_is_shared_and_matches_failed_remote_case() -> None:
    assert budget_contract.expected_selected_count(
        384,
        valid_len=251,
        dense_len=768,
        allow_short_valid_ratio_count=True,
    ) == 126
    assert policy.short_valid_ratio_budget(384, valid_len=251, dense_len=768) == 126


def test_oracle_budget_uses_smallest_bucket_meeting_quality_target() -> None:
    decision = policy.oracle_budget_from_quality_curve(
        {
            2: {"boundary_support_r1": 0.50, "action_positive_coverage": 0.25},
            4: {"boundary_support_r1": 1.00, "action_positive_coverage": 0.75},
            6: {"boundary_support_r1": 1.00, "action_positive_coverage": 0.90},
        },
        min_boundary_support=0.90,
        min_action_coverage=0.70,
    )

    assert decision.budget == 4
    assert decision.reason == "meets_quality_target"


def test_policy_row_adds_learned_fixed_and_dynamic_budget_strategies() -> None:
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 8,
        "strategy_selected_positions": {"delta_p_action": [2, 5]},
        "frame_signals": {"p_action": [0.1, 0.2, 0.9, 0.4, 0.7, 0.3, 0.8, 0.6]},
    }

    enriched = policy.add_policy_decision_to_sample_row(
        row,
        frame_values=[0.1, 0.8, 0.7, 0.2, 0.9, 0.3, 0.6, 0.5],
        fixed_budget=3,
        dynamic_budget_scores=[0.2, 0.9, 0.1],
        dynamic_budget_buckets=[2, 4, 6],
    )

    assert enriched is not row
    strategies = enriched["strategy_selected_positions"]
    assert strategies["learned_paction_gap_loss_value"] == [1, 2, 4]
    assert strategies["learned_paction_gap_loss_dynamic_budget"] == [1, 2, 4, 6]
    assert enriched["paction_policy"]["fixed_budget"] == 3
    assert enriched["paction_policy"]["dynamic_budget"] == 4
    assert enriched["paction_policy"]["selection_signal"] == "p_action_gap_loss_policy_value"
    assert enriched["paction_policy"]["gap_control"] == "learned_gap_hole_loss_no_uniform_fill"
    assert enriched["paction_policy"]["uses_uniform_scaffold"] is False
    assert enriched["paction_policy"]["uses_uniform_fill"] is False
