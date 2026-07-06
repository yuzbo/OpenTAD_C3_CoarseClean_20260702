from __future__ import annotations

import os

import pytest

from tools.bata import gas_vt_paction_policy as gas_vt
from tools.bata import train_gap_aware_acquisition_policy as train_gas_vt


def _source_row(*, split: str | None = "training") -> dict:
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 8,
        "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]},
        "paction_positive_provenance": {
            "p_action_source": "lowres_action_probe",
            "probe_model": "mobilenetv3_64px",
            "no_gt_generation": True,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_cache": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
            "prediction_uses_gt": False,
        },
        "action_target": [0, 1, 1, 1, 0, 1, 1, 0],
        "gt_boundaries": [1, 4, 5, 7],
    }
    if split is not None:
        row["split"] = split
    return row


def test_gap_aware_features_extend_paction_with_sequential_budget_state() -> None:
    features = gas_vt.build_gap_aware_feature_matrix(
        [0.10, 0.90, 0.40, 0.80, 0.20],
        valid=[True, True, True, True, False],
        selected_so_far=[1],
        target_budget=3,
    )

    assert "p_action" in gas_vt.GAS_VT_FEATURE_NAMES
    for name in (
        "local_density",
        "local_change",
        "distance_to_last_selection",
        "remaining_budget",
        "remaining_time",
        "budget_pressure",
        "gap_urgency",
    ):
        assert name in gas_vt.GAS_VT_FEATURE_NAMES
    assert len(features) == 5
    assert features[0][gas_vt.feature_index("distance_to_last_selection")] == 1.0
    assert features[2][gas_vt.feature_index("distance_to_last_selection")] == 1.0
    assert features[3][gas_vt.feature_index("remaining_budget")] == pytest.approx(2 / 3)
    assert features[3][gas_vt.feature_index("remaining_time")] == pytest.approx(1 / 4)
    assert features[4][gas_vt.feature_index("valid")] == 0.0


def test_gas_vt_hard_decoder_returns_named_fixed_and_dynamic_strategies_without_uniform_fill() -> None:
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 8,
        "valid_len": 8,
        "strategy_selected_positions": {"delta_p_action": [1, 4]},
    }

    enriched = gas_vt.add_gas_vt_decision_to_sample_row(
        row,
        frame_values=[0.95, 0.90, 0.85, 0.05, 0.04, 0.80, 0.79, 0.78],
        fixed_budgets=(3, 5),
        dynamic_budget_scores=[0.1, 0.9, 0.2],
        dynamic_budget_buckets=[2, 4, 6],
        max_unselected_hole=2,
    )

    strategies = enriched["strategy_selected_positions"]
    assert sorted(strategies) == ["delta_p_action", "gas_vt_dynamic", "gas_vt_fixed_384", "gas_vt_fixed_768"]
    assert len(strategies["gas_vt_fixed_384"]) == 3
    assert len(strategies["gas_vt_fixed_768"]) == 5
    assert len(strategies["gas_vt_dynamic"]) == 4
    assert gas_vt.max_unselected_hole(strategies["gas_vt_fixed_384"], valid_len=8) <= 2
    assert enriched["gas_vt_policy"]["policy_family"] == "GAS-VT"
    assert enriched["gas_vt_policy"]["decode_mode"] == "hard_gap_aware_topk"
    assert enriched["gas_vt_policy"]["uses_uniform_fill"] is False
    assert enriched["gas_vt_policy"]["uses_uniform_scaffold"] is False


def test_training_preparation_builds_action_interior_bins_for_objective() -> None:
    rows = [_source_row()]

    prepared = train_gas_vt._prepared_rows(rows, dynamic_budget_buckets=[2, 4], expected_split="training")

    bins = prepared[0]["action_interior_bins"]
    assert bins
    assert all(len(mask) == 8 for mask in bins)
    assert all(any(mask[pos] > 0 for mask in bins) for pos in (1, 2, 3, 5, 6))
    assert all(mask[0] == 0.0 and mask[4] == 0.0 and mask[7] == 0.0 for mask in bins)
    assert prepared[0]["inferred_split_from_source_path"] is False


def test_training_preparation_fails_closed_when_split_is_missing_by_default() -> None:
    with pytest.raises(ValueError, match="expected split training, got <missing>"):
        train_gas_vt._prepared_rows(
            [_source_row(split=None)],
            dynamic_budget_buckets=[2, 4],
            expected_split="training",
        )


def test_training_preparation_can_infer_missing_split_from_explicit_source_path() -> None:
    prepared = train_gas_vt._prepared_rows(
        [_source_row(split=None)],
        dynamic_budget_buckets=[2, 4],
        expected_split="training",
        allow_missing_split_from_source_path=True,
    )

    assert len(prepared) == 1
    assert prepared[0]["inferred_split_from_source_path"] is True


def test_training_preparation_can_allow_training_source_gt_diagnostics_only() -> None:
    row = _source_row()
    row["uses_gt_for_diagnostics"] = True

    prepared = train_gas_vt._prepared_rows(
        [row],
        dynamic_budget_buckets=[2, 4],
        expected_split="training",
        allow_gt_diagnostics_in_training_source=True,
    )

    assert len(prepared) == 1
    assert prepared[0]["allowed_gt_diagnostics_in_training_source"] is True


def test_training_preparation_rejects_gt_diagnostics_without_explicit_training_allowance() -> None:
    row = _source_row()
    row["uses_gt_for_diagnostics"] = True

    with pytest.raises(ValueError, match="forbidden p_action source flag uses_gt_for_diagnostics=true"):
        train_gas_vt._prepared_rows(
            [row],
            dynamic_budget_buckets=[2, 4],
            expected_split="training",
        )


def test_training_preparation_rejects_gt_diagnostics_outside_training_even_when_enabled() -> None:
    row = _source_row(split="validation")
    row["uses_gt_for_diagnostics"] = True

    with pytest.raises(ValueError, match="forbidden p_action source flag uses_gt_for_diagnostics=true"):
        train_gas_vt._prepared_rows(
            [row],
            dynamic_budget_buckets=[2, 4],
            expected_split="validation",
            allow_gt_diagnostics_in_training_source=True,
        )


def test_training_preparation_rejects_real_gt_source_even_when_diagnostics_are_enabled() -> None:
    row = _source_row()
    row["uses_gt"] = True
    row["uses_gt_for_diagnostics"] = True

    with pytest.raises(ValueError, match="forbidden p_action source flag uses_gt=true"):
        train_gas_vt._prepared_rows(
            [row],
            dynamic_budget_buckets=[2, 4],
            expected_split="training",
            allow_gt_diagnostics_in_training_source=True,
        )


def test_training_preparation_rejects_wrong_explicit_split_even_when_inference_enabled() -> None:
    with pytest.raises(ValueError, match="expected split training, got validation"):
        train_gas_vt._prepared_rows(
            [_source_row(split="validation")],
            dynamic_budget_buckets=[2, 4],
            expected_split="training",
            allow_missing_split_from_source_path=True,
        )


@pytest.mark.skipif(os.name == "nt", reason="torch objective tests run in Linux/remote OpenTAD")
def test_gas_vt_training_objective_uses_hard_or_st_mask_for_core_losses() -> None:
    torch = pytest.importorskip("torch")
    outputs = {
        "frame_value": torch.tensor([[2.0, -1.0, 1.5, -0.5]], dtype=torch.float32),
        "st_selected_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        "hard_selected_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        "budget_logits": torch.tensor([[0.0, 1.0]], dtype=torch.float32),
    }

    losses = gas_vt.gas_vt_training_objective(
        outputs,
        action_target=torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
        boundary_target=torch.tensor([[1.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
        valid=torch.tensor([[True, True, True, True]]),
        target_budget=torch.tensor([2.0]),
        action_interior_bins=torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]], dtype=torch.float32),
    )

    for key in (
        "value_bce_loss",
        "boundary_coverage_loss",
        "boundary_bracket_loss",
        "action_interior_bin_loss",
        "cvar_max_hole_loss",
        "budget_loss",
        "paction_dependence_loss",
        "total_loss",
    ):
        assert key in losses
    assert losses["selection_mask_source"] == "st_selected_mask"
