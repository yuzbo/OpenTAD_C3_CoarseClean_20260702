from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.bata import apply_detector_aware_acquisition_policy as apply_detector
from tools.bata import detector_aware_acquisition_policy as detector_policy
from tools.bata import train_detector_aware_acquisition_policy as train_detector


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _paction_provenance() -> dict:
    return {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }


def _sample_row() -> dict:
    return {
        "sample_id": "video_test_0001|0",
        "split": "training",
        "dense_len": 6,
        "valid_len": 6,
        "frame_signals": {"p_action": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6]},
        "paction_positive_provenance": _paction_provenance(),
        "action_target": [0, 1, 0, 1, 0, 1],
        "teacher_utility": {
            "utility_semantics": "signed_detector_utility_v1",
            "frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7],
            "signed_frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7],
        },
        "teacher_utility_provenance": {"split_scope": "train_only"},
    }


def test_detector_aware_training_preparation_targets_teacher_utility_not_action_labels() -> None:
    row = _sample_row()
    prepared = train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    assert len(prepared) == 1
    assert "detector_utility_target" in prepared[0]
    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]
    assert prepared[0]["positive_observation_gain_target"] == [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]
    assert prepared[0]["negative_observation_risk_target"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert prepared[0]["dynamic_budget_target"] == 2
    assert prepared[0]["action_target"] == [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    assert len(prepared[0]["features"][0]) == len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES)


def test_detector_aware_training_requires_explicit_action_target_for_action_local_loss() -> None:
    row = _sample_row()
    row.pop("action_target")

    with pytest.raises(ValueError, match="action_target is required"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")


def test_detector_aware_training_preserves_signed_utility_and_calibrated_gain_target() -> None:
    row = _sample_row()
    row["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7],
        "signed_frame_utility": [0.0, 0.9, -0.4, 1.0, -0.8, 0.7],
    }

    prepared = train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, -0.4, 1.0, -0.8, 0.7]
    assert prepared[0]["positive_observation_gain_target"] == [0.0, 0.9, 0.0, 1.0, 0.0, 0.7]
    assert prepared[0]["negative_observation_risk_target"] == [0.0, 0.0, 0.4, 0.0, 0.8, 0.0]
    assert prepared[0]["detector_marginal_gain_target"] == [0.0, 0.9, 0.0, 1.0, 0.0, 0.7]
    assert prepared[0]["dynamic_gain_calibration"]["score_semantics"] == "calibrated_marginal_gain"
    assert prepared[0]["dynamic_gain_calibration"]["calibration_fitted"] is True
    assert prepared[0]["dynamic_budget_target"] == 2


def test_detector_aware_main_training_requires_point_responsibility_contract() -> None:
    surrogate = _sample_row()
    surrogate["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "utility_source_type": "dense_detector_forward_test_proposal_score_surrogate_v1",
        "signed_frame_utility": [0.0, 0.9, -0.4, 1.0, -0.8, 0.7],
        "proposal_score_surrogate_utility": True,
        "point_responsibility_utility": False,
    }
    with pytest.raises(ValueError, match="signed_point_responsibility_utility_v1"):
        train_detector._require_point_responsibility_contract(train_detector._teacher_utility_contract([surrogate]))

    responsibility = _sample_row()
    responsibility["teacher_utility"] = {
        "utility_semantics": "signed_point_responsibility_utility_v1",
        "utility_source_type": "point_loss_gradient_responsibility_v1",
        "signed_frame_utility": [0.0, 0.9, -0.4, 1.0, -0.8, 0.7],
        "proposal_score_surrogate_utility": False,
        "point_responsibility_utility": True,
    }
    contract = train_detector._teacher_utility_contract([responsibility])
    train_detector._require_point_responsibility_contract(contract)
    assert contract["point_responsibility_utility"] is True
    assert contract["proposal_score_surrogate_utility"] is False


def test_detector_aware_main_checkpoint_rejects_surrogate_contract() -> None:
    surrogate_payload = {
        "utility_semantics": "signed_detector_utility_v1",
        "utility_source_type": "dense_detector_forward_test_proposal_score_surrogate_v1",
        "point_responsibility_utility": False,
        "proposal_score_surrogate_utility": True,
        "paper_main_target_allowed": False,
    }
    with pytest.raises(ValueError, match="signed_point_responsibility_utility_v1"):
        apply_detector._validate_checkpoint_utility_contract(
            surrogate_payload,
            require_point_responsibility_utility=True,
        )

    responsibility_payload = {
        "utility_semantics": "signed_point_responsibility_utility_v1",
        "utility_source_type": "point_loss_gradient_responsibility_v1",
        "point_responsibility_utility": True,
        "proposal_score_surrogate_utility": False,
        "paper_main_target_allowed": True,
    }
    apply_detector._validate_checkpoint_utility_contract(
        responsibility_payload,
        require_point_responsibility_utility=True,
    )


def test_detector_aware_training_can_ignore_train_source_gt_diagnostic_flag_when_explicitly_allowed() -> None:
    row = _sample_row()
    row["uses_gt_for_diagnostics"] = True

    with pytest.raises(ValueError, match="uses_gt_for_diagnostics"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    prepared = train_detector._prepared_rows(
        [row],
        dynamic_budget_buckets=[2, 4],
        expected_split="training",
        allow_gt_diagnostics_in_training_source=True,
    )

    assert prepared[0]["allowed_gt_diagnostics_in_training_source"] is True
    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]


def test_detector_aware_training_can_accept_train_only_teacher_artifact_when_explicitly_allowed() -> None:
    row = _sample_row()
    row["uses_teacher"] = True
    row["training_only"] = True

    with pytest.raises(ValueError, match="uses_teacher"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    prepared = train_detector._prepared_rows(
        [row],
        dynamic_budget_buckets=[2, 4],
        expected_split="training",
        allow_teacher_utility_training_artifact=True,
    )

    assert prepared[0]["allowed_teacher_utility_training_artifact"] is True
    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]


def test_detector_aware_training_rejects_legacy_abs_marginal_gain_target() -> None:
    row = _sample_row()
    row["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "signed_frame_utility": [0.0, 0.9, -0.4, 1.0, -0.8, 0.7],
        "marginal_gain_frame_utility": [0.0, 0.9, 0.4, 1.0, 0.8, 0.7],
    }

    with pytest.raises(ValueError, match="marginal_gain_frame_utility"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")


def test_detector_aware_training_rejects_unsigned_frame_utility_for_signed_claim() -> None:
    row = _sample_row()
    row["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7],
    }

    with pytest.raises(ValueError, match="signed_frame_utility"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")


def test_detector_aware_dynamic_budget_uses_train_global_gain_threshold_not_per_video_ranking() -> None:
    high_gain = _sample_row()
    high_gain["sample_id"] = "video_high|0"
    high_gain["frame_signals"]["p_action"] = [0.9, 0.8, 0.7, 0.6]
    high_gain["dense_len"] = 4
    high_gain["valid_len"] = 4
    high_gain["action_target"] = [1, 1, 1, 1]
    high_gain["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "signed_frame_utility": [0.9, 0.8, 0.7, 0.6],
    }
    low_gain = _sample_row()
    low_gain["sample_id"] = "video_low|0"
    low_gain["frame_signals"]["p_action"] = [0.9, 0.8, 0.7, 0.6]
    low_gain["dense_len"] = 4
    low_gain["valid_len"] = 4
    low_gain["action_target"] = [1, 1, 1, 1]
    low_gain["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "signed_frame_utility": [0.2, 0.1, 0.0, 0.0],
    }

    prepared = train_detector._prepared_rows(
        [high_gain, low_gain],
        dynamic_budget_buckets=[1, 3],
        expected_split="training",
    )

    assert [row["dynamic_budget_target"] for row in prepared] == [3, 1]
    calibration = prepared[0]["dynamic_gain_calibration"]
    assert calibration == prepared[1]["dynamic_gain_calibration"]
    assert calibration["fit_split"] == "training"
    assert calibration["calibration_scope"] == "cross_video_comparable"
    assert calibration["budget_target_rule"] == "count_positive_gain_at_global_threshold_then_nearest_bucket"


def test_detector_aware_policy_marks_dynamic_budget_uncalibrated_without_fit_evidence() -> None:
    row = {
        "sample_id": "video_test_0001|0",
        "dense_len": 4,
        "valid_len": 4,
    }

    enriched = detector_policy.add_detector_aware_decision_to_sample_row(
        row,
        frame_values=[0.9, 0.8, 0.1, 0.0],
        fixed_budgets=(1, 2),
        dynamic_budget_scores=[0.0, 1.0],
        dynamic_budget_buckets=[1, 3],
    )

    calibration = enriched["detector_aware_policy"]["dynamic_budget_calibration"]
    assert calibration["calibration_fitted"] is False
    assert calibration["calibrated_dynamic_claim_allowed"] is False
    assert calibration["score_semantics"] == "uncalibrated_dynamic_budget_score"


def test_detector_aware_fixed_decoder_fills_budget_instead_of_short_ratio_scaling() -> None:
    row = {
        "sample_id": "video_short_valid_0001|0",
        "dense_len": 12,
        "valid_len": 6,
    }

    enriched = detector_policy.add_detector_aware_decision_to_sample_row(
        row,
        frame_values=[0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 9.0, 9.0, 9.0, 9.0, 9.0, 9.0],
        fixed_budgets=(8, 10),
        dynamic_budget_scores=[0.0, 1.0],
        dynamic_budget_buckets=[2, 4],
        max_unselected_hole=0,
    )

    strategies = enriched["strategy_selected_positions"]
    assert strategies["detector_aware_fixed_384"] == [0, 1, 2, 3, 4, 5]
    assert strategies["detector_aware_fixed_768"] == [0, 1, 2, 3, 4, 5]
    assert enriched["detector_aware_policy"]["fixed_budget_contract"] == "min_requested_budget_or_valid_len_no_short_ratio"


def test_detector_aware_fixed_decoder_dilates_responsibility_scores_with_r2_then_r4() -> None:
    row = {
        "sample_id": "video_boundary_context_0001|0",
        "dense_len": 11,
        "valid_len": 11,
    }

    enriched = detector_policy.add_detector_aware_decision_to_sample_row(
        row,
        frame_values=[0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        fixed_budgets=(5, 9),
        dynamic_budget_scores=[0.0, 1.0],
        dynamic_budget_buckets=[5, 9],
    )

    strategies = enriched["strategy_selected_positions"]
    assert strategies["detector_aware_fixed_384"] == [3, 4, 5, 6, 7]
    assert strategies["detector_aware_fixed_768"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
    meta = enriched["detector_aware_policy"]
    assert meta["score_dilation_radii"] == [2, 4]
    assert meta["decode_mode"] == "hard_gap_aware_topk_after_learned_score_dilation_r2_r4"


def test_detector_aware_score_dilation_can_use_learned_continuous_context_radius() -> None:
    values = [0.0] * 21
    values[10] = 10.0
    valid = [True] * len(values)

    narrow = detector_policy.score_dilated_frame_values(
        values,
        valid=valid,
        context_radii=[2.0] * len(values),
    )
    wide = detector_policy.score_dilated_frame_values(
        values,
        valid=valid,
        context_radii=[16.0] * len(values),
    )

    assert detector_policy.gas_vt.hard_gap_aware_topk(narrow, valid=valid, budget=5) == [8, 9, 10, 11, 12]
    assert detector_policy.gas_vt.hard_gap_aware_topk(wide, valid=valid, budget=17) == [
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
    ]
    assert wide[0] > narrow[0]


def test_detector_aware_score_dilation_clamps_learned_radius_to_minimum_two() -> None:
    values = [0.0] * 11
    values[5] = 10.0
    valid = [True] * len(values)

    decoded = detector_policy.score_dilated_frame_values(
        values,
        valid=valid,
        context_radii=[0.0] * len(values),
    )

    assert detector_policy.gas_vt.hard_gap_aware_topk(decoded, valid=valid, budget=5) == [3, 4, 5, 6, 7]


def test_detector_aware_score_dilation_uses_local_peaks_not_all_positive_values() -> None:
    flat_positive = [0.1] * 9
    valid = [True] * len(flat_positive)

    decoded = detector_policy.score_dilated_frame_values(
        flat_positive,
        valid=valid,
        context_radii=[16.0] * len(flat_positive),
    )

    assert decoded == pytest.approx(flat_positive)


def test_detector_aware_apply_fails_closed_when_checkpoint_lacks_context_radius(monkeypatch) -> None:
    def legacy_scores(_model, p_action, *, valid, target_budget=None, device="cpu", **_kwargs):
        return [float(item) for item in p_action], [0.0, 1.0]

    monkeypatch.setattr(apply_detector, "checkpoint_policy_scores", legacy_scores)

    with pytest.raises(ValueError, match="context_radius"):
        apply_detector._checkpoint_policy_scores_with_context(
            object(),
            [0.1, 0.9],
            valid=[True, True],
            target_budget=2,
            device="cpu",
        )


def test_detector_aware_apply_allows_legacy_context_radius_only_with_explicit_opt_in(monkeypatch) -> None:
    def legacy_scores(_model, p_action, *, valid, target_budget=None, device="cpu", **_kwargs):
        return [float(item) for item in p_action], [0.0, 1.0]

    monkeypatch.setattr(apply_detector, "checkpoint_policy_scores", legacy_scores)

    frame_values, budget_scores, context_radius = apply_detector._checkpoint_policy_scores_with_context(
        object(),
        [0.1, 0.9],
        valid=[True, True],
        target_budget=2,
        device="cpu",
        allow_legacy_no_context_radius=True,
        legacy_context_radius=4.0,
    )

    assert frame_values == [0.1, 0.9]
    assert budget_scores == [0.0, 1.0]
    assert context_radius == [4.0, 4.0]


def test_detector_aware_context_radius_head_receives_gradient() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch DLL import is not reliable in this workspace")
    torch = pytest.importorskip("torch")
    model = detector_policy.DetectorAwareSequentialAcquisitionPolicy(hidden_dim=8, num_layers=1, dropout=0.0)
    features = torch.zeros((1, 8, len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES)), dtype=torch.float32)
    valid = torch.ones((1, 8), dtype=torch.bool)
    utility = torch.ones((1, 8), dtype=torch.float32)
    action = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)

    outputs = model(features, valid, target_budget=torch.tensor([4.0]))
    losses = detector_policy.detector_aware_training_objective(
        outputs,
        detector_utility_target=utility,
        valid=valid,
        target_budget=torch.tensor([4.0]),
        action_target=action,
    )
    losses["total_loss"].backward()

    radius_grad = [
        param.grad.detach().abs().sum().item()
        for param in model.context_radius_head.parameters()
        if param.grad is not None
    ]
    assert sum(radius_grad) > 0.0


def test_detector_aware_objective_penalizes_action_local_holes() -> None:
    if os.name == "nt":
        pytest.skip("local Windows torch DLL import is not reliable in this workspace")
    torch = pytest.importorskip("torch")
    frame_logits = torch.zeros((1, 8), dtype=torch.float32)
    valid = torch.ones((1, 8), dtype=torch.bool)
    utility = torch.ones((1, 8), dtype=torch.float32)
    action = torch.tensor([[0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]], dtype=torch.float32)
    sparse_outputs = {
        "frame_value": frame_logits,
        "st_selected_mask": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float32),
    }
    covered_outputs = {
        "frame_value": frame_logits,
        "st_selected_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
    }

    sparse = detector_policy.detector_aware_training_objective(
        sparse_outputs,
        detector_utility_target=utility,
        valid=valid,
        target_budget=torch.tensor([4.0]),
        action_target=action,
    )
    covered = detector_policy.detector_aware_training_objective(
        covered_outputs,
        detector_utility_target=utility,
        valid=valid,
        target_budget=torch.tensor([4.0]),
        action_target=action,
    )

    assert sparse["action_local_hole_loss"].item() > covered["action_local_hole_loss"].item()
    assert "learned_spacing_loss" in sparse


def test_detector_aware_training_rejects_non_train_teacher_utility() -> None:
    row = _sample_row()
    row["split"] = "validation"

    with pytest.raises(ValueError, match="expected split training"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    row = _sample_row()
    row["teacher_utility_provenance"] = {"split_scope": "validation"}
    with pytest.raises(ValueError, match="teacher utility must be train_only"):
        train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")


def test_detector_aware_apply_emits_strategies_and_strips_teacher_payload(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    output = tmp_path / "samples.detector_aware.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "validation",
                "dense_len": 8,
                "valid_len": 8,
                "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]},
                "paction_positive_provenance": _paction_provenance(),
            }
        ],
    )

    summary = apply_detector.run_policy_application(
        source,
        output,
        fixed_budgets=(3, 5),
        dynamic_budget_buckets=[2, 4, 6],
        strict_deploy_source=True,
        strip_deploy_invisible_payload=True,
        allow_bootstrap_for_tests=True,
    )
    rows = _read_jsonl(output)

    assert summary["decision"] == "C3_DETECTOR_AWARE_POLICY_APPLICATION_READY"
    assert summary["stage_label"] == "Stage-2 detector-aware offline selector"
    assert rows[0]["deploy_invisible_payload_stripped"] is True
    assert "teacher_utility" not in rows[0]
    assert "frame_signals" not in rows[0]
    assert sorted(rows[0]["strategy_selected_positions"]) == [
        "detector_aware_dynamic",
        "detector_aware_fixed_384",
        "detector_aware_fixed_768",
    ]
    meta = rows[0]["detector_aware_policy"]
    assert meta["source"] == "bootstrap_detector_aware_surrogate_policy"
    assert meta["selection_signal"] == "p_action_to_detector_utility"
    assert meta["stage_label"] == "Stage-2 detector-aware offline selector"
    assert meta["end_to_end"] is False
    assert meta["uses_uniform_fill"] is False
    assert meta["dynamic_budget_calibration"]["score_semantics"] == "uncalibrated_dynamic_budget_score"
    assert meta["dynamic_budget_calibration"]["calibrated_dynamic_claim_allowed"] is False


def test_detector_aware_checkpoint_apply_scores_each_strategy_with_its_target_budget(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "samples.jsonl"
    output = tmp_path / "samples.detector_aware.jsonl"
    checkpoint = tmp_path / "detector_policy.pth"
    checkpoint.write_bytes(b"fake detector-aware checkpoint")
    _write_jsonl(
        source,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "validation",
                "dense_len": 8,
                "valid_len": 8,
                "frame_signals": {"p_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]},
                "paction_positive_provenance": _paction_provenance(),
            }
        ],
    )
    seen_budgets: list[int | None] = []

    monkeypatch.setattr(
        apply_detector,
        "load_policy_checkpoint",
        lambda *args, **kwargs: (
            object(),
            {
                "dynamic_budget_buckets": [2, 4, 6],
                "dynamic_gain_calibration": {
                    "score_semantics": "calibrated_marginal_gain",
                    "calibration_scope": "cross_video_comparable",
                    "target_source": "positive_observation_gain_from_signed_detector_utility",
                    "calibration_fitted": True,
                    "fit_split": "training",
                    "budget_buckets": [2, 4, 6],
                    "gain_threshold": 0.5,
                    "calibrated_dynamic_claim_allowed": True,
                },
            },
        ),
    )

    def fake_checkpoint_scores(_model, p_action, *, valid, target_budget, device, return_context_radius=False):
        seen_budgets.append(target_budget)
        frame_values = [float(target_budget or 0) + float(idx) / 100.0 for idx, _ in enumerate(p_action)]
        budget_scores = [0.0, 0.0, 1.0]
        if return_context_radius:
            return frame_values, budget_scores, [4.0 for _ in p_action]
        return frame_values, budget_scores

    monkeypatch.setattr(apply_detector, "checkpoint_policy_scores", fake_checkpoint_scores)

    apply_detector.run_policy_application(
        source,
        output,
        fixed_budgets=(3, 5),
        checkpoint_path=checkpoint,
        device="cpu",
    )
    row = _read_jsonl(output)[0]

    assert seen_budgets == [6, 3, 5, 6]
    assert row["detector_aware_policy"]["apply_time_target_budgets"] == {
        "detector_aware_fixed_384": 3,
        "detector_aware_fixed_768": 5,
        "detector_aware_dynamic": 6,
    }
    assert row["detector_aware_policy"]["budget_conditioning_rule"] == "checkpoint_two_pass_strategy_specific_target_budget"
    assert row["detector_aware_policy"]["budget_conditioned_frame_values"] is True


def test_detector_aware_checkpoint_apply_conditions_fixed_scores_on_filled_budget_for_short_valid_rows(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "samples.jsonl"
    output = tmp_path / "samples.detector_aware.jsonl"
    checkpoint = tmp_path / "detector_policy.pth"
    checkpoint.write_bytes(b"fake detector-aware checkpoint")
    _write_jsonl(
        source,
        [
            {
                "sample_id": "video_short_valid_0001|0",
                "split": "validation",
                "dense_len": 12,
                "valid_len": 6,
                "frame_signals": {"p_action": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.9, 0.8]},
                "paction_positive_provenance": _paction_provenance(),
            }
        ],
    )
    seen_budgets: list[int | None] = []

    monkeypatch.setattr(
        apply_detector,
        "load_policy_checkpoint",
        lambda *args, **kwargs: (object(), {"dynamic_budget_buckets": [2, 4, 6]}),
    )

    def fake_checkpoint_scores(_model, p_action, *, valid, target_budget, device, return_context_radius=False):
        seen_budgets.append(target_budget)
        frame_values = [float(target_budget or 0) + float(idx) / 100.0 for idx, _ in enumerate(p_action)]
        budget_scores = [0.0, 0.0, 1.0]
        if return_context_radius:
            return frame_values, budget_scores, [4.0 for _ in p_action]
        return frame_values, budget_scores

    monkeypatch.setattr(apply_detector, "checkpoint_policy_scores", fake_checkpoint_scores)

    apply_detector.run_policy_application(
        source,
        output,
        fixed_budgets=(8, 10),
        checkpoint_path=checkpoint,
        device="cpu",
        max_unselected_hole=0,
    )
    row = _read_jsonl(output)[0]

    assert seen_budgets == [6, 6, 6, 6]
    assert row["detector_aware_policy"]["apply_time_target_budgets"] == {
        "detector_aware_fixed_384": 6,
        "detector_aware_fixed_768": 6,
        "detector_aware_dynamic": 6,
    }
    assert row["strategy_selected_positions"]["detector_aware_fixed_384"] == [0, 1, 2, 3, 4, 5]


def test_detector_aware_apply_rejects_teacher_payload_in_deploy_source(tmp_path: Path) -> None:
    source = tmp_path / "samples.jsonl"
    output = tmp_path / "samples.detector_aware.jsonl"
    row = {
        "sample_id": "video_test_0001|0",
        "split": "validation",
        "dense_len": 4,
        "valid_len": 4,
        "frame_signals": {"p_action": [0.1, 0.9, 0.2, 0.8]},
        "paction_positive_provenance": _paction_provenance(),
        "teacher_utility": {"frame_utility": [0.0, 1.0, 0.0, 0.8]},
    }
    _write_jsonl(source, [row])

    with pytest.raises(ValueError, match="teacher_utility"):
        apply_detector.run_policy_application(
            source,
            output,
            strict_deploy_source=True,
            strip_deploy_invisible_payload=True,
            allow_bootstrap_for_tests=True,
        )

    row.pop("teacher_utility")
    row["teacher_utility_provenance"] = {"split_scope": "train_only"}
    _write_jsonl(source, [row])
    with pytest.raises(ValueError, match="teacher_utility_provenance"):
        apply_detector.run_policy_application(
            source,
            output,
            strict_deploy_source=True,
            strip_deploy_invisible_payload=True,
            allow_bootstrap_for_tests=True,
        )
