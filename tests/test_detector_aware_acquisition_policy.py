from __future__ import annotations

import json
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
        "teacher_utility": {"frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]},
        "teacher_utility_provenance": {"split_scope": "train_only"},
    }


def test_detector_aware_training_preparation_targets_teacher_utility_not_action_labels() -> None:
    prepared = train_detector._prepared_rows([_sample_row()], dynamic_budget_buckets=[2, 4], expected_split="training")

    assert len(prepared) == 1
    assert "detector_utility_target" in prepared[0]
    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, 0.1, 1.0, 0.2, 0.7]
    assert prepared[0]["dynamic_budget_target"] == 4
    assert len(prepared[0]["features"][0]) == len(detector_policy.DETECTOR_AWARE_FEATURE_NAMES)


def test_detector_aware_training_preserves_signed_utility_and_calibrated_gain_target() -> None:
    row = _sample_row()
    row["teacher_utility"] = {
        "utility_semantics": "signed_detector_utility_v1",
        "frame_utility": [0.0, 0.9, 0.1, 1.0, 0.2, 0.7],
        "signed_frame_utility": [0.0, 0.9, -0.4, 1.0, -0.8, 0.7],
    }

    prepared = train_detector._prepared_rows([row], dynamic_budget_buckets=[2, 4], expected_split="training")

    assert prepared[0]["detector_utility_target"] == [0.0, 0.9, -0.4, 1.0, -0.8, 0.7]
    assert prepared[0]["detector_marginal_gain_target"] == [0.0, 0.9, 0.4, 1.0, 0.8, 0.7]
    assert prepared[0]["dynamic_gain_calibration"]["score_semantics"] == "calibrated_marginal_gain"
    assert prepared[0]["dynamic_budget_target"] == 4


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
    assert meta["dynamic_budget_calibration"]["score_semantics"] == "calibrated_marginal_gain"
    assert meta["dynamic_budget_calibration"]["calibration_scope"] == "cross_video_comparable"


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
