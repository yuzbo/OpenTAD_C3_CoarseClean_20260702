from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tools.bata import export_adatad_responsibility_utility as exporter
from tools.bata import export_adatad_responsibility_points_from_teacher as points_from_teacher
from tools.bata import train_detector_aware_acquisition_policy as train_detector
from tools.bata import validate_adatad_responsibility_utility as validator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _load_stage23_validator():
    root = Path(__file__).resolve().parents[1]
    validator_path = root / "tools" / "bata" / "validate_duca_stage23_precheck.py"
    spec = importlib.util.spec_from_file_location("validate_duca_stage23_precheck_test", validator_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _train_manifest(**extra: object) -> dict:
    manifest = {
        "split": "train",
        "uses_val_or_test_gt_for_selection": False,
        "uses_teacher_at_deploy": False,
        "uses_prediction_cache_at_deploy": False,
    }
    manifest.update(extra)
    return manifest


def _responsibility_point(**extra: object) -> dict:
    point = {
        "true_time_center": 2,
        "support_start": 1,
        "support_end": 3,
        "utility_source_type": "point_loss_gradient_responsibility_v1",
        "positive_gain": 0.7,
        "negative_risk": 0.2,
        "cls_loss": 0.4,
        "reg_loss": 0.1,
        "quality_loss": 0.05,
        "grad_norm": 1.5,
        "boundary_role": "center",
        "assigned_gt_id": "gt-1",
    }
    point.update(extra)
    return point


def _source_row(**extra: object) -> dict:
    row = {
        "sample_id": "video_train_0001|0",
        "split": "training",
        "dense_len": 5,
        "points": [_responsibility_point()],
    }
    row.update(extra)
    return row


def test_legal_responsibility_export_validates(tmp_path: Path) -> None:
    source = tmp_path / "responsibility_points.jsonl"
    output = tmp_path / "responsibility_utility.jsonl"
    summary = tmp_path / "responsibility_utility.summary.json"
    _write_jsonl(source, [_source_row()])

    export_summary = exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest())
    evidence = validator.validate_responsibility_utility_export(summary, output_jsonl=output)

    assert export_summary["decision"] == "ADATAD_RESPONSIBILITY_UTILITY_EXPORT_READY"
    assert evidence["decision"] == "ADATAD_RESPONSIBILITY_UTILITY_VALIDATION_PASS"
    assert evidence["row_count"] == 1


def test_proposal_score_surrogate_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "surrogate_points.jsonl"
    output = tmp_path / "surrogate_utility.jsonl"
    summary = tmp_path / "surrogate_utility.summary.json"
    _write_jsonl(source, [_source_row(points=[_responsibility_point(utility_source_type="proposal_score_surrogate")])])

    with pytest.raises(ValueError, match="proposal_score_surrogate"):
        exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest())


def test_val_or_test_manifest_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "val_points.jsonl"
    output = tmp_path / "val_utility.jsonl"
    summary = tmp_path / "val_utility.summary.json"
    _write_jsonl(source, [_source_row()])

    with pytest.raises(ValueError, match="split=train"):
        exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest(split="val"))

    with pytest.raises(ValueError, match="uses_val_or_test_gt_for_selection=False"):
        exporter.run_export(
            source,
            output,
            summary_json=summary,
            manifest=_train_manifest(uses_val_or_test_gt_for_selection=True),
        )


def test_val_or_test_source_rows_fail_closed_even_with_train_manifest(tmp_path: Path) -> None:
    source = tmp_path / "val_points.jsonl"
    output = tmp_path / "val_utility.jsonl"
    _write_jsonl(source, [_source_row(split="validation")])

    with pytest.raises(ValueError, match="source row 1: split must be train/training"):
        exporter.run_export(source, output, manifest=_train_manifest())


def test_base_paction_samples_must_be_train_split(tmp_path: Path) -> None:
    source = tmp_path / "points.jsonl"
    base = tmp_path / "base_samples.jsonl"
    output = tmp_path / "samples_with_responsibility_utility.jsonl"
    _write_jsonl(source, [_source_row(split="training")])
    _write_jsonl(
        base,
        [
            {
                "sample_id": "video_train_0001|0",
                "split": "validation",
                "dense_len": 5,
                "valid_len": 5,
                "action_target": [0.0, 1.0, 1.0, 1.0, 0.0],
                "frame_signals": {"p_action": [0.1, 0.7, 0.4, 0.2, 0.6]},
            }
        ],
    )

    with pytest.raises(ValueError, match="base_samples_jsonl:1: split must be train/training"):
        exporter.run_export(source, output, manifest=_train_manifest(), base_samples_jsonl=base)


def test_missing_point_fields_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "missing_fields.jsonl"
    output = tmp_path / "missing_fields.utility.jsonl"
    _write_jsonl(source, [_source_row(points=[_responsibility_point(support_end=None)])])

    with pytest.raises(ValueError, match="support_end"):
        exporter.run_export(source, output, manifest=_train_manifest())


def test_dense_utility_output_length_is_dense_len(tmp_path: Path) -> None:
    source = tmp_path / "points.jsonl"
    output = tmp_path / "utility.jsonl"
    summary = tmp_path / "utility.summary.json"
    _write_jsonl(
        source,
        [
            _source_row(
                dense_len=6,
                points=[
                    _responsibility_point(support_start=1, support_end=2, positive_gain=0.4, negative_risk=0.0),
                    _responsibility_point(true_time_center=5, support_start=5, support_end=5, positive_gain=0.1, negative_risk=0.6),
                ],
            )
        ],
    )

    exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest())
    validator.validate_responsibility_utility_export(summary, output_jsonl=output)
    row = _read_jsonl(output)[0]

    assert len(row["positive_observation_gain"]) == 6
    assert len(row["negative_observation_risk"]) == 6
    assert len(row["signed_frame_utility"]) == 6
    assert row["positive_observation_gain"][1] == pytest.approx(0.4)
    assert row["negative_observation_risk"][5] == pytest.approx(0.6)
    assert row["signed_frame_utility"][5] == pytest.approx(-0.5)


def test_responsibility_export_can_merge_base_paction_samples_for_stage2_training(tmp_path: Path) -> None:
    source = tmp_path / "points.jsonl"
    base = tmp_path / "base_samples.jsonl"
    output = tmp_path / "samples_with_responsibility_utility.jsonl"
    summary = tmp_path / "responsibility_utility.summary.json"
    _write_jsonl(source, [_source_row()])
    _write_jsonl(
        base,
        [
            {
                "sample_id": "video_train_0001|0",
                "split": "training",
                "dense_len": 5,
                "valid_len": 5,
                "action_target": [0.0, 1.0, 1.0, 1.0, 0.0],
                "frame_signals": {"p_action": [0.1, 0.7, 0.4, 0.2, 0.6]},
            }
        ],
    )

    exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest(), base_samples_jsonl=base)
    validator.validate_responsibility_utility_export(summary, output_jsonl=output)
    prepared = train_detector._prepared_rows(_read_jsonl(output), dynamic_budget_buckets=[2, 4], expected_split="training")

    assert prepared[0]["detector_utility_target"] == pytest.approx([0.0, 0.5, 0.5, 0.5, 0.0])
    assert prepared[0]["positive_observation_gain_target"] == pytest.approx([0.0, 0.5, 0.5, 0.5, 0.0])
    assert prepared[0]["negative_observation_risk_target"] == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


def test_responsibility_stage2_policy_summary_preserves_source_semantics(tmp_path: Path) -> None:
    source = tmp_path / "points.jsonl"
    base = tmp_path / "base_samples.jsonl"
    output = tmp_path / "samples_with_responsibility_utility.jsonl"
    summary = tmp_path / "responsibility_utility.summary.json"
    _write_jsonl(source, [_source_row()])
    _write_jsonl(
        base,
        [
            {
                "sample_id": "video_train_0001|0",
                "split": "training",
                "dense_len": 5,
                "valid_len": 5,
                "action_target": [0.0, 1.0, 1.0, 1.0, 0.0],
                "frame_signals": {"p_action": [0.1, 0.7, 0.4, 0.2, 0.6]},
            }
        ],
    )
    exporter.run_export(source, output, summary_json=summary, manifest=_train_manifest(), base_samples_jsonl=base)
    contract = train_detector._teacher_utility_contract(_read_jsonl(output))

    assert contract["utility_semantics"] == exporter.UTILITY_SEMANTICS
    assert contract["utility_source_type"] == exporter.UTILITY_SOURCE_TYPE
    assert contract["point_responsibility_utility"] is True
    assert contract["proposal_score_surrogate_utility"] is False


def test_teacher_points_can_generate_responsibility_utility(tmp_path: Path) -> None:
    teacher_points = tmp_path / "dense_teacher_points.jsonl"
    base = tmp_path / "base_samples.jsonl"
    responsibility_points = tmp_path / "responsibility_points.jsonl"
    responsibility_manifest = tmp_path / "responsibility_points.manifest.json"
    responsibility_utility = tmp_path / "samples_with_responsibility_utility.jsonl"
    responsibility_summary = tmp_path / "responsibility_utility.summary.json"
    _write_jsonl(
        teacher_points,
        [
            {
                "sample_id": "video_train_0001|0",
                "split": "training",
                "dense_len": 12,
                "valid_len": 12,
                "teacher_dense_points": [
                    {
                        "point_index": 4,
                        "proposal_score": 0.8,
                        "classification_score": 0.8,
                        "segment_start": 3.0,
                        "segment_end": 7.0,
                    },
                    {
                        "point_index": 10,
                        "proposal_score": 0.6,
                        "classification_score": 0.6,
                        "segment_start": 9.0,
                        "segment_end": 11.0,
                    },
                ],
            }
        ],
    )
    _write_jsonl(
        base,
        [
            {
                "sample_id": "video_train_0001|0",
                "split": "training",
                "dense_len": 12,
                "valid_len": 12,
                "action_target": [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                "gt_segments": [[3.0, 7.0]],
                "frame_signals": {"p_action": [0.1] * 12},
            }
        ],
    )

    points_manifest = points_from_teacher.convert_teacher_points_to_responsibility(
        teacher_points,
        base,
        responsibility_points,
        manifest_json=responsibility_manifest,
    )
    assert points_manifest["decision"] == points_from_teacher.READY

    export_summary = exporter.run_export(
        responsibility_points,
        responsibility_utility,
        summary_json=responsibility_summary,
        manifest=_train_manifest(),
        base_samples_jsonl=base,
    )
    evidence = validator.validate_responsibility_utility_export(
        responsibility_summary,
        output_jsonl=responsibility_utility,
    )
    rows = _read_jsonl(responsibility_utility)

    assert export_summary["utility_source_type"] == "point_loss_gradient_responsibility_v1"
    assert evidence["decision"] == "ADATAD_RESPONSIBILITY_UTILITY_VALIDATION_PASS"
    assert rows[0]["positive_observation_gain"][3] > 0.0
    assert rows[0]["negative_observation_risk"][10] > 0.0


def _write_responsibility_policy_summary(tmp_path: Path, **extra: object) -> Path:
    validator_module = _load_stage23_validator()
    train_jsonl = tmp_path / "train_with_responsibility.jsonl"
    checkpoint = tmp_path / "policy.pth"
    summary = tmp_path / "policy.summary.json"
    _write_jsonl(train_jsonl, [{"sample_id": "video_train_0001|0"}])
    checkpoint.write_bytes(b"detector-aware-policy")
    payload = {
        "decision": "C3_DETECTOR_AWARE_POLICY_TRAIN_READY",
        "policy_family": "detector_aware_offline_selector",
        "utility_semantics": exporter.UTILITY_SEMANTICS,
        "utility_source_type": exporter.UTILITY_SOURCE_TYPE,
        "point_responsibility_utility": True,
        "proposal_score_surrogate_utility": False,
        "signed_utility_supported": True,
        "teacher_target_scope": "train_only",
        "end_to_end": False,
        "dynamic_gain_calibration": {
            "schema_version": "c3_detector_aware_dynamic_gain_calibration_v1",
            "calibration_fitted": True,
            "fit_split": "training",
            "budget_target_rule": "count_positive_gain_at_global_threshold_then_nearest_bucket",
        },
        "train_jsonl": str(train_jsonl),
        "train_jsonl_sha256": validator_module._sha256_file(train_jsonl),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": validator_module._sha256_file(checkpoint),
    }
    payload.update(extra)
    summary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def test_stage2_precheck_accepts_responsibility_policy_summary(tmp_path: Path) -> None:
    validator_module = _load_stage23_validator()
    summary = _write_responsibility_policy_summary(tmp_path)

    evidence = validator_module._validate_policy_summary(summary, checkpoint_path=None)

    assert evidence["checkpoint_path"].endswith("policy.pth")


def test_stage2_precheck_rejects_mislabeled_responsibility_policy_summary(tmp_path: Path) -> None:
    validator_module = _load_stage23_validator()
    summary = _write_responsibility_policy_summary(
        tmp_path,
        utility_semantics="signed_detector_utility_v1",
    )

    with pytest.raises(AssertionError, match="point responsibility source requires"):
        validator_module._validate_policy_summary(summary, checkpoint_path=None)
