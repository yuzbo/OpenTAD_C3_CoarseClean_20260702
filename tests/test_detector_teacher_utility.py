from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import detector_teacher_utility as teacher_utility


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_generator_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "c3_detector_teacher_utility_generator_manifest_v1",
                "decision": "C3_DETECTOR_TEACHER_UTILITY_GENERATOR_MANIFEST_READY",
                "teacher_signal_source": "adatad_dense_teacher",
                "generator_source": "dense_detector_forward_train",
                "split_scope": "train_only",
                "input_split": "training",
                "uses_evaluator_outputs": False,
                "uses_raw_prediction": False,
                "uses_prediction_cache": False,
                "load_from_raw_predictions": False,
                "uses_val_or_test_gt_for_selection": False,
                "uses_gt_for_selection": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_dense_point_utility_maps_to_normalized_frame_utility_with_contract() -> None:
    points = [
        {"point_index": 1, "classification_score": 0.8, "localization_quality": 0.5, "centerness": 1.0},
        {"point_index": 3, "classification_score": 0.4, "localization_quality": 1.0, "centerness": 0.5},
        {"point_index": 4, "proposal_score": 0.9},
    ]

    frame_utility = teacher_utility.map_dense_points_to_frame_utility(points, dense_len=6, valid_len=5)

    assert len(frame_utility) == 6
    assert frame_utility[4] == pytest.approx(1.0)
    assert frame_utility[1] > frame_utility[3] > frame_utility[0]
    assert frame_utility[5] == 0.0


def test_signed_detector_utility_preserves_background_suppression_value() -> None:
    points = [
        {"point_index": 1, "classification_score": 0.8, "localization_quality": 0.5},
        {"point_index": 3, "signed_utility": -0.6, "utility_role": "background_suppression"},
    ]

    signed = teacher_utility.map_dense_points_to_signed_frame_utility(points, dense_len=5, valid_len=4)

    assert len(signed) == 5
    assert signed[1] > 0.0
    assert signed[3] < 0.0
    assert max(abs(value) for value in signed[:4]) == pytest.approx(1.0)
    assert signed[4] == 0.0


def test_actionformer_predictions_map_to_dense_teacher_points_without_gt() -> None:
    dense_points = teacher_utility.actionformer_predictions_to_dense_points(
        proposals=[[0.2, 1.8], [3.2, 4.7], [10.0, 12.0], ["bad", 2.0]],
        scores=[[0.1, 0.9], {"classification_score": 0.4}, 0.2, 1.0],
        dense_len=6,
        valid_len=5,
    )

    assert [point["point_index"] for point in dense_points] == [1, 4]
    assert dense_points[0]["proposal_score"] == pytest.approx(0.9)
    assert dense_points[0]["teacher_signal_source"] == "adatad_dense_teacher_actionformer_prediction"


def test_teacher_utility_export_accepts_actionformer_prediction_rows(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "actionformer_predictions.jsonl"
    output_jsonl = tmp_path / "teacher_utility.jsonl"
    _write_jsonl(
        input_jsonl,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 6,
                "valid_len": 5,
                "teacher_proposals": [[0.0, 2.0], [3.0, 5.0]],
                "teacher_scores": [[0.2, 0.8], [0.3, 0.6]],
            }
        ],
    )

    teacher_utility.run_export(input_jsonl, output_jsonl, expected_split="training")
    exported = _read_jsonl(output_jsonl)

    assert exported[0]["frame_utility"][1] == pytest.approx(1.0)
    assert exported[0]["frame_utility"][4] > 0.0
    assert exported[0]["uses_raw_prediction"] is False
    assert exported[0]["prediction_uses_gt"] is False


def test_teacher_utility_export_rejects_val_or_gt_leakage_and_writes_jsonl_npz(tmp_path: Path) -> None:
    input_jsonl = tmp_path / "dense_teacher.jsonl"
    output_jsonl = tmp_path / "teacher_utility.jsonl"
    output_npz = tmp_path / "teacher_utility.npz"
    summary_json = tmp_path / "summary.json"
    rows = [
        {
            "sample_id": "video_test_0001|0",
            "split": "training",
            "dense_len": 5,
            "valid_len": 5,
            "teacher_dense_points": [
                {"point_index": 0, "classification_score": 0.2, "localization_quality": 0.5},
                {"point_index": 2, "classification_score": 0.9, "localization_quality": 0.8},
            ],
        }
    ]
    _write_jsonl(input_jsonl, rows)

    summary = teacher_utility.run_export(
        input_jsonl,
        output_jsonl,
        summary_json=summary_json,
        output_npz=output_npz,
        expected_split="training",
    )
    exported = _read_jsonl(output_jsonl)

    assert summary["decision"] == "C3_DETECTOR_TEACHER_UTILITY_EXPORT_READY"
    assert summary["schema_version"] == "c3_detector_teacher_utility_export_v1"
    assert summary["stage_label"] == "Stage-2 detector-aware offline selector"
    assert summary["uses_val_or_test_gt_for_selection"] is False
    assert output_npz.is_file()
    assert exported[0]["schema_version"] == "c3_detector_teacher_utility_row_v1"
    assert exported[0]["teacher_utility_provenance"]["teacher_signal_source"] == "adatad_dense_teacher"
    assert exported[0]["teacher_utility_provenance"]["split_scope"] == "train_only"
    assert exported[0]["uses_gt"] is False
    assert exported[0]["uses_teacher"] is True
    assert exported[0]["training_only"] is True
    assert exported[0]["frame_utility"][2] == pytest.approx(1.0)
    assert exported[0]["teacher_utility"]["utility_semantics"] == "signed_detector_utility_v1"
    assert exported[0]["teacher_utility"]["signed_frame_utility"][2] == pytest.approx(1.0)
    assert summary["utility_semantics"] == "signed_detector_utility_v1"
    assert summary["signed_utility_supported"] is True

    rows[0]["split"] = "validation"
    _write_jsonl(input_jsonl, rows)
    with pytest.raises(ValueError, match="expected split training"):
        teacher_utility.run_export(input_jsonl, output_jsonl, expected_split="training")

    rows[0]["split"] = "training"
    rows[0]["uses_gt"] = True
    _write_jsonl(input_jsonl, rows)
    with pytest.raises(ValueError, match="forbidden teacher source flag uses_gt=true"):
        teacher_utility.run_export(input_jsonl, output_jsonl, expected_split="training")


def test_teacher_utility_export_merges_base_samples_and_validates_evidence(tmp_path: Path) -> None:
    dense_points = tmp_path / "teacher_dense_points.jsonl"
    base_samples = tmp_path / "train_paction_samples.jsonl"
    output_jsonl = tmp_path / "samples_with_teacher_utility.jsonl"
    summary_json = tmp_path / "teacher_utility_export.summary.json"
    checkpoint = tmp_path / "teacher.pth"
    config = tmp_path / "teacher.py"
    manifest = tmp_path / "teacher_generator.manifest.json"
    checkpoint.write_bytes(b"teacher checkpoint")
    config.write_text("model = dict(type='AdaTAD')\n", encoding="utf-8")
    _write_generator_manifest(manifest)
    _write_jsonl(
        dense_points,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 4,
                "valid_len": 4,
                "teacher_dense_points": [{"point_index": 1, "proposal_score": 0.8}],
            }
        ],
    )
    _write_jsonl(
        base_samples,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 4,
                "valid_len": 4,
                "frame_signals": {"p_action": [0.1, 0.8, 0.2, 0.7]},
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
            }
        ],
    )

    summary = teacher_utility.run_export(
        dense_points,
        output_jsonl,
        summary_json=summary_json,
        base_samples_jsonl=base_samples,
        teacher_checkpoint_path=checkpoint,
        teacher_config_path=config,
        generator_manifest_json=manifest,
        expected_split="training",
    )
    rows = _read_jsonl(output_jsonl)
    evidence = teacher_utility.validate_teacher_utility_export_evidence(
        summary_json,
        output_jsonl=output_jsonl,
        require_paction=True,
    )

    assert rows[0]["frame_signals"]["p_action"] == [0.1, 0.8, 0.2, 0.7]
    assert rows[0]["teacher_utility"]["frame_utility"][1] == pytest.approx(1.0)
    assert summary["teacher_checkpoint_sha256"] == teacher_utility._sha256_file(checkpoint)
    assert summary["teacher_config_sha256"] == teacher_utility._sha256_file(config)
    assert summary["generator_manifest_sha256"] == teacher_utility._sha256_file(manifest)
    assert evidence["decision"] == "C3_DETECTOR_TEACHER_UTILITY_EVIDENCE_PASS"

    output_jsonl.write_text(output_jsonl.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="output_jsonl_sha256 mismatch"):
        teacher_utility.validate_teacher_utility_export_evidence(summary_json, output_jsonl=output_jsonl)


def test_teacher_utility_evidence_gate_fails_closed_without_provenance(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "samples_with_teacher_utility.jsonl"
    summary_json = tmp_path / "teacher_utility_export.summary.json"
    _write_jsonl(
        output_jsonl,
        [
            {
                "schema_version": "c3_detector_teacher_utility_row_v1",
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 2,
                "valid_len": 2,
                "frame_utility": [0.0, 1.0],
                "teacher_utility": {"frame_utility": [0.0, 1.0]},
                "teacher_utility_provenance": {"split_scope": "train_only"},
            }
        ],
    )
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "c3_detector_teacher_utility_export_v1",
                "decision": "C3_DETECTOR_TEACHER_UTILITY_EXPORT_READY",
                "output_jsonl": str(output_jsonl),
                "output_jsonl_sha256": teacher_utility._sha256_file(output_jsonl),
                "teacher_signal_source": "not_adatad",
                "split_scope": "train_only",
                "uses_val_or_test_gt_for_selection": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="teacher_signal_source"):
        teacher_utility.validate_teacher_utility_export_evidence(summary_json, output_jsonl=output_jsonl)


def test_teacher_utility_evidence_requires_generator_manifest_when_claimed(tmp_path: Path) -> None:
    output_jsonl = tmp_path / "samples_with_teacher_utility.jsonl"
    summary_json = tmp_path / "teacher_utility_export.summary.json"
    checkpoint = tmp_path / "teacher.pth"
    config = tmp_path / "teacher.py"
    checkpoint.write_bytes(b"teacher")
    config.write_text("model = dict(type='AdaTAD')\n", encoding="utf-8")
    _write_jsonl(
        output_jsonl,
        [
            {
                "schema_version": "c3_detector_teacher_utility_row_v1",
                "sample_id": "video_test_0001|0",
                "split": "training",
                "dense_len": 2,
                "valid_len": 2,
                "frame_utility": [0.0, 1.0],
                "signed_frame_utility": [0.0, 1.0],
                "teacher_utility": {
                    "utility_semantics": "signed_detector_utility_v1",
                    "frame_utility": [0.0, 1.0],
                    "signed_frame_utility": [0.0, 1.0],
                },
                "teacher_utility_provenance": {
                    "teacher_signal_source": "adatad_dense_teacher",
                    "split_scope": "train_only",
                },
                "uses_teacher": True,
                "training_only": True,
                "end_to_end": False,
            }
        ],
    )
    summary_json.write_text(
        json.dumps(
            {
                "schema_version": "c3_detector_teacher_utility_export_v1",
                "decision": "C3_DETECTOR_TEACHER_UTILITY_EXPORT_READY",
                "stage_label": "Stage-2 detector-aware offline selector",
                "route_label": "DIVERGENT_INNOVATION_DETECTOR_AWARE_UTILITY_DO_NOT_MERGE_WITH_C3",
                "teacher_signal_source": "adatad_dense_teacher",
                "split_scope": "train_only",
                "utility_semantics": "signed_detector_utility_v1",
                "signed_utility_supported": True,
                "teacher_checkpoint_path": str(checkpoint),
                "teacher_checkpoint_sha256": teacher_utility._sha256_file(checkpoint),
                "teacher_config_path": str(config),
                "teacher_config_sha256": teacher_utility._sha256_file(config),
                "output_jsonl": str(output_jsonl),
                "output_jsonl_sha256": teacher_utility._sha256_file(output_jsonl),
                "row_count": 1,
                "uses_val_or_test_gt_for_selection": False,
                "uses_gt_for_selection": False,
                "uses_prediction_cache": False,
                "uses_raw_prediction": False,
                "load_from_raw_predictions": False,
                "end_to_end": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="generator_manifest"):
        teacher_utility.validate_teacher_utility_export_evidence(
            summary_json,
            output_jsonl=output_jsonl,
            require_generator_manifest=True,
        )
