from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import eval_zero_shot_actionness as actionness
from tools.bata import validate_zero_shot_actionness_eval as validator


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _toy_annotation() -> dict:
    return {
        "database": {
            "v_alpha": {
                "duration": 4.0,
                "subset": "test",
                "annotations": [{"segment": [2.0, 4.0], "label": "action"}],
            },
            "v_beta": {
                "duration": 4.0,
                "subset": "test",
                "annotations": [{"segment": [1.0, 3.0], "label": "action"}],
            },
        }
    }


def test_manual_jsonl_eval_writes_schema_and_threshold_free_metrics(tmp_path: Path) -> None:
    annotation = tmp_path / "anno.json"
    samples = tmp_path / "samples.jsonl"
    manual = tmp_path / "manual.jsonl"
    output = tmp_path / "actionness.jsonl"
    summary = tmp_path / "summary.json"
    _write_json(annotation, _toy_annotation())
    _write_jsonl(
        samples,
        [
            {"video_id": "v_alpha", "window_id": "a0", "time_index": 0, "original_time": 0.0},
            {"video_id": "v_alpha", "window_id": "a1", "time_index": 1, "original_time": 1.0},
            {"video_id": "v_alpha", "window_id": "a2", "time_index": 2, "original_time": 2.0},
            {"video_id": "v_alpha", "window_id": "a3", "time_index": 3, "original_time": 3.0},
            {"video_id": "v_beta", "window_id": "b0", "time_index": 0, "original_time": 0.0},
            {"video_id": "v_beta", "window_id": "b1", "time_index": 1, "original_time": 1.0},
            {"video_id": "v_beta", "window_id": "b2", "time_index": 2, "original_time": 2.0},
            {"video_id": "v_beta", "window_id": "b3", "time_index": 3, "original_time": 3.0},
        ],
    )
    _write_jsonl(
        manual,
        [
            {"video_id": "v_alpha", "window_id": "a0", "p_action": 0.10},
            {"video_id": "v_alpha", "window_id": "a1", "p_action": 0.20},
            {"video_id": "v_alpha", "window_id": "a2", "p_action": 0.95},
            {"video_id": "v_alpha", "window_id": "a3", "p_action": 0.90},
            {"video_id": "v_beta", "window_id": "b0", "p_action": 0.05},
            {"video_id": "v_beta", "window_id": "b1", "p_action": 0.85},
            {"video_id": "v_beta", "window_id": "b2", "p_action": 0.80},
            {"video_id": "v_beta", "window_id": "b3", "p_action": 0.15},
        ],
    )

    result = actionness.run_eval(
        annotation_json=annotation,
        sample_jsonl=samples,
        output_jsonl=output,
        summary_json=summary,
        source_mode="manual_jsonl",
        manual_jsonl=manual,
        recall_k=[4],
    )

    rows = _read_jsonl(output)
    assert len(rows) == 8
    required = {
        "video_id",
        "window_id",
        "time_index",
        "original_time",
        "p_action",
        "logit",
        "valid",
        "source_name",
        "source_provenance",
        "prompt_hash",
        "checkpoint_hash",
        "thumos_trained",
        "uses_labels",
        "uses_teacher",
        "calibration_split",
    }
    assert required.issubset(rows[0])
    assert {row["gt_action"] for row in rows} == {0, 1}
    assert all(row["uses_labels"] is False for row in rows)
    assert all(row["uses_teacher"] is False for row in rows)
    assert result["metrics"]["auroc"] == pytest.approx(1.0)
    assert result["metrics"]["auprc"] == pytest.approx(1.0)
    assert result["metrics"]["recall_at_k"]["4"] == pytest.approx(1.0)
    assert result["metrics"]["precision_at_k"]["4"] == pytest.approx(1.0)
    assert result["metrics"]["action_background_balance"] == {
        "action": 4,
        "background": 4,
        "action_fraction": 0.5,
    }
    assert _read_json(summary)["primary_metric_policy"] == "threshold_free"
    validator.validate_eval(actionness_jsonl=output, summary_json=summary)


def test_manual_unknown_provenance_does_not_claim_no_thumos_training(tmp_path: Path) -> None:
    annotation = tmp_path / "anno.json"
    samples = tmp_path / "samples.jsonl"
    manual = tmp_path / "manual.jsonl"
    output = tmp_path / "actionness.jsonl"
    summary = tmp_path / "summary.json"
    _write_json(annotation, _toy_annotation())
    _write_jsonl(samples, [{"video_id": "v_alpha", "window_id": "a0", "time_index": 0, "original_time": 0.0}])
    _write_jsonl(manual, [{"video_id": "v_alpha", "window_id": "a0", "p_action": 0.42}])

    actionness.run_eval(
        annotation_json=annotation,
        sample_jsonl=samples,
        output_jsonl=output,
        summary_json=summary,
        source_mode="manual_jsonl",
        manual_jsonl=manual,
    )

    row = _read_jsonl(output)[0]
    assert row["source_name"] == "manual_jsonl"
    assert row["thumos_trained"] is None
    assert row["source_provenance"]["thumos_trained"] is None


def test_manual_jsonl_preserves_declared_frozen_x3d_provenance(tmp_path: Path) -> None:
    annotation = tmp_path / "anno.json"
    samples = tmp_path / "samples.jsonl"
    manual = tmp_path / "manual.jsonl"
    output = tmp_path / "actionness.jsonl"
    summary = tmp_path / "summary.json"
    provenance = {
        "source_name": "frozen_kinetics_x3d_xs_actionness",
        "source_mode": "frozen_kinetics_classifier_confidence",
        "provider": "x3d_xs",
        "training_dataset": "Kinetics",
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_gt": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "calibration_split": "none",
        "checkpoint_hash": "pytorch_provider:x3d_xs:pretrained=True",
        "prompt_hash": None,
    }
    _write_json(annotation, _toy_annotation())
    _write_jsonl(samples, [{"video_id": "v_alpha", "window_id": "a0", "time_index": 0, "original_time": 0.0}])
    _write_jsonl(
        manual,
        [
            {
                "video_id": "v_alpha",
                "window_id": "a0",
                "p_action": 0.42,
                "source_name": provenance["source_name"],
                "source_provenance": provenance,
                "checkpoint_hash": provenance["checkpoint_hash"],
                "prompt_hash": None,
            }
        ],
    )

    actionness.run_eval(
        annotation_json=annotation,
        sample_jsonl=samples,
        output_jsonl=output,
        summary_json=summary,
        source_mode="manual_jsonl",
        manual_jsonl=manual,
    )

    row = _read_jsonl(output)[0]
    assert row["source_name"] == "frozen_kinetics_x3d_xs_actionness"
    assert row["source_provenance"]["provider"] == "x3d_xs"
    assert row["source_provenance"]["training_dataset"] == "Kinetics"
    assert row["thumos_trained"] is False
    assert row["checkpoint_hash"] == "pytorch_provider:x3d_xs:pretrained=True"


def test_actionness_validator_rejects_teacher_gt_cache_and_raw_prediction_payloads(tmp_path: Path) -> None:
    actionness_jsonl = tmp_path / "bad_actionness.jsonl"
    summary = tmp_path / "summary.json"
    clean = {
        "schema_version": actionness.OUTPUT_SCHEMA_VERSION,
        "video_id": "v_alpha",
        "window_id": "a0",
        "time_index": 0,
        "original_time": 0.0,
        "p_action": 0.5,
        "logit": 0.0,
        "valid": True,
        "source_name": "motion",
        "source_provenance": {
            "source_name": "motion",
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "uses_raw_prediction": False,
        },
        "prompt_hash": None,
        "checkpoint_hash": None,
        "thumos_trained": False,
        "uses_labels": False,
        "uses_teacher": False,
        "calibration_split": None,
    }
    _write_json(summary, {"schema_version": actionness.SUMMARY_SCHEMA_VERSION, "row_count": 1})

    for forbidden_key in ("teacher", "gt_segments", "prediction_cache", "raw_prediction"):
        bad = dict(clean)
        bad["source_payload"] = {forbidden_key: [1, 2, 3]}
        _write_jsonl(actionness_jsonl, [bad])
        with pytest.raises(ValueError, match=forbidden_key):
            validator.validate_eval(actionness_jsonl=actionness_jsonl, summary_json=summary)
