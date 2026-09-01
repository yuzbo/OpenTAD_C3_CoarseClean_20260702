from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import detector_deploy_leakage
from tools.bata import validate_duca_stage23_precheck as duca_precheck


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_detector_deploy_leakage_recurses_nested_teacher_payload() -> None:
    payload = {
        "sample_id": "video_test_0001|0",
        "meta": {
            "frames": [
                {"score": 0.1},
                {"debug": {"dense_teacher_points": [{"t": 3, "score": 0.9}]}},
            ]
        },
    }

    paths = detector_deploy_leakage.find_detector_deploy_forbidden_paths(payload)

    assert paths == ["meta.frames[1].debug.dense_teacher_points"]
    with pytest.raises(ValueError, match="dense_teacher_points"):
        detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
            payload,
            source_name="deploy-row",
        )


def test_detector_deploy_leakage_rejects_forbidden_string_values() -> None:
    payload = {
        "sample_id": "video_test_0001|0",
        "source_path": "/tmp/run/prediction_cache/video_test_0001.json",
    }

    with pytest.raises(ValueError, match="source_path"):
        detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
            payload,
            source_name="deploy-row",
        )


def test_detector_deploy_leakage_allows_false_teacher_safety_metadata() -> None:
    payload = {
        "detector_aware_policy": {
            "teacher_payload_visible_to_deploy": False,
            "selection_uses_teacher": False,
            "teacher_target_scope": "train_only",
        },
        "uses_raw_prediction": False,
        "uses_prediction_cache": False,
        "uses_teacher": False,
    }

    detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
        payload,
        source_name="deploy-row",
    )


def test_detector_deploy_strip_removes_nested_key_and_forbidden_string_value() -> None:
    payload = {
        "sample_id": "video_test_0001|0",
        "debug": [{"teacher_scores": [0.9]}, {"source": "raw_prediction_cache.pkl"}],
        "detector_aware_policy": {"teacher_payload_visible_to_deploy": False},
    }

    stripped = detector_deploy_leakage.strip_detector_deploy_forbidden_payloads(payload)

    assert stripped["debug"] == [{}, {}]
    assert stripped["detector_aware_policy"]["teacher_payload_visible_to_deploy"] is False
    detector_deploy_leakage.reject_detector_deploy_forbidden_payloads(
        stripped,
        source_name="stripped-row",
    )


def test_duca_stage2_val_test_leakage_validation_uses_recursive_scanner(tmp_path: Path) -> None:
    source = tmp_path / "val.jsonl"
    _write_jsonl(
        source,
        [
            {
                "sample_id": "video_test_0001|0",
                "split": "validation",
                "dense_len": 4,
                "audit": {"nested": [{"raw_predictions": [{"score": 0.9}]}]},
            }
        ],
    )

    with pytest.raises(ValueError, match="raw_predictions"):
        duca_precheck._validate_no_teacher_leakage_jsonl(source, split_name="val")
