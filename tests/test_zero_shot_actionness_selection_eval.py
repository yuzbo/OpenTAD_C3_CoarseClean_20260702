from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import run_zero_shot_actionness_selection_eval as selection_eval
from tools.bata import validate_zero_shot_selection_eval as validator


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


def _annotation() -> dict:
    return {
        "database": {
            "v_alpha": {
                "duration": 8.0,
                "annotations": [
                    {"segment": [1.0, 2.0], "label": "short"},
                    {"segment": [4.0, 7.0], "label": "long"},
                ],
            },
            "v_beta": {
                "duration": 6.0,
                "annotations": [{"segment": [2.0, 5.0], "label": "action"}],
            },
        }
    }


def _actionness_rows() -> list[dict]:
    rows: list[dict] = []
    scores = {
        "v_alpha": [0.05, 0.92, 0.20, 0.10, 0.85, 0.80, 0.75, 0.05],
        "v_beta": [0.05, 0.10, 0.88, 0.84, 0.76, 0.08],
    }
    for video_id, values in scores.items():
        for idx, value in enumerate(values):
            rows.append(
                {
                    "schema_version": "zero_shot_actionness_eval_v1",
                    "video_id": video_id,
                    "window_id": f"{video_id}_{idx}",
                    "time_index": idx,
                    "original_time": float(idx),
                    "p_action": float(value),
                    "boundary_score": 1.0 if idx in {1, 4} else 0.0,
                    "selection_priority_score": (1.0 if idx in {1, 4} else 0.0) + 0.05 * float(value),
                    "logit": float(value),
                    "valid": True,
                    "source_name": "manual_jsonl",
                    "source_provenance": {
                        "source_name": "manual_jsonl",
                        "uses_labels": False,
                        "uses_teacher": False,
                        "uses_gt": False,
                        "uses_prediction_cache": False,
                        "uses_raw_prediction": False,
                    },
                    "prompt_hash": None,
                    "checkpoint_hash": None,
                    "thumos_trained": None,
                    "uses_labels": False,
                    "uses_teacher": False,
                    "calibration_split": None,
                }
            )
    return rows


def _assert_sparse_grid_valid(row: dict) -> None:
    assert selection_eval.validate_sparse_temporal_grid_row(row) == "pass"


def test_selection_eval_generates_all_baselines_and_duca_valid_positions(tmp_path: Path) -> None:
    annotation = tmp_path / "anno.json"
    actionness_jsonl = tmp_path / "actionness.jsonl"
    audit_jsonl = tmp_path / "audit.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_json(annotation, _annotation())
    _write_jsonl(actionness_jsonl, _actionness_rows())

    summary = selection_eval.run_selection_eval(
        annotation_json=annotation,
        actionness_jsonl=actionness_jsonl,
        audit_jsonl=audit_jsonl,
        summary_json=summary_json,
        budget=3,
        baselines=["uniform", "random", "motion", "manual", "boundary-first", "oracle-actionness"],
        random_seed=7,
        boundary_radius=1,
    )

    rows = _read_jsonl(audit_jsonl)
    assert {(row["baseline"], row["video_id"]) for row in rows} == {
        ("uniform", "v_alpha"),
        ("uniform", "v_beta"),
        ("random", "v_alpha"),
        ("random", "v_beta"),
        ("motion", "v_alpha"),
        ("motion", "v_beta"),
        ("manual", "v_alpha"),
        ("manual", "v_beta"),
        ("boundary-first", "v_alpha"),
        ("boundary-first", "v_beta"),
        ("oracle-actionness", "v_alpha"),
        ("oracle-actionness", "v_beta"),
    }
    for row in rows:
        assert row["ledger_role"] == "audit_only_or_baseline_selection_artifact"
        assert row["selected_count"] <= 3
        assert row["budget_violation"] is False
        assert row["selected_positions"] == sorted(set(row["selected_positions"]))
        _assert_sparse_grid_valid(row)
        for key in (
            "action_touched_recall",
            "boundary_radius_recall",
            "short_action_recall",
            "action_interior_coverage",
            "max_hole",
            "p95_hole",
            "redundancy",
            "uniform_similarity",
        ):
            assert key in row
    oracle_rows = [row for row in rows if row["baseline"] == "oracle-actionness"]
    assert all(row["diagnostic_only"] is True for row in oracle_rows)
    assert all(row["uses_gt_for_selection"] is True for row in oracle_rows)
    assert summary["deployable_claim_baselines"] == ["uniform", "random", "motion", "manual", "boundary-first"]
    assert "oracle-actionness" not in summary["deployable_claim_baselines"]
    assert set(_read_json(summary_json)["baseline_summaries"]) == {
        "uniform",
        "random",
        "motion",
        "manual",
        "boundary-first",
        "oracle-actionness",
    }
    validator.validate_selection_eval(audit_jsonl=audit_jsonl, summary_json=summary_json)


def test_boundary_first_scores_prioritize_boundary_fields_over_actionness() -> None:
    rows = [
        {"video_id": "v", "p_action": 0.95, "boundary_score": 0.0},
        {"video_id": "v", "p_action": 0.10, "boundary_score": 1.0},
        {"video_id": "v", "p_action": 0.90, "boundary_score": 0.0},
        {"video_id": "v", "p_action": 0.80, "boundary_score": 0.0},
        {"video_id": "v", "p_action": 0.05, "boundary_score": 0.8},
    ]

    scores = selection_eval._selection_scores_for_baseline(
        baseline="boundary-first",
        rows=rows,
        labels=[0, 0, 1, 1, 0],
    )

    assert sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)[:2] == [1, 4]


def test_selection_validator_rejects_oracle_without_diagnostic_flag(tmp_path: Path) -> None:
    audit_jsonl = tmp_path / "audit.jsonl"
    summary_json = tmp_path / "summary.json"
    row = {
        "schema_version": selection_eval.AUDIT_SCHEMA_VERSION,
        "video_id": "v_alpha",
        "baseline": "oracle-actionness",
        "valid_len": 4,
        "budget": 2,
        "selected_positions": [1, 2],
        "selected_count": 2,
        "ledger_role": "audit_only_or_baseline_selection_artifact",
        "diagnostic_only": False,
        "uses_gt_for_selection": True,
        "uses_labels": False,
        "uses_teacher": False,
        "uses_raw_prediction": False,
        "budget_violation": False,
    }
    _write_jsonl(audit_jsonl, [row])
    _write_json(summary_json, {"schema_version": selection_eval.SUMMARY_SCHEMA_VERSION, "row_count": 1})

    with pytest.raises(ValueError, match="oracle-actionness"):
        validator.validate_selection_eval(audit_jsonl=audit_jsonl, summary_json=summary_json)
