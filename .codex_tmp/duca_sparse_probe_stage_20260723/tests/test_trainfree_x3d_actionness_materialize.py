from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata.materialize_trainfree_x3d_actionness import materialize_actionness


ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _grid_summary(tmp_path: Path, *, out_root: Path) -> Path:
    summary = {
        "schema_version": "trainfree_x3d_interval_grid_summary_v1",
        "decision": "TRAINFREE_X3D_INTERVAL_GRID_SUMMARY_READY",
        "subset": "validation",
        "rows": [
            {
                "provider": "x3d_xs",
                "clip_frames": 4,
                "frame_interval": 2,
                "crop_size": 160,
                "batch_size": 16,
                "out_root": str(out_root),
                "row_count": 2,
                "video_count": 1,
                "uses_original_x3d_clip_window": True,
                "source_provenance": {
                    "source_name": "frozen_kinetics_x3d_xs_actionness",
                    "thumos_trained": False,
                    "uses_labels": False,
                    "uses_teacher": False,
                    "uses_gt": False,
                    "uses_prediction_cache": False,
                    "calibration_split": "none",
                },
            }
        ],
    }
    path = tmp_path / "x3d_interval_grid.summary.json"
    _write_json(path, summary)
    return path


def test_materialize_preregistered_x3d_grid_cell_to_formal_downstream_jsonl(tmp_path: Path) -> None:
    cell_root = tmp_path / "x3d_xs_t4x2"
    rows = [
        {
            "schema_version": "zero_shot_actionness_samples_v1",
            "video_id": "video_test_0001",
            "window_id": "video_test_0001_0000",
            "time_index": 0,
            "p_action": 0.7,
            "logit": 0.847,
            "valid": True,
            "source_name": "frozen_kinetics_x3d_xs_actionness",
        },
        {
            "schema_version": "zero_shot_actionness_samples_v1",
            "video_id": "video_test_0001",
            "window_id": "video_test_0001_0001",
            "time_index": 1,
            "p_action": 0.2,
            "logit": -1.386,
            "valid": True,
            "source_name": "frozen_kinetics_x3d_xs_actionness",
        },
    ]
    _write_jsonl(cell_root / "x3d_xs_validation_actionness.jsonl", rows)
    grid_summary = _grid_summary(tmp_path, out_root=cell_root)
    output_jsonl = tmp_path / "best_x3d_actionness.jsonl"
    output_summary = tmp_path / "best_x3d_actionness.materialization.json"

    summary = materialize_actionness(
        grid_summary_json=grid_summary,
        output_jsonl=output_jsonl,
        summary_json=output_summary,
        provider="x3d_xs",
        frame_interval=2,
        selection_policy="pre_registered",
    )

    assert output_jsonl.read_text(encoding="utf-8") == (cell_root / "x3d_xs_validation_actionness.jsonl").read_text(
        encoding="utf-8"
    )
    assert summary["decision"] == "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED"
    assert summary["selection_policy"] == "pre_registered"
    assert summary["validation_metric_selection"] is False
    assert summary["downstream_detector_ready"] is True
    assert summary["output_jsonl"] == str(output_jsonl)
    assert summary["selected_cell"]["provider"] == "x3d_xs"
    assert json.loads(output_summary.read_text(encoding="utf-8"))["output_sha256"] == summary["output_sha256"]


def test_materialize_rejects_validation_best_selection_without_explicit_opt_in(tmp_path: Path) -> None:
    cell_root = tmp_path / "x3d_xs_t4x2"
    _write_jsonl(cell_root / "x3d_xs_validation_actionness.jsonl", [])
    grid_summary = _grid_summary(tmp_path, out_root=cell_root)

    with pytest.raises(ValueError, match="validation metric selection"):
        materialize_actionness(
            grid_summary_json=grid_summary,
            output_jsonl=tmp_path / "best_x3d_actionness.jsonl",
            selection_policy="best_by_metric",
            metric="coarse_auroc",
        )


def test_interval_grid_runner_materializes_formal_preregistered_jsonl() -> None:
    text = (ROOT / "scripts" / "run_duca_trainfree_x3d_interval_grid_gpu0.sh").read_text(encoding="utf-8")

    assert "materialize_trainfree_x3d_actionness.py" in text
    assert "best_x3d_actionness.jsonl" in text
    assert "--selection-policy pre_registered" in text
    assert "DUCA_X3D_FORMAL_PROVIDER" in text
    assert "DUCA_X3D_FORMAL_FRAME_INTERVAL" in text
    assert "--allow-validation-selection" not in text
