from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MONITOR = ROOT / "tools" / "bata" / "monitor_duca_x3d_formal_only.py"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _deployment(tmp_path: Path) -> Path:
    run_root = tmp_path / "x3d_formal"
    return _write_json(
        run_root / "deployment_summary_x3d_formal_only.json",
        {
            "schema_version": "duca_x3d_formal_only_deployment_v1",
            "branch": "codex/gas-vt-stage23-detector-aware-20260706",
            "commit": "019c183",
            "run_root": str(run_root),
            "formal_x3d_actionness_jsonl": str(tmp_path / "best_x3d_actionness.jsonl"),
            "formal_x3d_materialization_summary": str(tmp_path / "best_x3d_actionness.materialization.json"),
            "provider": "x3d_xs",
            "clip_frames": 4,
            "frame_interval": 2,
            "export_job": "201",
            "x3d_duca384_job": None,
            "x3d_must_job": None,
        },
    )


def test_x3d_formal_monitor_reports_export_progress_and_submit_limit(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    run_root = deployment.parent
    _write_text(
        run_root / "formal_x3d_xs_t4x2" / "export_x3d_xs.out",
        "\n".join(
            [
                "[FROZEN_KINETICS_ACTIONNESS] video=1/211 video_id=v1 rows=768 total_rows=768 elapsed_sec=10.0",
                "[FROZEN_KINETICS_ACTIONNESS] video=2/211 video_id=v2 rows=768 total_rows=1536 elapsed_sec=20.5",
            ]
        )
        + "\n",
    )
    _write_text(run_root / "formal_x3d_xs_t4x2" / "x3d_xs_validation_actionness.jsonl", "{}\n{}\n")
    _write_text(run_root / "sbatch" / "downstream_submit_state.env", "EXPORT_JOB=201\nFIXED_JOB=''\nMUST_JOB=''\n")
    _write_text(
        run_root / "sbatch" / "x3d_duca384_submit.err",
        "sbatch: error: AssocMaxSubmitJobLimit\n",
    )

    from tools.bata.monitor_duca_x3d_formal_only import monitor_formal_only

    summary = monitor_formal_only(
        deployment_summary=deployment,
        squeue_text="201|x3d_formal|RUNNING|g0013\n",
    )

    assert summary["schema_version"] == "duca_x3d_formal_only_monitor_v1"
    assert summary["status"] == "export_running"
    assert summary["materialized"] is False
    assert summary["submit_limit_hit"] is True
    assert "formal_x3d_not_materialized" in summary["blockers"]
    assert "x3d_downstream_not_submitted" in summary["blockers"]
    assert summary["export_progress"]["videos_done"] == 2
    assert summary["export_progress"]["videos_total"] == 211
    assert summary["export_progress"]["jsonl_row_count"] == 2
    assert summary["jobs"]["export"]["status"] == "running"
    assert summary["jobs"]["x3d_duca384"]["status"] == "not_submitted"


def test_x3d_formal_monitor_reports_materialized_downstream_submission_and_cli(tmp_path: Path) -> None:
    deployment = _deployment(tmp_path)
    payload = json.loads(deployment.read_text(encoding="utf-8"))
    payload["x3d_duca384_job"] = "202"
    payload["x3d_must_job"] = "203"
    deployment.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    formal_jsonl = Path(payload["formal_x3d_actionness_jsonl"])
    materialization = Path(payload["formal_x3d_materialization_summary"])
    _write_text(formal_jsonl, "{}\n{}\n{}\n")
    _write_json(
        materialization,
        {
            "schema_version": "trainfree_x3d_actionness_materialization_v1",
            "decision": "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED",
            "downstream_detector_ready": True,
            "train_free_baseline": True,
            "not_main_method": True,
        },
    )
    output = tmp_path / "monitor.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(MONITOR),
            "--deployment-summary",
            str(deployment),
            "--squeue-text",
            "202|x3d_duca384|PENDING|Dependency\n203|x3d_must|PENDING|Dependency\n",
            "--output-json",
            str(output),
        ],
        cwd=str(ROOT),
        check=True,
        text=True,
        capture_output=True,
    )

    stdout_payload = json.loads(completed.stdout)
    file_payload = json.loads(output.read_text(encoding="utf-8"))
    assert stdout_payload["status"] == "ready_for_downstream_results"
    assert file_payload["materialized"] is True
    assert file_payload["downstream_submitted"] is True
    assert file_payload["formal_jsonl_row_count"] == 3
    assert file_payload["train_free_baseline"] is True
    assert file_payload["not_main_method"] is True
    assert not file_payload["blockers"]
