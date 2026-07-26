from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "duca_x3d_formal_only_monitor_v1"
DEPLOYMENT_SCHEMA_VERSION = "duca_x3d_formal_only_deployment_v1"
MATERIALIZATION_SCHEMA_VERSION = "trainfree_x3d_actionness_materialization_v1"
READY_DECISION = "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED"


def _path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _maybe_read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return _read_json(path)


def _read_text_input(value: str | Path | None) -> str:
    if value is None:
        return ""
    path = _path(value)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace")
    return str(value)


def _parse_squeue(value: str | Path | None) -> dict[str, dict[str, str]]:
    text = _read_text_input(value)
    out: dict[str, dict[str, str]] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("jobid"):
            continue
        parts = [part.strip() for part in line.split("|")] if "|" in line else line.split()
        if not parts:
            continue
        job_id = parts[0]
        out[str(job_id)] = {
            "job_id": str(job_id),
            "name": parts[1] if len(parts) > 1 else "",
            "state": parts[2] if len(parts) > 2 else "",
            "reason": parts[3] if len(parts) > 3 else "",
        }
    return out


def _live_squeue_text(job_ids: list[str]) -> str:
    ids = [str(job_id).strip() for job_id in job_ids if str(job_id).strip()]
    if not ids:
        return ""
    try:
        completed = subprocess.run(
            ["squeue", "-h", "-j", ",".join(ids), "-o", "%i|%j|%T|%R"],
            check=False,
            text=True,
            capture_output=True,
            timeout=20,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout


def _state_to_status(state: str) -> str:
    normalized = str(state).strip().upper()
    if normalized in {"R", "RUNNING", "CG", "COMPLETING", "CONFIGURING"}:
        return "running"
    if normalized in {"PD", "PENDING", "CF"}:
        return "pending"
    if normalized in {"CD", "COMPLETED"}:
        return "completed"
    if normalized in {"F", "FAILED", "CA", "CANCELLED", "TO", "TIMEOUT", "NF", "NODE_FAIL", "OOM", "OUT_OF_MEMORY"}:
        return "failed"
    return "finished_or_unknown" if not normalized else "unknown"


def _parse_state_env(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        elif len(value) >= 2 and value[0] == value[-1] == '"':
            value = value[1:-1]
        out[key.strip()] = value
    return out


def _line_count(path: Path) -> int | None:
    if not path.is_file():
        return None
    count = 0
    with path.open("rb") as handle:
        for _ in handle:
            count += 1
    return count


def _export_progress(run_root: Path, deployment: dict[str, Any]) -> dict[str, Any]:
    provider = str(deployment.get("provider") or "x3d_xs")
    clip_frames = deployment.get("clip_frames")
    frame_interval = deployment.get("frame_interval")
    suffix = ""
    if clip_frames is not None and frame_interval is not None:
        suffix = f"_t{clip_frames}x{frame_interval}"
    out_root = run_root / f"formal_{provider}{suffix}"
    log_path = out_root / f"export_{provider}.out"
    jsonl_path = out_root / f"{provider}_validation_actionness.jsonl"
    summary_path = out_root / f"{provider}_validation_actionness.summary.json"
    progress: dict[str, Any] = {
        "out_root": str(out_root),
        "log_path": str(log_path),
        "jsonl_path": str(jsonl_path),
        "summary_path": str(summary_path),
        "jsonl_exists": jsonl_path.is_file(),
        "summary_exists": summary_path.is_file(),
        "jsonl_row_count": _line_count(jsonl_path),
    }
    if log_path.is_file():
        text = log_path.read_text(encoding="utf-8", errors="replace")
        matches = list(
            re.finditer(
                r"video=(\d+)/(\d+)\s+video_id=([^\s]+)\s+rows=(\d+)\s+total_rows=(\d+)\s+elapsed_sec=([0-9.]+)",
                text,
            )
        )
        if matches:
            last = matches[-1]
            done = int(last.group(1))
            total = int(last.group(2))
            progress.update(
                {
                    "videos_done": done,
                    "videos_total": total,
                    "progress_fraction": done / max(total, 1),
                    "last_video_id": last.group(3),
                    "last_video_rows": int(last.group(4)),
                    "total_rows_reported": int(last.group(5)),
                    "elapsed_sec": float(last.group(6)),
                }
            )
    return progress


def _submit_limit_hit(path: Path) -> bool:
    if not path.is_file():
        return False
    return "AssocMaxSubmitJobLimit" in path.read_text(encoding="utf-8", errors="replace")


def monitor_formal_only(
    *,
    deployment_summary: str | Path,
    squeue_text: str | Path | None = None,
) -> dict[str, Any]:
    deployment_path = _path(deployment_summary)
    deployment = _read_json(deployment_path)
    if deployment.get("schema_version") != DEPLOYMENT_SCHEMA_VERSION:
        raise ValueError(f"deployment_summary schema_version must be {DEPLOYMENT_SCHEMA_VERSION}")
    run_root = _path(deployment.get("run_root", deployment_path.parent))
    sbatch_root = run_root / "sbatch"
    materialization_path = _path(deployment.get("formal_x3d_materialization_summary", ""))
    formal_jsonl_path = _path(deployment.get("formal_x3d_actionness_jsonl", ""))
    materialization = _maybe_read_json(materialization_path)
    state = _parse_state_env(sbatch_root / "downstream_submit_state.env")
    export_job = str(state.get("EXPORT_JOB") or deployment.get("export_job") or "")
    fixed_job = str(state.get("FIXED_JOB") or deployment.get("x3d_duca384_job") or "")
    must_job = str(state.get("MUST_JOB") or deployment.get("x3d_must_job") or "")
    if squeue_text is None:
        squeue_text = _live_squeue_text([export_job, fixed_job, must_job])
    squeue = _parse_squeue(squeue_text)
    jobs = {
        "export": {
            "job_id": export_job,
            "status": _state_to_status(squeue.get(export_job, {}).get("state", "")),
            "squeue": squeue.get(export_job),
        },
        "x3d_duca384": {
            "job_id": fixed_job,
            "status": "not_submitted" if not fixed_job else _state_to_status(squeue.get(fixed_job, {}).get("state", "")),
            "squeue": squeue.get(fixed_job),
        },
        "x3d_must": {
            "job_id": must_job,
            "status": "not_submitted" if not must_job else _state_to_status(squeue.get(must_job, {}).get("state", "")),
            "squeue": squeue.get(must_job),
        },
    }
    progress = _export_progress(run_root, deployment)
    formal_jsonl_exists = formal_jsonl_path.is_file()
    materialized = bool(
        materialization
        and materialization.get("schema_version") == MATERIALIZATION_SCHEMA_VERSION
        and materialization.get("decision") == READY_DECISION
        and materialization.get("downstream_detector_ready") is True
        and formal_jsonl_exists
    )
    downstream_submitted = bool(fixed_job and must_job)
    submit_limit = _submit_limit_hit(sbatch_root / "x3d_duca384_submit.err") or _submit_limit_hit(
        sbatch_root / "x3d_must_submit.err"
    )
    blockers: list[str] = []
    if not materialized:
        blockers.append("formal_x3d_not_materialized")
    if not downstream_submitted:
        blockers.append("x3d_downstream_not_submitted")
    if submit_limit:
        blockers.append("slurm_submit_limit")
    status = "ready_for_downstream_results"
    if not materialized:
        status = "export_running" if jobs["export"]["status"] == "running" else "waiting_for_materialization"
    elif not downstream_submitted:
        status = "waiting_for_downstream_submit_limit" if submit_limit else "waiting_for_downstream_submission"
    return {
        "schema_version": SCHEMA_VERSION,
        "deployment_summary": str(deployment_path),
        "run_root": str(run_root),
        "commit": deployment.get("commit"),
        "branch": deployment.get("branch"),
        "status": status,
        "materialized": materialized,
        "formal_jsonl_path": str(formal_jsonl_path),
        "formal_jsonl_exists": formal_jsonl_exists,
        "formal_jsonl_row_count": _line_count(formal_jsonl_path),
        "materialization_summary_path": str(materialization_path),
        "materialization_summary_exists": materialization_path.is_file(),
        "materialization_decision": None if materialization is None else materialization.get("decision"),
        "train_free_baseline": bool(materialization.get("train_free_baseline")) if materialization else None,
        "not_main_method": bool(materialization.get("not_main_method")) if materialization else None,
        "export_progress": progress,
        "jobs": jobs,
        "downstream_submitted": downstream_submitted,
        "submit_limit_hit": submit_limit,
        "blockers": sorted(set(blockers)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor formal train-free X3D materialization and downstream jobs.")
    parser.add_argument("--deployment-summary", required=True)
    parser.add_argument("--squeue-text")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    summary = monitor_formal_only(deployment_summary=args.deployment_summary, squeue_text=args.squeue_text)
    if args.output_json:
        out = _path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary)
        payload["output_json"] = str(out)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = payload
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
