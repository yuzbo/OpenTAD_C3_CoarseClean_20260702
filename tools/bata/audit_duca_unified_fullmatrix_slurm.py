from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from tools.bata.aggregate_duca_unified_fullmatrix import build_summary


JOB_ID_KEYS = (
    "preflight_job_id",
    "train_eval_array_job_id",
    "cost_array_job_id",
    "bootstrap_array_job_id",
    "finalizer_job_id",
    "audit_afterany_job_id",
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _normalize_job_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.split(";", 1)[0]


def _query_sacct(job_ids: list[str]) -> dict[str, Any]:
    if not job_ids:
        return {"available": False, "reason": "no job ids"}
    sacct = shutil.which("sacct")
    if sacct is None:
        return {"available": False, "reason": "sacct not found"}
    command = [
        sacct,
        "-P",
        "-n",
        "-j",
        ",".join(job_ids),
        "--format=JobID,JobName,State,ExitCode,Elapsed,MaxRSS,AllocTRES%80",
    ]
    proc = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    rows = []
    if proc.stdout.strip():
        for line in proc.stdout.splitlines():
            parts = line.split("|")
            rows.append(
                {
                    "JobID": parts[0] if len(parts) > 0 else "",
                    "JobName": parts[1] if len(parts) > 1 else "",
                    "State": parts[2] if len(parts) > 2 else "",
                    "ExitCode": parts[3] if len(parts) > 3 else "",
                    "Elapsed": parts[4] if len(parts) > 4 else "",
                    "MaxRSS": parts[5] if len(parts) > 5 else "",
                    "AllocTRES": parts[6] if len(parts) > 6 else "",
                }
            )
    return {
        "available": True,
        "returncode": int(proc.returncode),
        "stderr": proc.stderr,
        "rows": rows,
    }


def build_audit(matrix_path: Path, run_root: Path) -> dict[str, Any]:
    manifest_path = run_root / "submission_manifest.json"
    submission = _read_json(manifest_path) if manifest_path.is_file() else {}
    job_ids = [
        job_id
        for job_id in (_normalize_job_id(submission.get(key)) for key in JOB_ID_KEYS)
        if job_id is not None
    ]
    artifact_summary = build_summary(
        matrix_path,
        run_root=run_root,
        bootstrap_dir=run_root / "bootstrap",
        audit_only=True,
    )
    return {
        "schema_version": "duca_unified_afterany_slurm_audit_v1",
        "matrix_id": artifact_summary.get("matrix_id"),
        "run_root": str(run_root),
        "submission_manifest_path": str(manifest_path),
        "submission_manifest": submission,
        "slurm_job_ids": job_ids,
        "sacct": _query_sacct(job_ids),
        "artifact_summary": artifact_summary,
        "status": artifact_summary.get("status", "INCOMPLETE"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit DUCA Unified Slurm DAG afterany.")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    payload = build_audit(args.matrix, args.run_root)
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
