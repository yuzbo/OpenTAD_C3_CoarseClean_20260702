"""Validate and dispatch the immutable multi-branch experiment DAG.

The dispatcher is deliberately fail-closed: a missing gate, unimplemented arm,
or missing Slurm endpoint produces a terminal reason instead of a fake job ID.
Actual sbatch command templates are supplied by the remote deployment wrapper;
this module owns the dependency and admission decision.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs" / "audits" / "DUCA_MULTIBRANCH_20260902" / "06_FULL_DAG_MANIFEST.json"
DEFAULT_ADMISSION = ROOT / "docs" / "audits" / "DUCA_MULTIBRANCH_20260902" / "05_ADMISSION_MATRIX.json"


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _blocked(admission: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    for branch in admission.get("branches", []):
        status = str(branch.get("status", "INVALID"))
        if status == "BLOCKED_UNIMPLEMENTED":
            reasons.append(f"{branch['id']}: BLOCKED_UNIMPLEMENTED")
        elif status not in {"CONDITIONAL", "PARITY_ONLY", "GEOMETRY_ONLY", "SCREEN_ONLY", "CHECKPOINT_DDP_ONLY"}:
            reasons.append(f"{branch.get('id', '<unknown>')}: invalid admission status {status}")
    for stage in manifest.get("stages", []):
        if stage.get("status") == "BLOCKED_UNIMPLEMENTED":
            reasons.append(f"stage {stage['id']}: BLOCKED_UNIMPLEMENTED")
    return reasons


def _plan(manifest: dict[str, Any], admission: dict[str, Any]) -> dict[str, Any]:
    blocked = _blocked(admission, manifest)
    return {
        "manifest_id": manifest.get("manifest_id"),
        "mode": "plan",
        "status": "BLOCKED" if blocked else "READY_FOR_REMOTE_GATES",
        "blocked_reasons": blocked,
        "stages": [
            {"id": stage.get("id"), "status": stage.get("status"), "tasks": stage.get("tasks", [])}
            for stage in manifest.get("stages", [])
        ],
        "job_ids": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--admission", type=Path, default=DEFAULT_ADMISSION)
    parser.add_argument("--mode", choices=("plan", "submit"), default="plan")
    parser.add_argument("--json", action="store_true", help="emit only JSON")
    args = parser.parse_args()

    manifest = _read(args.manifest)
    admission = _read(args.admission)
    receipt = _plan(manifest, admission)
    receipt["mode"] = args.mode
    if args.mode == "submit":
        if receipt["blocked_reasons"]:
            receipt["status"] = "BLOCKED_UNIMPLEMENTED"
            receipt["terminal_reason"] = "scientific implementation/admission gates are not open"
        elif shutil.which("sbatch") is None:
            receipt["status"] = "BLOCKED_INFRASTRUCTURE"
            receipt["terminal_reason"] = "sbatch is unavailable on this host; no job was submitted"
        else:
            receipt["status"] = "BLOCKED_CONFIGURATION"
            receipt["terminal_reason"] = "manifest has no remote launcher command templates"
    output = json.dumps(receipt, indent=2, sort_keys=True)
    print(output)
    if receipt["status"] != "READY_FOR_REMOTE_GATES" and args.mode == "plan":
        return 3
    if args.mode == "submit" and receipt["status"] != "SUCCEEDED":
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
