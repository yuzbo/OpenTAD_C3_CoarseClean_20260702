#!/usr/bin/env python3
"""Minute-level N16R4 supervisor for the gated DUCA multi-branch DAG.

The supervisor never bypasses the repository dispatcher or a route launcher's
exact-SHA/clean-tree checks. It records scheduler state and turns code or
contract failures into NEEDS_REPAIR instead of blindly retrying them.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows local dry-runs
    fcntl = None  # type: ignore[assignment]


DEFAULT_ROOT = Path("/data/run01/sczc063/yuzibo/projects/duca_multibranch_supervisor_20260902")
DEFAULT_DISPATCHER = Path("/data/run01/sczc063/yuzibo/projects/duca_multibranch_audit_20260902/coordination/tools/bata/dispatch_duca_multibranch.py")
DEFAULT_MANIFEST = Path("/data/run01/sczc063/yuzibo/projects/duca_multibranch_audit_20260902/coordination/docs/audits/DUCA_MULTIBRANCH_20260902/06_FULL_DAG_MANIFEST.json")
DEFAULT_ADMISSION = Path("/data/run01/sczc063/yuzibo/projects/duca_multibranch_audit_20260902/coordination/docs/audits/DUCA_MULTIBRANCH_20260902/05_ADMISSION_MATRIX.json")
JOB_RE = re.compile(r"\b\d+(?:_\d+)?(?:;\d+)?\b")
RETRYABLE = ("OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "TIMEOUT")


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run(argv: list[str], *, cwd: Path | None = None) -> tuple[int, str]:
    try:
        proc = subprocess.run(argv, cwd=cwd, text=True, capture_output=True, check=False)
    except OSError as exc:
        return 127, f"{type(exc).__name__}: {exc}"
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output[-20000:]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def scheduler_snapshot(user: str) -> dict[str, Any]:
    snapshot: dict[str, Any] = {"checked_at": now(), "user": user}
    for key, command in {
        "squeue": ["squeue", "-h", "-u", user, "-o", "%i|%T|%R|%j"],
        "sacct": ["sacct", "-X", "-n", "-P", "-u", user, "-S", (dt.datetime.now() - dt.timedelta(days=2)).strftime("%Y-%m-%d"), "-o", "JobID,State,ExitCode,JobName,Elapsed"],
    }.items():
        code, output = run(command)
        snapshot[key] = {"returncode": code, "output": output}
    return snapshot


def dispatcher_plan(dispatcher: Path, manifest: Path, admission: Path) -> dict[str, Any]:
    code, output = run([sys.executable, str(dispatcher), "--manifest", str(manifest), "--admission", str(admission), "--mode", "plan", "--json"])
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        payload = {"status": "BLOCKED_INFRASTRUCTURE", "terminal_reason": "dispatcher did not emit JSON", "returncode": code, "output": output}
    payload["returncode"] = code
    return payload


def classify_failure(output: str) -> str:
    upper = output.upper()
    if "OUT OF MEMORY" in upper or "CUDA OOM" in upper or "OUT_OF_MEMORY" in upper:
        return "OUT_OF_MEMORY"
    if "NODE_FAIL" in upper or "NODE FAILURE" in upper:
        return "NODE_FAIL"
    if "PREEMPT" in upper:
        return "PREEMPTED"
    if "TIME LIMIT" in upper or "TIMEOUT" in upper:
        return "TIMEOUT"
    if "ASSOCMAXSUBMITJOBLIMIT" in upper or "ASSOCGRPGRES" in upper or "QOS" in upper:
        return "RESOURCE_DEFERRED"
    return "SUBMIT_OR_RUNTIME_FAILURE"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def verify_checkout(entry: dict[str, Any]) -> tuple[bool, str]:
    workdir = Path(str(entry.get("workdir", "")))
    expected = str(entry.get("expected_sha", ""))
    if not workdir.is_dir() or not expected:
        return False, "workdir or expected_sha missing"
    code, head = run(["git", "rev-parse", "HEAD"], cwd=workdir)
    if code or head.strip() != expected:
        return False, f"HEAD mismatch: expected {expected}, got {head.strip()}"
    code, status = run(["git", "status", "--porcelain", "--untracked-files=normal"], cwd=workdir)
    if code or status.strip():
        return False, "checkout is not clean"
    return True, "exact clean checkout"


def parse_job_ids(output: str) -> list[str]:
    ids: list[str] = []
    for match in re.finditer(r"Submitted batch job (\d+(?:_\d+)?)|^\s*(\d+(?:_\d+)?)(?:;[^\s]+)?\s*$", output, flags=re.MULTILINE):
        value = match.group(1) or match.group(2)
        if value and value not in ids:
            ids.append(value)
    return ids


def slurm_state_for(job_ids: list[str], snapshot: dict[str, Any]) -> dict[str, str]:
    """Return the strongest observed state for each submitted job id."""
    states: dict[str, str] = {}
    wanted = set(job_ids)
    for row in str(snapshot.get("squeue", {}).get("output", "")).splitlines():
        fields = row.split("|", 3)
        if len(fields) >= 2 and fields[0] in wanted:
            states[fields[0]] = fields[1].split()[0].upper()
    for row in str(snapshot.get("sacct", {}).get("output", "")).splitlines():
        fields = row.split("|", 4)
        if not fields:
            continue
        job_id = fields[0].split(".", 1)[0]
        if job_id in wanted and len(fields) >= 2:
            state = fields[1].split()[0].upper()
            # sacct is authoritative for terminal jobs, while squeue is
            # authoritative for a still-running/pending job.
            if state:
                states[job_id] = state
    return states


def update_submitted_entry(current: dict[str, Any], snapshot: dict[str, Any]) -> None:
    job_ids = [str(value) for value in current.get("job_ids", [])]
    if not job_ids:
        return
    observed = slurm_state_for(job_ids, snapshot)
    if not observed:
        return
    current["job_states"] = observed
    values = list(observed.values())
    if any(value in {"RUNNING", "COMPLETING", "CONFIGURING", "STAGE_OUT"} for value in values):
        current["status"] = "RUNNING"
        return
    if any(value in {"PENDING", "SUSPENDED"} for value in values):
        current["status"] = "SUBMITTED"
        return
    if all(value in {"COMPLETED"} for value in values) and len(values) == len(job_ids):
        current["status"] = "SUCCEEDED"
        return
    if any(value in {"OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "TIMEOUT"} for value in values):
        current["failure_class"] = next(value for value in values if value in RETRYABLE)
        current["status"] = "PENDING_RETRY" if int(current.get("attempts", 0)) < int(current.get("max_attempts", 3)) else "NEEDS_REPAIR"
        return
    if any(value in {"FAILED", "CANCELLED", "BOOT_FAIL", "DEADLINE", "INVALID_DEPEND", "NONZERO_EXIT"} for value in values):
        current["failure_class"] = "REMOTE_JOB_FAILED"
        current["status"] = "NEEDS_REPAIR"


def poll_once(args: argparse.Namespace) -> int:
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    lock_path = root / "supervisor.lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        if fcntl is not None:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 0

        user = os.environ.get("USER", "sczc063")
        queue_path = Path(args.queue)
        state_path = root / "supervisor_state.json"
        queue = load_json(queue_path, {"schema_version": 1, "entries": []})
        state = load_json(state_path, {"schema_version": 1, "entries": {}, "history": []})
        plan = dispatcher_plan(Path(args.dispatcher), Path(args.manifest), Path(args.admission))
        sched = scheduler_snapshot(user)
        receipt: dict[str, Any] = {
            "checked_at": now(),
            "dispatcher": plan,
            "scheduler": sched,
            "entries": [],
        }
        entries = state.setdefault("entries", {})
        for raw in queue.get("entries", []):
            if not isinstance(raw, dict) or not raw.get("id"):
                continue
            entry_id = str(raw["id"])
            current = entries.setdefault(entry_id, {"status": "PENDING", "attempts": 0, "job_ids": []})
            current.setdefault("attempts", 0)
            current.setdefault("job_ids", [])
            current["route_id"] = raw.get("route_id")
            current["max_attempts"] = int(raw.get("max_attempts", 3))
            if current.get("status") in {"SUBMITTED", "RUNNING"}:
                update_submitted_entry(current, sched)
                receipt["entries"].append({"id": entry_id, **current})
                continue
            if current.get("status") == "SUCCEEDED":
                receipt["entries"].append({"id": entry_id, **current})
                continue
            if current.get("status") == "NEEDS_REPAIR":
                receipt["entries"].append({"id": entry_id, **current})
                continue
            if current.get("status") in {"PENDING_RETRY", "DEFERRED_RESOURCE"}:
                retry_at = float(current.get("retry_after_epoch", 0))
                if time.time() < retry_at:
                    receipt["entries"].append({"id": entry_id, **current})
                    continue
            if plan.get("status") != "READY_FOR_REMOTE_GATES":
                current["status"] = "BLOCKED_GATE"
                current["reason"] = "; ".join(plan.get("blocked_reasons", [])) or plan.get("terminal_reason", "dispatcher blocked")
                receipt["entries"].append({"id": entry_id, **current})
                continue
            ok, reason = verify_checkout(raw)
            if not ok:
                current.update({"status": "NEEDS_REPAIR", "reason": reason})
                receipt["entries"].append({"id": entry_id, **current})
                continue
            max_attempts = int(raw.get("max_attempts", 3))
            if int(current["attempts"]) >= max_attempts:
                current.update({"status": "NEEDS_REPAIR", "reason": "retry limit reached"})
                receipt["entries"].append({"id": entry_id, **current})
                continue
            command = str(raw.get("command", "")).strip()
            if not command:
                current.update({"status": "NEEDS_REPAIR", "reason": "submission command missing"})
                receipt["entries"].append({"id": entry_id, **current})
                continue
            current["attempts"] = int(current["attempts"]) + 1
            code, output = run(["bash", "-lc", command], cwd=Path(str(raw["workdir"])))
            job_ids = parse_job_ids(output)
            current["last_output"] = output
            current["last_checked_at"] = now()
            if code == 0 and job_ids:
                current.update({"status": "SUBMITTED", "job_ids": job_ids})
            else:
                failure = classify_failure(output)
                current["failure_class"] = failure
                if failure in RETRYABLE and current["attempts"] < max_attempts:
                    current["status"] = "PENDING_RETRY"
                    current["retry_after_epoch"] = time.time() + int(raw.get("retry_delay_seconds", 300))
                elif failure == "RESOURCE_DEFERRED":
                    current["status"] = "DEFERRED_RESOURCE"
                    current["retry_after_epoch"] = time.time() + int(raw.get("resource_retry_delay_seconds", 300))
                else:
                    current["status"] = "NEEDS_REPAIR"
            receipt["entries"].append({"id": entry_id, **current})
        state["last_checked_at"] = receipt["checked_at"]
        state["history"] = (state.get("history", []) + [receipt])[-120:]
        write_json(state_path, state)
        write_json(root / "latest_receipt.json", receipt)
        write_json(root / f"receipt_{receipt['checked_at'].replace(':', '').replace('+00:00', 'Z')}.json", receipt)
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--queue", default="")
    parser.add_argument("--dispatcher", default=str(DEFAULT_DISPATCHER))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--admission", default=str(DEFAULT_ADMISSION))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if not args.queue:
        args.queue = str(Path(args.root) / "submission_queue.json")
    if args.once:
        return poll_once(args)
    stop = False

    def _stop(_signum: int, _frame: Any) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    while not stop:
        poll_once(args)
        for _ in range(max(1, args.interval)):
            if stop:
                break
            time.sleep(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
