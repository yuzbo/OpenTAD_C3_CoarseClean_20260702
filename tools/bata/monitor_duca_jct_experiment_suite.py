from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "duca_jct_suite_monitor_v1"
X3D_READY = "TRAINFREE_X3D_ACTIONNESS_MATERIALIZED"
GRAD_PROOF_SCHEMA_VERSION = "duca_jct_one_step_grad_proof_v1"

JOB_SPECS = {
    "duca_jct_tests": {
        "summary_key": "duca_jct_tests_job",
        "slurm_name": "duca_jct_tests",
        "requires_result": False,
    },
    "duca384": {
        "summary_key": "duca384_job",
        "slurm_name": "duca_jct_384",
        "work_dir": ("duca384_jct", "work_dir"),
        "requires_result": True,
    },
    "duca_must": {
        "summary_key": "duca_must_job",
        "slurm_name": "duca_jct_must",
        "work_dir": ("duca_must_jct", "work_dir"),
        "requires_result": True,
    },
    "x3d_grid": {
        "summary_key": "x3d_grid_job",
        "slurm_name": "duca_x3d_grid",
        "requires_result": False,
    },
    "x3d_duca384": {
        "summary_key": "x3d_duca384_job",
        "slurm_name": "duca_x3d_384",
        "work_dir": ("x3d_duca384", "work_dir"),
        "requires_result": True,
        "requires_x3d": True,
    },
    "x3d_must": {
        "summary_key": "x3d_must_job",
        "slurm_name": "duca_x3d_must",
        "work_dir": ("x3d_must", "work_dir"),
        "requires_result": True,
        "requires_x3d": True,
    },
}

FAILURE_PATTERNS = (
    re.compile(r"\bTraceback\b"),
    re.compile(r"\bCUDA out of memory\b", re.IGNORECASE),
    re.compile(r"\bRuntimeError\b"),
    re.compile(r"\[FAIL\]"),
    re.compile(r"\bFAILED\b"),
    re.compile(r"\bNon-finite loss\b", re.IGNORECASE),
    re.compile(r"\bNaN\b"),
)

RECOVERABLE_NONFINITE_GRAD = re.compile(r"non[- ]finite parameter gradient", re.IGNORECASE)


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
        if "|" in line:
            parts = [part.strip() for part in line.split("|")]
        else:
            parts = line.split()
        if not parts:
            continue
        job_id = parts[0]
        name = parts[1] if len(parts) > 1 else ""
        state = parts[2] if len(parts) > 2 else ""
        reason = parts[3] if len(parts) > 3 else ""
        out[str(job_id)] = {
            "job_id": str(job_id),
            "name": name,
            "state": state,
            "reason": reason,
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
    if normalized in {"F", "FAILED", "CA", "CANCELLED", "TO", "TIMEOUT", "NF", "NODE_FAIL", "OOM", "OUT_OF_MEMORY"}:
        return "failed"
    if normalized in {"CD", "COMPLETED"}:
        return "completed"
    return "queued" if normalized else "unknown"


def _job_logs(run_root: Path, slurm_name: str, job_id: str) -> list[Path]:
    log_root = run_root / "slurm_logs"
    patterns = []
    if job_id:
        patterns.extend([f"{slurm_name}_{job_id}.out", f"{slurm_name}_{job_id}.err", f"{slurm_name}_*{job_id}*.out", f"{slurm_name}_*{job_id}*.err"])
    patterns.extend([f"{slurm_name}_*.out", f"{slurm_name}_*.err"])
    paths: list[Path] = []
    seen = set()
    if log_root.is_dir():
        for pattern in patterns:
            for path in sorted(log_root.glob(pattern)):
                if path not in seen:
                    paths.append(path)
                    seen.add(path)
    return paths


def _job_train_logs(run_root: Path, spec: Mapping[str, Any]) -> list[Path]:
    parts = spec.get("work_dir")
    if not parts:
        return []
    run_dir = run_root / str(parts[0])
    log_dir = run_dir / "logs"
    if not log_dir.is_dir():
        return []
    patterns = ("train.out", "train.log", "*.out", "*.log", "*.err")
    paths: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(log_dir.glob(pattern)):
            if path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def _scan_logs(paths: list[Path]) -> dict[str, Any]:
    combined = []
    failure_hits: list[str] = []
    nonfinite_grad_skips = 0
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        combined.append(text)
        nonfinite_grad_skips += len(RECOVERABLE_NONFINITE_GRAD.findall(text))
        for pattern in FAILURE_PATTERNS:
            for match in pattern.finditer(text):
                line_start = text.rfind("\n", 0, match.start()) + 1
                line_end = text.find("\n", match.end())
                if line_end < 0:
                    line_end = len(text)
                line = text[line_start:line_end].strip()
                if RECOVERABLE_NONFINITE_GRAD.search(line):
                    continue
                failure_hits.append(line or pattern.pattern)
    text_all = "\n".join(combined)
    success = bool(
        re.search(r"\bpassed\b", text_all)
        or X3D_READY in text_all
        or re.search(r"Average-mAP", text_all)
        or re.search(r"Training Finished|Finished training", text_all, re.IGNORECASE)
    )
    metrics = _extract_metrics(text_all)
    return {
        "has_logs": bool(paths),
        "failure_hits": failure_hits,
        "success_marker": success,
        "nonfinite_grad_skip_count": int(nonfinite_grad_skips),
        "metrics": metrics,
    }


def _extract_metrics(text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    average_matches = list(re.finditer(r"Average-mAP:\s*([0-9]+(?:\.[0-9]+)?)\s*\(%\)", text))
    if average_matches:
        metrics["average_mAP_percent"] = float(average_matches[-1].group(1))
        metrics["eval_block_count"] = float(len(average_matches))
    for match in re.finditer(r"mAP at tIoU\s+([0-9.]+)\s+is\s+([0-9]+(?:\.[0-9]+)?)%", text):
        metrics[f"mAP@{match.group(1)}_percent"] = float(match.group(2))
    train_matches = list(
        re.finditer(
            r"^.*?\[Train\]:\s+\[(\d+)\]\[(\d+)/(\d+)\].*?\bLoss=([-+0-9.eE]+).*$",
            text,
            flags=re.MULTILINE,
        )
    )
    if train_matches:
        last = train_matches[-1]
        last_line = last.group(0)
        metrics["latest_train_epoch"] = float(int(last.group(1)))
        metrics["latest_train_iter"] = float(int(last.group(2)))
        metrics["latest_train_iter_max"] = float(int(last.group(3)))
        metrics["latest_train_loss"] = float(last.group(4))
        for key in (
            "detector_loss",
            "budget_loss",
            "lagrangian_budget_loss",
            "marginal_monotonic_loss",
            "hard_budget_cap_loss",
            "dynamic_budget_mean_lossless_metric",
            "teacher_utility_loss",
            "boundary_coverage_loss",
            "actionness_bce_loss",
            "action_local_hole_loss",
            "redundancy_loss",
            "radius_cost_loss",
            "entropy_anti_collapse_loss",
            "total_loss",
            "cls_loss",
            "reg_loss",
            "lr_backbone",
            "lr_det",
            "mem",
            "duca_schedule_step",
            "duca_schedule_progress",
            "duca_detector_grad_w",
            "duca_actionness_w",
            "duca_hole_w",
            "duca_lagrangian_budget_w",
            "duca_requested_budget_mean",
            "duca_effective_budget_mean",
        ):
            match = re.search(rf"\b{re.escape(key)}=([-+0-9.eE]+)", last_line)
            if match:
                metric_key = f"latest_{key}"
                if key == "mem":
                    metric_key = "latest_mem_mb"
                metrics[metric_key] = float(match.group(1))
    return metrics


def _result_artifacts(run_root: Path, spec: Mapping[str, Any]) -> list[Path]:
    parts = spec.get("work_dir")
    if not parts:
        return []
    work_dir = run_root.joinpath(*parts)
    if not work_dir.exists():
        return []
    patterns = ("result_detection.json", "result_detection*.json", "*.summary.json", "*.validation.json")
    out: list[Path] = []
    seen = set()
    for pattern in patterns:
        for path in sorted(work_dir.rglob(pattern)):
            if path not in seen:
                out.append(path)
                seen.add(path)
    return out


def _formal_x3d_status(deployment: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = _path(deployment.get("formal_x3d_materialization_summary", ""))
    jsonl_path = _path(deployment.get("formal_x3d_actionness_jsonl", ""))
    materialization = _maybe_read_json(summary_path)
    decision = None if materialization is None else materialization.get("decision")
    downstream_ready = bool(materialization.get("downstream_detector_ready")) if materialization else False
    jsonl_exists = jsonl_path.is_file()
    ready = bool(decision == X3D_READY and downstream_ready and jsonl_exists)
    return {
        "ready": ready,
        "summary_path": str(summary_path),
        "summary_exists": summary_path.is_file(),
        "jsonl_path": str(jsonl_path),
        "jsonl_exists": jsonl_exists,
        "decision": decision,
        "downstream_detector_ready": downstream_ready,
        "train_free_baseline": bool(materialization.get("train_free_baseline")) if materialization else False,
        "not_main_method": bool(materialization.get("not_main_method")) if materialization else False,
    }


def _positive_number(value: Any) -> bool:
    if isinstance(value, bool) or value is None:
        return False
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _joint_grad_proof_status(deployment: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    proof_path = _path(deployment.get("duca_jct_one_step_grad_proof", run_root / "duca_jct_one_step_grad_proof.json"))
    payload = _maybe_read_json(proof_path)
    if payload is None:
        return {
            "ready": False,
            "path": str(proof_path),
            "exists": False,
            "proof_passed": False,
            "schema_version": None,
            "reason": "missing_duca_jct_one_step_grad_proof",
        }
    fixed = payload.get("fixed384") if isinstance(payload.get("fixed384"), Mapping) else {}
    must = payload.get("duca_must") if isinstance(payload.get("duca_must"), Mapping) else {}
    dual_update = must.get("dynamic_budget_dual_update") if isinstance(must.get("dynamic_budget_dual_update"), Mapping) else {}
    fixed_schedule_update = (
        fixed.get("loss_schedule_step_update")
        if isinstance(fixed.get("loss_schedule_step_update"), Mapping)
        else {}
    )
    must_schedule_update = (
        must.get("loss_schedule_step_update")
        if isinstance(must.get("loss_schedule_step_update"), Mapping)
        else {}
    )
    checks = {
        "schema": payload.get("schema_version") == GRAD_PROOF_SCHEMA_VERSION,
        "proof_passed": payload.get("proof_passed") is True,
        "fixed_coarse_probe_grad": _positive_number(fixed.get("coarse_probe_grad_sum")),
        "fixed_selector_encoder_grad": _positive_number(fixed.get("selector_encoder_grad_sum")),
        "fixed_loss_schedule_optimizer_step": fixed_schedule_update.get("source") == "optimizer_step"
        and fixed_schedule_update.get("updated") is True,
        "must_coarse_probe_grad": _positive_number(must.get("coarse_probe_grad_sum")),
        "must_selector_encoder_grad": _positive_number(must.get("selector_encoder_grad_sum")),
        "must_budget_controller_grad": _positive_number(must.get("budget_controller_grad_sum")),
        "must_loss_schedule_optimizer_step": must_schedule_update.get("source") == "optimizer_step"
        and must_schedule_update.get("updated") is True,
        "must_dual_update": dual_update.get("updated") is True,
    }
    failed = sorted(key for key, ok in checks.items() if not ok)
    return {
        "ready": not failed,
        "path": str(proof_path),
        "exists": True,
        "proof_passed": payload.get("proof_passed") is True,
        "schema_version": payload.get("schema_version"),
        "failed_checks": failed,
        "fixed_coarse_probe_grad_sum": fixed.get("coarse_probe_grad_sum"),
        "fixed_selector_encoder_grad_sum": fixed.get("selector_encoder_grad_sum"),
        "duca_must_coarse_probe_grad_sum": must.get("coarse_probe_grad_sum"),
        "duca_must_selector_encoder_grad_sum": must.get("selector_encoder_grad_sum"),
        "duca_must_budget_controller_grad_sum": must.get("budget_controller_grad_sum"),
        "fixed_loss_schedule_step_update": dict(fixed_schedule_update),
        "duca_must_loss_schedule_step_update": dict(must_schedule_update),
        "duca_must_dual_update": dict(dual_update),
    }


def monitor_suite(
    *,
    deployment_summary: str | Path,
    squeue_text: str | Path | None = None,
) -> dict[str, Any]:
    deployment_path = _path(deployment_summary)
    deployment = _read_json(deployment_path)
    if deployment.get("schema_version") != "duca_jct_experiment_suite_deployment_v1":
        raise ValueError("deployment_summary schema_version must be duca_jct_experiment_suite_deployment_v1")
    run_root = _path(deployment.get("run_root", deployment_path.parent))
    if squeue_text is None:
        squeue_text = _live_squeue_text(
            [str(deployment.get(spec["summary_key"], "") or "") for spec in JOB_SPECS.values()]
        )
    squeue = _parse_squeue(squeue_text)
    x3d_status = _formal_x3d_status(deployment)
    grad_proof_status = _joint_grad_proof_status(deployment, run_root)

    jobs: dict[str, dict[str, Any]] = {}
    hard_failures: list[str] = []
    running_jobs: list[str] = []
    pending_jobs: list[str] = []
    missing_results: list[str] = []
    missing_prerequisites: list[str] = []
    if not x3d_status["summary_exists"]:
        missing_prerequisites.append("formal_x3d_materialization_summary")
    if not x3d_status["jsonl_exists"]:
        missing_prerequisites.append("formal_x3d_actionness_jsonl")
    if not grad_proof_status["exists"]:
        missing_prerequisites.append("duca_jct_one_step_grad_proof")
    elif not grad_proof_status["ready"]:
        hard_failures.append("duca_jct_one_step_grad_proof")

    for label, spec in JOB_SPECS.items():
        job_id = str(deployment.get(spec["summary_key"], "") or "")
        slurm_name = str(spec["slurm_name"])
        logs = _job_logs(run_root, slurm_name, job_id) + _job_train_logs(run_root, spec)
        log_scan = _scan_logs(logs)
        artifacts = _result_artifacts(run_root, spec)
        squeue_row = squeue.get(job_id)
        status = "not_submitted" if not job_id else "finished_or_unknown"
        if spec.get("requires_x3d") and not x3d_status["ready"]:
            status = "blocked_missing_x3d_actionness"
        elif squeue_row is not None:
            status = _state_to_status(squeue_row.get("state", ""))
        elif log_scan["failure_hits"]:
            status = "failed"
        elif artifacts or log_scan["success_marker"]:
            status = "completed"

        if status == "failed":
            hard_failures.append(label)
        if status == "running":
            running_jobs.append(label)
        if status == "pending":
            pending_jobs.append(label)
        if (
            spec.get("requires_result")
            and status in {"completed", "finished_or_unknown"}
            and not any(path.name.startswith("result_detection") for path in artifacts)
        ):
            missing_results.append(label)

        jobs[label] = {
            "job_id": job_id,
            "slurm_name": slurm_name,
            "status": status,
            "squeue": squeue_row,
            "log_paths": [str(path) for path in logs],
            "failure_hits": list(log_scan["failure_hits"]),
            "nonfinite_grad_skip_count": int(log_scan["nonfinite_grad_skip_count"]),
            "success_marker": bool(log_scan["success_marker"]),
            "metrics": dict(log_scan["metrics"]),
            "result_artifacts": [str(path) for path in artifacts],
            "requires_result": bool(spec.get("requires_result", False)),
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "deployment_summary": str(deployment_path),
        "run_root": str(run_root),
        "commit": deployment.get("commit"),
        "branch": deployment.get("branch"),
        "formal_x3d_actionness": x3d_status,
        "joint_grad_proof": grad_proof_status,
        "jobs": jobs,
        "hard_failures": hard_failures,
        "running_jobs": running_jobs,
        "pending_jobs": pending_jobs,
        "missing_results": missing_results,
        "missing_prerequisites": missing_prerequisites,
    }


def _print_table(summary: Mapping[str, Any]) -> None:
    print("job\tjob_id\tstatus\tmetrics")
    for label, job in summary["jobs"].items():
        print(f"{label}\t{job.get('job_id', '')}\t{job.get('status', '')}\t{json.dumps(job.get('metrics', {}), sort_keys=True)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Monitor a submitted DUCA-JCT experiment suite.")
    parser.add_argument("--deployment-summary", required=True)
    parser.add_argument("--squeue-text")
    parser.add_argument("--output-json")
    parser.add_argument("--print-table", action="store_true")
    args = parser.parse_args(argv)

    summary = monitor_suite(deployment_summary=args.deployment_summary, squeue_text=args.squeue_text)
    if args.output_json:
        out = _path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(summary)
        payload["output_json"] = str(out)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summary = payload
    if args.print_table:
        _print_table(summary)
    else:
        print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
