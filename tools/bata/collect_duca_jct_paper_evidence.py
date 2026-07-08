from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCHEMA_VERSION = "duca_jct_paper_evidence_v1"
MAIN_METHODS = ("duca384", "duca_must")
TRAINFREE_METHODS = ("x3d_duca384", "x3d_must")
HIGH_IOU_KEYS = ("mAP@0.60_percent", "mAP@0.70_percent")
AVG_KEY = "average_mAP_percent"
TRAINING_DIAGNOSTIC_KEYS = (
    "latest_train_epoch",
    "latest_train_iter",
    "latest_train_loss",
    "latest_actionness_bce_loss",
    "latest_detector_utility_distribution_loss",
    "latest_action_local_hole_loss",
    "latest_lagrangian_budget_loss",
    "latest_cls_loss",
    "latest_reg_loss",
    "latest_lr_det",
    "latest_mem_mb",
    "latest_duca_schedule_step",
    "latest_duca_schedule_progress",
    "latest_duca_detector_grad_w",
    "latest_duca_actionness_w",
    "latest_duca_detector_utility_w",
    "latest_duca_hole_w",
    "latest_duca_lagrangian_budget_w",
    "latest_duca_requested_budget_mean",
    "latest_duca_effective_budget_mean",
)


def _path(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def _read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(_path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _metric(metrics: Mapping[str, Any], key: str) -> float | None:
    return _number(metrics.get(key))


def _job_metrics(monitor: Mapping[str, Any], method: str) -> dict[str, float]:
    job = monitor.get("jobs", {}).get(method, {})
    metrics = job.get("metrics", {}) if isinstance(job, Mapping) else {}
    if not isinstance(metrics, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in metrics.items():
        numeric = _number(value)
        if numeric is not None:
            out[str(key)] = float(numeric)
    return out


def _load_baseline(path: str | Path | None) -> tuple[str | None, dict[str, float]]:
    if path is None:
        return None, {}
    payload = _read_json(path)
    baselines = payload.get("baselines")
    if not isinstance(baselines, Mapping) or not baselines:
        raise ValueError("baseline_summary must contain a non-empty baselines object")
    primary = str(payload.get("primary_baseline") or next(iter(baselines)))
    row = baselines.get(primary)
    if not isinstance(row, Mapping):
        raise ValueError(f"primary baseline {primary!r} is missing")
    metrics: dict[str, float] = {}
    for key, value in row.items():
        numeric = _number(value)
        if numeric is not None:
            metrics[str(key)] = float(numeric)
    return primary, metrics


def _row(
    monitor: Mapping[str, Any],
    method: str,
    *,
    role: str,
    baseline: Mapping[str, float],
) -> dict[str, Any]:
    job = monitor.get("jobs", {}).get(method, {})
    if not isinstance(job, Mapping):
        job = {}
    metrics = _job_metrics(monitor, method)
    avg = _metric(metrics, AVG_KEY)
    baseline_avg = _metric(baseline, AVG_KEY)
    row = {
        "method": method,
        "role": role,
        "status": str(job.get("status", "missing")),
        "result_artifacts": list(job.get("result_artifacts", [])) if isinstance(job.get("result_artifacts", []), list) else [],
        AVG_KEY: _round(avg),
        "mAP@0.60_percent": _round(_metric(metrics, "mAP@0.60_percent")),
        "mAP@0.70_percent": _round(_metric(metrics, "mAP@0.70_percent")),
        "delta_vs_primary_average_mAP": _round(None if avg is None or baseline_avg is None else avg - baseline_avg),
        "delta_vs_primary_mAP@0.60": _round(
            None
            if _metric(metrics, "mAP@0.60_percent") is None or _metric(baseline, "mAP@0.60_percent") is None
            else _metric(metrics, "mAP@0.60_percent") - _metric(baseline, "mAP@0.60_percent")
        ),
        "delta_vs_primary_mAP@0.70": _round(
            None
            if _metric(metrics, "mAP@0.70_percent") is None or _metric(baseline, "mAP@0.70_percent") is None
            else _metric(metrics, "mAP@0.70_percent") - _metric(baseline, "mAP@0.70_percent")
        ),
    }
    for key in TRAINING_DIAGNOSTIC_KEYS:
        row[key] = _round(_metric(metrics, key))
    return row


def _has_result(row: Mapping[str, Any]) -> bool:
    return bool(row.get("result_artifacts"))


def _has_required_metrics(row: Mapping[str, Any]) -> bool:
    return _number(row.get(AVG_KEY)) is not None and all(_number(row.get(key)) is not None for key in HIGH_IOU_KEYS)


def collect_evidence(
    *,
    monitor_summary: str | Path,
    baseline_summary: str | Path | None = None,
    min_avg_delta: float = 0.7,
) -> dict[str, Any]:
    monitor = _read_json(monitor_summary)
    if monitor.get("schema_version") != "duca_jct_suite_monitor_v1":
        raise ValueError("monitor_summary schema_version must be duca_jct_suite_monitor_v1")
    primary_baseline, baseline_metrics = _load_baseline(baseline_summary)
    table_rows = [
        _row(monitor, method, role="main_joint_trainable", baseline=baseline_metrics) for method in MAIN_METHODS
    ]
    table_rows.extend(
        _row(monitor, method, role="trainfree_actionness_baseline", baseline=baseline_metrics)
        for method in TRAINFREE_METHODS
    )

    blockers: list[str] = []
    hard_failures = list(monitor.get("hard_failures", []))
    missing_results = list(monitor.get("missing_results", []))
    missing_prerequisites = list(monitor.get("missing_prerequisites", []))
    joint_grad_proof = monitor.get("joint_grad_proof") if isinstance(monitor.get("joint_grad_proof"), Mapping) else {}
    joint_grad_proof_ready = bool(joint_grad_proof.get("ready") is True and joint_grad_proof.get("proof_passed") is True)
    if hard_failures:
        blockers.append("hard_failures_present")
    if missing_results:
        blockers.append("missing_detector_result_artifacts")
    if missing_prerequisites:
        blockers.append("missing_prerequisites")
    if not joint_grad_proof_ready:
        blockers.append("missing_or_failed_joint_grad_proof")
    if not primary_baseline or not baseline_metrics:
        blockers.append("missing_matched_reference_baseline")
    for key in (AVG_KEY, *HIGH_IOU_KEYS):
        if baseline_metrics and _metric(baseline_metrics, key) is None:
            blockers.append(f"baseline_missing_metric:{key}")

    main_rows = [row for row in table_rows if row["role"] == "main_joint_trainable"]
    trainfree_rows = [row for row in table_rows if row["role"] == "trainfree_actionness_baseline"]
    for row in main_rows:
        if row["status"] != "completed":
            blockers.append(f"main_method_not_completed:{row['method']}")
        if not _has_result(row):
            blockers.append(f"main_method_missing_result:{row['method']}")
        if _number(row.get(AVG_KEY)) is None:
            blockers.append(f"missing_metric:{row['method']}:{AVG_KEY}")
    for key in HIGH_IOU_KEYS:
        if not all(_number(row.get(key)) is not None for row in main_rows):
            blockers.append(f"missing_high_iou_metric:{key}")
    for row in trainfree_rows:
        if row["status"] != "completed":
            blockers.append(f"trainfree_method_not_completed:{row['method']}")
        if not _has_result(row):
            blockers.append(f"trainfree_method_missing_result:{row['method']}")
        if _number(row.get(AVG_KEY)) is None:
            blockers.append(f"missing_metric:{row['method']}:{AVG_KEY}")
    for key in HIGH_IOU_KEYS:
        if not all(_number(row.get(key)) is not None for row in trainfree_rows):
            blockers.append(f"missing_trainfree_high_iou_metric:{key}")

    best = None
    for row in main_rows:
        avg = _number(row.get(AVG_KEY))
        if avg is None:
            continue
        if best is None or avg > _number(best.get(AVG_KEY)):
            best = row

    best_delta = None if best is None else _number(best.get("delta_vs_primary_average_mAP"))
    if baseline_metrics and best_delta is None:
        blockers.append("missing_average_mAP_delta")
    elif baseline_metrics and best_delta is not None and best_delta < float(min_avg_delta):
        blockers.append("average_mAP_delta_below_threshold")

    high_iou_deltas: dict[str, float | None] = {}
    if best is not None:
        high_iou_deltas = {
            "mAP@0.60": _number(best.get("delta_vs_primary_mAP@0.60")),
            "mAP@0.70": _number(best.get("delta_vs_primary_mAP@0.70")),
        }
        for key, value in high_iou_deltas.items():
            if baseline_metrics and value is None:
                blockers.append(f"missing_high_iou_delta:{key}")
            elif baseline_metrics and value is not None and value < 0.0:
                blockers.append(f"negative_high_iou_delta:{key}")

    blockers = sorted(set(blockers))
    main_complete = all(row["status"] == "completed" and _has_result(row) and _has_required_metrics(row) for row in main_rows)
    trainfree_complete = all(
        row["status"] == "completed" and _has_result(row) and _has_required_metrics(row) for row in trainfree_rows
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "monitor_summary": str(_path(monitor_summary)),
        "commit": monitor.get("commit"),
        "branch": monitor.get("branch"),
        "main_duca_results_complete": bool(main_complete),
        "trainfree_baseline_results_complete": bool(trainfree_complete),
        "joint_grad_proof_ready": bool(joint_grad_proof_ready),
        "paper_claim_allowed": not blockers,
        "table_rows": table_rows,
        "claim_gate": {
            "primary_baseline": primary_baseline,
            "min_avg_delta_required": float(min_avg_delta),
            "best_main_method": None if best is None else best["method"],
            "best_main_average_mAP_delta": _round(best_delta),
            "best_main_high_iou_deltas": {key: _round(value) for key, value in high_iou_deltas.items()},
            "blockers": blockers,
        },
    }


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return str(_round(value))
    return str(value)


def write_tsv(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    out = _path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method",
        "role",
        "status",
        AVG_KEY,
        "mAP@0.60_percent",
        "mAP@0.70_percent",
        "delta_vs_primary_average_mAP",
        "delta_vs_primary_mAP@0.60",
        "delta_vs_primary_mAP@0.70",
        *TRAINING_DIAGNOSTIC_KEYS,
    ]
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _format_cell(row.get(field)) for field in fields})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect DUCA-JCT paper-evidence metrics from a suite monitor summary.")
    parser.add_argument("--monitor-summary", required=True)
    parser.add_argument("--baseline-summary")
    parser.add_argument("--min-avg-delta", type=float, default=0.7)
    parser.add_argument("--output-json")
    parser.add_argument("--output-tsv")
    args = parser.parse_args(argv)

    summary = collect_evidence(
        monitor_summary=args.monitor_summary,
        baseline_summary=args.baseline_summary,
        min_avg_delta=float(args.min_avg_delta),
    )
    if args.output_json:
        _write_json(args.output_json, summary)
    if args.output_tsv:
        write_tsv(args.output_tsv, summary["table_rows"])
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
