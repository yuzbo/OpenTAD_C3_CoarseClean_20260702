from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


PRIMARY_CANDIDATE = "A11"
PRIMARY_CONTROL = "A10"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _work_dir(row: dict[str, Any], project_dir: Path) -> Path:
    work = Path(str(row["work_dir"]))
    if not work.is_absolute():
        work = project_dir / work
    return work / "gpu1_id0"


def _row_artifacts(row: dict[str, Any], project_dir: Path) -> dict[str, Any]:
    work = _work_dir(row, project_dir)
    metrics_path = work / "evaluation_metrics.json"
    result_path = work / "result_detection.json"
    checkpoint_path = work / "checkpoint" / "epoch_59.pth"
    metrics = _read_json(metrics_path) if metrics_path.is_file() else None
    complete = (
        isinstance(metrics, dict)
        and _as_float(metrics.get("average_mAP")) is not None
        and result_path.is_file()
        and checkpoint_path.is_file()
    )
    return {
        "task_id": row["task_id"],
        "arm_id": row["arm_id"],
        "phase": row["phase"],
        "seed": int(row["seed"]),
        "config_path": row["config_path"],
        "work_dir": str(work),
        "metrics_path": str(metrics_path),
        "result_path": str(result_path),
        "checkpoint_path": str(checkpoint_path),
        "complete": complete,
        "metrics": metrics,
    }


def _metric_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for item in rows:
        metrics = item.get("metrics") or {}
        table.append(
            {
                "task_id": item["task_id"],
                "arm_id": item["arm_id"],
                "phase": item["phase"],
                "seed": item["seed"],
                "complete": bool(item["complete"]),
                "average_mAP": _as_float(metrics.get("average_mAP")),
                "mAP@0.3": _as_float(metrics.get("mAP@0.3")),
                "mAP@0.4": _as_float(metrics.get("mAP@0.4")),
                "mAP@0.5": _as_float(metrics.get("mAP@0.5")),
                "mAP@0.6": _as_float(metrics.get("mAP@0.6")),
                "mAP@0.7": _as_float(metrics.get("mAP@0.7")),
            }
        )
    return table


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_by_arm(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in metric_rows:
        if not row["complete"] or row["average_mAP"] is None:
            continue
        key = row["arm_id"]
        bucket = out.setdefault(key, {"seeds": [], "average_mAP": [], "mAP@0.7": []})
        bucket["seeds"].append(int(row["seed"]))
        bucket["average_mAP"].append(float(row["average_mAP"]))
        if row["mAP@0.7"] is not None:
            bucket["mAP@0.7"].append(float(row["mAP@0.7"]))
    return {
        arm: {
            "seeds": values["seeds"],
            "n": len(values["average_mAP"]),
            "mean_average_mAP": _mean(values["average_mAP"]),
            "mean_mAP@0.7": _mean(values["mAP@0.7"]),
        }
        for arm, values in sorted(out.items())
    }


def _primary_delta(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_seed: dict[tuple[str, int], dict[str, Any]] = {
        (str(row["arm_id"]), int(row["seed"])): row
        for row in metric_rows
        if row["complete"] and row["average_mAP"] is not None
    }
    deltas = []
    for arm, seed in list(by_seed):
        if arm != PRIMARY_CANDIDATE:
            continue
        candidate = by_seed.get((PRIMARY_CANDIDATE, seed))
        control = by_seed.get((PRIMARY_CONTROL, seed))
        if candidate is None or control is None:
            continue
        deltas.append(
            {
                "seed": seed,
                "candidate": PRIMARY_CANDIDATE,
                "control": PRIMARY_CONTROL,
                "delta_average_mAP": float(candidate["average_mAP"]) - float(control["average_mAP"]),
                "delta_average_mAP_pp": 100.0 * (float(candidate["average_mAP"]) - float(control["average_mAP"])),
            }
        )
    return {
        "contrast": f"{PRIMARY_CANDIDATE}-{PRIMARY_CONTROL}",
        "complete_seed_count": len(deltas),
        "deltas": deltas,
        "mean_delta_average_mAP_pp": _mean([float(row["delta_average_mAP_pp"]) for row in deltas]),
        "all_seeds_positive": bool(deltas) and all(float(row["delta_average_mAP"]) > 0.0 for row in deltas),
    }


def _bootstrap_artifacts(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_dir():
        return {"complete": False, "reason": f"bootstrap dir missing: {path}", "shards": []}
    shards = sorted(path.glob("bootstrap_*.json"))
    payloads = []
    for shard in shards:
        try:
            payloads.append(_read_json(shard))
        except json.JSONDecodeError:
            payloads.append({"path": str(shard), "complete": False, "reason": "json_decode_error"})
    return {
        "complete": bool(payloads) and all(item.get("complete", False) for item in payloads),
        "shard_count": len(payloads),
        "shards": payloads,
    }


def build_summary(
    matrix_path: Path,
    *,
    run_root: Path,
    bootstrap_dir: Path | None = None,
    audit_only: bool = False,
    cost_index: int | None = None,
) -> dict[str, Any]:
    matrix = _read_json(matrix_path)
    project_dir = matrix_path.resolve().parents[2]
    artifacts = [_row_artifacts(row, project_dir) for row in matrix["rows"]]
    metric_rows = _metric_table(artifacts)
    incomplete = [row for row in artifacts if not row["complete"]]

    if cost_index is not None:
        return {
            "schema_version": "duca_unified_cost_task_v1",
            "matrix_id": matrix["matrix_id"],
            "cost_index": int(cost_index),
            "run_root": str(run_root),
            "status": "PENDING_MEASUREMENT",
            "reason": "cost benchmark requires N16R4 runtime counters and completed terminal checkpoints",
        }

    payload = {
        "schema_version": "duca_unified_fullmatrix_summary_v1",
        "matrix_id": matrix["matrix_id"],
        "base_revision": matrix["base_revision"],
        "run_root": str(run_root),
        "status": "INCOMPLETE" if incomplete else "COMPLETE",
        "audit_only": bool(audit_only),
        "task_count": len(artifacts),
        "complete_task_count": len(artifacts) - len(incomplete),
        "incomplete_task_ids": [str(row["task_id"]) for row in incomplete],
        "metrics": metric_rows,
        "by_arm": _summarize_by_arm(metric_rows),
        "primary_contrast": _primary_delta(metric_rows),
        "claim_boundary": {
            "historical_values_descriptive_only": True,
            "primary_claim_requires_confirmation_bootstrap": True,
            "validation_best_checkpoint_forbidden": True,
        },
    }
    bootstrap = _bootstrap_artifacts(bootstrap_dir)
    if bootstrap is not None:
        payload["bootstrap"] = bootstrap
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate DUCA unified full-matrix outputs")
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--bootstrap-dir", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--cost-index", type=int, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = build_summary(
        args.matrix,
        run_root=args.run_root,
        bootstrap_dir=args.bootstrap_dir,
        audit_only=args.audit_only,
        cost_index=args.cost_index,
    )
    _write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
