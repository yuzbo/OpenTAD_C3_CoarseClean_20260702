from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Mapping
from uuid import uuid4

from tools.bata.duca_full_stack_cost import validate_and_rebuild_profile_summary


SCHEMA = "duca_r5_paper_matrix_results_v1"
IOU_KEYS = tuple(f"mAP@{value:.1f}" for value in (0.3, 0.4, 0.5, 0.6, 0.7))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"R5 aggregation failed: {message}")


def _load_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _finite_metric(metrics: Mapping[str, Any], key: str, label: str) -> float:
    value = metrics.get(key)
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} lacks finite {key}",
    )
    return float(value)


def _validate_evaluation(
    *, cell: Mapping[str, Any], root: Path, expected_commit: str
) -> dict[str, Any]:
    cell_id = str(cell["id"])
    path = root / "results" / f"{cell_id}.terminal_evaluation.json"
    payload = _load_json(path, f"{cell_id} terminal evaluation")
    unsigned = dict(payload)
    observed_self_hash = unsigned.pop("evaluation_sha256", None)
    _require(
        observed_self_hash == _canonical_sha256(unsigned),
        f"{cell_id} evaluation self-hash drift",
    )
    config = Path(str(cell["config"])).resolve()
    _require(config.is_file(), f"{cell_id} config is missing")
    _require(
        _sha256(config) == cell["config_sha256"] == payload.get("config_sha256"),
        f"{cell_id} config identity drift",
    )
    _require(
        Path(str(payload.get("config_path", ""))).resolve() == config,
        f"{cell_id} evaluation points at another config",
    )
    _require(payload.get("git_commit") == expected_commit, f"{cell_id} commit drift")
    _require(payload.get("task") == "offline_temporal_action_detection", f"{cell_id} task drift")
    _require(
        payload.get("schema_version") == "duca_r5_terminal_evaluation_v1",
        f"{cell_id} is not an R5 terminal evaluation",
    )
    identity = payload.get("training_identity")
    _require(
        isinstance(identity, Mapping)
        and identity.get("variant") == cell_id
        and identity.get("seed") == int(cell["seed"])
        and identity.get("successful_optimizer_updates") == 6000,
        f"{cell_id} lacks the exact successful-update identity",
    )
    runtime_cell = payload.get("r5_cell")
    _require(
        isinstance(runtime_cell, Mapping)
        and all(runtime_cell.get(key) == cell[key] for key in ("backend", "arm", "budget", "seed")),
        f"{cell_id} runtime cell identity drift",
    )
    _require(payload.get("checkpoint_epoch") == 59, f"{cell_id} is not terminal epoch 59")
    _require(payload.get("checkpoint_state_key") == "state_dict_ema", f"{cell_id} is not EMA")
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    _require(checkpoint.is_file(), f"{cell_id} checkpoint is missing")
    _require(_sha256(checkpoint) == payload.get("checkpoint_sha256"), f"{cell_id} checkpoint drift")
    _require(int(payload.get("result_count", 0)) > 0, f"{cell_id} has no detections")
    _require(int(payload.get("video_count", 0)) > 0, f"{cell_id} has no evaluated videos")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), f"{cell_id} metrics are missing")
    row = {
        key: cell[key] for key in ("id", "backend", "arm", "budget", "seed")
    }
    row["average_mAP"] = _finite_metric(metrics, "average_mAP", cell_id)
    row["iou_mAP"] = {
        key: _finite_metric(metrics, key, cell_id) for key in IOU_KEYS
    }
    row["evaluation_path"] = str(path.resolve())
    row["evaluation_sha256"] = _sha256(path)
    row["checkpoint_path"] = str(checkpoint)
    row["checkpoint_sha256"] = payload["checkpoint_sha256"]
    return row


def _validate_cost(row: Mapping[str, Any], expected_commit: str) -> dict[str, Any]:
    path = Path(str(row["summary"])).resolve()
    payload = _load_json(path, str(row["id"]))
    validate_and_rebuild_profile_summary(payload)
    _require(payload.get("config_commit") == expected_commit, f"{row['id']} commit drift")
    _require(payload.get("random_init") is False, f"{row['id']} used random initialization")
    latency = _finite_metric(
        payload.get("stages", {}).get("end_to_end_serial_ms", {}),
        "p50",
        str(row["id"]),
    )
    return {
        "id": row["id"],
        "source_cell": row["source_cell"],
        "summary_path": str(path),
        "summary_sha256": _sha256(path),
        "end_to_end_serial_ms_p50": latency,
        "selected_count_p50": _finite_metric(
            payload.get("selected_count", {}), "p50", str(row["id"])
        ),
        "peak_gpu_memory_mb_p50": _finite_metric(
            payload.get("resources", {}).get("peak_gpu_memory_mb", {}),
            "p50",
            str(row["id"]),
        ),
    }


def aggregate_matrix(
    *, matrix_summary: str | Path, expected_commit: str
) -> dict[str, Any]:
    summary_path = Path(matrix_summary).expanduser().resolve()
    summary = _load_json(summary_path, "R5 matrix summary")
    _require(len(expected_commit) == 40, "exact commit is required")
    _require(
        summary.get("schema") == "duca_r5_paper_matrix_v1"
        and summary.get("task") == "offline_temporal_action_detection"
        and summary.get("git_commit") == expected_commit,
        "matrix summary protocol/commit drift",
    )
    cells = summary.get("cells")
    costs = summary.get("costs")
    _require(isinstance(cells, list) and len(cells) == 24, "matrix must contain 24 cells")
    _require(isinstance(costs, list) and len(costs) == 4, "matrix must contain four cost profiles")
    root = summary_path.parent
    rows = [
        _validate_evaluation(cell=cell, root=root, expected_commit=expected_commit)
        for cell in cells
    ]
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["backend"], row["arm"], int(row["budget"]))].append(row)
        by_pair[(row["backend"], int(row["budget"]), int(row["seed"]))][
            row["arm"]
        ] = row
    aggregates = []
    for (backend, arm, budget), group in sorted(grouped.items()):
        values = [float(row["average_mAP"]) for row in group]
        _require(len(values) == 3, f"{backend}/{arm}/K{budget} lacks three seeds")
        aggregates.append(
            {
                "backend": backend,
                "arm": arm,
                "budget": budget,
                "seed_count": len(values),
                "average_mAP_mean": mean(values),
                "average_mAP_std": stdev(values),
                "average_mAP_by_seed": {
                    str(row["seed"]): row["average_mAP"] for row in group
                },
            }
        )
    paired = []
    for (backend, budget, seed), pair in sorted(by_pair.items()):
        _require(set(pair) == {"uniform", "learned"}, f"unmatched pair {backend}/K{budget}/s{seed}")
        paired.append(
            {
                "backend": backend,
                "budget": budget,
                "seed": seed,
                "learned_minus_uniform_average_mAP": (
                    pair["learned"]["average_mAP"]
                    - pair["uniform"]["average_mAP"]
                ),
            }
        )
    payload = {
        "schema": SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "matrix_summary_path": str(summary_path),
        "matrix_summary_sha256": _sha256(summary_path),
        "cell_count": len(rows),
        "cost_count": len(costs),
        "rows": rows,
        "three_seed_aggregates": aggregates,
        "paired_deltas": paired,
        "costs": [_validate_cost(row, expected_commit) for row in costs],
        "paper_claim_allowed": False,
        "status": "r5_raw_evidence_complete_pending_claim_adjudication",
    }
    payload["result_sha256"] = _canonical_sha256(payload)
    return payload


def _atomic_write(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve()
    _require(not output.exists(), f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate the complete DUCA R5 matrix")
    parser.add_argument("--matrix-summary", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = aggregate_matrix(
        matrix_summary=args.matrix_summary,
        expected_commit=args.expected_commit,
    )
    _atomic_write(Path(args.output_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
