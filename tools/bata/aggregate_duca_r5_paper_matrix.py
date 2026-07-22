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
from tools.bata.profile_duca_full_stack_cost import (
    load_r5_terminal_cost_binding,
)


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


def _load_sha256_file(path: Path, *, target: Path, label: str) -> str:
    _require(path.is_file(), f"{label} SHA256 file is missing: {path}")
    digest = path.read_text(encoding="utf-8").strip()
    _require(
        len(digest) == 64 and all(char in "0123456789abcdef" for char in digest),
        f"{label} SHA256 file is invalid",
    )
    _require(target.is_file() and _sha256(target) == digest, f"{label} content drift")
    return digest


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
    *,
    cell: Mapping[str, Any],
    root: Path,
    expected_commit: str,
    matrix_summary_path: Path,
    matrix_summary_sha256: str,
    mechanism_gate_path: Path,
    mechanism_gate_sha256: str,
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
        and all(
            runtime_cell.get(key) == cell[key]
            for key in (
                "backend",
                "arm",
                "budget",
                "max_unselected_hole",
                "seed",
            )
        ),
        f"{cell_id} runtime cell identity drift",
    )
    _require(payload.get("checkpoint_epoch") == 59, f"{cell_id} is not terminal epoch 59")
    _require(payload.get("checkpoint_state_key") == "state_dict_ema", f"{cell_id} is not EMA")
    checkpoint = Path(str(payload.get("checkpoint_path", ""))).resolve()
    _require(checkpoint.is_file(), f"{cell_id} checkpoint is missing")
    _require(_sha256(checkpoint) == payload.get("checkpoint_sha256"), f"{cell_id} checkpoint drift")
    training_binding = load_r5_terminal_cost_binding(
        method_name=cell_id,
        config_path=config,
        checkpoint_path=checkpoint,
        expected_commit=expected_commit,
        matrix_summary_path=matrix_summary_path,
        matrix_summary_sha256=matrix_summary_sha256,
        mechanism_gate_path=mechanism_gate_path,
        mechanism_gate_sha256=mechanism_gate_sha256,
        expected_resolved_config_sha256=str(
            payload.get("resolved_config_sha256", "")
        ),
        expected_training_identity=identity,
        expected_evaluation=payload,
    )
    _require(int(payload.get("result_count", 0)) > 0, f"{cell_id} has no detections")
    _require(int(payload.get("video_count", 0)) > 0, f"{cell_id} has no evaluated videos")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), f"{cell_id} metrics are missing")
    row = {
        key: cell[key]
        for key in (
            "id",
            "backend",
            "arm",
            "budget",
            "max_unselected_hole",
            "seed",
        )
    }
    row["average_mAP"] = _finite_metric(metrics, "average_mAP", cell_id)
    row["iou_mAP"] = {
        key: _finite_metric(metrics, key, cell_id) for key in IOU_KEYS
    }
    row["evaluation_path"] = str(path.resolve())
    row["evaluation_sha256"] = _sha256(path)
    row["checkpoint_path"] = str(checkpoint)
    row["checkpoint_sha256"] = payload["checkpoint_sha256"]
    row["training_binding"] = training_binding
    return row


def _validate_cost(
    row: Mapping[str, Any],
    *,
    source: Mapping[str, Any],
    expected_commit: str,
) -> dict[str, Any]:
    _require(row.get("kind") == "r5_cell", f"{row['id']} cost kind drift")
    path = Path(str(row["summary"])).resolve()
    payload = _load_json(path, str(row["id"]))
    validate_and_rebuild_profile_summary(payload)
    _require(payload.get("config_commit") == expected_commit, f"{row['id']} commit drift")
    _require(payload.get("random_init") is False, f"{row['id']} used random initialization")
    _require(payload.get("method") == row["source_cell"], f"{row['id']} method drift")
    _require(payload.get("uses_ema") is True, f"{row['id']} did not use EMA")
    binding = payload.get("r5_cost_binding")
    _require(isinstance(binding, Mapping), f"{row['id']} lacks the R5 cost binding")
    unsigned_binding = dict(binding)
    binding_self_hash = unsigned_binding.pop("binding_sha256", None)
    _require(
        binding_self_hash == _canonical_sha256(unsigned_binding),
        f"{row['id']} R5 terminal binding self-hash drift",
    )
    _require(
        payload.get("r5_cost_binding_sha256") == _canonical_sha256(binding),
        f"{row['id']} R5 cost binding self-hash drift",
    )
    _require(
        binding == source.get("training_binding"),
        f"{row['id']} is not bound to its terminal source cell",
    )
    for key, expected in (
        ("checkpoint_path", source["checkpoint_path"]),
        ("checkpoint_sha256", source["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
    ):
        observed = payload.get(key)
        if key == "checkpoint_path":
            observed = str(Path(str(observed or "")).resolve())
        _require(observed == expected, f"{row['id']} source checkpoint drift: {key}")
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
        "source_evaluation_sha256": source["evaluation_sha256"],
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "r5_cost_binding_sha256": payload["r5_cost_binding_sha256"],
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


def _validate_dense_cost(
    row: Mapping[str, Any],
    *,
    expected_evidence_commit: str,
) -> dict[str, Any]:
    _require(row.get("kind") == "dense_baseline", "dense cost kind drift")
    path = Path(str(row["summary"])).resolve()
    payload = _load_json(path, str(row["id"]))
    validate_and_rebuild_profile_summary(payload)
    trained_commit = str(row.get("trained_commit", ""))
    _require(len(trained_commit) == 40, "dense baseline lacks an exact trained commit")
    _require(payload.get("method") == "dense-adatad", "dense baseline method drift")
    _require(payload.get("config_commit") == trained_commit, "dense config commit drift")
    _require(payload.get("trained_commit") == trained_commit, "dense trained commit drift")
    _require(
        payload.get("evidence_git_commit") == expected_evidence_commit,
        "dense evidence commit drift",
    )
    _require(payload.get("random_init") is False, "dense baseline used random initialization")
    _require(payload.get("uses_ema") is True, "dense baseline did not use EMA")

    code_binding = payload.get("inference_code_tree_binding")
    config_git_binding = payload.get("profile_config_git_binding")
    _require(
        isinstance(code_binding, Mapping)
        and code_binding.get("profile_model_loaded_from_trained_repository") is True
        and code_binding.get("profile_configs_loaded_from_trained_repository") is True,
        "dense baseline was not executed from its trained repository",
    )
    _require(
        isinstance(config_git_binding, Mapping)
        and config_git_binding.get("trained_commit") == trained_commit,
        "dense profile config lacks its trained Git binding",
    )
    trained_repository = Path(
        str(config_git_binding.get("trained_repository", ""))
    ).resolve()
    _require(
        Path(str(code_binding.get("execution_repository", ""))).resolve()
        == trained_repository
        and Path(str(code_binding.get("loaded_opentad_root", ""))).resolve()
        == trained_repository / "opentad",
        "dense model import escaped its trained repository",
    )

    expected_paths = {
        "config_path": Path(str(row["config"])).resolve(),
        "checkpoint_path": Path(str(row["checkpoint"])).resolve(),
    }
    for key, expected in expected_paths.items():
        _require(
            Path(str(payload.get(key, ""))).resolve() == expected,
            f"dense baseline {key} drift",
        )
    for key, expected in (
        ("profile_config_sha256", row["config_sha256"]),
        ("checkpoint_sha256", row["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
    ):
        _require(payload.get(key) == expected, f"dense baseline {key} drift")

    binding = payload.get("trained_checkpoint_binding")
    _require(isinstance(binding, Mapping), "dense baseline lacks checkpoint evidence")
    _require(
        payload.get("trained_checkpoint_binding_sha256")
        == _canonical_sha256(binding),
        "dense checkpoint binding canonical hash drift",
    )
    for key, expected in (
        ("role", "dense_adatad_baseline"),
        ("git_commit", trained_commit),
        ("config_path", str(expected_paths["config_path"])),
        ("config_sha256", row["config_sha256"]),
        ("checkpoint_path", str(expected_paths["checkpoint_path"])),
        ("checkpoint_sha256", row["checkpoint_sha256"]),
        ("checkpoint_epoch", 59),
        ("checkpoint_state_key", "state_dict_ema"),
        ("path", str(Path(str(row["checkpoint_evidence"])).resolve())),
        ("sha256", row["checkpoint_evidence_sha256"]),
    ):
        _require(binding.get(key) == expected, f"dense checkpoint binding drift: {key}")
    _require(
        binding.get("resolved_config_sha256")
        == payload.get("profile_resolved_config_sha256"),
        "dense resolved config binding drift",
    )

    latency = _finite_metric(
        payload.get("stages", {}).get("end_to_end_serial_ms", {}),
        "p50",
        str(row["id"]),
    )
    selected_count = _finite_metric(
        payload.get("selected_count", {}), "p50", str(row["id"])
    )
    _require(selected_count == 768.0, "dense baseline is not the full 768-point input")
    return {
        "id": row["id"],
        "kind": "dense_baseline",
        "summary_path": str(path),
        "summary_sha256": _sha256(path),
        "trained_commit": trained_commit,
        "checkpoint_sha256": row["checkpoint_sha256"],
        "checkpoint_evidence_sha256": row["checkpoint_evidence_sha256"],
        "end_to_end_serial_ms_p50": latency,
        "selected_count_p50": selected_count,
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
    _require(isinstance(costs, list) and len(costs) == 9, "matrix must contain nine cost profiles")
    r5_cost_rows = [row for row in costs if row.get("kind") == "r5_cell"]
    dense_cost_rows = [row for row in costs if row.get("kind") == "dense_baseline"]
    _require(len(r5_cost_rows) == 8, "matrix must profile all seed-3407 R5 cells")
    _require(len(dense_cost_rows) == 1, "matrix must contain one dense-768 cost baseline")
    root = summary_path.parent
    matrix_digest = _load_sha256_file(
        Path(str(summary.get("matrix_summary_sha256_file", ""))).resolve(),
        target=summary_path,
        label="R5 matrix summary",
    )
    mechanism_gate_path = Path(
        str(summary.get("mechanism_gate_output", ""))
    ).resolve()
    mechanism_gate_digest = _load_sha256_file(
        Path(str(summary.get("mechanism_gate_sha256_file", ""))).resolve(),
        target=mechanism_gate_path,
        label="R5 mechanism gate",
    )
    rows = [
        _validate_evaluation(
            cell=cell,
            root=root,
            expected_commit=expected_commit,
            matrix_summary_path=summary_path,
            matrix_summary_sha256=matrix_digest,
            mechanism_gate_path=mechanism_gate_path,
            mechanism_gate_sha256=mechanism_gate_digest,
        )
        for cell in cells
    ]
    rows_by_id = {str(row["id"]): row for row in rows}
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
    validated_r5_costs = [
        _validate_cost(
            row,
            source=rows_by_id[str(row["source_cell"])],
            expected_commit=expected_commit,
        )
        for row in r5_cost_rows
    ]
    dense_cost = _validate_dense_cost(
        dense_cost_rows[0], expected_evidence_commit=expected_commit
    )
    dense_latency = float(dense_cost["end_to_end_serial_ms_p50"])
    _require(dense_latency > 0.0, "dense latency must be positive")
    cost_comparisons = []
    for cost in validated_r5_costs:
        latency = float(cost["end_to_end_serial_ms_p50"])
        _require(latency > 0.0, f"{cost['source_cell']} latency must be positive")
        cost_comparisons.append(
            {
                "source_cell": cost["source_cell"],
                "latency_ratio_vs_dense": latency / dense_latency,
                "latency_reduction_vs_dense": 1.0 - latency / dense_latency,
                "speedup_vs_dense": dense_latency / latency,
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
        "costs": validated_r5_costs,
        "dense_baseline_cost": dense_cost,
        "cost_comparisons": cost_comparisons,
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
