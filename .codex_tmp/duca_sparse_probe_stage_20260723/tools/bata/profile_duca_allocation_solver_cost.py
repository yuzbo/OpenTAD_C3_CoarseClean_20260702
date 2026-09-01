from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time
from typing import Any, Sequence

from tools.bata.diagnose_duca_allocation_family_ceiling import (
    axis_from_record,
    read_input_records,
)
from tools.bata.duca_allocation_families import resolve_physical_cap
from tools.bata.duca_exact_physical_solver import solve_additive_exact_k_physical
from tools.bata.export_duca_allocation_ceiling_inputs import (
    canonical_sha256,
    sha256,
    write_json_exclusive,
)


def profile_solver(
    *,
    input_jsonl: str | Path,
    output_samples_jsonl: str | Path,
    output_summary_json: str | Path,
    score_key: str,
    cap_policy: str,
    cap_value: float | None,
    quantization_scale: int,
    warmup_samples: int,
    samples: int,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    samples_path = Path(output_samples_jsonl).resolve()
    summary_path = Path(output_summary_json).resolve()
    if samples_path.exists() or summary_path.exists():
        raise FileExistsError("allocation solver cost profile never overwrites artifacts")
    if warmup_samples < 0 or samples < 1:
        raise ValueError("warmup_samples must be non-negative and samples must be positive")
    records = read_input_records(input_path)
    if not records:
        raise ValueError("solver cost profile requires input records")

    def solve(row):
        axis = axis_from_record(row)
        scores = row["scores"][score_key]
        cap = resolve_physical_cap(
            axis,
            requested_budget=int(row["requested_budget"]),
            policy=cap_policy,
            value=cap_value,
        )
        return solve_additive_exact_k_physical(
            axis,
            scores,
            requested_budget=int(row["requested_budget"]),
            cap=cap,
            quantization_scale=quantization_scale,
        )

    for index in range(warmup_samples):
        solve(records[index % len(records)])

    rows: list[dict[str, Any]] = []
    for index in range(samples):
        record = records[index % len(records)]
        start = time.perf_counter_ns()
        result = solve(record)
        end = time.perf_counter_ns()
        latency_ms = (end - start) / 1.0e6
        if not math.isfinite(latency_ms) or latency_ms <= 0:
            raise RuntimeError("solver latency measurement must be finite and positive")
        row = {
            "schema_version": "duca_allocation_solver_cost_sample_v1",
            "sample_index": index,
            "sample_id": record["sample_id"],
            "latency_ms": latency_ms,
            "selected_positions_sha256": canonical_sha256(result.positions),
            "quantized_objective": result.quantized_objective,
            "solver_status": result.solver_status,
        }
        row["record_sha256"] = canonical_sha256(row)
        rows.append(row)

    samples_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = samples_path.with_suffix(samples_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale solver-cost partial artifact exists: {temporary}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(samples_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    latencies = sorted(float(row["latency_ms"]) for row in rows)
    summary = {
        "schema_version": "duca_allocation_solver_cost_summary_v1",
        "diagnostic_role": "cpu_exact_decoder_incremental_cost_not_full_stack",
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256(input_path),
        "samples_jsonl": str(samples_path),
        "samples_jsonl_sha256": sha256(samples_path),
        "score_key": score_key,
        "cap_policy": cap_policy,
        "cap_value": cap_value,
        "quantization_scale": int(quantization_scale),
        "warmup_samples": int(warmup_samples),
        "sample_count": len(rows),
        "latency_ms": {
            "mean": statistics.fmean(latencies),
            "median": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
            "min": min(latencies),
            "max": max(latencies),
        },
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "processor": platform.processor(),
        },
        "contract": {
            "full_stack_claim": False,
            "frontend_included": False,
            "backbone_included": False,
            "detector_included": False,
            "exact_decoder_only": True,
            "sample_records_hash_bound": True,
        },
    }
    write_json_exclusive(summary_path, summary)
    return summary


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires non-empty values")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    rank = fraction * (len(sorted_values) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    weight = rank - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile exact family-D allocation decoder CPU cost.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-samples-jsonl", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--score-key", default="transition_policy_scores")
    parser.add_argument(
        "--cap-policy",
        default="uniform_reference",
        choices=["uniform_reference", "explicit_frames", "explicit_seconds"],
    )
    parser.add_argument("--cap-value", type=float)
    parser.add_argument("--quantization-scale", type=int, default=1_000_000)
    parser.add_argument("--warmup-samples", type=int, default=5)
    parser.add_argument("--samples", type=int, default=100)
    args = parser.parse_args(argv)
    result = profile_solver(
        input_jsonl=args.input_jsonl,
        output_samples_jsonl=args.output_samples_jsonl,
        output_summary_json=args.output_summary_json,
        score_key=args.score_key,
        cap_policy=args.cap_policy,
        cap_value=args.cap_value,
        quantization_scale=args.quantization_scale,
        warmup_samples=args.warmup_samples,
        samples=args.samples,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
