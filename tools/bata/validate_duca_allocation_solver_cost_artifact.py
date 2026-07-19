from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

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
from tools.bata.profile_duca_allocation_solver_cost import _percentile


_SAMPLE_KEYS = {
    "schema_version",
    "sample_index",
    "sample_id",
    "latency_ms",
    "selected_positions_sha256",
    "quantized_objective",
    "solver_status",
    "record_sha256",
}


def validate_solver_cost_artifact(
    *,
    input_jsonl: str | Path,
    samples_jsonl: str | Path,
    summary_json: str | Path,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    samples_path = Path(samples_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    records = read_input_records(input_path)
    summary = _load_mapping(summary_path)
    if summary.get("schema_version") != "duca_allocation_solver_cost_summary_v1":
        raise ValueError("solver-cost summary schema mismatch")
    if Path(str(summary.get("input_jsonl"))).resolve() != input_path:
        raise ValueError("solver-cost input path mismatch")
    if summary.get("input_jsonl_sha256") != sha256(input_path):
        raise ValueError("solver-cost input SHA-256 mismatch")
    if Path(str(summary.get("samples_jsonl"))).resolve() != samples_path:
        raise ValueError("solver-cost sample path mismatch")
    if summary.get("samples_jsonl_sha256") != sha256(samples_path):
        raise ValueError("solver-cost sample SHA-256 mismatch")
    contract = summary.get("contract")
    required_contract = {
        "full_stack_claim": False,
        "frontend_included": False,
        "backbone_included": False,
        "detector_included": False,
        "exact_decoder_only": True,
        "sample_records_hash_bound": True,
    }
    if not isinstance(contract, Mapping):
        raise ValueError("solver-cost contract is missing")
    for key, value in required_contract.items():
        if contract.get(key) is not value:
            raise ValueError(f"solver-cost contract mismatch: {key}")

    score_key = str(summary.get("score_key"))
    cap_policy = str(summary.get("cap_policy"))
    cap_value = summary.get("cap_value")
    quantization_scale = int(summary.get("quantization_scale", 0))
    if quantization_scale < 1:
        raise ValueError("solver-cost quantization scale must be positive")
    if cap_policy not in {"uniform_reference", "explicit_frames", "explicit_seconds"}:
        raise ValueError("solver-cost cap policy is invalid")
    if score_key not in records[0]["scores"]:
        raise ValueError("solver-cost score key is absent from its input")

    rows = _read_rows(samples_path)
    if int(summary.get("sample_count", -1)) != len(rows):
        raise ValueError("solver-cost sample count mismatch")
    for expected_index, row in enumerate(rows):
        if set(row) != _SAMPLE_KEYS:
            raise ValueError("strict solver-cost sample fields mismatch")
        if row.get("schema_version") != "duca_allocation_solver_cost_sample_v1":
            raise ValueError("solver-cost sample schema mismatch")
        unhashed = dict(row)
        recorded_hash = unhashed.pop("record_sha256", None)
        if not isinstance(recorded_hash, str) or canonical_sha256(unhashed) != recorded_hash:
            raise ValueError("solver-cost sample record SHA-256 mismatch")
        if int(row.get("sample_index", -1)) != expected_index:
            raise ValueError("solver-cost sample indices must be contiguous and ordered")
        latency = float(row.get("latency_ms", float("nan")))
        if not math.isfinite(latency) or latency <= 0.0:
            raise ValueError("solver-cost latency must be finite and positive")
        source = records[expected_index % len(records)]
        if row.get("sample_id") != source["sample_id"]:
            raise ValueError("solver-cost sample order differs from the registered replay order")
        axis = axis_from_record(source)
        cap = resolve_physical_cap(
            axis,
            requested_budget=int(source["requested_budget"]),
            policy=cap_policy,
            value=cap_value,
        )
        solved = solve_additive_exact_k_physical(
            axis,
            source["scores"][score_key],
            requested_budget=int(source["requested_budget"]),
            cap=cap,
            quantization_scale=quantization_scale,
        )
        if row.get("solver_status") != solved.solver_status:
            raise ValueError("solver-cost replay status mismatch")
        if int(row.get("quantized_objective")) != solved.quantized_objective:
            raise ValueError("solver-cost replay objective mismatch")
        if row.get("selected_positions_sha256") != canonical_sha256(solved.positions):
            raise ValueError("solver-cost replay selection mismatch")

    latencies = sorted(float(row["latency_ms"]) for row in rows)
    expected_latency = {
        "mean": statistics.fmean(latencies),
        "median": statistics.median(latencies),
        "p95": _percentile(latencies, 0.95),
        "min": min(latencies),
        "max": max(latencies),
    }
    actual_latency = summary.get("latency_ms")
    if not isinstance(actual_latency, Mapping) or set(actual_latency) != set(expected_latency):
        raise ValueError("solver-cost latency summary fields mismatch")
    for key, value in expected_latency.items():
        actual = float(actual_latency[key])
        if not math.isfinite(actual) or not math.isclose(
            actual,
            value,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ):
            raise ValueError(f"solver-cost latency summary mismatch: {key}")
    if int(summary.get("warmup_samples", -1)) < 0:
        raise ValueError("solver-cost warmup count is invalid")
    return {
        "schema_version": "duca_allocation_solver_cost_validation_v1",
        "validation_passed": True,
        "sample_count": len(rows),
        "input_jsonl_sha256": sha256(input_path),
        "samples_jsonl_sha256": sha256(samples_path),
        "summary_json_sha256": sha256(summary_path),
        "replayed_every_selection": True,
        "latency_values_recomputed": True,
        "full_stack_claim": False,
    }


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: solver-cost row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError("solver-cost artifact has no samples")
    return rows


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected an object")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay and validate exact DUCA allocation solver-cost evidence."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--samples-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--validation-json")
    args = parser.parse_args(argv)
    result = validate_solver_cost_artifact(
        input_jsonl=args.input_jsonl,
        samples_jsonl=args.samples_jsonl,
        summary_json=args.summary_json,
    )
    if args.validation_json:
        write_json_exclusive(args.validation_json, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
