#!/usr/bin/env python3
"""Analyze validated GeoRoute-AdaTAD paper records without changing them.

The program deliberately emits descriptive summaries and paired deltas, not a
paper verdict.  A caller must apply the preregistered decision rule after
inspecting raw records, matched controls, and scope-complete cost evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from georoute_result_schema import (
    GeoRouteResultSchemaError,
    canonical_json_sha256,
    grouped_records,
    load_records,
    validate_records,
)


def _mean_std(values: Sequence[float]) -> Dict[str, float]:
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def _record_summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    metrics = record["metrics"]
    cost = record["cost"]
    return {
        "seed": record["seed"],
        "average_map": float(metrics["average_map"]),
        "map_03": float(metrics["map_by_tiou"]["0.3"]),
        "map_04": float(metrics["map_by_tiou"]["0.4"]),
        "map_05": float(metrics["map_by_tiou"]["0.5"]),
        "map_06": float(metrics["map_by_tiou"]["0.6"]),
        "map_07": float(metrics["map_by_tiou"]["0.7"]),
        "high_iou_mean": (
            float(metrics["map_by_tiou"]["0.6"])
            + float(metrics["map_by_tiou"]["0.7"])
        )
        / 2.0,
        "end_to_end_p50_ms": float(cost["end_to_end_p50_ms"]),
        "end_to_end_p95_ms": float(cost["end_to_end_p95_ms"]),
        "peak_memory_mb": float(cost["peak_memory_mb"]),
        "gross_gpu_energy_j": float(cost["gross_gpu_energy_j"]),
        "runtime_commit": record["evidence"]["runtime_commit"],
        "run_receipt_sha256": record["evidence"]["run_receipt_sha256"],
    }


def summarize_groups(records: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    for key, group in sorted(grouped_records(records).items(), key=lambda item: item[0]):
        stage, split_role, dataset, detector, variant, budget = key
        rows = [_record_summary(record) for record in sorted(group, key=lambda item: item["seed"])]
        summaries.append(
            {
                "stage": stage,
                "split_role": split_role,
                "dataset": dataset,
                "detector": detector,
                "variant": variant,
                "tokens_per_tubelet": budget,
                "raw_rows": rows,
                "average_map": _mean_std([row["average_map"] for row in rows]),
                "high_iou_mean": _mean_std([row["high_iou_mean"] for row in rows]),
                "map_06": _mean_std([row["map_06"] for row in rows]),
                "map_07": _mean_std([row["map_07"] for row in rows]),
                "end_to_end_p50_ms": _mean_std(
                    [row["end_to_end_p50_ms"] for row in rows]
                ),
                "end_to_end_p95_ms": _mean_std(
                    [row["end_to_end_p95_ms"] for row in rows]
                ),
                "peak_memory_mb": _mean_std([row["peak_memory_mb"] for row in rows]),
                "gross_gpu_energy_j": _mean_std(
                    [row["gross_gpu_energy_j"] for row in rows]
                ),
            }
        )
    return summaries


def pareto_frontier(summaries: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Return non-dominated group means: maximize Avg-mAP, minimize p50."""

    frontier: List[Dict[str, Any]] = []
    for summary in summaries:
        dominated = False
        score = summary["average_map"]["mean"]
        cost = summary["end_to_end_p50_ms"]["mean"]
        for other in summaries:
            if other is summary:
                continue
            other_score = other["average_map"]["mean"]
            other_cost = other["end_to_end_p50_ms"]["mean"]
            weakly_better = other_score >= score and other_cost <= cost
            strictly_better = other_score > score or other_cost < cost
            if weakly_better and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "stage": summary["stage"],
                    "dataset": summary["dataset"],
                    "detector": summary["detector"],
                    "variant": summary["variant"],
                    "tokens_per_tubelet": summary["tokens_per_tubelet"],
                    "average_map_mean": score,
                    "p50_ms_mean": cost,
                }
            )
    return frontier


def paired_deltas(
    records: Iterable[Mapping[str, Any]], *, structured_variant: str, free_variant: str
) -> List[Dict[str, Any]]:
    """Pair structured and free routes only when every matching key/seed exists."""

    keyed: Dict[Tuple[Any, ...], Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for record in records:
        key = (
            record["stage"],
            record["split_role"],
            record["dataset"],
            record["detector"],
            record["budget"]["tokens_per_tubelet"],
            record["seed"],
        )
        keyed[key][record["variant"]] = record
    pairs: List[Dict[str, Any]] = []
    for key, by_variant in sorted(keyed.items(), key=lambda item: item[0]):
        if structured_variant not in by_variant or free_variant not in by_variant:
            continue
        structured = _record_summary(by_variant[structured_variant])
        free = _record_summary(by_variant[free_variant])
        pairs.append(
            {
                "stage": key[0],
                "split_role": key[1],
                "dataset": key[2],
                "detector": key[3],
                "tokens_per_tubelet": key[4],
                "seed": key[5],
                "structured_variant": structured_variant,
                "free_variant": free_variant,
                "delta_average_map": structured["average_map"] - free["average_map"],
                "delta_high_iou_mean": structured["high_iou_mean"] - free["high_iou_mean"],
                "delta_p50_ms": structured["end_to_end_p50_ms"] - free["end_to_end_p50_ms"],
                "delta_energy_j": structured["gross_gpu_energy_j"] - free["gross_gpu_energy_j"],
            }
        )
    return pairs


def analyze(
    records: Iterable[Mapping[str, Any]], *, development_only: bool, structured_variant: str
) -> Dict[str, Any]:
    validated = validate_records(records, development_only=development_only)
    summaries = summarize_groups(validated)
    pairs = paired_deltas(
        validated,
        structured_variant=structured_variant,
        free_variant="free_token_select",
    )
    pair_summary: Dict[str, Dict[str, float]] = {}
    if pairs:
        for metric in (
            "delta_average_map",
            "delta_high_iou_mean",
            "delta_p50_ms",
            "delta_energy_j",
        ):
            pair_summary[metric] = _mean_std([pair[metric] for pair in pairs])
    output = {
        "analysis_schema_version": "georoute-paper-analysis-v1",
        "input_schema_version": validated[0]["schema_version"],
        "input_records_sha256": canonical_json_sha256(validated),
        "development_only": development_only,
        "raw_record_count": len(validated),
        "group_summaries": summaries,
        "accuracy_cost_pareto": pareto_frontier(summaries),
        "paired_structured_vs_free": {
            "structured_variant": structured_variant,
            "free_variant": "free_token_select",
            "raw_pairs": pairs,
            "summary": pair_summary,
            "interpretation_guard": (
                "Descriptive paired deltas only. Apply preregistered thresholds and "
                "inspect matched raw rows before any paper claim."
            ),
        },
        "non_claims": [
            "No mAP theorem follows from this report.",
            "No official-test conclusion is produced by development-only analysis.",
            "No causal routing claim follows without matched controls.",
        ],
    }
    output["analysis_sha256"] = canonical_json_sha256(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="JSON or JSONL result records")
    parser.add_argument("--output", required=True, type=Path, help="analysis JSON output path")
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="reject official-test evidence to keep development selection result-blind",
    )
    parser.add_argument(
        "--structured-variant",
        default="roi_residual",
        choices=("roi_only", "roi_residual", "roi_residual_amod"),
        help="structured route paired with free TokenSelect",
    )
    args = parser.parse_args()
    try:
        output = analyze(
            load_records(args.input),
            development_only=args.development_only,
            structured_variant=args.structured_variant,
        )
    except (GeoRouteResultSchemaError, OSError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"analysis_sha256": output["analysis_sha256"], "records": output["raw_record_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
