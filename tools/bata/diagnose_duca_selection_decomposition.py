from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.analyze_duca_selection_quality import (
    RADII,
    _boundaries,
    _selection_metrics,
    _validated_segments,
    analyze_record,
    exact_uniform_positions,
)
from tools.bata.duca_ceiling_utils import mean, read_jsonl, sha256, write_csv, write_json


def gt_oracle_positions(row: Mapping[str, Any]) -> list[int]:
    valid_len = int(row["valid_len"])
    budget = min(int(row["budget"]), valid_len)
    segments = _validated_segments(row, valid_len)
    anchors: list[int] = []
    for boundary in _boundaries(valid_len, segments):
        position = min(valid_len - 1, max(0, int(round(boundary))))
        if position not in anchors:
            anchors.append(position)
    if len(anchors) > budget:
        anchors = anchors[:budget]
    for position in exact_uniform_positions(valid_len, budget):
        if len(anchors) >= budget:
            break
        if position not in anchors:
            anchors.append(position)
    if len(anchors) < budget:
        for position in range(valid_len):
            if position not in anchors:
                anchors.append(position)
                if len(anchors) == budget:
                    break
    return sorted(anchors)


def _clustering(positions: Sequence[int]) -> dict[str, float | None]:
    selected = sorted(set(int(value) for value in positions))
    if len(selected) < 2:
        return {"adjacent_fraction": None, "gap_cv": None}
    gaps = [right - left for left, right in zip(selected, selected[1:])]
    gap_mean = sum(gaps) / len(gaps)
    variance = sum((gap - gap_mean) ** 2 for gap in gaps) / len(gaps)
    return {
        "adjacent_fraction": sum(gap == 1 for gap in gaps) / len(gaps),
        "gap_cv": 0.0 if gap_mean == 0 else variance**0.5 / gap_mean,
    }


def decompose_record(row: Mapping[str, Any]) -> dict[str, Any]:
    analyzed = analyze_record(row)
    valid_len = int(row["valid_len"])
    segments = _validated_segments(row, valid_len)
    boundaries = _boundaries(valid_len, segments)
    methods = {
        "exact_uniform": analyzed["selection"]["uniform"]["selected_positions"],
        "learned": analyzed["selection"]["learned"]["selected_positions"],
        "gt_informed_heuristic_evaluation_only": gt_oracle_positions(row),
    }
    repair = row.get("decode_diagnostics", {})
    out: dict[str, Any] = {"sample_id": row["sample_id"], "video_id": analyzed["video_id"], "methods": {}}
    for name, positions in methods.items():
        metrics = _selection_metrics(valid_len=valid_len, positions=positions, segments=segments, boundaries=boundaries)
        metrics.update(_clustering(positions))
        metrics["repair_count"] = int(repair.get("repair_count", repair.get("repair_fill_count", 0))) if name == "learned" else 0
        metrics["repair_ratio"] = metrics["repair_count"] / max(1, metrics["selected_count"])
        out["methods"][name] = metrics
    out["coarse"] = analyzed["coarse"]
    return out


def _flat(row: Mapping[str, Any], method: str) -> dict[str, Any]:
    metrics = row["methods"][method]
    result = {"sample_id": row["sample_id"], "video_id": row["video_id"], "method": method}
    for radius in RADII:
        if radius <= 4:
            result[f"boundary_recall_r{radius}"] = metrics["boundary_recall"][f"r{radius}"]
            result[f"both_endpoint_r{radius}"] = metrics["both_endpoint_coverage"][f"r{radius}"]
    for key in ("mean_endpoint_distance", "max_unselected_hole", "repair_count", "repair_ratio", "adjacent_fraction", "gap_cv"):
        result[key] = metrics[key]
    return result


def run(records_jsonl: str | Path, output_dir: str | Path) -> dict[str, Any]:
    analyzed = [decompose_record(row) for row in read_jsonl(records_jsonl)]
    methods = ("exact_uniform", "learned", "gt_informed_heuristic_evaluation_only")
    csv_rows = [_flat(row, method) for row in analyzed for method in methods]
    aggregate: dict[str, Any] = {}
    for method in methods:
        rows = [item for item in csv_rows if item["method"] == method]
        aggregate[method] = {key: mean(row[key] for row in rows) for key in rows[0] if key not in {"sample_id", "video_id", "method"}}
    coarse = {
        "auroc": mean(row["coarse"]["auroc"] for row in analyzed),
        "auprc": mean(row["coarse"]["auprc"] for row in analyzed),
        "ece": mean(row["coarse"]["ece"] for row in analyzed),
    }
    summary = {
        "schema_version": "duca_selection_quality_decomposition_v1",
        "sample_count": len(analyzed),
        "required_metrics": ["boundary_recall_r0/r1/r2/r4", "both_endpoint", "endpoint_distance", "max_hole", "repair", "clustering", "AUROC", "AUPRC", "ECE"],
        "coarse_macro": coarse,
        "selection_macro": aggregate,
        "input_sha256": sha256(records_jsonl),
        "contract": {"gt_informed_heuristic_evaluation_only": True, "is_optimized_oracle": False, "deployable": False, "matched_valid_len_and_k": True},
    }
    out = Path(output_dir)
    write_json(out / "selection_quality_decomposition.json", summary)
    write_csv(out / "selection_quality_decomposition_per_sample.csv", csv_rows)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Exact-uniform/learned/GT-informed heuristic decomposition.")
    parser.add_argument("--records-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.records_jsonl, args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
