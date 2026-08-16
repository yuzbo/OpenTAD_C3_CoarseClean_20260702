"""Summarize a complete Q-route telemetry replay without reading labels."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return float(ordered[index])


def summarize(payload: dict) -> dict:
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("telemetry contains no route records")

    all_k: list[int] = []
    window_k_cv: list[float] = []
    zero_tubelets = 0
    tubelets = 0
    area_means: list[float] = []
    width_floor: list[float] = []
    height_floor: list[float] = []
    role_totals = {"context": 0, "roi": 0, "residual": 0}

    for record in records:
        route = record["route"]
        k_values = [int(value) for value in route["k_t"]["values"]]
        if not k_values:
            raise ValueError("route record has no k_t values")
        all_k.extend(k_values)
        tubelets += len(k_values)
        zero_tubelets += sum(value == 0 for value in k_values)
        mean_k = statistics.fmean(k_values)
        window_k_cv.append(
            statistics.pstdev(k_values) / mean_k if mean_k > 0.0 else 0.0
        )
        geometry = route["geometry"]
        area_means.append(float(geometry["area"]["mean"]))
        width_floor.append(float(geometry["width_floor_saturation_rate"]))
        height_floor.append(float(geometry["height_floor_saturation_rate"]))
        for role, count in route["roles"]["aggregate_counts"].items():
            role_totals[role] += int(count)

    selected = sum(role_totals.values())
    return {
        "schema_version": "zoomtoken_q_telemetry_summary_v001",
        "record_count": len(records),
        "dataset_count": int(payload.get("dataset_count", -1)),
        "tubelet_count": tubelets,
        "k_t": {
            "min": min(all_k),
            "p05": _quantile([float(v) for v in all_k], 0.05),
            "p50": _quantile([float(v) for v in all_k], 0.50),
            "p95": _quantile([float(v) for v in all_k], 0.95),
            "max": max(all_k),
            "mean": statistics.fmean(all_k),
            "zero_fraction": zero_tubelets / float(tubelets),
            "within_window_cv_mean": statistics.fmean(window_k_cv),
        },
        "geometry": {
            "area_mean": statistics.fmean(area_means),
            "width_floor_saturation_mean": statistics.fmean(width_floor),
            "height_floor_saturation_mean": statistics.fmean(height_floor),
        },
        "role_fractions": {
            role: count / float(selected) for role, count in role_totals.items()
        },
        "labels_or_ground_truth_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.telemetry.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = summarize(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()

