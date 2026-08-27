#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics

from opentad.models.chronotransport.cost_lookup import CostLookupKey, ScheduleCostLookup


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * fraction)))
    return ordered[index]


def build(input_path: Path) -> dict:
    groups = {}
    for line in input_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = CostLookupKey(
            hardware=row["hardware"],
            precision=row["precision"],
            batch_size=int(row["batch_size"]),
            candidate_schedule=row["candidate_schedule"],
            selected_rows_per_group=tuple(row["selected_rows_per_group"]),
        )
        groups.setdefault(key, []).append(float(row["latency_ms"]))
    entries = [
        (key, statistics.median(values), _percentile(values, 0.95))
        for key, values in groups.items()
    ]
    return ScheduleCostLookup.payload(entries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build(args.input), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
