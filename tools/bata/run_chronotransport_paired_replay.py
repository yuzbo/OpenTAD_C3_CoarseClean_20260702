#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.replay import (
    canonical_record_line,
    paired_detector_losses,
    records_sha256,
)


def _factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


def _compact_runtime_payload(detector) -> tuple[dict, dict]:
    runtimes = [
        module
        for module in detector.modules()
        if module.__class__.__name__ == "ChronoTransportRuntime"
    ]
    if len(runtimes) != 1 or not isinstance(runtimes[0].latest_summary, dict):
        raise RuntimeError("paired replay requires one executed ChronoTransportRuntime")
    summary = runtimes[0].latest_summary
    signals = {
        key: summary[key]
        for key in (
            "selected_schedule_names",
            "action_counts",
            "requested_action_counts",
            "schedule_repair_count",
            "first_chunk_forced_recompute",
        )
        if key in summary
    }
    cost = {
        key: summary[key]
        for key in ("recompute_rows", "transport_rows", "hold_rows", "profile")
        if key in summary
    }
    return signals, cost


def run(factory_spec: str, output: Path, schedule: str, limit: int | None = None) -> dict:
    detector, batches = _factory(factory_spec)()
    records = []
    for index, batch in enumerate(batches):
        if limit is not None and index >= limit:
            break
        sample_id = str(batch.pop("sample_id"))
        split = str(batch.pop("split", "train"))
        result = paired_detector_losses(
            detector,
            batch,
            counterfactual_schedule=schedule,
            track_counterfactual_grad=False,
        )
        signals, cost = _compact_runtime_payload(detector)
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "schedule": schedule,
                "signals": signals,
                "pooled_targets": {
                    "dense_loss": float(result.dense_total.detach().cpu()),
                    "counterfactual_loss": float(result.counterfactual_total.detach().cpu()),
                },
                "cost": cost,
                "regret": float(result.regret.detach().cpu()),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(canonical_record_line(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return {"rows": len(records), "sha256": records_sha256(records), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--factory", required=True, help="module:function returning (detector, batches)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--schedule", default="periodic2_transport")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.factory, args.output, args.schedule, args.limit), indent=2))


if __name__ == "__main__":
    main()
