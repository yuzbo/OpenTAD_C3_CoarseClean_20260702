#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from opentad.models.chronotransport.replay import (
    canonical_record_line,
    paired_detector_losses,
    records_sha256,
)


def _factory(spec: str):
    module_name, function_name = spec.split(":", 1)
    return getattr(importlib.import_module(module_name), function_name)


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
        )
        records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "schedule": schedule,
                "cost": {},
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
