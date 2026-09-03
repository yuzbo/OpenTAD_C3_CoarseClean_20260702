#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.replay import validate_compact_record


def _single_record(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError(f"dense gate ledger must contain exactly one row: {path}")
    return raw, validate_compact_record(json.loads(lines[0]))


def validate(first: Path, second: Path, *, tolerance: float = 1e-6) -> dict:
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("dense gate tolerance must be finite and non-negative")
    first_raw, first_row = _single_record(Path(first))
    second_raw, second_row = _single_record(Path(second))
    if first_raw != second_raw:
        raise ValueError("dense gate ledgers must be byte-identical")
    if first_row["schedule"] != "dense" or second_row["schedule"] != "dense":
        raise ValueError("dense gate requires schedule=dense")
    targets = first_row.get("pooled_targets")
    if not isinstance(targets, dict):
        raise ValueError("dense gate requires pooled_targets")
    dense_loss = float(targets["dense_loss"])
    counterfactual_loss = float(targets["counterfactual_loss"])
    regret = float(first_row["regret"])
    values = (dense_loss, counterfactual_loss, regret)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("dense gate values must be finite")
    loss_delta = abs(counterfactual_loss - dense_loss)
    if regret < 0.0 or regret > tolerance or loss_delta > tolerance:
        raise ValueError("dense gate regret/loss delta exceeds tolerance")
    return {
        "status": "PASS",
        "sha256": hashlib.sha256(first_raw).hexdigest(),
        "regret": regret,
        "absolute_loss_delta": loss_delta,
        "tolerance": tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    print(json.dumps(validate(args.first, args.second, tolerance=args.tolerance), indent=2))


if __name__ == "__main__":
    main()
