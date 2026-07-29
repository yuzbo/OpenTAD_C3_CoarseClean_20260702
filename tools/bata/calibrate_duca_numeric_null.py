from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.duca_evidence_io import (
    canonical_sha256,
    with_content_sha256,
    write_json_exclusive_atomic,
)


def calibrate_numeric_null(
    rows: Sequence[Mapping[str, Any]],
    *,
    git_commit: str,
    safety_multiplier: float,
    absolute_floor: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("numeric null calibration requires at least one run")
    safety_multiplier = float(safety_multiplier)
    absolute_floor = float(absolute_floor)
    if (
        not math.isfinite(safety_multiplier)
        or safety_multiplier < 1.0
        or not math.isfinite(absolute_floor)
        or absolute_floor < 0.0
    ):
        raise ValueError("invalid numeric null calibration factors")
    maxima: dict[str, float] = {}
    run_ids = []
    for index, row in enumerate(rows):
        if (
            row.get("split_scope") not in {"training", "train_only_calibration"}
            or row.get("uses_official_final") is not False
        ):
            raise ValueError(f"numeric null run {index} violates split scope")
        run_id = str(row.get("run_id", "")).strip()
        if not run_id or run_id in run_ids:
            raise ValueError("numeric null run IDs must be nonempty and unique")
        run_ids.append(run_id)
        errors = row.get("metric_errors")
        if not isinstance(errors, Mapping) or not errors:
            raise ValueError(f"numeric null run {index} lacks metric errors")
        for key, value in errors.items():
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"numeric null metric {key!r} is invalid")
            maxima[str(key)] = max(maxima.get(str(key), 0.0), numeric)
    thresholds = {
        key: max(absolute_floor, value * safety_multiplier)
        for key, value in sorted(maxima.items())
    }
    payload = {
        "schema": "duca_numeric_null_calibration_v1",
        "status": "frozen",
        "git_commit": str(git_commit),
        "fit_scope": "training_or_train_only_calibration",
        "uses_official_final": False,
        "frozen_before_candidate_development": True,
        "safety_multiplier": safety_multiplier,
        "absolute_floor": absolute_floor,
        "run_ids": run_ids,
        "run_count": len(rows),
        "source_rows_sha256": canonical_sha256([dict(row) for row in rows]),
        "observed_maxima": maxima,
        "thresholds": thresholds,
        "scientific_noninferiority_margin": None,
        "paper_claim_allowed": False,
    }
    return with_content_sha256(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze DUCA admission-v2 numeric null thresholds."
    )
    parser.add_argument("--runs-jsonl", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--safety-multiplier", type=float, default=2.0)
    parser.add_argument("--absolute-floor", type=float, default=1.0e-7)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in Path(args.runs_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    payload = calibrate_numeric_null(
        rows,
        git_commit=args.git_commit,
        safety_multiplier=args.safety_multiplier,
        absolute_floor=args.absolute_floor,
    )
    write_json_exclusive_atomic(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
