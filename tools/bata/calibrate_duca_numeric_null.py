from __future__ import annotations

import argparse
from typing import Any, Mapping, Sequence


def calibrate_numeric_null(
    rows: Sequence[Mapping[str, Any]],
    *,
    git_commit: str,
    safety_multiplier: float,
    absolute_floor: float,
) -> dict[str, Any]:
    raise RuntimeError(
        "Admission v2 formal numeric calibration is disabled: synthetic/head-only "
        "rows cannot authorize an experiment. Use the real-video, grouped, "
        "holdout Admission v2.1 implementation."
    )


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
    calibrate_numeric_null(
        (),
        git_commit=args.git_commit,
        safety_multiplier=args.safety_multiplier,
        absolute_floor=args.absolute_floor,
    )
    raise AssertionError("disabled calibrator unexpectedly returned")


if __name__ == "__main__":
    raise SystemExit(main())
