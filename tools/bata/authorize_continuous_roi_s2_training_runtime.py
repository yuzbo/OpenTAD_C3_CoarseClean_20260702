from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_runtime_gate import (  # noqa: E402
    build_runtime_authorization,
)


def _publish_once(path: Path, payload: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorize the exact Continuous-RoI S2 3x3 training campaign"
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--full-model-gate", type=Path, required=True)
    parser.add_argument("--training-runtime-precheck", type=Path, required=True)
    parser.add_argument(
        "--completion", type=Path, action="append", required=True
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite runtime authorization")
        report = build_runtime_authorization(
            expected_commit=args.expected_commit,
            full_model_gate_path=args.full_model_gate,
            training_runtime_precheck_path=args.training_runtime_precheck,
            completion_paths=args.completion,
        )
        _publish_once(args.output, report)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "authorization_sha256": report["authorization_sha256"],
                "campaign_namespace": report["campaign_namespace"],
                "canonical_experiment_root": report[
                    "canonical_experiment_root"
                ],
                "output": str(args.output.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
