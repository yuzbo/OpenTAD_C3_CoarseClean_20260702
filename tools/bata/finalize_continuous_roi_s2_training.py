from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_training import (
    build_training_completion,
    load_pure_data_config,
    validate_training_completion,
)


def _publish_once(path: Path, report: dict) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate and seal one completed Continuous-RoI S2 run"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        cfg = load_pure_data_config(args.config)
        if args.output.exists():
            existing = json.loads(args.output.read_text(encoding="utf-8"))
            existing = validate_training_completion(
                existing,
                cfg=cfg,
                seed=args.seed,
                checkpoint_path=args.checkpoint,
            )
            print(json.dumps(existing, indent=2, sort_keys=True))
            return 0
        report = build_training_completion(
            cfg=cfg,
            seed=args.seed,
            checkpoint_path=args.checkpoint,
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
            )
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
