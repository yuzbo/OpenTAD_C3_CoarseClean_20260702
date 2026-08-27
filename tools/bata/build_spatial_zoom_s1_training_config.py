from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_training import bind_s1_training_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a fail-closed manifest-bound S1 training config"
    )
    parser.add_argument("source_config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError(
                "refusing to overwrite a frozen S1 materialized config"
            )
        cfg = bind_s1_training_config(
            source_config_path=args.source_config,
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            seed=args.seed,
            work_dir=args.work_dir,
            precheck_path=args.precheck,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(args.output))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
