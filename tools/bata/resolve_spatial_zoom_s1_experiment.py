from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    resolve_s1_formal_experiment_identity,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Resolve the unique formal S1 experiment namespace"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--precheck", type=Path, required=True)
    parser.add_argument("--path-only", action="store_true")
    args = parser.parse_args(argv)
    try:
        identity = resolve_s1_formal_experiment_identity(
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            precheck_path=args.precheck,
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    if args.path_only:
        print(identity["canonical_experiment_root"])
    else:
        print(json.dumps({"status": "PASS", **identity}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
