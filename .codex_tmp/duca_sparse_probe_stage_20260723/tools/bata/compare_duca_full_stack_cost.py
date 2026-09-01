from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bata.duca_full_stack_cost import build_cost_matrix, write_cost_matrix_artifacts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare auditable DUCA full-stack cost profiles")
    parser.add_argument("--baseline", required=True, help="dense baseline summary JSON")
    parser.add_argument("--candidate", action="append", required=True, help="candidate summary JSON")
    parser.add_argument("--output-prefix", required=True)
    return parser


def _read_json(path: str) -> dict:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"cost profile missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"cost profile must be a JSON object: {source}")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    baseline = _read_json(args.baseline)
    candidates = [_read_json(path) for path in args.candidate]
    matrix = build_cost_matrix(baseline, candidates)
    paths = write_cost_matrix_artifacts(matrix, args.output_prefix)
    print(json.dumps({key: str(value) for key, value in paths.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
