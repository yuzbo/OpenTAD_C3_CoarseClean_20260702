from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_test_open import (
    build_test_open_certificate,
    create_global_test_open_marker,
    recover_global_test_open_certificate,
)
from tools.bata.spatial_zoom_s1_contract import atomic_publish_json
from tools.bata.spatial_zoom_s1_training import require_clean_git_checkout


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Open the sealed S1 test split once after all 3x3 selections freeze"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--selection", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        recovered = recover_global_test_open_certificate(
            output_path=args.output,
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            selection_paths=args.selection,
        )
        if recovered is not None:
            require_clean_git_checkout(expected_commit=recovered["code_commit"])
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "output": str(args.output),
                        "recovered_from_global_commit": True,
                    },
                    indent=2,
                )
            )
            return 0
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an S1 test-open certificate")
        certificate = build_test_open_certificate(
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            selection_paths=args.selection,
        )
        require_clean_git_checkout(expected_commit=certificate["code_commit"])
        expected_output = (
            Path(certificate["canonical_experiment_root"])
            / "test_open"
            / "test_open_certificate.json"
        ).resolve()
        if args.output.resolve() != expected_output:
            raise ValueError(
                f"S1 test-open certificate path must be canonical: {expected_output}"
            )
        create_global_test_open_marker(certificate)
        atomic_publish_json(args.output, certificate)
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
