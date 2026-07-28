from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    build_s1_manifest,
    write_s1_manifest_bundle,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Freeze the THUMOS14 S1 fit/gate/test protocol"
    )
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build_s1_manifest(
        args.annotation,
    )
    paths = write_s1_manifest_bundle(manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": "PASS",
                "manifest_sha256": manifest["manifest_sha256"],
                "fit_videos": len(manifest["splits"]["fit"]),
                "gate_videos": len(manifest["splits"]["gate"]),
                "sealed_test_videos": len(manifest["splits"]["test"]),
                "outputs": {key: str(value) for key, value in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
