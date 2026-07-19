from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.native_crop_s1_contract import (
    NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256,
    NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT,
    NATIVE_CROP_SOURCE_ANNOTATION_SHA256,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file


def build_development_annotation(source_path: Path) -> tuple[str, dict]:
    if sha256_file(source_path) != NATIVE_CROP_SOURCE_ANNOTATION_SHA256:
        raise ValueError("Native-Crop source annotation SHA-256 mismatch")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    database = source.get("database")
    if not isinstance(database, dict):
        raise ValueError("Native-Crop source annotation has no database mapping")
    development_database = {
        str(video_id): video
        for video_id, video in database.items()
        if video.get("subset") == "training"
    }
    if len(development_database) != NATIVE_CROP_DEVELOPMENT_VIDEO_COUNT:
        raise ValueError("Native-Crop source annotation has an unexpected training split")
    if any(
        video.get("subset") != "training"
        for video in development_database.values()
    ):
        raise ValueError("Native-Crop development annotation retained another subset")
    output = dict(source)
    output["database"] = development_database
    text = json.dumps(output, indent=2, sort_keys=True) + "\n"
    output_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if output_sha256 != NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256:
        raise ValueError("Native-Crop development annotation is not reproducible")
    return text, {
        "source_annotation_sha256": NATIVE_CROP_SOURCE_ANNOTATION_SHA256,
        "development_annotation_sha256": output_sha256,
        "development_video_count": len(development_database),
        "retained_subsets": ["training"],
        "official_test_video_files_opened": 0,
        "preparation_reads_full_annotation_metadata": True,
    }


def publish_immutable(path: Path, text: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != (
            NATIVE_CROP_DEVELOPMENT_ANNOTATION_SHA256
        ):
            raise FileExistsError(f"existing development annotation differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the immutable development-only THUMOS annotation used by "
            "the Native-Crop gate; this is not a gate or test entrypoint."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    text, report = build_development_annotation(args.source)
    publish_immutable(args.output, text)
    report["output"] = str(args.output.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
