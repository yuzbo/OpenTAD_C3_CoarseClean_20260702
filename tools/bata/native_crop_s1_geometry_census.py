from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.native_crop_s1_contract import (  # noqa: E402
    NATIVE_CROP_GEOMETRY_SCHEMA,
    finalize_self_hash,
    quantiles,
    validate_development_only_manifest,
)
from tools.bata.spatial_zoom_s1_contract import sha256_file  # noqa: E402


def _reject_test_root(video_root: Path) -> None:
    lowered = str(video_root.resolve()).replace("\\", "/").lower()
    forbidden = ("/test data/", "/th14_test_set_mp4", "/sealed_test")
    if any(token in lowered for token in forbidden):
        raise ValueError(
            "Native-Crop geometry census is development-only and refuses a test root: "
            f"{video_root}"
        )


def probe_video_geometry(video_path: Path, ffprobe: str = "ffprobe") -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate:stream_tags=rotate",
        "-of",
        "json",
        str(video_path),
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout)
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ValueError(f"expected one video stream in {video_path}, got {len(streams)}")
    stream = streams[0]
    width = int(stream["width"])
    height = int(stream["height"])
    rotation = int(stream.get("tags", {}).get("rotate", 0)) % 360
    if rotation in {90, 270}:
        width, height = height, width
    if min(width, height) <= 0:
        raise ValueError(f"invalid source geometry for {video_path}: {width}x{height}")
    return {
        "width": width,
        "height": height,
        "rotation_degrees": rotation,
        "nb_frames": stream.get("nb_frames"),
        "avg_frame_rate": stream.get("avg_frame_rate"),
    }


def summarize_records(records: list[dict], crop_sizes: list[int]) -> dict:
    if not records:
        raise ValueError("geometry census received no development videos")
    heights = [row["height"] for row in records]
    widths = [row["width"] for row in records]
    aspects = [row["width"] / row["height"] for row in records]
    crops = {}
    for crop_size in crop_sizes:
        no_padding = [
            min(row["height"], row["width"]) >= crop_size for row in records
        ]
        valid_crop_areas = [
            min(row["height"], crop_size) * min(row["width"], crop_size)
            for row in records
        ]
        area_fractions = [
            float(valid_area) / float(row["height"] * row["width"])
            for row, valid_area in zip(records, valid_crop_areas)
        ]
        valid_pixel_fractions = [
            float(valid_area) / float(crop_size * crop_size)
            for valid_area in valid_crop_areas
        ]
        crops[str(crop_size)] = {
            "no_padding_count": int(sum(no_padding)),
            "no_padding_rate": float(sum(no_padding)) / float(len(records)),
            "padding_count": int(len(records) - sum(no_padding)),
            "crop_area_over_source": quantiles(area_fractions),
            "valid_pixels_if_padded": quantiles(valid_pixel_fractions),
        }
    return {
        "video_count": len(records),
        "height": quantiles(heights),
        "width": quantiles(widths),
        "aspect_ratio_w_over_h": quantiles(aspects),
        "crop_sizes": crops,
    }


def build_geometry_census(
    *,
    manifest_path: Path,
    video_root: Path,
    crop_sizes: list[int],
    ffprobe: str = "ffprobe",
) -> dict:
    _reject_test_root(video_root)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = validate_development_only_manifest(manifest)
    records = []
    by_split = {}
    for split_name in ("fit", "gate"):
        split_records = []
        for video_id in splits[split_name]:
            video_path = video_root / f"{video_id}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)
            geometry = probe_video_geometry(video_path, ffprobe=ffprobe)
            record = {
                "split": split_name,
                "video_id": video_id,
                "path": str(video_path.resolve()),
                "file_size_bytes": int(video_path.stat().st_size),
                **geometry,
            }
            split_records.append(record)
            records.append(record)
        by_split[split_name] = summarize_records(split_records, crop_sizes)
    return finalize_self_hash(
        {
            "schema_version": NATIVE_CROP_GEOMETRY_SCHEMA,
            "manifest_path": str(manifest_path.resolve()),
            "manifest_file_sha256": sha256_file(manifest_path),
            "manifest_sha256": splits["manifest_sha256"],
            "video_root": str(video_root.resolve()),
            "development_splits_probed": ["fit", "gate"],
            "sealed_test_identity_count": len(splits["sealed_test"]),
            "sealed_test_files_probed": 0,
            "annotation_or_gt_read": False,
            "crop_sizes": crop_sizes,
            "summary": {
                "combined": summarize_records(records, crop_sizes),
                **by_split,
            },
            "records": records,
        },
        "census_sha256",
    )


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Development-only source-video geometry census for Native-Crop S1"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--crop-sizes", type=int, nargs="+", default=[96, 112, 128])
    parser.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args(argv)
    crop_sizes = sorted(set(args.crop_sizes))
    if any(size <= 0 or size % 16 for size in crop_sizes):
        raise ValueError("crop sizes must be positive multiples of patch size 16")
    report = build_geometry_census(
        manifest_path=args.manifest,
        video_root=args.video_root,
        crop_sizes=crop_sizes,
        ffprobe=args.ffprobe,
    )
    _atomic_write_json(args.output, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "output": str(args.output.resolve()),
                "census_sha256": report["census_sha256"],
                "video_count": report["summary"]["combined"]["video_count"],
                "sealed_test_files_probed": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
