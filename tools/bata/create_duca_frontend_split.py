from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _write_exact(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different split artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_split(
    annotation_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 3407,
    holdout_fraction: float = 0.20,
) -> dict[str, Any]:
    annotation = Path(annotation_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    if not 0.0 < float(holdout_fraction) < 0.5:
        raise ValueError("holdout_fraction must lie in (0, 0.5)")
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, dict):
        raise ValueError("THUMOS annotation must contain a database mapping")
    videos = sorted(
        str(video_name)
        for video_name, info in database.items()
        if isinstance(info, dict) and str(info.get("subset")) == "training"
    )
    if len(videos) < 2:
        raise ValueError("training subset must contain at least two videos")
    ranked = sorted(
        videos,
        key=lambda name: hashlib.sha256(f"{int(seed)}|{name}".encode("utf-8")).hexdigest(),
    )
    holdout_count = max(1, min(len(videos) - 1, round(len(videos) * holdout_fraction)))
    holdout_videos = sorted(ranked[:holdout_count])
    train_videos = sorted(ranked[holdout_count:])

    train_block_list = output / "frontend_train_block_list.txt"
    holdout_block_list = output / "frontend_holdout_block_list.txt"
    manifest_path = output / "frontend_split_manifest.json"
    _write_exact(train_block_list, "".join(f"{name}\n" for name in holdout_videos))
    _write_exact(holdout_block_list, "".join(f"{name}\n" for name in train_videos))
    manifest = {
        "schema": "duca_frontend_train_holdout_split_v1",
        "dataset": "THUMOS14",
        "source_subset": "training",
        "seed": int(seed),
        "holdout_fraction": float(holdout_fraction),
        "annotation_path": str(annotation),
        "annotation_sha256": _sha256_file(annotation),
        "video_count": len(videos),
        "train_video_count": len(train_videos),
        "holdout_video_count": len(holdout_videos),
        "train_videos": train_videos,
        "holdout_videos": holdout_videos,
        "train_block_list": str(train_block_list),
        "train_block_list_sha256": _sha256_file(train_block_list),
        "holdout_block_list": str(holdout_block_list),
        "holdout_block_list_sha256": _sha256_file(holdout_block_list),
        "test_subset_consumed": False,
    }
    manifest["assignment_sha256"] = _canonical_sha256(
        {
            "seed": manifest["seed"],
            "train_videos": train_videos,
            "holdout_videos": holdout_videos,
        }
    )
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _write_exact(manifest_path, manifest_text)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a deterministic train-only DUCA frontend split."
    )
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    manifest = create_split(
        args.annotation,
        args.output_dir,
        seed=args.seed,
        holdout_fraction=args.holdout_fraction,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
