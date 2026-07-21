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


def _resolved_file(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"split manifest {field} must be a nonempty absolute path")
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"split manifest {field} must be absolute")
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _nonempty_unique_lines(path: Path, *, field: str) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"{field} must contain nonempty unique video names")
    return values


def validate_split_manifest(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
    annotation_path: str | Path | None = None,
    train_block_list: str | Path | None = None,
    holdout_block_list: str | Path | None = None,
) -> dict[str, Any]:
    manifest_file = Path(manifest_path).expanduser().resolve()
    if not manifest_file.is_file():
        raise FileNotFoundError(manifest_file)
    manifest_sha256 = _sha256_file(manifest_file)
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise ValueError("frontend split manifest SHA256 mismatch")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("schema") != "duca_frontend_train_holdout_split_v2":
        raise ValueError("frontend split manifest schema mismatch")
    if manifest.get("dataset") != "THUMOS14" or manifest.get("source_subset") != "training":
        raise ValueError("frontend split dataset/subset contract mismatch")
    if manifest.get("test_subset_consumed") is not False:
        raise ValueError("frontend split must not consume the test subset")

    annotation = _resolved_file(manifest.get("annotation_path"), field="annotation_path")
    train_block = _resolved_file(manifest.get("train_block_list"), field="train_block_list")
    holdout_block = _resolved_file(manifest.get("holdout_block_list"), field="holdout_block_list")
    for path, sha_field in (
        (annotation, "annotation_sha256"),
        (train_block, "train_block_list_sha256"),
        (holdout_block, "holdout_block_list_sha256"),
    ):
        if _sha256_file(path) != manifest.get(sha_field):
            raise ValueError(f"frontend split artifact hash drift: {sha_field}")

    expected_paths = (
        (annotation_path, annotation, "annotation_path"),
        (train_block_list, train_block, "train_block_list"),
        (holdout_block_list, holdout_block, "holdout_block_list"),
    )
    for expected, recorded, field in expected_paths:
        if expected is not None and Path(expected).expanduser().resolve() != recorded:
            raise ValueError(f"frontend split runtime path disagrees with {field}")

    train_videos = [str(value) for value in manifest.get("train_videos", [])]
    holdout_videos = [str(value) for value in manifest.get("holdout_videos", [])]
    if not train_videos or not holdout_videos:
        raise ValueError("frontend split video assignments must be nonempty")
    if len(train_videos) != len(set(train_videos)) or len(holdout_videos) != len(set(holdout_videos)):
        raise ValueError("frontend split video assignments must be unique")
    if set(train_videos) & set(holdout_videos):
        raise ValueError("frontend train and holdout assignments overlap")
    if int(manifest.get("train_video_count", -1)) != len(train_videos):
        raise ValueError("frontend train video count drift")
    if int(manifest.get("holdout_video_count", -1)) != len(holdout_videos):
        raise ValueError("frontend holdout video count drift")
    if int(manifest.get("video_count", -1)) != len(train_videos) + len(holdout_videos):
        raise ValueError("frontend total video count drift")

    # A block list contains the videos excluded from that consumer.
    if sorted(_nonempty_unique_lines(train_block, field="train block list")) != sorted(holdout_videos):
        raise ValueError("frontend train block list disagrees with holdout assignment")
    if sorted(_nonempty_unique_lines(holdout_block, field="holdout block list")) != sorted(train_videos):
        raise ValueError("frontend holdout block list disagrees with train assignment")
    expected_assignment_sha256 = _canonical_sha256(
        {
            "seed": int(manifest["seed"]),
            "train_videos": train_videos,
            "holdout_videos": holdout_videos,
        }
    )
    if manifest.get("assignment_sha256") != expected_assignment_sha256:
        raise ValueError("frontend split assignment hash drift")
    return {
        "ok": True,
        "schema": manifest["schema"],
        "manifest_path": str(manifest_file),
        "manifest_sha256": manifest_sha256,
        "annotation_path": str(annotation),
        "annotation_sha256": manifest["annotation_sha256"],
        "train_block_list": str(train_block),
        "train_block_list_sha256": manifest["train_block_list_sha256"],
        "holdout_block_list": str(holdout_block),
        "holdout_block_list_sha256": manifest["holdout_block_list_sha256"],
        "assignment_sha256": expected_assignment_sha256,
    }


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
        "schema": "duca_frontend_train_holdout_split_v2",
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
    parser.add_argument("--annotation")
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-manifest")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--train-block-list")
    parser.add_argument("--holdout-block-list")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    args = parser.parse_args()
    if args.validate_manifest:
        manifest = validate_split_manifest(
            args.validate_manifest,
            expected_manifest_sha256=args.expected_manifest_sha256,
            annotation_path=args.annotation,
            train_block_list=args.train_block_list,
            holdout_block_list=args.holdout_block_list,
        )
    else:
        if not args.annotation or not args.output_dir:
            parser.error("--annotation and --output-dir are required when creating a split")
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
