from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA = "duca_rime_video_split_manifest_v1"
TRAIN_ROLES = (
    "detector_selector_train",
    "hard_label_generation",
    "utility_risk_fit",
    "dual_risk_calibration",
    "certification_development",
)
DEFAULT_FRACTIONS = (0.50, 0.15, 0.15, 0.10, 0.10)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    )


def _write_immutable(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite a different RIME split: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _largest_remainder_counts(total: int, fractions: Sequence[float]) -> list[int]:
    raw = [total * float(value) for value in fractions]
    counts = [int(value) for value in raw]
    remainder = total - sum(counts)
    order = sorted(
        range(len(raw)),
        key=lambda index: (-(raw[index] - counts[index]), index),
    )
    for index in order[:remainder]:
        counts[index] += 1
    if total >= len(counts) and any(value <= 0 for value in counts):
        raise RuntimeError("largest-remainder split unexpectedly produced an empty role")
    return counts


def _validate_fractions(fractions: Sequence[float]) -> tuple[float, ...]:
    values = tuple(float(value) for value in fractions)
    if len(values) != len(TRAIN_ROLES):
        raise ValueError(f"exactly {len(TRAIN_ROLES)} train-role fractions are required")
    if any(value <= 0.0 for value in values):
        raise ValueError("all RIME split fractions must be positive")
    if abs(sum(values) - 1.0) > 1.0e-12:
        raise ValueError("RIME split fractions must sum exactly to one")
    return values


def create_rime_splits(
    annotation_path: str | Path,
    output_dir: str | Path,
    *,
    seed: int = 3407,
    fractions: Sequence[float] = DEFAULT_FRACTIONS,
) -> dict[str, Any]:
    annotation = Path(annotation_path).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    fractions = _validate_fractions(fractions)
    payload = json.loads(annotation.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("THUMOS annotation must contain a database mapping")
    train_videos = sorted(
        str(video_id)
        for video_id, row in database.items()
        if isinstance(row, Mapping) and str(row.get("subset")) == "training"
    )
    final_rows = [
        (str(video_id), str(row.get("subset")))
        for video_id, row in database.items()
        if isinstance(row, Mapping)
        and str(row.get("subset")) in {"validation", "testing", "test"}
    ]
    final_videos = sorted(
        video_id for video_id, _subset in final_rows
    )
    if len(train_videos) < len(TRAIN_ROLES):
        raise ValueError("THUMOS training subset is too small for five nonempty roles")
    if not final_videos:
        raise ValueError("THUMOS annotation contains no official testing videos")
    ranked = sorted(
        train_videos,
        key=lambda name: hashlib.sha256(
            f"duca-rime|{int(seed)}|{name}".encode("utf-8")
        ).hexdigest(),
    )
    counts = _largest_remainder_counts(len(ranked), fractions)
    roles: dict[str, list[str]] = {}
    cursor = 0
    for role, count in zip(TRAIN_ROLES, counts):
        roles[role] = sorted(ranked[cursor : cursor + count])
        cursor += count
    if cursor != len(ranked):
        raise RuntimeError("RIME split allocation did not consume the training subset")

    all_train = set(train_videos)
    artifact_rows = {}
    for role in TRAIN_ROLES:
        include = roles[role]
        block = sorted(all_train - set(include))
        include_path = output / f"{role}_videos.txt"
        block_path = output / f"{role}_block_list.txt"
        _write_immutable(include_path, "".join(f"{value}\n" for value in include))
        _write_immutable(block_path, "".join(f"{value}\n" for value in block))
        artifact_rows[role] = {
            "video_count": len(include),
            "videos": include,
            "videos_path": str(include_path),
            "videos_sha256": _sha256_file(include_path),
            "block_list_path": str(block_path),
            "block_list_sha256": _sha256_file(block_path),
        }
    final_path = output / "official_final_evaluation_videos.txt"
    _write_immutable(final_path, "".join(f"{value}\n" for value in final_videos))
    manifest = {
        "schema": SCHEMA,
        "dataset": "THUMOS14",
        "seed": int(seed),
        "annotation_path": str(annotation),
        "annotation_sha256": _sha256_file(annotation),
        "train_source_subset": "training",
        "train_video_count": len(train_videos),
        "train_role_fractions": {
            role: fraction for role, fraction in zip(TRAIN_ROLES, fractions)
        },
        "train_roles": artifact_rows,
        "official_final_evaluation": {
            "source_subset": sorted({subset for _video, subset in final_rows}),
            "consumed_during_method_development": False,
            "video_count": len(final_videos),
            "videos": final_videos,
            "videos_path": str(final_path),
            "videos_sha256": _sha256_file(final_path),
        },
    }
    manifest["assignment_sha256"] = _canonical_sha256(
        {
            "seed": manifest["seed"],
            "train_roles": {
                role: artifact_rows[role]["videos"] for role in TRAIN_ROLES
            },
            "official_final_evaluation": final_videos,
        }
    )
    manifest_path = output / "duca_rime_split_manifest.json"
    _write_immutable(
        manifest_path,
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_sha256"] = _sha256_file(manifest_path)
    return manifest


def validate_rime_splits(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = _sha256_file(path)
    if expected_sha256 is not None and actual_sha != str(expected_sha256):
        raise ValueError("RIME split manifest SHA-256 mismatch")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("dataset") != "THUMOS14":
        raise ValueError("RIME split manifest schema/dataset mismatch")
    annotation = Path(manifest["annotation_path"]).resolve()
    if not annotation.is_file() or _sha256_file(annotation) != manifest.get(
        "annotation_sha256"
    ):
        raise ValueError("RIME annotation path/hash drift")
    role_sets = {}
    for role in TRAIN_ROLES:
        row = manifest.get("train_roles", {}).get(role)
        if not isinstance(row, Mapping):
            raise ValueError(f"RIME split role is missing: {role}")
        videos = [str(value) for value in row.get("videos", ())]
        if not videos or len(videos) != len(set(videos)):
            raise ValueError(f"RIME split role {role} is empty or duplicated")
        include_path = Path(row["videos_path"]).resolve()
        block_path = Path(row["block_list_path"]).resolve()
        if (
            not include_path.is_file()
            or _sha256_file(include_path) != row.get("videos_sha256")
            or not block_path.is_file()
            or _sha256_file(block_path) != row.get("block_list_sha256")
        ):
            raise ValueError(f"RIME split artifacts drifted for role {role}")
        include_lines = [
            value.strip()
            for value in include_path.read_text(encoding="utf-8").splitlines()
            if value.strip()
        ]
        if include_lines != videos:
            raise ValueError(f"RIME include list disagrees with role {role}")
        role_sets[role] = set(videos)
    for left_index, left in enumerate(TRAIN_ROLES):
        for right in TRAIN_ROLES[left_index + 1 :]:
            if role_sets[left] & role_sets[right]:
                raise ValueError(f"RIME video leakage between {left} and {right}")
    all_train = set().union(*role_sets.values())
    if len(all_train) != int(manifest.get("train_video_count", -1)):
        raise ValueError("RIME train video count drift")
    for role in TRAIN_ROLES:
        row = manifest["train_roles"][role]
        blocked = {
            value.strip()
            for value in Path(row["block_list_path"])
            .read_text(encoding="utf-8")
            .splitlines()
            if value.strip()
        }
        if blocked != all_train - role_sets[role]:
            raise ValueError(f"RIME block list disagrees with role {role}")
    final = manifest.get("official_final_evaluation")
    if not isinstance(final, Mapping) or final.get(
        "consumed_during_method_development"
    ) is not False:
        raise ValueError("RIME official final split contract is missing")
    final_videos = set(str(value) for value in final.get("videos", ()))
    if not final_videos or final_videos & all_train:
        raise ValueError("RIME official final videos are empty or overlap training")
    final_path = Path(final["videos_path"]).resolve()
    if (
        not final_path.is_file()
        or _sha256_file(final_path) != final.get("videos_sha256")
    ):
        raise ValueError("RIME official final artifact drifted")
    final_lines = [
        value.strip()
        for value in final_path.read_text(encoding="utf-8").splitlines()
        if value.strip()
    ]
    if (
        final_lines != [str(value) for value in final.get("videos", ())]
        or len(final_lines) != int(final.get("video_count", -1))
        or len(final_lines) != len(set(final_lines))
    ):
        raise ValueError("RIME official final file/content/count mismatch")
    expected_assignment = _canonical_sha256(
        {
            "seed": int(manifest["seed"]),
            "train_roles": {
                role: manifest["train_roles"][role]["videos"] for role in TRAIN_ROLES
            },
            "official_final_evaluation": manifest["official_final_evaluation"][
                "videos"
            ],
        }
    )
    if expected_assignment != manifest.get("assignment_sha256"):
        raise ValueError("RIME split assignment hash drift")
    return {
        "ok": True,
        "schema": SCHEMA,
        "manifest_path": str(path),
        "manifest_sha256": actual_sha,
        "assignment_sha256": expected_assignment,
        "train_video_count": len(all_train),
        "official_final_video_count": len(final_videos),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create or validate the six-role DUCA-RIME video split."
    )
    parser.add_argument("--annotation")
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-manifest")
    parser.add_argument("--expected-sha256")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--fractions",
        nargs=len(TRAIN_ROLES),
        type=float,
        default=DEFAULT_FRACTIONS,
        metavar=tuple(role.upper() for role in TRAIN_ROLES),
    )
    args = parser.parse_args()
    if args.validate_manifest:
        result = validate_rime_splits(
            args.validate_manifest,
            expected_sha256=args.expected_sha256,
        )
    else:
        if not args.annotation or not args.output_dir:
            parser.error("--annotation and --output-dir are required for creation")
        result = create_rime_splits(
            args.annotation,
            args.output_dir,
            seed=args.seed,
            fractions=args.fractions,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
