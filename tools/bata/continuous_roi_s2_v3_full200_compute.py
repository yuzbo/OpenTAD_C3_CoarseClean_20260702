from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config


PROTOCOL_ID = "ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FULL200-COMPUTE-PARETO-3X3-v001"
ARMS = ("D160", "G96", "U128-A0")
SEEDS = (4407, 4408, 4409)
EXPECTED_TRAINING_IDENTITIES = 200
EXPECTED_EVALUATION_VIDEOS = 211
EXPECTED_EVALUATION_WINDOWS = 792
EXPECTED_UPDATES_PER_EPOCH = 100
EXPECTED_EPOCHS = 60
EXPECTED_TOTAL_UPDATES = 6000
EXPECTED_WORLD_SIZE = 2
FEATURE_STRIDE = 4
WINDOW_SIZE = 768
WINDOW_OVERLAP_RATIO = 0.5
WINDOW_STRIDE = 384
SHORT_Q1_SCHEMA = "ZT_SHORT_Q1_LINEAR_V1"


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_publish_json(path: str | Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(canonical_json_bytes(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _finite_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _read_class_map(path: str | Path) -> list[str]:
    rows = [line.rstrip("\r\n") for line in Path(path).read_text(encoding="utf-8").splitlines()]
    if not rows or any(not row for row in rows) or len(rows) != len(set(rows)):
        raise ValueError("class map must contain unique non-empty foreground labels")
    return rows


def _linear_q1(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("short-Q1 requires at least one foreground duration")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in ordered):
        raise ValueError("short-Q1 duration population must be finite and positive")
    h = (len(ordered) - 1) * 0.25
    index = math.floor(h)
    fraction = h - index
    if index == len(ordered) - 1:
        return ordered[index]
    return (1.0 - fraction) * ordered[index] + fraction * ordered[index + 1]


def _window_rows(
    video_rows: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ordinal = 0
    for video_id, info in video_rows:
        frame_count = int(info["frame"])
        duration = _finite_float(info["duration"], field=f"{video_id}.duration")
        if frame_count <= 0 or duration <= 0:
            raise ValueError(f"{video_id} has invalid frame or duration metadata")
        snippet_centers = list(range(0, frame_count, FEATURE_STRIDE))
        if not snippet_centers:
            raise ValueError(f"{video_id} has no snippet center")
        snippet_count = len(snippet_centers)
        last_window = False
        for window_index in range(max(1, snippet_count // WINDOW_STRIDE)):
            start_index = window_index * WINDOW_STRIDE
            end_index = start_index + WINDOW_SIZE
            if end_index > snippet_count:
                end_index = snippet_count
                start_index = max(0, end_index - WINDOW_SIZE)
                last_window = True
            centers = snippet_centers[start_index:end_index]
            if not centers:
                raise ValueError(f"{video_id} produced an empty evaluation window")
            rows.append(
                {
                    "ordinal": ordinal,
                    "video_id": video_id,
                    "video_window_index": window_index,
                    "window_start_frame": centers[0],
                    "window_end_frame": centers[-1],
                    "snippet_count": len(centers),
                }
            )
            ordinal += 1
            if last_window:
                break
    return rows


def _media_inventory(
    media_root: str | Path,
    database: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    root = Path(media_root).resolve()
    candidates: dict[str, list[Path]] = {}
    for path in sorted(Path(media_root).rglob("*.mp4"), key=lambda item: item.as_posix()):
        candidates.setdefault(path.stem, []).append(path)
    expected_ids = set(map(str, database))
    if set(candidates) != expected_ids:
        missing = sorted(expected_ids - set(candidates))
        extra = sorted(set(candidates) - expected_ids)
        raise ValueError(f"media inventory identity mismatch: missing={missing} extra={extra}")
    rows: list[dict[str, Any]] = []
    for video_id, info in database.items():
        paths = candidates[str(video_id)]
        if len(paths) != 1:
            raise ValueError(f"{video_id} maps to {len(paths)} MP4 paths")
        path = paths[0]
        if not path.exists():
            raise ValueError(f"broken media path: {path}")
        resolved = path.resolve(strict=True)
        rows.append(
            {
                "video_id": str(video_id),
                "subset": str(info.get("subset")),
                "relative_path": path.resolve().relative_to(root).as_posix()
                if path.resolve().is_relative_to(root)
                else path.relative_to(media_root).as_posix(),
                "resolved_path": resolved.as_posix(),
                "size_bytes": resolved.stat().st_size,
                "sha256": sha256_file(resolved),
            }
        )
    rows.sort(key=lambda row: (row["subset"], row["video_id"], row["resolved_path"]))
    return rows


def build_full_data_bundle(
    annotation_path: str | Path,
    class_map_path: str | Path,
    media_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Seal the complete train/evaluation population before formal execution."""

    annotation_path = Path(annotation_path).resolve()
    class_map_path = Path(class_map_path).resolve()
    media_root = Path(media_root).resolve()
    output_dir = Path(output_dir).resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("annotation must contain a database mapping")
    database = {str(video_id): info for video_id, info in database.items()}
    training_rows = [
        (video_id, info)
        for video_id, info in database.items()
        if str(info.get("subset")) == "training"
    ]
    validation_rows = [
        (video_id, info)
        for video_id, info in database.items()
        if str(info.get("subset")) == "validation"
    ]
    if len(training_rows) != EXPECTED_TRAINING_IDENTITIES:
        raise ValueError(f"training identity count is {len(training_rows)}, expected 200")
    if len(validation_rows) != EXPECTED_EVALUATION_VIDEOS:
        raise ValueError(f"validation identity count is {len(validation_rows)}, expected 211")
    if set(video_id for video_id, _ in training_rows) & set(
        video_id for video_id, _ in validation_rows
    ):
        raise ValueError("training and validation identities overlap")

    classes = _read_class_map(class_map_path)
    class_index = {label: index for index, label in enumerate(classes)}
    duration_rows: list[dict[str, Any]] = []
    excluded_labels: dict[str, int] = {}
    for video_id, info in training_rows:
        annotations = info.get("annotations")
        if not isinstance(annotations, list):
            raise ValueError(f"{video_id} training annotations are missing")
        for annotation_ordinal, annotation in enumerate(annotations):
            label = str(annotation.get("label"))
            if label not in class_index:
                excluded_labels[label] = excluded_labels.get(label, 0) + 1
                continue
            segment = annotation.get("segment")
            if not isinstance(segment, list) or len(segment) != 2:
                raise ValueError(f"{video_id} annotation {annotation_ordinal} has invalid segment")
            start = _finite_float(segment[0], field="annotation start")
            end = _finite_float(segment[1], field="annotation end")
            duration = end - start
            if not math.isfinite(duration) or duration <= 0:
                raise ValueError(f"{video_id} annotation {annotation_ordinal} has non-positive duration")
            duration_rows.append(
                {
                    "video_id": video_id,
                    "annotation_ordinal": annotation_ordinal,
                    "class_index": class_index[label],
                    "start_float64_hex": start.hex(),
                    "end_float64_hex": end.hex(),
                    "duration_float64_hex": duration.hex(),
                }
            )
    duration_rows.sort(key=lambda row: (row["video_id"], row["annotation_ordinal"]))
    q1 = _linear_q1([float.fromhex(row["duration_float64_hex"]) for row in duration_rows])
    windows = _window_rows(validation_rows)
    if len(windows) != EXPECTED_EVALUATION_WINDOWS:
        raise ValueError(f"evaluation window count is {len(windows)}, expected 792")
    media_rows = _media_inventory(media_root, database)
    if len(media_rows) != EXPECTED_TRAINING_IDENTITIES + EXPECTED_EVALUATION_VIDEOS:
        raise ValueError("media inventory is not the complete 411-video population")

    training_only = copy.deepcopy(payload)
    training_only["database"] = {
        video_id: copy.deepcopy(info) for video_id, info in training_rows
    }
    heldout_inference = {
        "database": {
            video_id: {
                "subset": "validation",
                "frame": int(info["frame"]),
                "duration": _finite_float(info["duration"], field=f"{video_id}.duration"),
            }
            for video_id, info in validation_rows
        }
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    training_path = output_dir / "training_only_annotation.json"
    heldout_path = output_dir / "heldout_inference_annotation.json"
    atomic_publish_json(training_path, training_only)
    atomic_publish_json(heldout_path, heldout_inference)
    manifest: dict[str, Any] = {
        "schema_version": "thumos_full200_heldout211_v1",
        "protocol_id": PROTOCOL_ID,
        "annotation": {
            "path": annotation_path.as_posix(),
            "sha256": sha256_file(annotation_path),
        },
        "class_map": {
            "path": class_map_path.as_posix(),
            "sha256": sha256_file(class_map_path),
            "classes": classes,
        },
        "media": {
            "root": media_root.as_posix(),
            "count": len(media_rows),
            "records_sha256": canonical_sha256(media_rows),
            "records": media_rows,
        },
        "training": {
            "identity_count": len(training_rows),
            "identity_order": [video_id for video_id, _ in training_rows],
            "identity_order_sha256": canonical_sha256(
                [video_id for video_id, _ in training_rows]
            ),
            "training_only_annotation": training_path.as_posix(),
            "training_only_annotation_sha256": sha256_file(training_path),
        },
        "evaluation": {
            "video_count": len(validation_rows),
            "video_order": [video_id for video_id, _ in validation_rows],
            "video_cluster_order": sorted(video_id for video_id, _ in validation_rows),
            "video_cluster_order_sha256": canonical_sha256(
                sorted(video_id for video_id, _ in validation_rows)
            ),
            "ordered_window_count": len(windows),
            "ordered_windows_sha256": canonical_sha256(windows),
            "ordered_windows": windows,
            "heldout_inference_annotation": heldout_path.as_posix(),
            "heldout_inference_annotation_sha256": sha256_file(heldout_path),
        },
        "short_q1": {
            "schema": SHORT_Q1_SCHEMA,
            "source_split": "training",
            "source_video_count": len(training_rows),
            "source_annotation_sha256": sha256_file(annotation_path),
            "source_class_map_sha256": sha256_file(class_map_path),
            "training_video_manifest_sha256": canonical_sha256(
                [video_id for video_id, _ in training_rows]
            ),
            "duration_unit": "seconds",
            "foreground_membership": "frozen_class_map",
            "invalid_duration_policy": "OBJECTIVE_BLOCKER",
            "excluded_non_foreground_labels": dict(sorted(excluded_labels.items())),
            "duration_count": len(duration_rows),
            "duration_rows_sha256": canonical_sha256(duration_rows),
            "quantile": 0.25,
            "quantile_formula": "h=(n-1)q;linear_interpolation",
            "q1_float64_hex": q1.hex(),
            "q1_decimal17": format(q1, ".17g"),
            "short_membership": "0<duration<=q1",
        },
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    atomic_publish_json(output_dir / "full_data_manifest.json", manifest)
    return manifest


def require_clean_commit(expected_commit: str, root: str | Path) -> None:
    root = Path(root).resolve()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if actual != expected_commit:
        raise RuntimeError(f"candidate commit mismatch: {actual} != {expected_commit}")
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if status:
        raise RuntimeError("formal execution requires a clean candidate checkout")


def config_path(root: str | Path, arm: str, seed: int) -> Path:
    names = {
        "D160": "d160",
        "G96": "g96",
        "U128-A0": "u128_a0",
    }
    if arm not in names or seed not in SEEDS:
        raise ValueError(f"unsupported cell arm={arm!r} seed={seed!r}")
    return (
        Path(root)
        / "configs"
        / "adatad"
        / "thumos"
        / f"continuous_roi_s2_v3_{names[arm]}_seed{seed}.py"
    )


def _pipeline_step(cfg: Config, split: str, type_name: str) -> Any:
    for step in cfg.dataset[split].pipeline:
        if step.type == type_name:
            return step
    raise ValueError(f"{split} pipeline is missing {type_name}")


def validate_cell_config(path: str | Path, *, arm: str, seed: int) -> dict[str, Any]:
    path = Path(path)
    cfg = Config.fromfile(path)
    binding = cfg.continuous_roi_s2_v3_full200_compute
    if binding.protocol != PROTOCOL_ID or binding.arm != arm or int(binding.seed) != seed:
        raise ValueError("cell protocol, arm, or seed binding changed")
    if cfg.dataset.train.subset_name != "training":
        raise ValueError("formal training must use the complete training split")
    for split in ("val", "test"):
        dataset = cfg.dataset[split]
        if (
            dataset.subset_name != "validation"
            or int(dataset.window_size) != 768
            or float(dataset.window_overlap_ratio) != 0.5
            or not bool(dataset.test_mode)
        ):
            raise ValueError("formal evaluation population geometry changed")
    if (
        int(cfg.solver.train.batch_size) != 2
        or not bool(cfg.solver.ema)
        or int(cfg.workflow.max_train_iters) != EXPECTED_UPDATES_PER_EPOCH
        or int(cfg.workflow.end_epoch) != EXPECTED_EPOCHS
        or str(cfg.workflow.checkpoint_policy) != "final_only"
        or int(cfg.inference.test_epoch) != 59
    ):
        raise ValueError("formal schedule or final-EMA rule changed")
    if not bool(cfg.workflow.schedule_and_ema_on_success_only):
        raise ValueError("scheduler and EMA must advance only on successful updates")
    if not bool(cfg.workflow.fail_on_skipped_update):
        raise ValueError("a cell must fail rather than silently lose an update")

    if arm == "D160":
        step = _pipeline_step(cfg, "train", "FullFrameLetterboxView")
        expected = 160
    elif arm == "G96":
        step = _pipeline_step(cfg, "train", "FullFrameLetterboxView")
        expected = 96
    else:
        step = _pipeline_step(cfg, "train", "NativeCropSourceViews")
        if (
            int(step.global_size) != 96
            or int(step.local_size) != 128
            or bool(step.allow_local_padding)
        ):
            raise ValueError("U128-A0 fixed source-native view changed")
        custom = cfg.model.backbone.custom
        if (
            custom.wrapper_type != "native_crop_shared_videomae"
            or custom.native_crop_fusion_mode != "fixed_mean"
        ):
            raise ValueError("U128-A0 must use one shared backbone and parameter-free fusion")
        expected = None
    if expected is not None and int(step.output_size) != expected:
        raise ValueError(f"{arm} spatial resolution changed")

    forbidden = ("fit160", "gate40", "129-window", "sobol")
    lowered = path.read_text(encoding="utf-8").lower()
    if any(token in lowered for token in forbidden):
        raise ValueError("a superseded partial-data route entered the formal config")
    return {
        "arm": arm,
        "seed": seed,
        "config": str(path.resolve()),
        "work_dir": str(cfg.work_dir),
        "final_checkpoint": "checkpoint/epoch_59.pth:state_dict_ema",
    }


def validate_matrix(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    cells = [
        validate_cell_config(config_path(root, arm, seed), arm=arm, seed=seed)
        for arm in ARMS
        for seed in SEEDS
    ]
    return {
        "protocol_id": PROTOCOL_ID,
        "cells": cells,
        "cell_count": len(cells),
        "training_identities": EXPECTED_TRAINING_IDENTITIES,
        "evaluation_videos": EXPECTED_EVALUATION_VIDEOS,
        "evaluation_ordered_windows": EXPECTED_EVALUATION_WINDOWS,
        "successful_updates_per_cell": EXPECTED_TOTAL_UPDATES,
        "world_size": EXPECTED_WORLD_SIZE,
    }


def parameter_surface(cfg: Config) -> dict[str, Any]:
    """Return the parameter-bearing config after removing U128-A0 wrapper metadata."""

    model = copy.deepcopy(cfg.model.to_dict())
    custom = model["backbone"]["custom"]
    for key in list(custom):
        if key == "wrapper_type" or key.startswith("native_crop_"):
            custom.pop(key)
    optimizer = copy.deepcopy(cfg.optimizer.to_dict())
    return {"model": model, "optimizer": optimizer}


def validate_parameter_fairness(root: str | Path) -> None:
    root = Path(root)
    reference = Config.fromfile(config_path(root, "D160", 4407))
    expected = parameter_surface(reference)
    for arm in ARMS:
        cfg = Config.fromfile(config_path(root, arm, 4407))
        if parameter_surface(cfg) != expected:
            raise ValueError(f"{arm} changed the trainable parameter surface")
    u128 = Config.fromfile(config_path(root, "U128-A0", 4407))
    if u128.model.backbone.custom.native_crop_fusion_mode != "fixed_mean":
        raise ValueError("U128-A0 fusion is not parameter-free fixed_mean")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the full-data S2-v3 matrix")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--manifest-dir", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_args = (args.annotation, args.class_map, args.media_root, args.manifest_dir)
    if any(value is not None for value in manifest_args):
        if not all(value is not None for value in manifest_args):
            raise ValueError(
                "--annotation, --class-map, --media-root and --manifest-dir are required together"
            )
        manifest = build_full_data_bundle(
            args.annotation, args.class_map, args.media_root, args.manifest_dir
        )
        print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
        return 0
    receipt = validate_matrix(args.root)
    validate_parameter_fairness(args.root)
    receipt["parameter_fairness"] = "PASS"
    if args.output is not None:
        atomic_publish_json(args.output, receipt)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
