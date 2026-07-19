from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any

from tools.bata.export_duca_selection_quality import (
    _checkpoint_state,
    selector_state_dict,
)


SCHEMA_VERSION = "duca_allocation_ceiling_input_v1"
SUMMARY_SCHEMA_VERSION = "duca_allocation_ceiling_export_summary_v1"
SCORE_KEYS = (
    "p_action",
    "actionness_logits",
    "transition_policy_scores",
    "raw_transition_scores",
    "abs_delta_p_action",
    "uncertainty",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json_exclusive(path: str | Path, payload: Any) -> None:
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".partial")
    if target.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite JSON artifact: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=True,
                    allow_nan=False,
                )
                + "\n"
            )
            handle.flush()
        temporary.replace(target)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def git_state(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if commit.returncode != 0 or status.returncode != 0:
        raise RuntimeError("failed to resolve Git provenance")
    return {
        "git_commit": commit.stdout.strip(),
        "git_clean": not bool(status.stdout.strip()),
    }


def dataset_provenance(
    dataset_config: Mapping[str, Any],
    dataset: Any,
) -> dict[str, Any]:
    required_paths = {
        "annotation": dataset_config.get("ann_file"),
        "class_map": dataset_config.get("class_map"),
        "data": dataset_config.get("data_path"),
    }
    resolved: dict[str, Path] = {}
    for key, value in required_paths.items():
        if not isinstance(value, (str, Path)) or not str(value):
            raise ValueError(f"dataset config is missing {key} path")
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"dataset {key} path is missing: {path}")
        resolved[key] = path
    if not resolved["annotation"].is_file() or not resolved["class_map"].is_file():
        raise ValueError("dataset annotation and class map must be files")
    if not resolved["data"].is_dir():
        raise ValueError("dataset data path must be a directory")
    data_provenance = data_directory_provenance(resolved["data"])

    data_list = getattr(dataset, "data_list", None)
    if not isinstance(data_list, Sequence) or isinstance(data_list, (str, bytes)):
        raise ValueError("allocation ceiling requires an auditable dataset data_list")
    window_manifest: list[dict[str, Any]] = []
    for index, item in enumerate(data_list):
        if not isinstance(item, Sequence) or isinstance(item, (str, bytes)) or len(item) < 4:
            raise ValueError("allocation ceiling requires sliding-window dataset entries")
        video_id = str(item[0])
        raw_window = _to_python(item[3])
        if not isinstance(raw_window, Sequence) or isinstance(raw_window, (str, bytes)) or not raw_window:
            raise ValueError("allocation ceiling dataset window coordinates are missing")
        window = [float(value) for value in raw_window]
        if any(not math.isfinite(value) for value in window):
            raise ValueError("allocation ceiling dataset window coordinates must be finite")
        window_manifest.append(
            {
                "dataset_index": index,
                "video_id": video_id,
                "window_start_frame": window[0],
                "window_end_frame": window[-1],
                "window_length": len(window),
            }
        )
    return {
        "annotation_path": str(resolved["annotation"]),
        "annotation_sha256": sha256(resolved["annotation"]),
        "class_map_path": str(resolved["class_map"]),
        "class_map_sha256": sha256(resolved["class_map"]),
        "data_path": str(resolved["data"]),
        **data_provenance,
        "dataset_subset_name": str(dataset_config.get("subset_name", "")),
        "dataset_test_mode": bool(dataset_config.get("test_mode", False)),
        "dataset_filter_gt": bool(dataset_config.get("filter_gt", False)),
        "dataset_ioa_thresh": float(dataset_config.get("ioa_thresh", 0.75)),
        "dataset_feature_stride": int(dataset_config.get("feature_stride", -1)),
        "dataset_sample_stride": int(dataset_config.get("sample_stride", 1)),
        "dataset_window_size": int(dataset_config.get("window_size", -1)),
        "dataset_window_overlap_ratio": float(
            dataset_config.get("window_overlap_ratio", 0.25)
        ),
        "dataset_offset_frames": int(dataset_config.get("offset_frames", 0)),
        "dataset_config_sha256": canonical_sha256(_json_plain(dataset_config)),
        "dataset_window_manifest_sha256": canonical_sha256(window_manifest),
        "dataset_window_count": len(window_manifest),
        "dataset_window_deduplication": "exact_video_start_identity_keep_first",
        "dataset_duplicate_window_count_removed": int(
            getattr(dataset, "_duca_duplicate_window_count_removed", 0)
        ),
    }


def data_directory_provenance(data_path: str | Path) -> dict[str, Any]:
    root = Path(data_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"dataset data path must be a directory: {root}")
    manifest: list[dict[str, Any]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            target_text = path.readlink().as_posix()
            target = path.resolve(strict=True)
            if not target.is_file():
                raise ValueError(
                    f"dataset data symlink must resolve to a regular file: {path}"
                )
            stat = target.stat()
            relative = path.relative_to(root).as_posix()
            manifest.append(
                {
                    "relative_path": relative,
                    "entry_type": "symlink_to_regular_file",
                    "symlink_target": target_text,
                    "resolved_target": str(target),
                    "size_bytes": int(stat.st_size),
                    "sha256": sha256(target),
                }
            )
            total_bytes += int(stat.st_size)
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"dataset data provenance found a non-regular entry: {path}")
        stat = path.stat()
        relative = path.relative_to(root).as_posix()
        manifest.append(
            {
                "relative_path": relative,
                "entry_type": "regular_file",
                "size_bytes": int(stat.st_size),
                "sha256": sha256(path),
            }
        )
        total_bytes += int(stat.st_size)
    if not manifest:
        raise ValueError(f"dataset data directory contains no regular files: {root}")
    return {
        "dataset_data_manifest_sha256": canonical_sha256(manifest),
        "dataset_data_file_count": len(manifest),
        "dataset_data_total_bytes": total_bytes,
        "dataset_data_hash_algorithm": "sha256_full_file_and_symlink_target_v1",
    }


def deduplicate_sliding_windows(dataset: Any) -> int:
    data_list = getattr(dataset, "data_list", None)
    if not isinstance(data_list, Sequence) or isinstance(data_list, (str, bytes)):
        raise ValueError("allocation ceiling requires an auditable dataset data_list")
    unique: list[Any] = []
    seen: dict[str, tuple[float, ...]] = {}
    removed = 0
    for item in data_list:
        if (
            not isinstance(item, Sequence)
            or isinstance(item, (str, bytes))
            or len(item) < 4
        ):
            raise ValueError("allocation ceiling requires sliding-window dataset entries")
        window = _to_python(item[3])
        if (
            not isinstance(window, Sequence)
            or isinstance(window, (str, bytes))
            or not window
        ):
            raise ValueError("allocation ceiling dataset window coordinates are missing")
        coordinates = tuple(float(value) for value in window)
        identity = f"{item[0]}|{int(round(coordinates[0]))}"
        previous = seen.get(identity)
        if previous is None:
            seen[identity] = coordinates
            unique.append(item)
            continue
        if coordinates != previous:
            raise ValueError(
                "dataset emits one sample identity with conflicting windows: "
                f"{identity}"
            )
        removed += 1
    dataset.data_list = unique
    dataset._duca_duplicate_window_count_removed = removed
    return removed


def _json_plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_plain(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"dataset config contains a non-serializable value: {type(value)!r}")


def _to_python(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def extract_center_frame_indices(meta: Mapping[str, Any], valid_len: int) -> list[float]:
    raw = _to_python(meta.get("frame_inds"))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ValueError("metadata must retain frame_inds")
    if len(raw) < valid_len:
        raise ValueError("frame_inds is shorter than the contiguous valid prefix")
    if valid_len < 1:
        raise ValueError("valid_len must be positive")
    first = raw[0]
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        centers: list[float] = []
        for row in raw[:valid_len]:
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or not row:
                raise ValueError("frame_inds rows must be non-empty clip-index sequences")
            centers.append(float(row[len(row) // 2]))
    else:
        centers = [float(value) for value in raw[:valid_len]]
    if any(not math.isfinite(value) for value in centers):
        raise ValueError("center frame indices must be finite")
    if any(right <= left for left, right in zip(centers, centers[1:])):
        raise ValueError("center frame indices must be strictly increasing on the valid prefix")
    return centers


def audit_regular_grid(
    center_frames: Sequence[float],
    *,
    window_start_frame: float,
    snippet_stride: float,
    tolerance_frames: float,
) -> dict[str, Any]:
    start = float(window_start_frame)
    stride = float(snippet_stride)
    tolerance = float(tolerance_frames)
    if not math.isfinite(start):
        raise ValueError("window_start_frame must be finite")
    if not math.isfinite(stride) or stride <= 0:
        raise ValueError("snippet_stride must be finite and positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("coordinate tolerance must be finite and non-negative")
    expected = [start + index * stride for index in range(len(center_frames))]
    errors = [abs(float(actual) - reference) for actual, reference in zip(center_frames, expected)]
    maximum = max(errors, default=0.0)
    if maximum > tolerance + 1.0e-9:
        raise ValueError(
            "decoded center-frame grid differs from registered regular grid: "
            f"max_abs_error={maximum}, tolerance={tolerance}"
        )
    return {
        "expected_source_frames": expected,
        "max_abs_error_frames": maximum,
        "tolerance_frames": tolerance,
        "passed": True,
    }


def audit_timeline_alignment(
    *,
    decoder_fps: float,
    annotation_fps: float,
    total_frames: int,
) -> dict[str, Any]:
    decoder = float(decoder_fps)
    annotation = float(annotation_fps)
    if not math.isfinite(decoder) or decoder <= 0:
        raise ValueError("decoder_fps must be finite and positive")
    if not math.isfinite(annotation) or annotation <= 0:
        raise ValueError("annotation_fps must be finite and positive")
    frames = int(total_frames)
    if frames < 1:
        raise ValueError("total_frames must be positive")
    absolute_error = abs(decoder - annotation)
    cumulative_drift_frames = (
        float(max(frames - 1, 0)) * abs(annotation / decoder - 1.0)
    )
    tolerance_frames = 1.0
    tolerance_fps = (
        decoder
        if frames == 1
        else tolerance_frames * decoder / float(frames - 1)
    )
    if cumulative_drift_frames > tolerance_frames + 1.0e-9:
        raise ValueError(
            "decoded and annotation timelines are not aligned closely enough: "
            f"decoder_fps={decoder}, annotation_fps={annotation}, "
            f"cumulative_drift_frames={cumulative_drift_frames}, "
            f"tolerance_frames={tolerance_frames}"
        )
    return {
        "decoder_fps": decoder,
        "annotation_fps": annotation,
        "absolute_fps_error": absolute_error,
        "tolerance_fps": tolerance_fps,
        "cumulative_drift_frames": cumulative_drift_frames,
        "tolerance_frames": tolerance_frames,
        "passed": True,
    }


def canonical_gt_segments(
    value: Any,
    *,
    valid_len: int,
) -> list[list[float]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("gt_segments must be a sequence")
    upper = float(valid_len - 1)
    canonical: list[list[float]] = []
    for index, pair in enumerate(value):
        if (
            not isinstance(pair, Sequence)
            or isinstance(pair, (str, bytes))
            or len(pair) != 2
        ):
            raise ValueError(f"gt segment {index} must contain exactly two endpoints")
        start, end = float(pair[0]), float(pair[1])
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError(f"gt segment {index} must be finite")
        if start < 0.0 or end > upper or end < start:
            raise ValueError(
                f"gt segment {index} is outside the exported valid prefix: "
                f"[{start}, {end}] vs [0, {upper}]"
            )
        canonical.append([start, end])
    return canonical


def _strict_score_row(value: Any, batch_index: int, valid_len: int, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"selector output is missing {name}")
    row = _to_python(value[batch_index])
    if not isinstance(row, Sequence) or len(row) < valid_len:
        raise ValueError(f"selector score {name} is shorter than valid_len")
    values = [float(item) for item in row[:valid_len]]
    if any(not math.isfinite(item) for item in values):
        raise ValueError(f"selector score {name} contains non-finite values")
    return values


def _contiguous_valid_len(mask: Any) -> int:
    values = [bool(value) for value in _to_python(mask)]
    valid_len = sum(values)
    if values != [True] * valid_len + [False] * (len(values) - valid_len):
        raise ValueError("allocation ceiling requires one contiguous valid prefix")
    if valid_len < 1:
        raise ValueError("allocation ceiling rejects empty valid prefixes")
    return valid_len


def build_record(
    *,
    selector_output: Mapping[str, Any],
    masks: Any,
    gt_segments: Sequence[Any],
    metas: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    split: str,
    requested_budget: int,
    seen_count: int,
    coordinate_tolerance_frames: float,
) -> list[dict[str, Any]]:
    state = selector_output.get("selector_outputs")
    if not isinstance(state, Mapping):
        raise ValueError("frame selector output is missing selector_outputs")
    score_sources = {
        "p_action": state.get("p_action"),
        "actionness_logits": state.get("actionness_logits"),
        "transition_policy_scores": state.get(
            "transition_policy_scores",
            state.get("center_scores"),
        ),
        "raw_transition_scores": state.get("transition_score"),
        "abs_delta_p_action": state.get("abs_delta_p_action"),
        "uncertainty": state.get("uncertainty"),
    }
    records: list[dict[str, Any]] = []
    for batch_index, meta in enumerate(metas):
        valid_len = _contiguous_valid_len(masks[batch_index])
        center_frames = extract_center_frame_indices(meta, valid_len)
        decoder_fps = float(meta.get("avg_fps", 0.0))
        annotation_fps = float(meta.get("fps", 0.0))
        if not math.isfinite(decoder_fps) or decoder_fps <= 0:
            raise ValueError("metadata avg_fps must be finite and positive")
        if not math.isfinite(annotation_fps) or annotation_fps <= 0:
            raise ValueError("metadata fps must be finite and positive")
        total_frames = int(meta.get("total_frames", 0))
        timeline_audit = audit_timeline_alignment(
            decoder_fps=decoder_fps,
            annotation_fps=annotation_fps,
            total_frames=total_frames,
        )
        window_start = float(meta.get("window_start_frame", 0.0))
        stride = float(meta.get("snippet_stride", 0.0))
        coordinate_audit = audit_regular_grid(
            center_frames,
            window_start_frame=window_start,
            snippet_stride=stride,
            tolerance_frames=coordinate_tolerance_frames,
        )
        video_id = str(meta.get("video_name") or meta.get("video_id") or f"sample_{seen_count + batch_index:06d}")
        sample_id = f"{video_id}|{int(round(window_start))}"
        gt = _to_python(gt_segments[batch_index]) if batch_index < len(gt_segments) else []
        canonical_gt = canonical_gt_segments(gt, valid_len=valid_len)
        if split != "train" and canonical_gt:
            raise ValueError("validation/test allocation export must contain no runtime GT")
        scores = {
            key: _strict_score_row(value, batch_index, valid_len, key)
            for key, value in score_sources.items()
        }
        record = {
            "schema_version": SCHEMA_VERSION,
            "sample_id": sample_id,
            "video_id": video_id,
            "split": str(split),
            "valid_len": valid_len,
            "requested_budget": int(requested_budget),
            "physical_axis": {
                "dense_ordinals": list(range(valid_len)),
                "source_frames": center_frames,
                "seconds": [value / decoder_fps for value in center_frames],
                "decoder_fps": decoder_fps,
                "annotation_fps": annotation_fps,
                "total_frames": total_frames,
            },
            "coordinate_audit": coordinate_audit,
            "timeline_audit": timeline_audit,
            "scores": scores,
            "gt_segments": canonical_gt,
            "gt_segments_unit": "dense_ordinal_aligned_to_exported_physical_axis",
            "gt_role": "privileged_diagnostic_only_never_score_generation",
            "source": dict(source),
            "decision_contract": {
                "offline_full_window": True,
                "gt_passed_to_selector": False,
                "teacher_passed_to_selector": False,
                "detector_backbone_executed": False,
                "valid_prefix_only": True,
            },
        }
        record["record_sha256"] = canonical_sha256(record)
        records.append(record)
    return records


def export_records(
    *,
    config: str | Path,
    checkpoint: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    split: str,
    requested_budget: int,
    device: str = "cuda:0",
    use_ema: str = "auto",
    use_amp: bool = True,
    batch_size: int | None = None,
    num_workers: int | None = None,
    limit_batches: int = 0,
    coordinate_tolerance_frames: float = 0.0,
    validation_authorized: bool = False,
) -> dict[str, Any]:
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models.builder import build_selector

    config_path = Path(config).expanduser().resolve()
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    output_path = Path(output_jsonl).expanduser().resolve()
    summary_path = Path(summary_json).expanduser().resolve()
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("allocation ceiling export never overwrites existing artifacts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(config_path))
    if split not in cfg.dataset or split not in cfg.solver:
        raise ValueError(f"config must define dataset.{split} and solver.{split}")
    dataset_cfg = cfg.dataset[split]
    subset_name = str(dataset_cfg.get("subset_name", "")).strip().lower()
    test_mode = bool(dataset_cfg.get("test_mode", False))
    if split == "train":
        if validation_authorized:
            raise ValueError("training-side export must not carry validation authorization")
        if test_mode:
            raise ValueError("training-side export requires test_mode=False")
        if subset_name != "training":
            raise ValueError("training-side export requires subset_name='training'")
    else:
        if not validation_authorized:
            raise ValueError("validation/test export requires explicit authorization")
        if test_mode is not True:
            raise ValueError("validation/test export requires test_mode=True and no runtime GT")
        if subset_name not in {"validation", "testing", "test"}:
            raise ValueError(
                "validation/test export requires a validation/testing subset_name"
            )
    dataset = build_dataset(dataset_cfg, default_args=dict(logger=None))
    deduplicate_sliding_windows(dataset)
    dataset_source = dataset_provenance(dataset_cfg, dataset)
    loader_cfg = dict(cfg.solver[split])
    if batch_size is not None:
        loader_cfg["batch_size"] = int(batch_size)
    if num_workers is not None:
        loader_cfg["num_workers"] = int(num_workers)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **loader_cfg,
    )
    selector = build_selector(cfg.model.frame_selector)
    checkpoint_payload = torch.load(str(checkpoint_path), map_location="cpu")
    if not isinstance(checkpoint_payload, Mapping):
        raise ValueError("checkpoint must be a mapping")
    state_key, full_state = _checkpoint_state(checkpoint_payload, use_ema=use_ema)
    selector.load_state_dict(selector_state_dict(full_state), strict=True)
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)
    selector = selector.to(torch_device).eval()
    repo_root = config_path.parents[3]
    source = {
        **git_state(repo_root),
        **dataset_source,
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_state_key": state_key,
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "split": split,
        "selector_only_inference": True,
        "detector_backbone_executed": False,
        "uses_gt_for_score_generation": False,
        "validation_authorized": bool(validation_authorized),
    }

    sample_count = 0
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial artifact exists: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8") as handle, torch.no_grad():
            for batch_index, data in enumerate(loader):
                if limit_batches > 0 and batch_index >= limit_batches:
                    break
                if split != "train" and (
                    "gt_segments" in data or "gt_labels" in data
                ):
                    raise ValueError(
                        "validation/test loader exposed runtime GT to allocation export"
                    )
                if split == "train" and (
                    "gt_segments" not in data or "gt_labels" not in data
                ):
                    raise ValueError("training allocation export requires GT diagnostics")
                inputs = data["inputs"].to(torch_device, non_blocking=True)
                masks = data["masks"].to(torch_device, non_blocking=True)
                metas = [dict(item) for item in data.get("metas", [{} for _ in range(inputs.shape[0])])]
                gt_segments = data.get(
                    "gt_segments",
                    [[] for _ in range(inputs.shape[0])],
                )
                with torch.cuda.amp.autocast(
                    dtype=torch.float16,
                    enabled=bool(use_amp and torch_device.type == "cuda"),
                ):
                    output = selector.forward_test(inputs=inputs, masks=masks, metas=metas)
                records = build_record(
                    selector_output=output,
                    masks=masks,
                    gt_segments=gt_segments,
                    metas=metas,
                    source=source,
                    split=split,
                    requested_budget=requested_budget,
                    seen_count=sample_count,
                    coordinate_tolerance_frames=coordinate_tolerance_frames,
                )
                for record in records:
                    handle.write(
                        json.dumps(
                            record,
                            sort_keys=True,
                            ensure_ascii=True,
                            allow_nan=False,
                        )
                        + "\n"
                    )
                sample_count += len(records)
                if batch_index % 20 == 0:
                    print(
                        json.dumps({"batch": batch_index, "samples": sample_count}, sort_keys=True),
                        flush=True,
                    )
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": sha256(output_path),
        "sample_count": sample_count,
        "requested_budget": int(requested_budget),
        "limit_batches": int(limit_batches),
        "coordinate_tolerance_frames": float(coordinate_tolerance_frames),
        "source": source,
        "decision_contract": {
            "offline_full_window": True,
            "gt_passed_to_selector": False,
            "teacher_passed_to_selector": False,
            "detector_backbone_executed": False,
            "actual_decoded_frame_indices_retained": True,
            "validation_authorized": bool(validation_authorized),
        },
    }
    write_json_exclusive(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Export fail-closed physical-axis DUCA allocation-ceiling inputs."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--requested-budget", type=int, default=384)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--use-ema", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--coordinate-tolerance-frames", type=float, default=0.0)
    parser.add_argument("--validation-authorized", action="store_true")
    args = parser.parse_args(argv)
    summary = export_records(
        config=args.config,
        checkpoint=args.checkpoint,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        split=args.split,
        requested_budget=args.requested_budget,
        device=args.device,
        use_ema=args.use_ema,
        use_amp=not args.no_amp,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        limit_batches=args.limit_batches,
        coordinate_tolerance_frames=args.coordinate_tolerance_frames,
        validation_authorized=args.validation_authorized,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
