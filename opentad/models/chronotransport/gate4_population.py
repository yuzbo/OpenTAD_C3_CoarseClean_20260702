"""Immutable official full-video/sliding-window Gate-4 population contract."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from .gate4 import ARM_ORDERS
from .protocol import R2_PROTOCOL_ID, canonical_json_bytes, canonical_sha256


GATE4_POPULATION_SCHEMA = "chronotransport-r2-gate4-population-v1"
GATE4_CONFIG_RELATIVE = "configs/adatad/thumos/c3_chronotransport_r2_stage_c.py"
_CONTRACT_FIELDS = {
    "dataset_type",
    "subset",
    "test_mode",
    "feature_stride",
    "sample_stride",
    "offset_frames",
    "window_size",
    "window_overlap_ratio",
    "scale_factor",
    "test_pipeline_sha256",
    "regret_pipeline_sha256",
    "inference_sha256",
    "post_processing_sha256",
    "evaluation_sha256",
}
_VIDEO_FIELDS = {
    "official_video_id",
    "media_path",
    "media_bytes",
    "media_sha256",
    "frame",
    "duration",
}
_INVOCATION_FIELDS = {
    "official_video_id",
    "invocation_id",
    "invocation_index",
    "video_invocation_index",
    "sampled_frame_indices",
    "valid_count",
    "window_start_frame",
    "window_end_frame",
    "invocation_sha256",
}
_BLOCK_FIELDS = {
    "official_video_id",
    "invocation_id",
    "repetition_id",
    "invocation_order_index",
    "arm_order",
    "block_sha256",
}
_ARTIFACT_FIELDS = {
    "schema",
    "protocol",
    "config_relative",
    "config_sources_sha256",
    "annotation",
    "class_map",
    "data_root",
    "dataset_contract",
    "official_video_ids",
    "videos",
    "unique_invocation_count",
    "unique_invocations",
    "unique_invocation_order_sha256",
    "timing_block_count",
    "timing_blocks",
    "timing_block_order_sha256",
    "ground_truth",
    "ground_truth_sha256",
    "fit_manifest_sha256",
    "fit_duration_quartile_thresholds",
    "artifact_sha256",
}


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _relative(value: Any, label: str) -> str:
    text = _text(value, label)
    pure = PurePosixPath(text)
    if (
        pure.is_absolute()
        or "\\" in text
        or any(part in ("", ".", "..") for part in pure.parts)
        or pure.as_posix() != text
    ):
        raise ValueError(f"{label} must be a canonical relative POSIX path")
    return text


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0 or (positive and result <= 0.0):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'non-negative'}")
    return result


def _dataset_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _CONTRACT_FIELDS:
        raise ValueError("Gate4 dataset contract fields mismatch")
    contract = dict(value)
    if (
        contract["dataset_type"] != "ThumosSlidingDataset"
        or contract["subset"] != "validation"
        or contract["test_mode"] is not True
        or _integer(contract["feature_stride"], "Gate4 feature_stride", minimum=1) != 4
        or _integer(contract["sample_stride"], "Gate4 sample_stride", minimum=1) != 1
        or _integer(contract["offset_frames"], "Gate4 offset_frames") != 0
        or _integer(contract["window_size"], "Gate4 window_size", minimum=1) != 768
        or _number(contract["window_overlap_ratio"], "Gate4 overlap") != 0.5
        or _integer(contract["scale_factor"], "Gate4 scale_factor", minimum=1) != 1
    ):
        raise ValueError("Gate4 dataset contract differs from official r2 population")
    for field in (
        "test_pipeline_sha256",
        "regret_pipeline_sha256",
        "inference_sha256",
        "post_processing_sha256",
        "evaluation_sha256",
    ):
        _sha(contract[field], f"Gate4 {field}")
    return contract


def _videos(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("Gate4 videos must be a non-empty sequence")
    rows = []
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != _VIDEO_FIELDS:
            raise ValueError(f"Gate4 video {ordinal} fields mismatch")
        row = {
            "official_video_id": _text(raw["official_video_id"], "Gate4 video ID"),
            "media_path": _relative(raw["media_path"], "Gate4 media path"),
            "media_bytes": _integer(raw["media_bytes"], "Gate4 media bytes", minimum=1),
            "media_sha256": _sha(raw["media_sha256"], "Gate4 media SHA-256"),
            "frame": _integer(raw["frame"], "Gate4 frame count", minimum=1),
            "duration": _number(raw["duration"], "Gate4 duration", positive=True),
        }
        rows.append(row)
    ids = [row["official_video_id"] for row in rows]
    paths = [row["media_path"] for row in rows]
    if len(ids) != len(set(ids)) or len(paths) != len(set(paths)):
        raise ValueError("Gate4 video IDs and media paths must be unique")
    return rows


def _enumerate_invocations(
    videos: Sequence[Mapping[str, Any]], contract: Mapping[str, Any]
) -> list[dict[str, Any]]:
    feature_stride = int(contract["feature_stride"] * contract["sample_stride"])
    window_size = int(contract["window_size"])
    window_stride = int(window_size * (1.0 - contract["window_overlap_ratio"]))
    rows: list[dict[str, Any]] = []
    for video in videos:
        centers = list(range(0, int(video["frame"]), feature_stride))
        video_rows = []
        last_window = False
        for index in range(max(1, len(centers) // window_stride)):
            start = index * window_stride
            end = start + window_size
            if end > len(centers):
                end = len(centers)
                start = max(0, end - window_size)
                last_window = True
            selected = centers[start:end]
            if not selected:
                raise ValueError("Gate4 official invocation has no sampled frames")
            invocation_id = (
                f"gate4/{len(rows):06d}/{video['official_video_id']}/"
                f"{selected[0]}-{selected[-1]}"
            )
            row: dict[str, Any] = {
                "official_video_id": video["official_video_id"],
                "invocation_id": invocation_id,
                "invocation_index": len(rows),
                "video_invocation_index": len(video_rows),
                "sampled_frame_indices": selected,
                "valid_count": len(selected),
                "window_start_frame": selected[0],
                "window_end_frame": selected[-1],
            }
            row["invocation_sha256"] = canonical_sha256(row)
            rows.append(row)
            video_rows.append(row)
            if last_window:
                break
    if len(rows) < 200:
        raise ValueError("Gate4 official population requires at least 200 invocations")
    return rows


def _timing_blocks(invocations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocks = []
    repetitions: dict[str, int] = {}

    def append(invocation: Mapping[str, Any]) -> None:
        invocation_id = str(invocation["invocation_id"])
        repetition = repetitions.get(invocation_id, 0)
        repetitions[invocation_id] = repetition + 1
        ordinal = len(blocks)
        row: dict[str, Any] = {
            "official_video_id": invocation["official_video_id"],
            "invocation_id": invocation_id,
            "repetition_id": repetition,
            "invocation_order_index": ordinal,
            "arm_order": list(ARM_ORDERS[ordinal % len(ARM_ORDERS)]),
        }
        row["block_sha256"] = canonical_sha256(row)
        blocks.append(row)

    for invocation in invocations:
        append(invocation)
    padding = (-len(blocks)) % len(ARM_ORDERS)
    hash_order = sorted(invocations, key=lambda row: row["invocation_sha256"])
    for invocation in hash_order[:padding]:
        append(invocation)
    if len(blocks) % 6 != 0:
        raise RuntimeError("Gate4 timing block padding failed")
    return blocks


def _ground_truth(value: Any, *, video_ids: Sequence[str]) -> list[dict[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise ValueError("Gate4 ground truth must be a non-empty sequence")
    allowed = set(video_ids)
    seen = set()
    rows = []
    for ordinal, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {
            "official_video_id",
            "label",
            "segment",
        }:
            raise ValueError(f"Gate4 ground truth {ordinal} fields mismatch")
        video = _text(raw["official_video_id"], "Gate4 ground-truth video")
        label = _text(raw["label"], "Gate4 ground-truth label")
        segment = raw["segment"]
        if not isinstance(segment, list) or len(segment) != 2:
            raise ValueError("Gate4 ground-truth segment must have two values")
        start = _number(segment[0], "Gate4 ground-truth start")
        end = _number(segment[1], "Gate4 ground-truth end", positive=True)
        if video not in allowed or end <= start:
            raise ValueError("Gate4 ground truth lies outside population or has invalid duration")
        key = (video, label, start, end)
        if key in seen:
            raise ValueError("Gate4 ground truth contains a duplicate instance")
        seen.add(key)
        rows.append(
            {
                "official_video_id": video,
                "label": label,
                "segment": [start, end],
            }
        )
    return rows


def build_gate4_population_artifact(
    *,
    config_sources_sha256: Mapping[str, str],
    annotation: Mapping[str, str],
    class_map: Mapping[str, str],
    data_root: str,
    dataset_contract: Mapping[str, Any],
    videos: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    fit_manifest_sha256: str,
    fit_duration_quartile_thresholds: Sequence[float],
) -> dict[str, Any]:
    if not isinstance(config_sources_sha256, Mapping) or not config_sources_sha256:
        raise ValueError("Gate4 config source identity must be non-empty")
    sources = {
        _relative(path, "Gate4 config source"): _sha(digest, "Gate4 config source SHA")
        for path, digest in sorted(config_sources_sha256.items())
    }
    if GATE4_CONFIG_RELATIVE not in sources:
        raise ValueError("Gate4 population must bind its resolved top-level config source")
    for value, label in ((annotation, "annotation"), (class_map, "class map")):
        if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
            raise ValueError(f"Gate4 {label} identity fields mismatch")
        _text(value["path"], f"Gate4 {label} path")
        _sha(value["sha256"], f"Gate4 {label} SHA-256")
    root = _text(data_root, "Gate4 data root")
    contract = _dataset_contract(dataset_contract)
    video_rows = _videos(videos)
    ids = [row["official_video_id"] for row in video_rows]
    invocations = _enumerate_invocations(video_rows, contract)
    blocks = _timing_blocks(invocations)
    gt = _ground_truth(ground_truth, video_ids=ids)
    fit_manifest_sha256 = _sha(fit_manifest_sha256, "Gate4 fit manifest SHA-256")
    if (
        not isinstance(fit_duration_quartile_thresholds, Sequence)
        or len(fit_duration_quartile_thresholds) != 3
    ):
        raise ValueError("Gate4 fit duration quartiles require Q1/Q2/Q3")
    quartiles = [
        _number(value, f"Gate4 fit duration Q{index}", positive=True)
        for index, value in enumerate(fit_duration_quartile_thresholds, 1)
    ]
    if not quartiles[0] < quartiles[1] < quartiles[2]:
        raise ValueError("Gate4 fit duration quartiles must be strictly increasing")
    artifact: dict[str, Any] = {
        "schema": GATE4_POPULATION_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "config_relative": GATE4_CONFIG_RELATIVE,
        "config_sources_sha256": sources,
        "annotation": dict(annotation),
        "class_map": dict(class_map),
        "data_root": root,
        "dataset_contract": contract,
        "official_video_ids": ids,
        "videos": video_rows,
        "unique_invocation_count": len(invocations),
        "unique_invocations": invocations,
        "unique_invocation_order_sha256": canonical_sha256(
            [row["invocation_sha256"] for row in invocations]
        ),
        "timing_block_count": len(blocks),
        "timing_blocks": blocks,
        "timing_block_order_sha256": canonical_sha256(
            [row["block_sha256"] for row in blocks]
        ),
        "ground_truth": gt,
        "ground_truth_sha256": canonical_sha256(gt),
        "fit_manifest_sha256": fit_manifest_sha256,
        "fit_duration_quartile_thresholds": quartiles,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_gate4_population_artifact(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _ARTIFACT_FIELDS:
        raise ValueError("Gate4 population artifact fields mismatch")
    if value["schema"] != GATE4_POPULATION_SCHEMA or value["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("Gate4 population schema/protocol mismatch")
    rebuilt = build_gate4_population_artifact(
        config_sources_sha256=value["config_sources_sha256"],
        annotation=value["annotation"],
        class_map=value["class_map"],
        data_root=value["data_root"],
        dataset_contract=value["dataset_contract"],
        videos=value["videos"],
        ground_truth=value["ground_truth"],
        fit_manifest_sha256=value["fit_manifest_sha256"],
        fit_duration_quartile_thresholds=value["fit_duration_quartile_thresholds"],
    )
    if rebuilt != dict(value):
        raise ValueError("Gate4 population differs from exact recomputation")
    return rebuilt


def gate4_population_exact_bytes(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(validate_gate4_population_artifact(value)) + b"\n"


def file_size_sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


__all__ = [
    "GATE4_CONFIG_RELATIVE",
    "GATE4_POPULATION_SCHEMA",
    "build_gate4_population_artifact",
    "file_size_sha256",
    "gate4_population_exact_bytes",
    "validate_gate4_population_artifact",
]
