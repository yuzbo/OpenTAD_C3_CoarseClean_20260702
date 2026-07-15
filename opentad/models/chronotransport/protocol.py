"""Canonical, result-independent contracts for ChronoTransport r2.

This module deliberately contains no dataset annotations, detector outputs, or
learned state.  It is safe to use before any formal gate is unlocked.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import PurePosixPath
import re
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


R2_PROTOCOL_ID = "CT-P3R-3S-r2"
R2_SEEDS = (3407, 3408, 3409)
R2_SEED_OFFSETS = {3407: 0, 3408: 4, 3409: 8}
R2_NON_DENSE_CANDIDATES = 16
R2_WINDOW_WIDTH = 768

_SPLIT_PREFIX = b"CT-P3R-3S-r2-split-v1\0" + b"3407\0"
_WINDOW_PREFIX = b"CT-P3R-3S-r2-window-v1\0"
_LOWER_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_WINDOWS_DRIVE_PATH = re.compile(r"^[A-Za-z]:/")
_MANIFEST_SCHEMA = "chronotransport-r2-window-manifest-v1"
_REGISTRY_SCHEMA = "chronotransport-r2-label-free-media-registry-v1"
_CONFIG_IDENTITY_SCHEMA = "chronotransport-r2-config-identity-v1"
_STAGE_B_EXPOSURE_SCHEMA = "chronotransport-r2-stage-b-exposures-v1"

_REGISTRY_KEYS = {"schema", "data_sha256", "annotation_sha256", "records"}
_REGISTRY_RECORD_KEYS = {
    "video_id",
    "media_registry_id",
    "media_path",
    "media_sha256",
    "source_total_frames",
    "fps",
    "sampled_frame_indices",
}
_CONFIG_IDENTITY_KEYS = {
    "schema",
    "config_sha256",
    "snippet_stride",
    "scale_factor",
    "rounding",
    "clipping",
}
_FORBIDDEN_REGISTRY_FIELDS = {
    "annotation",
    "annotations",
    "class",
    "classes",
    "detector_output",
    "detector_outputs",
    "gt",
    "gt_segment",
    "gt_segments",
    "label",
    "labels",
    "prediction",
    "predictions",
    "raw_prediction",
    "raw_predictions",
    "result",
    "results",
    "teacher",
}
_WINDOW_HASH_EXCLUDED_KEYS = {"annotation_sha256", "split", "window_id", "window_sha256"}


def normalize_nfc(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("canonical text fields must be strings")
    return unicodedata.normalize("NFC", value)


def _normalize_json(value: Any) -> Any:
    if isinstance(value, str):
        return normalize_nfc(value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise ValueError("canonical JSON forbids non-finite floats")
        return value
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = normalize_nfc(key)
            if normalized_key in normalized:
                raise ValueError("NFC normalization produced duplicate JSON keys")
            normalized[normalized_key] = _normalize_json(item)
        return normalized
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the r2 canonical JSON representation as UTF-8 without BOM."""

    normalized = _normalize_json(value)
    text = json.dumps(
        normalized,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return text.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_exact_bytes(manifest: Mapping[str, Any]) -> bytes:
    """Serialize a manifest to its immutable on-disk representation."""

    return canonical_json_bytes(manifest) + b"\n"


def _require_lower_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _LOWER_SHA256.fullmatch(value):
        raise ValueError(f"{field} must be 64 lowercase hexadecimal ASCII characters")
    return value


def _require_video_id(value: Any, *, field: str = "video_id") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a non-empty NUL-free string")
    return normalize_nfc(value)


def _require_relative_posix_path(value: Any, *, field: str = "media_path") -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a canonical relative POSIX path")
    posix_path = PurePosixPath(value)
    if (
        "\\" in value
        or posix_path.is_absolute()
        or _WINDOWS_DRIVE_PATH.match(value)
        or any(part in ("", ".", "..") for part in posix_path.parts)
        or posix_path.as_posix() != value
    ):
        raise ValueError(f"{field} must be a canonical relative POSIX path")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, where: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{where} fields mismatch: missing={missing}, extra={extra}")


def _find_forbidden_registry_field(value: Any, *, path: str = "registry") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = normalize_nfc(str(key)).casefold()
            if normalized in _FORBIDDEN_REGISTRY_FIELDS or normalized.startswith("gt_"):
                raise ValueError(f"forbidden result/annotation field at {path}.{key}")
            _find_forbidden_registry_field(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _find_forbidden_registry_field(item, path=f"{path}[{index}]")


def _validated_config_identity(config_identity: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(config_identity, Mapping):
        raise TypeError("config identity must be a mapping")
    _require_exact_keys(config_identity, _CONFIG_IDENTITY_KEYS, where="config identity")
    if config_identity["schema"] != _CONFIG_IDENTITY_SCHEMA:
        raise ValueError("config identity schema mismatch")
    _require_lower_sha256(config_identity["config_sha256"], field="config SHA-256")
    stride = config_identity["snippet_stride"]
    if isinstance(stride, bool) or not isinstance(stride, int) or stride <= 0:
        raise ValueError("snippet_stride must be a positive integer")
    scale_factor = config_identity["scale_factor"]
    if isinstance(scale_factor, bool) or not isinstance(scale_factor, (int, float)):
        raise ValueError("scale_factor must be numeric")
    if not math.isfinite(float(scale_factor)) or float(scale_factor) <= 0.0:
        raise ValueError("scale_factor must be finite and positive")
    if not float(scale_factor).is_integer():
        raise ValueError("scale_factor must be integer-valued for the fixed frame-index rule")
    scale = int(float(scale_factor))
    if stride < scale or stride % scale != 0:
        raise ValueError("snippet_stride must be at least and divisible by scale_factor")
    if config_identity["rounding"] != "floor":
        raise ValueError("rounding must equal the frozen 'floor' rule")
    if config_identity["clipping"] != "source_bounds":
        raise ValueError("clipping must equal the frozen 'source_bounds' rule")
    return _normalize_json(dict(config_identity))


def _expected_sampled_frame_indices(
    source_total_frames: int,
    config_identity: Mapping[str, Any],
) -> list[int]:
    """Reproduce OpenTAD LoadFrames' label-free pre-truncation index vector."""

    frame_stride = int(config_identity["snippet_stride"]) // int(
        float(config_identity["scale_factor"])
    )
    return list(range(0, int(source_total_frames), frame_stride))


def _validated_registry(registry: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(registry, Mapping):
        raise TypeError("media registry must be a mapping")
    _require_exact_keys(registry, _REGISTRY_KEYS, where="media registry")
    if registry["schema"] != _REGISTRY_SCHEMA:
        raise ValueError("media registry schema mismatch")
    _require_lower_sha256(registry["data_sha256"], field="data SHA-256")
    _require_lower_sha256(registry["annotation_sha256"], field="annotation SHA-256")
    records = registry["records"]
    if not isinstance(records, list) or len(records) != 200:
        raise ValueError("r2 media registry must contain exactly 200 records")
    normalized_records: list[dict[str, Any]] = []
    video_ids: list[str] = []
    for record_index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"media registry record {record_index} must be a mapping")
        _find_forbidden_registry_field(record, path=f"registry.records[{record_index}]")
        _require_exact_keys(record, _REGISTRY_RECORD_KEYS, where=f"media registry record {record_index}")
        normalized = _normalize_json(dict(record))
        video_id = _require_video_id(normalized["video_id"])
        if not isinstance(normalized["media_registry_id"], str) or not normalized["media_registry_id"]:
            raise ValueError("media_registry_id must be a non-empty string")
        if "\x00" in normalized["media_registry_id"]:
            raise ValueError("media_registry_id must not contain NUL")
        _require_relative_posix_path(normalized["media_path"])
        _validate_media_sha256(normalized["media_sha256"])
        total_frames = normalized["source_total_frames"]
        if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError("source_total_frames must be a positive integer")
        fps = normalized["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)):
            raise ValueError("fps must be numeric")
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("fps must be finite and positive")
        indices = normalized["sampled_frame_indices"]
        if not isinstance(indices, list) or not indices:
            raise ValueError("sampled_frame_indices must be a non-empty list")
        if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
            raise ValueError("sampled_frame_indices must contain integers")
        if any(index < 0 or index >= total_frames for index in indices):
            raise ValueError("sampled_frame_indices must remain within source bounds")
        if any(left > right for left, right in zip(indices, indices[1:])):
            raise ValueError("sampled_frame_indices must be non-decreasing")
        video_ids.append(video_id)
        normalized["video_id"] = video_id
        normalized_records.append(normalized)
    if len(set(video_ids)) != 200:
        raise ValueError("r2 media registry requires exactly 200 unique video IDs")
    identity = {
        "schema": registry["schema"],
        "data_sha256": registry["data_sha256"],
        "annotation_sha256": registry["annotation_sha256"],
    }
    return identity, normalized_records


def split_digest(video_id: str) -> bytes:
    return hashlib.sha256(_SPLIT_PREFIX + normalize_nfc(video_id).encode("utf-8")).digest()


def split_video_ids(video_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
    normalized = [normalize_nfc(video_id) for video_id in video_ids]
    if len(normalized) != 200 or len(set(normalized)) != 200:
        raise ValueError("r2 split requires exactly 200 unique video IDs")
    ordered = sorted(
        normalized,
        key=lambda video_id: (split_digest(video_id), video_id.encode("utf-8")),
    )
    return {
        "fit": tuple(ordered[:140]),
        "calibration": tuple(ordered[140:170]),
        "evaluation": tuple(ordered[170:200]),
    }


def _validate_media_sha256(media_sha256: str) -> str:
    if not isinstance(media_sha256, str) or not _LOWER_SHA256.fullmatch(media_sha256):
        raise ValueError("media SHA-256 must be 64 lowercase hexadecimal ASCII characters")
    return media_sha256


def window_digest(video_id: str, media_sha256: str, sampled_length: int) -> bytes:
    sampled_length = int(sampled_length)
    if sampled_length <= 0:
        raise ValueError("sampled-index vector must be non-empty")
    video_bytes = normalize_nfc(video_id).encode("utf-8")
    media_bytes = _validate_media_sha256(media_sha256).encode("ascii")
    length_bytes = str(sampled_length).encode("ascii")
    return hashlib.sha256(
        _WINDOW_PREFIX + video_bytes + b"\0" + media_bytes + b"\0" + length_bytes
    ).digest()


def build_window_payload(
    video_id: str,
    media_sha256: str,
    sampled_frame_indices: Sequence[int],
    *,
    width: int = R2_WINDOW_WIDTH,
) -> dict[str, Any]:
    """Build the label-free temporal part of one manifested window."""

    indices = [int(index) for index in sampled_frame_indices]
    if not indices:
        raise ValueError("sampled-index vector must be non-empty")
    width = int(width)
    if width <= 0:
        raise ValueError("window width must be positive")
    digest = window_digest(video_id, media_sha256, len(indices))
    start = 0 if len(indices) <= width else int.from_bytes(digest[:8], "big") % (len(indices) - width + 1)
    selected = indices[start : start + width]
    valid_count = len(selected)
    if valid_count < width:
        selected.extend([selected[-1]] * (width - valid_count))
    valid_mask = [position < valid_count for position in range(width)]
    payload: dict[str, Any] = {
        "protocol": R2_PROTOCOL_ID,
        "video_id": normalize_nfc(video_id),
        "media_sha256": _validate_media_sha256(media_sha256),
        "source_sampled_index_length": len(indices),
        "window_width": width,
        "window_start": start,
        "sampled_frame_indices": selected,
        "valid_mask": valid_mask,
        "padding_positions": [position for position, valid in enumerate(valid_mask) if not valid],
    }
    payload["payload_sha256"] = canonical_sha256(payload)
    return payload


def _window_hash_payload(window: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in window.items()
        if key not in _WINDOW_HASH_EXCLUDED_KEYS
    }


def build_r2_manifest(
    registry: Mapping[str, Any],
    config_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable one-label-free-window-per-video r2 manifest."""

    data_identity, records = _validated_registry(registry)
    config = _validated_config_identity(config_identity)
    for record in records:
        expected_indices = _expected_sampled_frame_indices(
            record["source_total_frames"], config
        )
        if record["sampled_frame_indices"] != expected_indices:
            raise ValueError(
                "sampled_frame_indices do not match the fixed config-derived source vector"
            )
    video_splits = split_video_ids(record["video_id"] for record in records)
    record_by_id = {record["video_id"]: record for record in records}
    split_by_video = {
        video_id: split
        for split, video_ids in video_splits.items()
        for video_id in video_ids
    }
    ordered_video_ids = tuple(
        video_id
        for split in ("fit", "calibration", "evaluation")
        for video_id in video_splits[split]
    )
    windows: list[dict[str, Any]] = []
    for video_id in ordered_video_ids:
        record = record_by_id[video_id]
        temporal = build_window_payload(
            video_id,
            record["media_sha256"],
            record["sampled_frame_indices"],
        )
        window: dict[str, Any] = {
            "protocol": R2_PROTOCOL_ID,
            "video_id": video_id,
            "media_registry_id": record["media_registry_id"],
            "media_path": record["media_path"],
            "media_sha256": record["media_sha256"],
            "source_total_frames": record["source_total_frames"],
            "fps": record["fps"],
            "snippet_stride": config["snippet_stride"],
            "scale_factor": config["scale_factor"],
            "rounding": config["rounding"],
            "clipping": config["clipping"],
            "source_sampled_index_length": temporal["source_sampled_index_length"],
            "window_width": temporal["window_width"],
            "window_start": temporal["window_start"],
            "sampled_frame_indices": temporal["sampled_frame_indices"],
            "valid_mask": temporal["valid_mask"],
            "padding_positions": temporal["padding_positions"],
            "data_sha256": data_identity["data_sha256"],
            "config_sha256": config["config_sha256"],
            # Annotation identity is provenance only and is intentionally
            # excluded from window selection and window-identity hashes.
            "annotation_sha256": data_identity["annotation_sha256"],
            "temporal_payload_sha256": temporal["payload_sha256"],
            "split": split_by_video[video_id],
        }
        window_sha256 = canonical_sha256(_window_hash_payload(window))
        window["window_sha256"] = window_sha256
        window["window_id"] = f"ct-r2-window-{window_sha256}"
        windows.append(window)

    window_id_by_video = {window["video_id"]: window["window_id"] for window in windows}
    splits = {
        split: [window_id_by_video[video_id] for video_id in video_splits[split]]
        for split in ("fit", "calibration", "evaluation")
    }
    split_hashes = {split: canonical_sha256(window_ids) for split, window_ids in splits.items()}
    manifest: dict[str, Any] = {
        "schema": _MANIFEST_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "population_size": 200,
        "data_identity": data_identity,
        "config_identity": config,
        "split_video_ids": {split: list(video_splits[split]) for split in splits},
        "splits": splits,
        "split_hashes": split_hashes,
        "windows": windows,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    return manifest


def _validate_temporal_window(window: Mapping[str, Any]) -> None:
    width = window["window_width"]
    if isinstance(width, bool) or not isinstance(width, int) or width != R2_WINDOW_WIDTH:
        raise ValueError("window width must be exactly 768")
    sampled_length = window["source_sampled_index_length"]
    if isinstance(sampled_length, bool) or not isinstance(sampled_length, int) or sampled_length <= 0:
        raise ValueError("source sampled-index length must be positive")
    expected_source_indices = _expected_sampled_frame_indices(
        window["source_total_frames"], window
    )
    if sampled_length != len(expected_source_indices):
        raise ValueError("source sampled-index length does not match the fixed config")
    expected_start = 0
    if sampled_length > width:
        digest = window_digest(window["video_id"], window["media_sha256"], sampled_length)
        expected_start = int.from_bytes(digest[:8], "big") % (sampled_length - width + 1)
    if isinstance(window["window_start"], bool) or not isinstance(window["window_start"], int):
        raise ValueError("window_start must be an integer")
    if window["window_start"] != expected_start:
        raise ValueError("window start digest mismatch")
    indices = window["sampled_frame_indices"]
    mask = window["valid_mask"]
    padding = window["padding_positions"]
    if not isinstance(indices, list) or len(indices) != width:
        raise ValueError("each manifested window must contain exactly 768 sampled frame indices")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("manifest sampled frame indices must be integers")
    if any(index < 0 or index >= window["source_total_frames"] for index in indices):
        raise ValueError("manifest sampled frame indices exceed source bounds")
    if any(left > right for left, right in zip(indices, indices[1:])):
        raise ValueError("manifest sampled frame indices must be non-decreasing")
    if not isinstance(mask, list) or any(type(value) is not bool for value in mask):
        raise ValueError("valid_mask must contain only boolean values")
    if not isinstance(padding, list) or any(
        isinstance(value, bool) or not isinstance(value, int) for value in padding
    ):
        raise ValueError("padding_positions must contain only integers")
    expected_indices = expected_source_indices[expected_start : expected_start + width]
    if len(expected_indices) < width:
        expected_indices.extend([expected_indices[-1]] * (width - len(expected_indices)))
    if indices != expected_indices:
        raise ValueError("manifest sampled_frame_indices do not match the fixed config window")
    valid_count = min(sampled_length, width)
    expected_mask = [position < valid_count for position in range(width)]
    expected_padding = list(range(valid_count, width))
    if mask != expected_mask or padding != expected_padding:
        raise ValueError("window valid mask or padding positions mismatch")
    if expected_padding and any(indices[position] != indices[valid_count - 1] for position in expected_padding):
        raise ValueError("window padding must use edge-repeat semantics")
    temporal = {
        "protocol": R2_PROTOCOL_ID,
        "video_id": window["video_id"],
        "media_sha256": window["media_sha256"],
        "source_sampled_index_length": sampled_length,
        "window_width": width,
        "window_start": expected_start,
        "sampled_frame_indices": indices,
        "valid_mask": mask,
        "padding_positions": padding,
    }
    if window["temporal_payload_sha256"] != canonical_sha256(temporal):
        raise ValueError("temporal window payload hash mismatch")


def validate_r2_manifest(
    manifest: Mapping[str, Any],
    *,
    registry: Mapping[str, Any] | None = None,
    config_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Deeply validate a manifest, optionally re-deriving it from source metadata."""

    if not isinstance(manifest, Mapping):
        raise TypeError("r2 manifest must be a mapping")
    normalized_manifest = _normalize_json(dict(manifest))
    if normalized_manifest != dict(manifest):
        raise ValueError("r2 manifest tree must be NFC-canonical")
    expected_manifest_keys = {
        "schema",
        "protocol",
        "population_size",
        "data_identity",
        "config_identity",
        "split_video_ids",
        "splits",
        "split_hashes",
        "windows",
        "manifest_sha256",
    }
    _require_exact_keys(manifest, expected_manifest_keys, where="r2 manifest")
    if manifest["schema"] != _MANIFEST_SCHEMA or manifest["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("r2 manifest schema or protocol mismatch")
    if (
        isinstance(manifest["population_size"], bool)
        or not isinstance(manifest["population_size"], int)
        or manifest["population_size"] != 200
    ):
        raise ValueError("r2 manifest population must be exactly 200")
    data_identity = manifest["data_identity"]
    if not isinstance(data_identity, Mapping):
        raise ValueError("manifest data identity must be a mapping")
    _require_exact_keys(
        data_identity,
        {"schema", "data_sha256", "annotation_sha256"},
        where="manifest data identity",
    )
    if data_identity["schema"] != _REGISTRY_SCHEMA:
        raise ValueError("manifest data identity schema mismatch")
    _require_lower_sha256(data_identity["data_sha256"], field="manifest data SHA-256")
    _require_lower_sha256(data_identity["annotation_sha256"], field="manifest annotation SHA-256")
    config = _validated_config_identity(manifest["config_identity"])
    windows = manifest["windows"]
    if not isinstance(windows, list) or len(windows) != 200:
        raise ValueError("r2 manifest must contain exactly 200 windows")
    expected_window_keys = {
        "protocol",
        "video_id",
        "media_registry_id",
        "media_path",
        "media_sha256",
        "source_total_frames",
        "fps",
        "snippet_stride",
        "scale_factor",
        "rounding",
        "clipping",
        "source_sampled_index_length",
        "window_width",
        "window_start",
        "sampled_frame_indices",
        "valid_mask",
        "padding_positions",
        "data_sha256",
        "config_sha256",
        "annotation_sha256",
        "temporal_payload_sha256",
        "split",
        "window_sha256",
        "window_id",
    }
    video_ids: list[str] = []
    window_ids: list[str] = []
    for index, window in enumerate(windows):
        if not isinstance(window, Mapping):
            raise TypeError(f"manifest window {index} must be a mapping")
        _require_exact_keys(window, expected_window_keys, where=f"manifest window {index}")
        if window["protocol"] != R2_PROTOCOL_ID:
            raise ValueError("window protocol mismatch")
        video_id = _require_video_id(window["video_id"], field="window video_id")
        if video_id != window["video_id"]:
            raise ValueError("window video ID must be NFC-normalized")
        if not isinstance(window["media_registry_id"], str) or not window["media_registry_id"] or "\x00" in window["media_registry_id"]:
            raise ValueError("window media_registry_id must be a non-empty NUL-free string")
        _require_relative_posix_path(window["media_path"], field="window media_path")
        _validate_media_sha256(window["media_sha256"])
        _require_lower_sha256(window["window_sha256"], field="window hash")
        if window["window_id"] != f"ct-r2-window-{window['window_sha256']}":
            raise ValueError("window ID does not bind the window hash")
        if window["data_sha256"] != data_identity["data_sha256"]:
            raise ValueError("window data identity mismatch")
        if window["annotation_sha256"] != data_identity["annotation_sha256"]:
            raise ValueError("window annotation identity mismatch")
        if window["config_sha256"] != config["config_sha256"]:
            raise ValueError("window config identity mismatch")
        for field in ("snippet_stride", "scale_factor", "rounding", "clipping"):
            if type(window[field]) is not type(config[field]) or window[field] != config[field]:
                raise ValueError(f"window {field} does not match config identity")
        total_frames = window["source_total_frames"]
        if isinstance(total_frames, bool) or not isinstance(total_frames, int) or total_frames <= 0:
            raise ValueError("window source total frames must be positive")
        fps = window["fps"]
        if isinstance(fps, bool) or not isinstance(fps, (int, float)) or not math.isfinite(float(fps)) or float(fps) <= 0:
            raise ValueError("window fps must be finite and positive")
        if window["split"] not in ("fit", "calibration", "evaluation"):
            raise ValueError("window split is not canonical")
        _validate_temporal_window(window)
        if window["window_sha256"] != canonical_sha256(_window_hash_payload(window)):
            raise ValueError("window hash mismatch")
        video_ids.append(video_id)
        window_ids.append(window["window_id"])
    if len(set(video_ids)) != 200:
        raise ValueError("manifest requires exactly 200 unique video IDs")
    if len(set(window_ids)) != 200:
        raise ValueError("manifest requires exactly 200 unique window IDs")

    expected_video_splits = split_video_ids(video_ids)
    expected_order = [
        video_id
        for split in ("fit", "calibration", "evaluation")
        for video_id in expected_video_splits[split]
    ]
    if video_ids != expected_order:
        raise ValueError("manifest windows are not in canonical split/digest order")
    window_id_by_video = dict(zip(video_ids, window_ids))
    expected_splits = {
        split: [window_id_by_video[video_id] for video_id in expected_video_splits[split]]
        for split in ("fit", "calibration", "evaluation")
    }
    expected_split_video_ids = {
        split: list(expected_video_splits[split])
        for split in ("fit", "calibration", "evaluation")
    }
    if manifest["split_video_ids"] != expected_split_video_ids:
        raise ValueError("manifest split video membership mismatch")
    if manifest["splits"] != expected_splits:
        raise ValueError("manifest split window membership mismatch")
    expected_membership = {
        window_id: split for split, split_windows in expected_splits.items() for window_id in split_windows
    }
    if any(window["split"] != expected_membership[window["window_id"]] for window in windows):
        raise ValueError("window split field mismatch")
    expected_split_hashes = {
        split: canonical_sha256(split_windows) for split, split_windows in expected_splits.items()
    }
    if manifest["split_hashes"] != expected_split_hashes:
        raise ValueError("manifest split hash mismatch")
    unsigned = dict(manifest)
    manifest_sha256 = unsigned.pop("manifest_sha256")
    _require_lower_sha256(manifest_sha256, field="manifest SHA-256")
    if manifest_sha256 != canonical_sha256(unsigned):
        raise ValueError("manifest hash mismatch")
    if (registry is None) != (config_identity is None):
        raise ValueError("registry and config_identity must be supplied together")
    if registry is not None and config_identity is not None:
        rebuilt = build_r2_manifest(registry, config_identity)
        if _normalize_json(dict(manifest)) != rebuilt:
            raise ValueError("manifest does not exactly match the supplied registry/config identity")
    return dict(manifest)


def _candidate(seed: int, ordinal: int) -> int:
    if int(seed) not in R2_SEED_OFFSETS:
        raise ValueError(f"unsupported r2 seed: {seed}")
    ordinal = int(ordinal)
    if ordinal < 0:
        raise ValueError("exposure ordinal must be non-negative")
    block, position = divmod(ordinal, R2_NON_DENSE_CANDIDATES)
    return (position + 5 * block + R2_SEED_OFFSETS[int(seed)]) % R2_NON_DENSE_CANDIDATES


def stage_b_exposure_matrix() -> dict[int, tuple[dict[str, int], ...]]:
    return {
        seed: tuple(
            {
                "successful_update": update,
                "canonical_window_index": update,
                "candidate": _candidate(seed, update),
            }
            for update in range(140)
        )
        for seed in R2_SEEDS
    }


def validate_stage_b_exposures(
    matrix: Mapping[int, Sequence[Mapping[str, int]]],
) -> None:
    if not isinstance(matrix, Mapping):
        raise TypeError("Stage B exposure matrix must be a mapping")
    matrix_keys = tuple(matrix)
    if (
        any(type(seed) is not int for seed in matrix_keys)
        or matrix_keys != R2_SEEDS
    ):
        raise ValueError("Stage B exposure matrix must use canonical seed order")
    expected_tails = {
        3407: [8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3],
        3408: [12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7],
        3409: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
    aggregate = Counter()
    for seed in R2_SEEDS:
        rows = matrix[seed]
        if len(rows) != 140:
            raise ValueError("each Stage B seed must have exactly 140 rows")
        candidates = []
        expected_row_keys = {"successful_update", "canonical_window_index", "candidate"}
        for update, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError("Stage B exposure rows must be mappings")
            _require_exact_keys(row, expected_row_keys, where="Stage B primitive exposure row")
            for field in expected_row_keys:
                if isinstance(row[field], bool) or not isinstance(row[field], int):
                    raise ValueError(f"Stage B primitive exposure {field} must be an integer")
            if row["successful_update"] != update or row["canonical_window_index"] != update:
                raise ValueError("Stage B primitive update/index mismatch")
            candidates.append(row["candidate"])
        if candidates != [_candidate(seed, update) for update in range(140)]:
            raise ValueError("Stage B candidate formula mismatch")
        for start in range(0, 128, 16):
            if sorted(candidates[start : start + 16]) != list(range(16)):
                raise ValueError("full Stage B blocks must be candidate permutations")
        counts = Counter(candidates)
        if sorted(counts.values()) != [8] * 4 + [9] * 12:
            raise ValueError("Stage B per-seed exposure counts mismatch")
        if candidates[-12:] != expected_tails[seed]:
            raise ValueError("Stage B exact tail mismatch")
        for candidate in range(16):
            positions = Counter(index % 4 for index in range(128) if candidates[index] == candidate)
            if positions != Counter({0: 2, 1: 2, 2: 2, 3: 2}):
                raise ValueError("Stage B position-mod-4 balance mismatch")
        aggregate.update(candidates)
    expected_aggregate = {candidate: 27 if candidate < 4 else 26 for candidate in range(16)}
    if dict(sorted(aggregate.items())) != expected_aggregate:
        raise ValueError("Stage B aggregate exposure counts mismatch")
    for update in range(140):
        if len({int(matrix[seed][update]["candidate"]) for seed in R2_SEEDS}) != 3:
            raise ValueError("each Stage B window must receive three different candidates")


def build_stage_b_exposure_artifact(fit_window_ids: Sequence[str]) -> dict[str, Any]:
    """Bind the frozen exposure formula to the exact canonical fit-window order."""

    fit_ids = [normalize_nfc(window_id) for window_id in fit_window_ids]
    if len(fit_ids) != 140 or len(set(fit_ids)) != 140:
        raise ValueError("Stage B requires exactly 140 unique fit window IDs")
    base = stage_b_exposure_matrix()
    matrices: dict[str, list[dict[str, Any]]] = {}
    for seed in R2_SEEDS:
        matrices[str(seed)] = [
            {
                "seed": seed,
                "successful_update": update,
                "canonical_window_index": update,
                "window_id": fit_ids[update],
                "candidate": int(base[seed][update]["candidate"]),
            }
            for update in range(140)
        ]
    per_seed_hashes = {
        seed: canonical_sha256(rows)
        for seed, rows in matrices.items()
    }
    artifact: dict[str, Any] = {
        "schema": _STAGE_B_EXPOSURE_SCHEMA,
        "protocol": R2_PROTOCOL_ID,
        "seeds": list(R2_SEEDS),
        "seed_offsets": {str(seed): R2_SEED_OFFSETS[seed] for seed in R2_SEEDS},
        "candidate_count": R2_NON_DENSE_CANDIDATES,
        "fit_window_order_sha256": canonical_sha256(fit_ids),
        "matrices": matrices,
        "per_seed_sha256": per_seed_hashes,
        "combined_matrix_sha256": canonical_sha256(
            {
                "fit_window_order_sha256": canonical_sha256(fit_ids),
                "matrices": matrices,
            }
        ),
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def validate_stage_b_exposure_artifact(
    artifact: Mapping[str, Any],
    *,
    fit_window_ids: Sequence[str],
) -> dict[str, Any]:
    if not isinstance(artifact, Mapping):
        raise TypeError("Stage B exposure artifact must be a mapping")
    expected_keys = {
        "schema",
        "protocol",
        "seeds",
        "seed_offsets",
        "candidate_count",
        "fit_window_order_sha256",
        "matrices",
        "per_seed_sha256",
        "combined_matrix_sha256",
        "artifact_sha256",
    }
    _require_exact_keys(artifact, expected_keys, where="Stage B exposure artifact")
    if artifact["schema"] != _STAGE_B_EXPOSURE_SCHEMA or artifact["protocol"] != R2_PROTOCOL_ID:
        raise ValueError("Stage B exposure artifact schema/protocol mismatch")
    if (
        not isinstance(artifact["seeds"], list)
        or any(type(seed) is not int for seed in artifact["seeds"])
        or artifact["seeds"] != list(R2_SEEDS)
    ):
        raise ValueError("Stage B exposure seeds mismatch")
    expected_offsets = {str(seed): R2_SEED_OFFSETS[seed] for seed in R2_SEEDS}
    seed_offsets = artifact["seed_offsets"]
    if (
        not isinstance(seed_offsets, Mapping)
        or tuple(seed_offsets) != tuple(expected_offsets)
        or any(type(value) is not int for value in seed_offsets.values())
        or dict(seed_offsets) != expected_offsets
    ):
        raise ValueError("Stage B exposure seed offsets mismatch")
    if (
        type(artifact["candidate_count"]) is not int
        or artifact["candidate_count"] != R2_NON_DENSE_CANDIDATES
    ):
        raise ValueError("Stage B exposure candidate count mismatch")
    fit_ids = [normalize_nfc(window_id) for window_id in fit_window_ids]
    if len(fit_ids) != 140 or len(set(fit_ids)) != 140:
        raise ValueError("Stage B requires exactly 140 unique fit window IDs")
    if artifact["fit_window_order_sha256"] != canonical_sha256(fit_ids):
        raise ValueError("Stage B fit-window order hash mismatch")
    matrices = artifact["matrices"]
    if not isinstance(matrices, Mapping) or tuple(matrices) != tuple(str(seed) for seed in R2_SEEDS):
        raise ValueError("Stage B matrices must use canonical seed order")
    primitive: dict[int, tuple[dict[str, int], ...]] = {}
    expected_row_keys = {
        "seed",
        "successful_update",
        "canonical_window_index",
        "window_id",
        "candidate",
    }
    for seed in R2_SEEDS:
        rows = matrices[str(seed)]
        if not isinstance(rows, list) or len(rows) != 140:
            raise ValueError("each Stage B artifact seed must contain exactly 140 rows")
        primitive_rows: list[dict[str, int]] = []
        for update, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError("Stage B exposure rows must be mappings")
            _require_exact_keys(row, expected_row_keys, where="Stage B exposure row")
            for field in ("seed", "successful_update", "canonical_window_index", "candidate"):
                if isinstance(row[field], bool) or not isinstance(row[field], int):
                    raise ValueError(f"Stage B exposure {field} must be an integer")
            if row["window_id"] != fit_ids[update]:
                raise ValueError("Stage B exposure window binding mismatch")
            if row["seed"] != seed:
                raise ValueError("Stage B exposure seed binding mismatch")
            if row["successful_update"] != update or row["canonical_window_index"] != update:
                raise ValueError("Stage B exposure update/index mismatch")
            expected_candidate = _candidate(seed, update)
            if row["candidate"] != expected_candidate:
                raise ValueError("Stage B exposure candidate formula mismatch")
            primitive_rows.append(
                {
                    "successful_update": update,
                    "canonical_window_index": update,
                    "candidate": expected_candidate,
                }
            )
        primitive[seed] = tuple(primitive_rows)
    validate_stage_b_exposures(primitive)
    expected_per_seed = {
        str(seed): canonical_sha256(matrices[str(seed)]) for seed in R2_SEEDS
    }
    if artifact["per_seed_sha256"] != expected_per_seed:
        raise ValueError("Stage B per-seed exposure hash mismatch")
    expected_combined = canonical_sha256(
        {
            "fit_window_order_sha256": canonical_sha256(fit_ids),
            "matrices": matrices,
        }
    )
    if artifact["combined_matrix_sha256"] != expected_combined:
        raise ValueError("Stage B combined exposure matrix hash mismatch")
    unsigned = dict(artifact)
    artifact_sha256 = unsigned.pop("artifact_sha256")
    if artifact_sha256 != canonical_sha256(unsigned):
        raise ValueError("Stage B exposure artifact hash mismatch")
    return dict(artifact)


def stage_c_exposure_matrix() -> dict[int, tuple[dict[str, int], ...]]:
    return {
        seed: tuple(
            {
                "successful_update": exposure // 2,
                "batch_position": exposure % 2,
                "window_exposure_ordinal": exposure,
                "candidate": _candidate(seed, exposure),
            }
            for exposure in range(8400)
        )
        for seed in R2_SEEDS
    }


def stage_c_batch_exposures(
    seed: int, successful_update: int
) -> tuple[dict[str, int], dict[str, int]]:
    """Return the two canonical r2 exposures for one successful Stage-C update."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in R2_SEEDS:
        raise ValueError("Stage C seed must be one of the three frozen r2 seeds")
    if (
        isinstance(successful_update, bool)
        or not isinstance(successful_update, int)
        or not 0 <= successful_update < 4200
    ):
        raise ValueError("Stage C successful update must be an integer in [0, 4199]")
    rows = []
    for batch_position in range(2):
        exposure = 2 * successful_update + batch_position
        rows.append(
            {
                "successful_update": successful_update,
                "batch_position": batch_position,
                "window_exposure_ordinal": exposure,
                "candidate": _candidate(seed, exposure),
            }
        )
    return rows[0], rows[1]


def validate_stage_c_exposures(
    matrix: Mapping[int | str, Sequence[Mapping[str, int]]],
    *,
    next_cursor: Mapping[int | str, int] | None = None,
) -> None:
    """Validate the complete 8,400-exposure Stage-C formula and resume cursor."""

    if not isinstance(matrix, Mapping):
        raise TypeError("Stage C exposure matrix must be a mapping")
    def canonical_seed_key(value: int | str) -> int:
        if isinstance(value, bool):
            raise ValueError("Stage C seed keys must be canonical integers or decimal strings")
        if isinstance(value, int) and value in R2_SEEDS:
            return value
        if isinstance(value, str) and value in tuple(str(seed) for seed in R2_SEEDS):
            return int(value)
        raise ValueError("Stage C seed keys must be canonical integers or decimal strings")

    normalized_keys = tuple(canonical_seed_key(seed) for seed in matrix)
    if normalized_keys != R2_SEEDS:
        raise ValueError("Stage C exposure matrix must use canonical seed order")
    normalized_matrix = {int(seed): matrix[seed] for seed in matrix}
    expected_row_keys = {
        "successful_update",
        "batch_position",
        "window_exposure_ordinal",
        "candidate",
    }
    for seed in R2_SEEDS:
        rows = normalized_matrix[seed]
        if len(rows) != 8400:
            raise ValueError("each Stage C seed must contain exactly 8400 exposures")
        counts = Counter()
        for exposure, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise TypeError("Stage C exposure rows must be mappings")
            _require_exact_keys(row, expected_row_keys, where="Stage C exposure row")
            for field in expected_row_keys:
                if isinstance(row[field], bool) or not isinstance(row[field], int):
                    label = field.replace("_", " ")
                    raise ValueError(f"Stage C {label} must be an integer")
            if row["successful_update"] != exposure // 2:
                raise ValueError("Stage C successful update mismatch")
            if row["batch_position"] != exposure % 2:
                raise ValueError("Stage C batch position mismatch")
            if row["window_exposure_ordinal"] != exposure:
                raise ValueError("Stage C window exposure ordinal mismatch")
            expected_candidate = _candidate(seed, exposure)
            if row["candidate"] != expected_candidate:
                raise ValueError("Stage C candidate formula mismatch")
            counts.update([expected_candidate])
        if counts != Counter({candidate: 525 for candidate in range(16)}):
            raise ValueError("Stage C candidate balance mismatch")
    for exposure in range(8400):
        candidates = {normalized_matrix[seed][exposure]["candidate"] for seed in R2_SEEDS}
        if len(candidates) != 3:
            raise ValueError("each Stage C exposure must receive three distinct seed candidates")
    if next_cursor is not None:
        if not isinstance(next_cursor, Mapping):
            raise TypeError("Stage C next cursor must be a mapping")
        normalized_cursor = {
            canonical_seed_key(seed): value for seed, value in next_cursor.items()
        }
        if tuple(normalized_cursor) != R2_SEEDS:
            raise ValueError("Stage C cursor must use canonical seed order")
        if any(
            isinstance(normalized_cursor[seed], bool)
            or not isinstance(normalized_cursor[seed], int)
            for seed in R2_SEEDS
        ):
            raise ValueError("Stage C cursor values must be integers")
        if any(normalized_cursor[seed] != 8400 for seed in R2_SEEDS):
            raise ValueError("Stage C cursor must equal 8400 after complete exposure")
