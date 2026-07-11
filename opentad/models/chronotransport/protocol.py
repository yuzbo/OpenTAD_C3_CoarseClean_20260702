"""Canonical, result-independent contracts for ChronoTransport r2.

This module deliberately contains no dataset annotations, detector outputs, or
learned state.  It is safe to use before any formal gate is unlocked.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
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
    if tuple(matrix) != R2_SEEDS:
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
        candidates = [int(row["candidate"]) for row in rows]
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

