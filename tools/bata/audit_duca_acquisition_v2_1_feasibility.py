from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.duca_evidence_io import (
    canonical_sha256,
    with_content_sha256,
    write_json_exclusive_atomic,
)


PROTOCOL_ID = "DUCA-ADMISSION-V2.1-REALVIDEO-CROSSED-NULL"
SOURCE_SPLIT_SCHEMA = "duca_rime_video_split_manifest_v1"
FEASIBILITY_SCHEMA = "duca_acquisition_v2_1_feasibility_receipt_v1"
SOURCE_ROLE = "detector_selector_train"
NUMERIC_ROLE_COUNT = 3
VIDEOS_PER_ROLE = 32
WINDOW_SIZE = 768
WINDOW_OVERLAP_RATIO = 0.5
SNIPPET_STRIDE = 4
FPS = -1.0
SHORT_BINS = ((1, 256), (257, 512), (513, 767))
MIN_SHORT_VIDEOS_PER_BIN = 8


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def enumerate_natural_window_valid_lengths(
    *,
    frame_count: int,
    duration: float | None = None,
    fps: float = FPS,
    snippet_stride: int = SNIPPET_STRIDE,
    window_size: int = WINDOW_SIZE,
    window_overlap_ratio: float = WINDOW_OVERLAP_RATIO,
) -> tuple[int, ...]:
    """Mirror SlidingWindowDataset.split_video_to_windows without decoding."""

    frame_count = int(frame_count)
    fps = float(fps)
    snippet_stride = int(snippet_stride)
    window_size = int(window_size)
    overlap = float(window_overlap_ratio)
    if not math.isfinite(fps):
        raise ValueError("fps must be finite")
    if fps > 0:
        if duration is None or not math.isfinite(float(duration)) or float(duration) <= 0:
            raise ValueError("positive-fps enumeration requires a positive finite duration")
        effective_frame_count = int(float(duration) * fps)
    else:
        effective_frame_count = frame_count
    if effective_frame_count <= 0 or snippet_stride <= 0 or window_size <= 0:
        raise ValueError(
            "effective frame_count, snippet_stride and window_size must be positive"
        )
    if not math.isfinite(overlap) or not 0.0 <= overlap < 1.0:
        raise ValueError("window_overlap_ratio must be finite in [0, 1)")
    window_stride = int(window_size * (1.0 - overlap))
    if window_stride <= 0:
        raise ValueError("window overlap produces a non-positive stride")

    snippet_count = (
        effective_frame_count + snippet_stride - 1
    ) // snippet_stride
    valid_lengths: list[int] = []
    for index in range(max(1, snippet_count // window_stride)):
        window_start = index * window_stride
        window_end = window_start + window_size
        last_window = False
        if window_end > snippet_count:
            window_end = snippet_count
            window_start = max(0, window_end - window_size)
            last_window = True
        valid_len = window_end - window_start
        if valid_len <= 0 or valid_len > window_size:
            raise ValueError("derived natural window length is invalid")
        valid_lengths.append(valid_len)
        if last_window:
            break
    return tuple(valid_lengths)


def _short_bin(valid_len: int) -> str | None:
    for lower, upper in SHORT_BINS:
        if lower <= valid_len <= upper:
            return f"short_{lower}_{upper}"
    return None


def _build_feasibility_receipt(
    *,
    split_manifest: Mapping[str, Any],
    split_manifest_sha256: str,
    annotation: Mapping[str, Any],
    annotation_sha256: str,
) -> dict[str, Any]:
    if split_manifest.get("schema") != SOURCE_SPLIT_SCHEMA:
        raise ValueError("unsupported split manifest schema")
    roles = split_manifest.get("train_roles")
    if not isinstance(roles, Mapping) or SOURCE_ROLE not in roles:
        raise ValueError("detector_selector_train role is missing")
    role = roles[SOURCE_ROLE]
    if not isinstance(role, Mapping):
        raise ValueError("detector_selector_train role is invalid")
    videos = role.get("videos")
    if not isinstance(videos, Sequence) or isinstance(videos, (str, bytes)):
        raise ValueError("detector_selector_train videos must be a sequence")
    video_ids = [str(video_id) for video_id in videos]
    if not video_ids or len(video_ids) != len(set(video_ids)):
        raise ValueError("detector_selector_train video IDs must be unique")
    if int(role.get("video_count", -1)) != len(video_ids):
        raise ValueError("detector_selector_train video count drift")
    if split_manifest.get("annotation_sha256") != annotation_sha256:
        raise ValueError("annotation SHA-256 drift")
    database = annotation.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("annotation database is missing")

    rows: list[dict[str, Any]] = []
    short_counts: Counter[str] = Counter()
    for video_id in video_ids:
        video = database.get(video_id)
        if not isinstance(video, Mapping):
            raise ValueError(f"annotation is missing source video {video_id}")
        if str(video.get("subset")) != str(split_manifest.get("train_source_subset")):
            raise ValueError(f"annotation subset drift for {video_id}")
        frame_count = int(video.get("frame", 0))
        valid_lengths = enumerate_natural_window_valid_lengths(
            frame_count=frame_count,
            duration=float(video.get("duration", 0.0)),
            fps=FPS,
        )
        has_full = WINDOW_SIZE in valid_lengths
        short_lengths = [value for value in valid_lengths if value < WINDOW_SIZE]
        bins = sorted(
            {
                label
                for value in short_lengths
                if (label := _short_bin(value)) is not None
            }
        )
        for label in bins:
            short_counts[label] += 1
        rows.append(
            {
                "video_id": video_id,
                "frame_count": frame_count,
                "snippet_count": (
                    frame_count + SNIPPET_STRIDE - 1
                )
                // SNIPPET_STRIDE,
                "natural_window_valid_lengths": list(valid_lengths),
                "has_natural_full_window": has_full,
                "has_natural_short_window": bool(short_lengths),
                "short_bins": bins,
            }
        )

    required_video_count = NUMERIC_ROLE_COUNT * VIDEOS_PER_ROLE
    full_count = sum(row["has_natural_full_window"] for row in rows)
    short_count = sum(row["has_natural_short_window"] for row in rows)
    both_count = sum(
        row["has_natural_full_window"] and row["has_natural_short_window"]
        for row in rows
    )
    reason_codes: list[str] = []
    if len(rows) < required_video_count:
        reason_codes.append("insufficient_training_only_videos_for_three_roles")
    if both_count != len(rows):
        reason_codes.append(
            "natural_full_and_short_per_video_infeasible_under_current_sliding_enumerator"
        )
    for lower, upper in SHORT_BINS:
        label = f"short_{lower}_{upper}"
        if short_counts[label] < MIN_SHORT_VIDEOS_PER_BIN:
            reason_codes.append(f"insufficient_{label}_video_coverage")

    payload = {
        "schema": FEASIBILITY_SCHEMA,
        "status": "passed" if not reason_codes else "failed",
        "reason_codes": reason_codes,
        "protocol_id": PROTOCOL_ID,
        "formal_data_artifact": False,
        "formal_inventory_ready": False,
        "phase1_v2_authorized": False,
        "admission_effect": False,
        "paper_claim_allowed": False,
        "uses_development": False,
        "uses_official_final": False,
        "phase4_submission_enabled": False,
        "official_final_sealed": True,
        "source": {
            "split_manifest_sha256": str(split_manifest_sha256),
            "split_assignment_sha256": str(
                split_manifest.get("assignment_sha256", "")
            ),
            "annotation_sha256": str(annotation_sha256),
            "source_subset": str(split_manifest.get("train_source_subset")),
            "source_role": SOURCE_ROLE,
            "source_video_ids_sha256": canonical_sha256(video_ids),
        },
        "frozen_proposal_checked": {
            "numeric_role_count": NUMERIC_ROLE_COUNT,
            "videos_per_role": VIDEOS_PER_ROLE,
            "required_video_count": required_video_count,
            "window_size": WINDOW_SIZE,
            "window_overlap_ratio": WINDOW_OVERLAP_RATIO,
            "snippet_stride": SNIPPET_STRIDE,
            "fps": FPS,
            "per_video_natural_full_required": True,
            "per_video_natural_short_required": True,
            "short_bins": [list(bounds) for bounds in SHORT_BINS],
            "min_distinct_videos_per_short_bin": MIN_SHORT_VIDEOS_PER_BIN,
        },
        "observed_metadata_only_counts": {
            "source_video_count": len(rows),
            "videos_with_natural_full": full_count,
            "videos_with_natural_short": short_count,
            "videos_with_both": both_count,
            "short_bin_video_counts": {
                f"short_{lower}_{upper}": short_counts[
                    f"short_{lower}_{upper}"
                ]
                for lower, upper in SHORT_BINS
            },
        },
        "enumerator_contract": {
            "implementation": (
                "SlidingWindowDataset.split_video_to_windows-compatible"
            ),
            "terminal_behavior": (
                "back_shift_to_full_window_when_snippet_count_at_least_window_size"
            ),
            "decoded_frames_consumed": False,
            "candidate_outputs_consumed": False,
        },
        "video_contract_rows": rows,
    }
    return with_content_sha256(payload)


def audit_feasibility(
    *,
    split_manifest_path: str | Path,
    expected_split_manifest_sha256: str,
) -> dict[str, Any]:
    split_path = Path(split_manifest_path).expanduser().resolve()
    if not split_path.is_file():
        raise FileNotFoundError(split_path)
    split_sha = _sha256_file(split_path)
    if split_sha != str(expected_split_manifest_sha256):
        raise ValueError("split manifest SHA-256 drift")
    split_manifest = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(split_manifest, Mapping):
        raise ValueError("split manifest must be a JSON object")
    annotation_path = Path(
        str(split_manifest.get("annotation_path", ""))
    ).expanduser()
    if not annotation_path.is_absolute():
        annotation_path = split_path.parent / annotation_path
    annotation_path = annotation_path.resolve()
    if not annotation_path.is_file():
        raise FileNotFoundError(annotation_path)
    annotation_sha = _sha256_file(annotation_path)
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(annotation, Mapping):
        raise ValueError("annotation must be a JSON object")
    return _build_feasibility_receipt(
        split_manifest=split_manifest,
        split_manifest_sha256=split_sha,
        annotation=annotation,
        annotation_sha256=annotation_sha,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether the proposed Admission v2.1 role/window contract is "
            "feasible from immutable training-only metadata."
        )
    )
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = audit_feasibility(
        split_manifest_path=args.split_manifest,
        expected_split_manifest_sha256=args.expected_split_manifest_sha256,
    )
    write_json_exclusive_atomic(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0 if payload["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
