#!/usr/bin/env python3
"""Independent NumPy/float64 closure for frozen PhysTime decode artifacts.

This file intentionally does not import OpenTAD decode, NMS, or evaluation
implementations. It only consumes their sealed artifact schemas.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np


OUTPUT_SCHEMA = "phystime_independent_recompute_v1"
POLICY_SCHEMA = "phystime_independent_nms_policy_v1"
COMPLETION_SCHEMA = "phystime_decode_cross_completion_v1"
CAPTURE_SCHEMA = "phystime_decode_replay_inputs_v2"
AXIS_ARRAYS = {
    "uniform_rank_seconds": "uniform_axis_sec",
    "physical_time_seconds": "physical_axis_sec",
}
AXES = tuple(AXIS_ARRAYS)
METRIC_KEYS = (
    "average_mAP",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "mAP@0.6",
    "mAP@0.7",
)
TIOS = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)


class IndependentClosureError(ValueError):
    """Raised when a sealed artifact or independent closure is invalid."""


def require(condition, message):
    if not condition:
        raise IndependentClosureError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_nonfinite_json(value):
    raise IndependentClosureError(f"JSON contains non-finite numeric token: {value}")


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=_reject_nonfinite_json,
    )


def load_gzip_json(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle, parse_constant=_reject_nonfinite_json)


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def validate_artifact_record(record, description):
    require(isinstance(record, dict), f"{description} record is not an object")
    path = Path(record.get("path", "")).resolve()
    require(path.is_file(), f"missing {description}: {path}")
    require(
        record.get("sha256") == sha256_file(path),
        f"{description} SHA-256 mismatch",
    )
    if "size_bytes" in record:
        require(
            int(record["size_bytes"]) == path.stat().st_size,
            f"{description} size mismatch",
        )
    return path


def validate_policy(policy):
    require(policy.get("schema_version") == POLICY_SCHEMA, "policy schema mismatch")
    require(policy.get("subset") == "test", "THUMOS14 evaluation subset must be test")
    require(
        policy.get("annotation_subset") == "validation",
        "OpenTAD THUMOS14 annotation subset must be validation",
    )
    require(
        int(policy.get("expected_annotation_video_count", -1)) == 211,
        "THUMOS14 validation video count contract mismatch",
    )
    require(
        int(policy.get("expected_annotation_gt_count", -1)) == 3325,
        "THUMOS14 validation ground-truth count contract mismatch",
    )
    require(
        int(policy.get("expected_annotation_class_count", -1)) == 20,
        "THUMOS14 validation class count contract mismatch",
    )
    thresholds = [float(value) for value in policy.get("tiou_thresholds", [])]
    require(thresholds == TIOS.tolist(), "policy tIoU thresholds mismatch")
    pre_nms_thresh = float(policy["pre_nms_thresh"])
    require(
        math.isfinite(pre_nms_thresh) and pre_nms_thresh >= 0.0,
        "pre-NMS threshold must be finite and non-negative",
    )
    require(int(policy["pre_nms_topk"]) > 0, "pre-NMS top-k must be positive")
    minimum_duration = float(policy["proposal_min_duration"])
    require(
        math.isfinite(minimum_duration) and minimum_duration >= 0.0,
        "proposal minimum duration must be finite and non-negative",
    )
    require(
        int(policy["segment_round_digits"]) >= 0
        and int(policy["score_round_digits"]) >= 0,
        "rounding digits must be non-negative",
    )
    nms = policy.get("nms", {})
    require(nms.get("use_soft_nms") is True, "closure expects Soft-NMS")
    require(nms.get("multiclass") is True, "closure expects multiclass Soft-NMS")
    require(int(nms.get("method", 2)) == 2, "closure expects Gaussian Soft-NMS")
    sigma = float(nms["sigma"])
    require(
        math.isfinite(sigma) and sigma > 0.0,
        "Soft-NMS sigma must be finite and positive",
    )
    require(int(nms["max_seg_num"]) > 0, "max segment count must be positive")
    min_score = float(nms.get("min_score", 0.0))
    require(
        math.isfinite(min_score) and min_score >= 0.0,
        "NMS score floor must be finite and non-negative",
    )
    return policy


def validate_annotation_contract(annotation, policy):
    database = annotation.get("database")
    require(isinstance(database, dict), "annotation database is missing")
    subset = policy["annotation_subset"]
    records = [
        (video_name, record)
        for video_name, record in database.items()
        if record.get("subset") == subset
    ]
    require(
        len(records) == int(policy["expected_annotation_video_count"]),
        (
            f"annotation subset {subset} video count mismatch: "
            f"expected {policy['expected_annotation_video_count']}, "
            f"observed {len(records)}"
        ),
    )
    annotations = []
    for video_name, record in records:
        rows = record.get("annotations")
        require(
            isinstance(rows, list),
            f"annotation rows are not a list: {video_name}",
        )
        annotations.extend(rows)
    require(
        len(annotations) == int(policy["expected_annotation_gt_count"]),
        (
            f"annotation subset {subset} ground-truth count mismatch: "
            f"expected {policy['expected_annotation_gt_count']}, "
            f"observed {len(annotations)}"
        ),
    )
    labels = {row.get("label") for row in annotations}
    require(None not in labels, "annotation contains a missing class label")
    require(
        len(labels) == int(policy["expected_annotation_class_count"]),
        (
            f"annotation subset {subset} class count mismatch: "
            f"expected {policy['expected_annotation_class_count']}, "
            f"observed {len(labels)}"
        ),
    )
    subset_histogram = {}
    for record in database.values():
        name = record.get("subset")
        subset_histogram[name] = subset_histogram.get(name, 0) + 1
    return {
        "logical_evaluation_subset": policy["subset"],
        "annotation_subset": subset,
        "video_count": len(records),
        "ground_truth_count": len(annotations),
        "class_count": len(labels),
        "video_subset_histogram": subset_histogram,
    }


def load_capture(completion):
    manifest_path = validate_artifact_record(
        completion["artifacts"]["capture_manifest"],
        "capture manifest",
    )
    npz_path = validate_artifact_record(
        completion["artifacts"]["capture_npz"],
        "capture NPZ",
    )
    manifest = load_json(manifest_path)
    require(manifest.get("schema_version") == CAPTURE_SCHEMA, "capture schema mismatch")
    require(
        manifest.get("artifact", {}).get("sha256") == sha256_file(npz_path),
        "capture manifest/NPZ binding mismatch",
    )
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    contracts = manifest.get("array_contract", {})
    require(set(arrays) == set(contracts), "capture array contract set mismatch")
    for name, array in arrays.items():
        contract = contracts[name]
        require(contract.get("dtype") == str(array.dtype), f"{name} dtype mismatch")
        require(
            contract.get("shape") == list(array.shape),
            f"{name} shape mismatch",
        )
        require(
            contract.get("canonical_sha256") == array_sha256(array),
            f"{name} canonical SHA-256 mismatch",
        )
    require(
        len(manifest.get("windows", [])) == arrays["reg_distances"].shape[0],
        "capture window count mismatch",
    )
    return manifest, arrays


def map_rank_to_seconds(coords, positions, domain_start, domain_end):
    """Piecewise-linear selected-rank map implemented only with NumPy/float64."""
    coords = np.asarray(coords, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64).reshape(-1)
    domain_start = float(domain_start)
    domain_end = float(domain_end)
    require(np.isfinite(coords).all(), "rank coordinates contain non-finite values")
    require(
        math.isfinite(domain_start)
        and math.isfinite(domain_end)
        and domain_start < domain_end,
        "decode domain must be finite and strictly increasing",
    )
    require(positions.size > 0, "axis positions cannot be empty")
    require(np.isfinite(positions).all(), "axis positions contain non-finite values")
    require(
        positions.size == 1 or np.all(np.diff(positions) > 0.0),
        "axis positions must be strictly increasing",
    )
    ranks = np.arange(positions.size, dtype=np.float64)
    xp = np.concatenate(
        (
            np.asarray([-0.5], dtype=np.float64),
            ranks,
            np.asarray([positions.size - 0.5], dtype=np.float64),
        )
    )
    fp = np.concatenate(
        (
            np.asarray([domain_start], dtype=np.float64),
            positions,
            np.asarray([domain_end], dtype=np.float64),
        )
    )
    values = np.clip(coords.reshape(-1), -0.5, positions.size - 0.5)
    upper = np.searchsorted(xp, values, side="right")
    upper = np.clip(upper, 1, xp.size - 1)
    lower = upper - 1
    denominator = np.maximum(xp[upper] - xp[lower], 1.0e-12)
    fraction = (values - xp[lower]) / denominator
    mapped = fp[lower] + fraction * (fp[upper] - fp[lower])
    return mapped.reshape(coords.shape)


def recompute_dense_decode(capture_arrays, axis_name):
    require(axis_name in AXIS_ARRAYS, f"unknown decode axis: {axis_name}")
    base = np.asarray(capture_arrays["base_points"], dtype=np.float64)
    reg = np.asarray(capture_arrays["reg_distances"], dtype=np.float64)
    base_mask = np.asarray(capture_arrays["base_mask"], dtype=np.bool_)
    counts = np.asarray(capture_arrays["native_valid_count"], dtype=np.int64)
    domains = np.asarray(capture_arrays["domain_sec"], dtype=np.float64)
    axis_values = np.asarray(
        capture_arrays[AXIS_ARRAYS[axis_name]],
        dtype=np.float64,
    )
    require(base.ndim == 2 and base.shape[1] == 4, "base point shape mismatch")
    require(
        reg.ndim == 3 and reg.shape[1:] == (base.shape[0], 2),
        "regression shape mismatch",
    )
    batch_size = reg.shape[0]
    require(
        base_mask.shape == (batch_size, base.shape[0]),
        "base mask shape mismatch",
    )
    require(counts.shape == (batch_size,), "native count shape mismatch")
    require(domains.shape == (batch_size, 2), "decode domain shape mismatch")
    require(
        axis_values.ndim == 2 and axis_values.shape[0] == batch_size,
        "axis values shape mismatch",
    )
    require(np.isfinite(base).all(), "base points contain non-finite values")
    require(np.isfinite(reg).all(), "regression distances contain non-finite values")
    require(np.isfinite(domains).all(), "decode domains contain non-finite values")
    require(np.all(reg >= 0.0), "regression distances must be non-negative")
    dense = np.empty(reg.shape[:2] + (2,), dtype=np.float64)
    points_all = np.empty(reg.shape[:2] + (4,), dtype=np.float64)
    masks = np.empty(reg.shape[:2], dtype=np.bool_)
    center = base[:, 0]
    nominal_stride = np.maximum(base[:, 3], 1.0e-12)
    for window_idx in range(reg.shape[0]):
        count = int(counts[window_idx])
        require(0 < count <= axis_values.shape[1], "invalid native count")
        positions = axis_values[window_idx, :count]
        padding = axis_values[window_idx, count:]
        require(
            np.isfinite(positions).all(),
            "axis valid prefix contains non-finite values",
        )
        require(
            positions.size == 1 or np.all(np.diff(positions) > 0.0),
            "axis valid prefix must be strictly increasing",
        )
        require(np.isnan(padding).all(), "axis padding must contain only NaN")
        start, end = domains[window_idx]
        require(start < end, "decode domain must be strictly increasing")
        mapped_center = map_rank_to_seconds(center, positions, start, end)
        mapped_left = map_rank_to_seconds(
            center - 0.5 * nominal_stride,
            positions,
            start,
            end,
        )
        mapped_right = map_rank_to_seconds(
            center + 0.5 * nominal_stride,
            positions,
            start,
            end,
        )
        physical_stride = np.maximum(mapped_right - mapped_left, 1.0e-12)
        scale = physical_stride / nominal_stride
        points = base.copy()
        points[:, 0] = mapped_center
        points[:, 1] = base[:, 1] * scale
        points[:, 2] = base[:, 2] * scale
        points[:, 3] = physical_stride
        proposals = np.stack(
            (
                mapped_center - reg[window_idx, :, 0] * physical_stride,
                mapped_center + reg[window_idx, :, 1] * physical_stride,
            ),
            axis=-1,
        )
        proposals[:, 0] = np.clip(proposals[:, 0], start, end)
        proposals[:, 1] = np.clip(proposals[:, 1], start, end)
        dense[window_idx] = proposals
        points_all[window_idx] = points
        masks[window_idx] = base_mask[window_idx] & (center < float(count))
    return dense, masks, points_all


def build_pre_cross(dense_proposals, valid_mask, scores, capture, policy):
    """Independently rebuild per-window top-k detections with stable ranking."""
    class_map = list(capture["class_map"])
    threshold = float(policy["pre_nms_thresh"])
    topk = int(policy["pre_nms_topk"])
    round_before = bool(policy["round_before_cross_window_nms"])
    segment_digits = int(policy["segment_round_digits"])
    score_digits = int(policy["score_round_digits"])
    results = {}
    tied_boundaries = 0
    require(
        dense_proposals.ndim == 3 and dense_proposals.shape[-1] == 2,
        "dense proposal shape mismatch",
    )
    require(
        valid_mask.shape == dense_proposals.shape[:2],
        "valid mask shape mismatch",
    )
    require(
        scores.ndim == 3
        and scores.shape[:2] == dense_proposals.shape[:2]
        and scores.shape[2] == len(class_map),
        "score tensor shape mismatch",
    )
    require(
        np.isfinite(dense_proposals).all(),
        "dense proposals contain non-finite values",
    )
    require(np.isfinite(scores).all(), "scores contain non-finite values")
    require(
        len(capture.get("windows", [])) == dense_proposals.shape[0],
        "capture window metadata count mismatch",
    )
    for window_idx, window in enumerate(capture["windows"]):
        mask = valid_mask[window_idx]
        segments = dense_proposals[window_idx, mask]
        window_scores = scores[window_idx, mask]
        flat_scores = window_scores.reshape(-1)
        kept_flat_indices = np.flatnonzero(flat_scores > threshold)
        kept_scores = flat_scores[kept_flat_indices]
        order = np.argsort(-kept_scores, kind="stable")
        if order.size > topk:
            sorted_scores = kept_scores[order]
            if sorted_scores[topk - 1] == sorted_scores[topk]:
                tied_boundaries += 1
        order = order[:topk]
        kept_flat_indices = kept_flat_indices[order]
        kept_scores = kept_scores[order]
        point_indices = kept_flat_indices // len(class_map)
        class_indices = kept_flat_indices % len(class_map)
        kept_segments = segments[point_indices].astype(np.float64, copy=True)
        duration = float(window["duration"])
        kept_segments = np.clip(kept_segments, 0.0, duration)
        detections = []
        for segment, class_idx, score in zip(
            kept_segments,
            class_indices,
            kept_scores,
        ):
            segment_value = [float(segment[0]), float(segment[1])]
            score_value = float(score)
            if round_before:
                segment_value = [
                    round(value, segment_digits) for value in segment_value
                ]
                score_value = round(score_value, score_digits)
            detections.append(
                {
                    "segment": segment_value,
                    "label": class_map[int(class_idx)],
                    "score": score_value,
                }
            )
        results.setdefault(window["video_name"], []).extend(detections)
    return results, {"topk_tied_boundary_windows": tied_boundaries}


def _validate_detections(detections, minimum_duration):
    valid = []
    invalid = 0
    for item in detections:
        try:
            start = float(item["segment"][0])
            end = float(item["segment"][1])
            score = float(item["score"])
            label = item["label"]
        except (KeyError, TypeError, ValueError, IndexError, OverflowError):
            invalid += 1
            continue
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or not math.isfinite(score)
            or end - start <= minimum_duration
            or label is None
        ):
            invalid += 1
            continue
        valid.append(
            {
                "segment": [start, end],
                "label": label,
                "score": score,
            }
        )
    return valid, invalid


def gaussian_soft_nms(segments, scores, original_indices, *, sigma, min_score):
    """Stable NumPy/float64 port of the sealed Gaussian Soft-NMS semantics."""
    segments = np.asarray(segments, dtype=np.float64).copy()
    scores = np.asarray(scores, dtype=np.float64).copy()
    original_indices = np.asarray(original_indices, dtype=np.int64).copy()
    require(segments.shape == (scores.size, 2), "Soft-NMS shape mismatch")
    require(original_indices.shape == scores.shape, "Soft-NMS index mismatch")
    require(np.isfinite(segments).all(), "Soft-NMS segments contain non-finite values")
    require(np.isfinite(scores).all(), "Soft-NMS scores contain non-finite values")
    require(np.all(segments[:, 1] > segments[:, 0]), "Soft-NMS segments are invalid")
    require(math.isfinite(sigma) and sigma > 0.0, "invalid Soft-NMS sigma")
    require(
        math.isfinite(min_score) and min_score >= 0.0,
        "invalid Soft-NMS score floor",
    )
    n = scores.size
    output_segments = np.empty((n, 2), dtype=np.float64)
    output_scores = np.empty(n, dtype=np.float64)
    output_indices = np.empty(n, dtype=np.int64)
    areas = segments[:, 1] - segments[:, 0] + 1.0e-6
    i = 0
    while i < n:
        tail = scores[i:n]
        max_offset = int(np.argmax(tail))
        max_pos = i + max_offset
        if max_pos != i:
            segments[[i, max_pos]] = segments[[max_pos, i]]
            scores[[i, max_pos]] = scores[[max_pos, i]]
            areas[[i, max_pos]] = areas[[max_pos, i]]
            original_indices[[i, max_pos]] = original_indices[[max_pos, i]]
        selected = segments[i].copy()
        selected_score = float(scores[i])
        output_segments[i] = selected
        output_scores[i] = selected_score
        output_indices[i] = original_indices[i]
        if i + 1 < n:
            left = np.maximum(selected[0], segments[i + 1 : n, 0])
            right = np.minimum(selected[1], segments[i + 1 : n, 1])
            intersection = np.maximum(0.0, right - left)
            overlap = intersection / (
                areas[i] + areas[i + 1 : n] - intersection
            )
            scores[i + 1 : n] *= np.exp(-(overlap * overlap) / sigma)
            if min_score > 0.0:
                pos = i + 1
                while pos < n:
                    if scores[pos] < min_score:
                        last = n - 1
                        segments[pos] = segments[last]
                        scores[pos] = scores[last]
                        areas[pos] = areas[last]
                        original_indices[pos] = original_indices[last]
                        n -= 1
                    else:
                        pos += 1
        i += 1
    return (
        output_segments[:n],
        output_scores[:n],
        output_indices[:n],
    )


def independent_cross_window_nms(pre_cross, policy):
    nms = policy["nms"]
    minimum_duration = float(policy["proposal_min_duration"])
    round_before = bool(policy["round_before_cross_window_nms"])
    round_after = bool(policy["round_after_cross_window_nms"])
    segment_digits = int(policy["segment_round_digits"])
    score_digits = int(policy["score_round_digits"])
    filter_invalid = bool(policy["filter_invalid_proposals"])
    max_segments = int(nms["max_seg_num"])
    sigma = float(nms["sigma"])
    min_score = float(nms.get("min_score", 0.0))
    merged = {}
    audit = {
        "input_detections": 0,
        "invalid_detections": 0,
        "post_nms_detections": 0,
        "equal_score_groups": 0,
    }
    for video_name, detections in pre_cross.items():
        audit["input_detections"] += len(detections)
        valid, invalid = _validate_detections(detections, minimum_duration)
        audit["invalid_detections"] += invalid
        if invalid and not filter_invalid:
            raise IndependentClosureError(
                f"invalid detections reached unfiltered NMS: {video_name}"
            )
        if round_before:
            rounded = []
            for item in valid:
                rounded.append(
                    {
                        "segment": [
                            round(value, segment_digits)
                            for value in item["segment"]
                        ],
                        "label": item["label"],
                        "score": round(item["score"], score_digits),
                    }
                )
            valid, induced_invalid = _validate_detections(
                rounded,
                minimum_duration,
            )
            audit["invalid_detections"] += induced_invalid

        class_order = []
        by_class = {}
        for original_index, item in enumerate(valid):
            label = item["label"]
            if label not in by_class:
                by_class[label] = []
                class_order.append(label)
            by_class[label].append((original_index, item))

        class_outputs = []
        for label in class_order:
            rows = by_class[label]
            segments = np.asarray(
                [item["segment"] for _, item in rows],
                dtype=np.float64,
            )
            scores = np.asarray(
                [item["score"] for _, item in rows],
                dtype=np.float64,
            )
            indices = np.asarray([index for index, _ in rows], dtype=np.int64)
            _, counts = np.unique(scores, return_counts=True)
            audit["equal_score_groups"] += int(np.sum(counts > 1))
            out_segments, out_scores, out_indices = gaussian_soft_nms(
                segments,
                scores,
                indices,
                sigma=sigma,
                min_score=min_score,
            )
            class_limit = min(max_segments, out_scores.size)
            for segment, score, original_index in zip(
                out_segments[:class_limit],
                out_scores[:class_limit],
                out_indices[:class_limit],
            ):
                class_outputs.append(
                    (float(score), int(original_index), label, segment)
                )

        # Stable score ordering with original sequence index as the tie breaker.
        class_outputs.sort(key=lambda row: (-row[0], row[1]))
        class_outputs = class_outputs[:max_segments]
        video_output = []
        for score, _, label, segment in class_outputs:
            segment_value = [float(segment[0]), float(segment[1])]
            score_value = float(score)
            if round_after:
                segment_value = [
                    round(value, segment_digits) for value in segment_value
                ]
                score_value = round(score_value, score_digits)
            video_output.append(
                {
                    "segment": segment_value,
                    "label": label,
                    "score": score_value,
                }
            )
        valid_output, invalid_output = _validate_detections(
            video_output,
            minimum_duration,
        )
        require(
            invalid_output == 0,
            f"independent NMS produced invalid detections: {video_name}",
        )
        merged[video_name] = valid_output
        audit["post_nms_detections"] += len(valid_output)
    return merged, audit


def remove_duplicate_annotations(annotations, tolerance=1.0e-3):
    valid = []
    for annotation in annotations:
        start = float(annotation["segment"][0])
        end = float(annotation["segment"][1])
        label = annotation["label"]
        require(
            math.isfinite(start) and math.isfinite(end),
            "ground-truth segment contains non-finite values",
        )
        if end - start <= 0.0:
            continue
        duplicate = any(
            abs(start - previous["segment"][0]) <= tolerance
            and abs(end - previous["segment"][1]) <= tolerance
            and label == previous["label"]
            for previous in valid
        )
        if not duplicate:
            valid.append(
                {"segment": [start, end], "label": label}
            )
    return valid


def segment_iou(segment, candidates):
    segment = np.asarray(segment, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    require(segment.shape == (2,), "query segment shape mismatch")
    require(
        candidates.ndim == 2 and candidates.shape[1] == 2,
        "candidate segment shape mismatch",
    )
    require(
        np.isfinite(segment).all() and np.isfinite(candidates).all(),
        "segment IoU input contains non-finite values",
    )
    intersection = np.maximum(
        0.0,
        np.minimum(segment[1], candidates[:, 1])
        - np.maximum(segment[0], candidates[:, 0]),
    )
    union = (
        candidates[:, 1]
        - candidates[:, 0]
        + segment[1]
        - segment[0]
        - intersection
    )
    return intersection / np.maximum(union, 1.0e-8)


def interpolated_ap(precision, recall):
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    padded_precision = np.concatenate(([0.0], precision, [0.0]))
    padded_recall = np.concatenate(([0.0], recall, [1.0]))
    for index in range(padded_precision.size - 2, -1, -1):
        padded_precision[index] = max(
            padded_precision[index],
            padded_precision[index + 1],
        )
    changes = np.flatnonzero(padded_recall[1:] != padded_recall[:-1]) + 1
    return float(
        np.sum(
            (padded_recall[changes] - padded_recall[changes - 1])
            * padded_precision[changes]
        )
    )


def independent_thumos_evaluate(annotation, predictions, subset="test"):
    database = annotation.get("database")
    require(isinstance(database, dict), "annotation database is missing")
    ground_truth = {}
    class_order = []
    for video_name, record in database.items():
        if record.get("subset") != subset:
            continue
        for item in remove_duplicate_annotations(record.get("annotations", [])):
            label = item["label"]
            if label not in ground_truth:
                ground_truth[label] = []
                class_order.append(label)
            ground_truth[label].append(
                {
                    "video_name": video_name,
                    "segment": item["segment"],
                }
            )
    require(class_order, "no THUMOS14 ground truth found for the requested subset")

    per_class = {}
    ap_matrix = np.zeros((TIOS.size, len(class_order)), dtype=np.float64)
    for class_index, label in enumerate(class_order):
        gt_rows = ground_truth[label]
        gt_by_video = {}
        for global_index, row in enumerate(gt_rows):
            gt_by_video.setdefault(row["video_name"], []).append(
                (global_index, row["segment"])
            )
        prediction_rows = []
        sequence_index = 0
        for video_name, items in predictions.items():
            for item in items:
                if item["label"] == label:
                    prediction_rows.append(
                        {
                            "video_name": video_name,
                            "segment": [
                                float(item["segment"][0]),
                                float(item["segment"][1]),
                            ],
                            "score": float(item["score"]),
                            "sequence_index": sequence_index,
                        }
                    )
                sequence_index += 1
        prediction_rows.sort(
            key=lambda row: (-row["score"], row["sequence_index"])
        )
        tp = np.zeros((TIOS.size, len(prediction_rows)), dtype=np.float64)
        fp = np.zeros_like(tp)
        locks = -np.ones((TIOS.size, len(gt_rows)), dtype=np.int64)
        for prediction_index, prediction in enumerate(prediction_rows):
            candidates = gt_by_video.get(prediction["video_name"])
            if not candidates:
                fp[:, prediction_index] = 1.0
                continue
            candidate_segments = np.asarray(
                [segment for _, segment in candidates],
                dtype=np.float64,
            )
            overlaps = segment_iou(prediction["segment"], candidate_segments)
            overlap_order = np.argsort(-overlaps, kind="stable")
            for threshold_index, threshold in enumerate(TIOS):
                matched = False
                for local_index in overlap_order:
                    if overlaps[local_index] < threshold:
                        break
                    global_index = candidates[int(local_index)][0]
                    if locks[threshold_index, global_index] >= 0:
                        continue
                    locks[threshold_index, global_index] = prediction_index
                    tp[threshold_index, prediction_index] = 1.0
                    matched = True
                    break
                if not matched:
                    fp[threshold_index, prediction_index] = 1.0
        class_ap = np.zeros(TIOS.size, dtype=np.float64)
        for threshold_index in range(TIOS.size):
            cumulative_tp = np.cumsum(tp[threshold_index])
            cumulative_fp = np.cumsum(fp[threshold_index])
            recall = cumulative_tp / float(len(gt_rows))
            precision = cumulative_tp / np.maximum(
                cumulative_tp + cumulative_fp,
                1.0e-12,
            )
            class_ap[threshold_index] = interpolated_ap(precision, recall)
        ap_matrix[:, class_index] = class_ap
        per_class[label] = {
            f"mAP@{threshold}": float(value)
            for threshold, value in zip(TIOS, class_ap)
        }
    mean_ap = ap_matrix.mean(axis=1)
    metrics = {"average_mAP": float(mean_ap.mean())}
    metrics.update(
        {
            f"mAP@{threshold}": float(value)
            for threshold, value in zip(TIOS, mean_ap)
        }
    )
    return metrics, per_class


def _metric_values(payload):
    require(isinstance(payload, dict), "metrics artifact is not an object")
    require("evaluation_epoch" in payload, "metrics evaluation epoch is missing")
    values = {key: float(payload[key]) for key in METRIC_KEYS}
    require(
        all(math.isfinite(value) for value in values.values()),
        "metrics artifact contains non-finite values",
    )
    return values


def _metric_deltas(observed, expected):
    return {
        key: float(observed[key]) - float(expected[key])
        for key in METRIC_KEYS
    }


def _validate_result_payload(payload, description):
    require(isinstance(payload, dict), f"{description} is not an object")
    require(
        isinstance(payload.get("results"), dict),
        f"{description} results are not an object",
    )
    require(
        "evaluation_epoch" in payload,
        f"{description} evaluation epoch is missing",
    )
    return payload["results"]


def compare_detection_maps(expected, observed, *, segment_atol, score_atol):
    """Compare ordered prediction maps without relying on production code."""
    require(isinstance(expected, dict), "expected detections are not an object")
    require(isinstance(observed, dict), "observed detections are not an object")
    issues = []
    max_segment_error = 0.0
    max_score_error = 0.0
    if set(expected) != set(observed):
        issues.append("video key set differs")
    for video_name in sorted(set(expected) & set(observed)):
        expected_rows = expected[video_name]
        observed_rows = observed[video_name]
        require(
            isinstance(expected_rows, list) and isinstance(observed_rows, list),
            f"detection rows are not lists: {video_name}",
        )
        if len(expected_rows) != len(observed_rows):
            issues.append(
                f"{video_name}: detection count {len(observed_rows)} "
                f"!= {len(expected_rows)}"
            )
        for row_index, (expected_row, observed_row) in enumerate(
            zip(expected_rows, observed_rows)
        ):
            expected_valid, expected_invalid = _validate_detections(
                [expected_row], -1.0
            )
            observed_valid, observed_invalid = _validate_detections(
                [observed_row], -1.0
            )
            require(
                expected_invalid == 0 and observed_invalid == 0,
                f"malformed detection at {video_name}[{row_index}]",
            )
            expected_item = expected_valid[0]
            observed_item = observed_valid[0]
            if expected_item["label"] != observed_item["label"]:
                issues.append(f"{video_name}[{row_index}]: label differs")
            segment_error = float(
                np.max(
                    np.abs(
                        np.asarray(expected_item["segment"], dtype=np.float64)
                        - np.asarray(observed_item["segment"], dtype=np.float64)
                    )
                )
            )
            score_error = abs(expected_item["score"] - observed_item["score"])
            max_segment_error = max(max_segment_error, segment_error)
            max_score_error = max(max_score_error, score_error)
            if segment_error > segment_atol:
                issues.append(
                    f"{video_name}[{row_index}]: segment error "
                    f"{segment_error} > {segment_atol}"
                )
            if score_error > score_atol:
                issues.append(
                    f"{video_name}[{row_index}]: score error "
                    f"{score_error} > {score_atol}"
                )
    return {
        "match": not issues,
        "issues": issues,
        "max_abs_segment_error_seconds": max_segment_error,
        "max_abs_score_error": max_score_error,
        "canonical_exact_match": canonical_sha256(expected)
        == canonical_sha256(observed),
    }


def validate_completion(
    completion_path,
    *,
    annotation,
    policy,
    proposal_atol,
    metric_atol,
):
    completion_path = Path(completion_path).resolve()
    completion = load_json(completion_path)
    require(
        completion.get("schema_version") == COMPLETION_SCHEMA,
        "formal completion schema mismatch",
    )
    require(
        completion.get("validation_pass") is True
        and completion.get("status") == "tested",
        "formal completion did not pass",
    )
    require(completion.get("new_training") is False, "completion trained a model")
    capture, capture_arrays = load_capture(completion)
    require(
        capture.get("new_training") is False,
        "capture is not a frozen-tensor artifact",
    )

    issues = []
    axis_reports = {}
    for axis_name in AXES:
        mode_records = completion["mode_artifacts"][axis_name]
        candidate_path = validate_artifact_record(
            mode_records["decoded_candidates"],
            f"{axis_name} candidates",
        )
        pre_cross_path = validate_artifact_record(
            mode_records["pre_cross"],
            f"{axis_name} pre-cross",
        )
        result_path = validate_artifact_record(
            mode_records["result"],
            f"{axis_name} result",
        )
        metrics_path = validate_artifact_record(
            mode_records["metrics"],
            f"{axis_name} metrics",
        )
        with np.load(candidate_path, allow_pickle=False) as archive:
            candidate_arrays = {name: archive[name] for name in archive.files}
        require(
            set(candidate_arrays) == {"proposals", "valid_mask", "scores"},
            "candidate array set mismatch",
        )
        recomputed, recomputed_mask, recomputed_points = recompute_dense_decode(
            capture_arrays,
            axis_name,
        )
        mask_match = np.array_equal(
            recomputed_mask,
            candidate_arrays["valid_mask"],
        )
        score_match = np.array_equal(
            capture_arrays["cls_scores"],
            candidate_arrays["scores"],
        )
        require(
            np.isfinite(candidate_arrays["proposals"]).all(),
            f"{axis_name}: candidates contain non-finite proposals",
        )
        require(
            np.isfinite(candidate_arrays["scores"]).all(),
            f"{axis_name}: candidates contain non-finite scores",
        )
        proposal_error = float(
            np.max(
                np.abs(
                    recomputed
                    - candidate_arrays["proposals"].astype(np.float64)
                )
            )
        )
        if not mask_match:
            issues.append(f"{axis_name}: candidate mask differs")
        if not score_match:
            issues.append(f"{axis_name}: candidate scores differ")
        if not math.isfinite(proposal_error) or proposal_error > proposal_atol:
            issues.append(
                f"{axis_name}: proposal error {proposal_error} > {proposal_atol}"
            )

        rebuilt_pre_cross, ranking_audit = build_pre_cross(
            recomputed,
            recomputed_mask,
            capture_arrays["cls_scores"],
            capture,
            policy,
        )
        stored_pre_payload = load_gzip_json(pre_cross_path)
        require(
            stored_pre_payload.get("schema_version")
            == "phystime_decode_cross_pre_cross_v1",
            "pre-cross schema mismatch",
        )
        stored_pre = stored_pre_payload["results"]
        pre_cross_comparison = compare_detection_maps(
            rebuilt_pre_cross,
            stored_pre,
            segment_atol=proposal_atol,
            score_atol=metric_atol,
        )
        if not pre_cross_comparison["match"]:
            issues.append(
                f"{axis_name}: rebuilt pre-cross differs: "
                f"{pre_cross_comparison['issues'][:10]}"
            )
        independently_merged, nms_audit = independent_cross_window_nms(
            rebuilt_pre_cross,
            policy,
        )
        independent_metrics, per_class = independent_thumos_evaluate(
            annotation,
            independently_merged,
            subset=policy["annotation_subset"],
        )
        stored_result_payload = load_json(result_path)
        stored_result = _validate_result_payload(
            stored_result_payload,
            f"{axis_name} result artifact",
        )
        stored_metrics_payload = load_json(metrics_path)
        stored_metrics = _metric_values(stored_metrics_payload)
        require(
            int(stored_result_payload["evaluation_epoch"])
            == int(completion["evaluation_epoch"])
            == int(stored_metrics_payload["evaluation_epoch"]),
            f"{axis_name}: evaluation epoch binding mismatch",
        )
        result_comparison = compare_detection_maps(
            independently_merged,
            stored_result,
            segment_atol=proposal_atol,
            score_atol=metric_atol,
        )
        if not result_comparison["match"]:
            issues.append(
                f"{axis_name}: independent result differs: "
                f"{result_comparison['issues'][:10]}"
            )
        metric_deltas = _metric_deltas(independent_metrics, stored_metrics)
        max_metric_delta = max(abs(value) for value in metric_deltas.values())
        if not math.isfinite(max_metric_delta) or max_metric_delta > metric_atol:
            issues.append(
                f"{axis_name}: metric delta {max_metric_delta} > {metric_atol}"
            )
        rebuilt_counts = {
            name: len(items) for name, items in rebuilt_pre_cross.items()
        }
        stored_counts = {
            name: len(items) for name, items in stored_pre.items()
        }
        native_point_error = (
            float(
                np.max(
                    np.abs(
                        recomputed_points
                        - capture_arrays["native_points"].astype(np.float64)
                    )
                )
            )
            if axis_name == completion["native_axis"]
            else None
        )
        if native_point_error is not None and (
            not math.isfinite(native_point_error)
            or native_point_error > proposal_atol
        ):
            issues.append(
                f"{axis_name}: native point error {native_point_error} "
                f"> {proposal_atol}"
            )
        axis_reports[axis_name] = {
            "proposal_max_abs_error_seconds": proposal_error,
            "mask_exact_match": mask_match,
            "scores_exact_match": score_match,
            "native_point_max_abs_error_seconds": native_point_error,
            "rebuilt_pre_cross_canonical_sha256": canonical_sha256(
                rebuilt_pre_cross
            ),
            "stored_pre_cross_canonical_sha256": canonical_sha256(stored_pre),
            "rebuilt_pre_cross_exact_match": pre_cross_comparison[
                "canonical_exact_match"
            ],
            "rebuilt_pre_cross_tolerance_match": pre_cross_comparison["match"],
            "rebuilt_pre_cross_comparison": pre_cross_comparison,
            "rebuilt_pre_cross_video_counts_match": (
                rebuilt_counts == stored_counts
            ),
            "ranking_audit": ranking_audit,
            "independent_nms_audit": nms_audit,
            "independent_result_canonical_sha256": canonical_sha256(
                independently_merged
            ),
            "stored_result_canonical_sha256": canonical_sha256(stored_result),
            "independent_result_exact_match": result_comparison[
                "canonical_exact_match"
            ],
            "independent_result_tolerance_match": result_comparison["match"],
            "independent_result_comparison": result_comparison,
            "independent_metrics": independent_metrics,
            "stored_metrics": stored_metrics,
            "metric_deltas": metric_deltas,
            "max_abs_metric_delta": max_metric_delta,
            "per_class_metrics": per_class,
        }

    independent_delta = {
        key: axis_reports["physical_time_seconds"]["independent_metrics"][key]
        - axis_reports["uniform_rank_seconds"]["independent_metrics"][key]
        for key in METRIC_KEYS
    }
    stored_delta = completion["physical_minus_uniform_fraction"]
    sign_mismatches = []
    for key, independent_value in independent_delta.items():
        stored_value = float(stored_delta[key])
        if not math.isfinite(stored_value) or not math.isfinite(independent_value):
            sign_mismatches.append(key)
            continue
        if (
            abs(independent_value) > metric_atol
            and abs(stored_value) > metric_atol
            and math.copysign(1.0, independent_value)
            != math.copysign(1.0, stored_value)
        ):
            sign_mismatches.append(key)
    if sign_mismatches:
        issues.append(f"decode delta sign mismatch: {sign_mismatches}")

    return {
        "completion_path": str(completion_path),
        "completion_sha256": sha256_file(completion_path),
        "arm": completion["arm"],
        "weights_source": completion["weights_source"],
        "runtime_commit": completion["runtime_commit"],
        "runtime_tree": completion["runtime_tree"],
        "source_commit": completion["source_commit"],
        "source_tree": completion["source_tree"],
        "evaluation_epoch": completion["evaluation_epoch"],
        "capture_npz_sha256": completion["artifacts"]["capture_npz"]["sha256"],
        "axis_reports": axis_reports,
        "independent_physical_minus_uniform_fraction": independent_delta,
        "stored_physical_minus_uniform_fraction": {
            key: float(stored_delta[key]) for key in METRIC_KEYS
        },
        "delta_sign_mismatches": sign_mismatches,
        "issues": issues,
        "validation_pass": not issues,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Independently recompute frozen PhysTime decode evidence."
    )
    parser.add_argument("--completion", action="append", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--proposal-atol", type=float, default=1.0e-4)
    parser.add_argument("--metric-atol", type=float, default=1.0e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    require(args.proposal_atol >= 0.0, "proposal tolerance must be non-negative")
    require(args.metric_atol >= 0.0, "metric tolerance must be non-negative")
    policy_path = Path(args.policy).resolve()
    annotation_path = Path(args.annotation).resolve()
    require(policy_path.is_file(), "NMS policy file is missing")
    require(annotation_path.is_file(), "annotation file is missing")
    policy = validate_policy(load_json(policy_path))
    annotation = load_json(annotation_path)
    annotation_contract = validate_annotation_contract(annotation, policy)
    reports = [
        validate_completion(
            completion,
            annotation=annotation,
            policy=policy,
            proposal_atol=args.proposal_atol,
            metric_atol=args.metric_atol,
        )
        for completion in args.completion
    ]
    identities = {
        (
            report["runtime_commit"],
            report["runtime_tree"],
            report["source_commit"],
            report["source_tree"],
            report["evaluation_epoch"],
        )
        for report in reports
    }
    conditions = {
        (report["arm"], report["weights_source"])
        for report in reports
    }
    issues = []
    if len(identities) != 1:
        issues.append("formal completions do not share one frozen identity")
    if len(reports) == 4 and conditions != {
        ("selected_axis", "online"),
        ("selected_axis", "ema"),
        ("physical_metric", "online"),
        ("physical_metric", "ema"),
    }:
        issues.append("four-condition completion set is incomplete or duplicated")
    for report in reports:
        issues.extend(
            f"{report['arm']}/{report['weights_source']}: {issue}"
            for issue in report["issues"]
        )
    payload = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "tested" if not issues else "failed",
        "validation_pass": not issues,
        "new_training": False,
        "independent_implementation": {
            "language": "python_numpy",
            "geometry_dtype": "float64",
            "nms_dtype": "float64",
            "stable_tie_breaker": "original_sequence_index",
            "imports_opentad_decode_nms_or_evaluator": False,
        },
        "proposal_atol_seconds": args.proposal_atol,
        "metric_atol_fraction": args.metric_atol,
        "annotation": {
            "path": str(annotation_path),
            "sha256": sha256_file(annotation_path),
            **annotation_contract,
        },
        "policy": {
            "path": str(policy_path),
            "sha256": sha256_file(policy_path),
            "canonical_sha256": canonical_sha256(policy),
        },
        "completion_reports": reports,
        "issues": issues,
        "claim_boundary": (
            "diagnostic_only_independent_closure_passed"
            if not issues
            else "implementation_or_evaluation_issue_requires_resolution"
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if issues:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
