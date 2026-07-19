from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from functools import lru_cache
import json
import math
from pathlib import Path
import re
from typing import Any

from tools.bata.duca_allocation_families import (
    AllocationContractError,
    PhysicalAxis,
    physical_gap_report,
    resolve_physical_cap,
    select_family_a,
    select_family_b,
    select_family_c,
    validate_physical_selection,
)
from tools.bata.duca_exact_physical_solver import (
    GroundTruthObjectiveSpec,
    select_family_d,
    solve_ground_truth_lexicographic,
)
from tools.bata.export_duca_allocation_ceiling_inputs import (
    SCHEMA_VERSION as INPUT_SCHEMA_VERSION,
    SCORE_KEYS,
    canonical_sha256,
    sha256,
    write_json_exclusive,
)


OUTPUT_SCHEMA_VERSION = "duca_allocation_family_ceiling_record_v1"
SUMMARY_SCHEMA_VERSION = "duca_allocation_family_ceiling_summary_v1"
_INPUT_KEYS = {
    "schema_version",
    "sample_id",
    "video_id",
    "split",
    "valid_len",
    "requested_budget",
    "physical_axis",
    "coordinate_audit",
    "timeline_audit",
    "scores",
    "gt_segments",
    "gt_segments_unit",
    "gt_role",
    "source",
    "decision_contract",
    "record_sha256",
}
_PHYSICAL_AXIS_KEYS = {
    "dense_ordinals",
    "source_frames",
    "seconds",
    "decoder_fps",
    "annotation_fps",
    "total_frames",
}
_COORDINATE_AUDIT_KEYS = {
    "expected_source_frames",
    "max_abs_error_frames",
    "tolerance_frames",
    "passed",
}
_TIMELINE_AUDIT_KEYS = {
    "decoder_fps",
    "annotation_fps",
    "absolute_fps_error",
    "tolerance_fps",
    "cumulative_drift_frames",
    "tolerance_frames",
    "passed",
}
_SOURCE_KEYS = {
    "git_commit",
    "git_clean",
    "annotation_path",
    "annotation_sha256",
    "class_map_path",
    "class_map_sha256",
    "data_path",
    "dataset_data_manifest_sha256",
    "dataset_data_file_count",
    "dataset_data_total_bytes",
    "dataset_data_hash_algorithm",
    "dataset_subset_name",
    "dataset_test_mode",
    "dataset_filter_gt",
    "dataset_ioa_thresh",
    "dataset_feature_stride",
    "dataset_sample_stride",
    "dataset_window_size",
    "dataset_window_overlap_ratio",
    "dataset_offset_frames",
    "dataset_config_sha256",
    "dataset_window_manifest_sha256",
    "dataset_window_count",
    "dataset_window_deduplication",
    "dataset_duplicate_window_count_removed",
    "config",
    "config_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "checkpoint_state_key",
    "checkpoint_epoch",
    "split",
    "selector_only_inference",
    "detector_backbone_executed",
    "uses_gt_for_score_generation",
    "validation_authorized",
}
_DECISION_KEYS = {
    "offline_full_window",
    "gt_passed_to_selector",
    "teacher_passed_to_selector",
    "detector_backbone_executed",
    "valid_prefix_only",
}


def read_input_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            validate_input_record(row, context=f"{path}:{line_number}")
            sample_id = str(row["sample_id"])
            if sample_id in sample_ids:
                raise ValueError(f"{path}:{line_number}: duplicate sample_id {sample_id}")
            sample_ids.add(sample_id)
            records.append(row)
    if not records:
        raise ValueError(f"{path}: no input records")
    return records


def validate_input_record(row: Mapping[str, Any], *, context: str) -> None:
    if row.get("schema_version") != INPUT_SCHEMA_VERSION:
        raise ValueError(f"{context}: unsupported input schema")
    unknown = set(row) - _INPUT_KEYS
    missing = _INPUT_KEYS - set(row)
    if unknown or missing:
        raise ValueError(f"{context}: strict input fields mismatch: unknown={unknown}, missing={missing}")
    recorded_hash = row.get("record_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError(f"{context}: record_sha256 is required")
    unhashed = dict(row)
    unhashed.pop("record_sha256", None)
    if canonical_sha256(unhashed) != recorded_hash:
        raise ValueError(f"{context}: record_sha256 mismatch")
    if row.get("gt_role") != "privileged_diagnostic_only_never_score_generation":
        raise ValueError(f"{context}: GT role is not fail-closed")
    if not str(row.get("sample_id", "")).strip() or not str(row.get("video_id", "")).strip():
        raise ValueError(f"{context}: sample and video identities are required")
    if row.get("split") not in {"train", "val", "test"}:
        raise ValueError(f"{context}: split is invalid")
    if int(row.get("valid_len", 0)) < 1 or int(row.get("requested_budget", 0)) < 1:
        raise ValueError(f"{context}: valid length and requested budget must be positive")
    contract = row.get("decision_contract")
    if not isinstance(contract, Mapping):
        raise ValueError(f"{context}: decision contract is required")
    if set(contract) != _DECISION_KEYS:
        raise ValueError(f"{context}: strict decision-contract fields mismatch")
    forbidden_true = (
        "gt_passed_to_selector",
        "teacher_passed_to_selector",
        "detector_backbone_executed",
    )
    if any(contract.get(key) is not False for key in forbidden_true):
        raise ValueError(f"{context}: deploy-visible score generation has leakage")
    if contract.get("offline_full_window") is not True:
        raise ValueError(f"{context}: task must be offline full-window TAD")
    if contract.get("valid_prefix_only") is not True:
        raise ValueError(f"{context}: allocation input is not restricted to the valid prefix")

    physical_axis = row.get("physical_axis")
    if not isinstance(physical_axis, Mapping) or set(physical_axis) != _PHYSICAL_AXIS_KEYS:
        raise ValueError(f"{context}: strict physical-axis fields mismatch")
    if int(physical_axis.get("total_frames", 0)) < 1:
        raise ValueError(f"{context}: decoded video total_frames must be positive")
    coordinate_audit = row.get("coordinate_audit")
    if not isinstance(coordinate_audit, Mapping) or set(coordinate_audit) != _COORDINATE_AUDIT_KEYS:
        raise ValueError(f"{context}: strict coordinate-audit fields mismatch")
    if coordinate_audit.get("passed") is not True:
        raise ValueError(f"{context}: physical coordinate audit did not pass")
    actual_frames = tuple(float(value) for value in physical_axis.get("source_frames", []))
    expected_frames = tuple(float(value) for value in coordinate_audit.get("expected_source_frames", []))
    if len(actual_frames) != int(row["valid_len"]) or len(expected_frames) != len(actual_frames):
        raise ValueError(f"{context}: physical coordinate lengths mismatch")
    tolerance = float(coordinate_audit.get("tolerance_frames", -1))
    maximum_error = max(
        (abs(actual - expected) for actual, expected in zip(actual_frames, expected_frames)),
        default=0.0,
    )
    if (
        not math.isfinite(tolerance)
        or tolerance < 0
        or not math.isclose(
            maximum_error,
            float(coordinate_audit.get("max_abs_error_frames", math.inf)),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
        or maximum_error > tolerance + 1.0e-9
    ):
        raise ValueError(f"{context}: physical coordinate audit payload is inconsistent")
    total_frames = int(physical_axis.get("total_frames", 0))
    if any(value < 0.0 or value > total_frames - 1 for value in actual_frames):
        raise ValueError(f"{context}: decoded source frame lies outside the video")
    expected_sample_id = (
        f"{row['video_id']}|{int(round(actual_frames[0]))}"
        if actual_frames
        else ""
    )
    if row.get("sample_id") != expected_sample_id:
        raise ValueError(f"{context}: sample identity differs from the physical window")
    timeline_audit = row.get("timeline_audit")
    if not isinstance(timeline_audit, Mapping) or set(timeline_audit) != _TIMELINE_AUDIT_KEYS:
        raise ValueError(f"{context}: strict timeline-audit fields mismatch")
    decoder_fps = float(physical_axis.get("decoder_fps", 0.0))
    annotation_fps = float(physical_axis.get("annotation_fps", 0.0))
    absolute_fps_error = abs(decoder_fps - annotation_fps)
    cumulative_drift_frames = (
        float(max(total_frames - 1, 0)) * abs(annotation_fps / decoder_fps - 1.0)
    )
    tolerance_frames = 1.0
    tolerance_fps = (
        decoder_fps
        if total_frames == 1
        else tolerance_frames * decoder_fps / float(total_frames - 1)
    )
    expected_timeline = {
        "decoder_fps": decoder_fps,
        "annotation_fps": annotation_fps,
        "absolute_fps_error": absolute_fps_error,
        "tolerance_fps": tolerance_fps,
        "cumulative_drift_frames": cumulative_drift_frames,
        "tolerance_frames": tolerance_frames,
        "passed": True,
    }
    if (
        timeline_audit != expected_timeline
        or cumulative_drift_frames > tolerance_frames + 1.0e-9
    ):
        raise ValueError(f"{context}: decoded and annotation timelines are misaligned")
    if row.get("gt_segments_unit") != "dense_ordinal_aligned_to_exported_physical_axis":
        raise ValueError(f"{context}: GT coordinate unit is missing or unsupported")
    canonical_gt = _canonical_gt_segments(
        row.get("gt_segments", []),
        int(row["valid_len"]),
    )

    scores = row.get("scores")
    if not isinstance(scores, Mapping) or set(scores) != set(SCORE_KEYS):
        raise ValueError(f"{context}: strict score fields mismatch")
    for name, values in scores.items():
        if (
            not isinstance(values, Sequence)
            or isinstance(values, (str, bytes))
            or len(values) != int(row["valid_len"])
            or any(not math.isfinite(float(value)) for value in values)
        ):
            raise ValueError(f"{context}: score vector {name} is invalid")

    source = row.get("source")
    if not isinstance(source, Mapping) or set(source) != _SOURCE_KEYS:
        raise ValueError(f"{context}: strict source fields mismatch")
    if re.fullmatch(r"[0-9a-f]{40}", str(source.get("git_commit", ""))) is None:
        raise ValueError(f"{context}: source Git commit is invalid")
    if source.get("git_clean") is not True:
        raise ValueError(f"{context}: source Git tree was not clean")
    for key in ("config_sha256", "checkpoint_sha256"):
        if re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))) is None:
            raise ValueError(f"{context}: source {key} is invalid")
    for key in (
        "annotation_sha256",
        "class_map_sha256",
        "dataset_data_manifest_sha256",
        "dataset_config_sha256",
        "dataset_window_manifest_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", str(source.get(key, ""))) is None:
            raise ValueError(f"{context}: source {key} is invalid")
    if int(source.get("dataset_window_count", 0)) < 1:
        raise ValueError(f"{context}: source dataset window count is invalid")
    if int(source.get("dataset_data_file_count", 0)) < 1:
        raise ValueError(f"{context}: source dataset data file count is invalid")
    if int(source.get("dataset_data_total_bytes", -1)) < 0:
        raise ValueError(f"{context}: source dataset data byte count is invalid")
    if (
        source.get("dataset_data_hash_algorithm")
        != "sha256_full_file_and_symlink_target_v1"
    ):
        raise ValueError(f"{context}: source dataset data hash algorithm is invalid")
    if source.get("dataset_window_deduplication") != "exact_video_start_identity_keep_first":
        raise ValueError(f"{context}: source window deduplication contract is invalid")
    if int(source.get("dataset_duplicate_window_count_removed", -1)) < 0:
        raise ValueError(f"{context}: source duplicate-window count is invalid")
    if source.get("split") != row.get("split"):
        raise ValueError(f"{context}: source split differs from record split")
    if row.get("split") == "train":
        if (
            str(source.get("dataset_subset_name", "")).lower() != "training"
            or source.get("dataset_test_mode") is not False
        ):
            raise ValueError(f"{context}: training split/dataset contract mismatch")
        expected_gt = _reconstruct_training_gt_segments(row, source)
        if not _segments_close(canonical_gt, expected_gt):
            raise ValueError(
                f"{context}: GT segments do not match the bound source annotation"
            )
    else:
        if (
            str(source.get("dataset_subset_name", "")).lower()
            not in {"validation", "testing", "test"}
            or source.get("dataset_test_mode") is not True
        ):
            raise ValueError(f"{context}: validation split/dataset contract mismatch")
        if canonical_gt:
            raise ValueError(f"{context}: validation/test input contains runtime GT")
    required_source_flags = {
        "selector_only_inference": True,
        "detector_backbone_executed": False,
        "uses_gt_for_score_generation": False,
    }
    for key, value in required_source_flags.items():
        if source.get(key) is not value:
            raise ValueError(f"{context}: source contract mismatch: {key}")
    expected_validation_authorized = row.get("split") != "train"
    if source.get("validation_authorized") is not expected_validation_authorized:
        raise ValueError(f"{context}: split/validation authorization mismatch")


def _segments_close(
    actual: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
) -> bool:
    return len(actual) == len(expected) and all(
        math.isclose(float(left[0]), float(right[0]), rel_tol=0.0, abs_tol=1.0e-6)
        and math.isclose(float(left[1]), float(right[1]), rel_tol=0.0, abs_tol=1.0e-6)
        for left, right in zip(actual, expected)
    )


def _reconstruct_training_gt_segments(
    row: Mapping[str, Any],
    source: Mapping[str, Any],
) -> list[list[float]]:
    annotation_path = Path(str(source.get("annotation_path", ""))).resolve()
    class_map_path = Path(str(source.get("class_map_path", ""))).resolve()
    payload, labels = _load_bound_annotation(
        str(annotation_path),
        str(source.get("annotation_sha256", "")),
        str(class_map_path),
        str(source.get("class_map_sha256", "")),
    )
    database = payload.get("database") if isinstance(payload, Mapping) else None
    video_id = str(row.get("video_id", ""))
    if not isinstance(database, Mapping) or video_id not in database:
        raise ValueError(f"bound annotation lacks video {video_id}")
    video = database[video_id]
    if not isinstance(video, Mapping):
        raise ValueError("bound video annotation must be an object")
    if str(video.get("subset", "")).lower() != "training":
        raise ValueError("training input is not bound to a training annotation")
    duration = float(video.get("duration", 0.0))
    video_frames = int(video.get("frame", 0))
    if not math.isfinite(duration) or duration <= 0.0 or video_frames < 1:
        raise ValueError("bound video annotation has invalid duration/frame count")
    physical_axis = row["physical_axis"]
    if video_frames != int(physical_axis["total_frames"]):
        raise ValueError("bound annotation frame count differs from decoded metadata")
    annotation_fps = video_frames / duration
    if not math.isclose(
        annotation_fps,
        float(physical_axis["annotation_fps"]),
        rel_tol=0.0,
        abs_tol=1.0e-9,
    ):
        raise ValueError("bound annotation FPS differs from exported metadata")
    source_frames = [float(value) for value in physical_axis["source_frames"]]
    window_start = source_frames[0]
    window_end = source_frames[-1]
    if len(source_frames) > 1:
        snippet_stride = source_frames[1] - source_frames[0]
    else:
        snippet_stride = float(
            int(source.get("dataset_feature_stride", -1))
            * int(source.get("dataset_sample_stride", 1))
        )
    expected_stride = int(source.get("dataset_feature_stride", -1)) * int(
        source.get("dataset_sample_stride", 1)
    )
    if (
        expected_stride <= 0
        or not math.isclose(
            snippet_stride,
            float(expected_stride),
            rel_tol=0.0,
            abs_tol=1.0e-9,
        )
    ):
        raise ValueError("bound dataset stride differs from the exported axis")
    if int(source.get("dataset_offset_frames", 0)) != 0:
        raise ValueError("allocation ceiling currently requires zero frame offset")
    ioa_thresh = float(source.get("dataset_ioa_thresh", -1.0))
    if not math.isfinite(ioa_thresh) or ioa_thresh <= 0.0:
        raise ValueError("training GT reconstruction requires positive ioa_thresh")
    if bool(source.get("dataset_filter_gt", True)):
        raise ValueError("allocation ceiling currently requires filter_gt=False")

    deduplicated: list[tuple[int, int, int]] = []
    for annotation in video.get("annotations", []):
        if not isinstance(annotation, Mapping):
            raise ValueError("bound action annotation must be an object")
        label = str(annotation.get("label", ""))
        if label == "Ambiguous":
            continue
        if label not in labels:
            raise ValueError(f"bound annotation label is absent from class map: {label}")
        segment = annotation.get("segment")
        if (
            not isinstance(segment, Sequence)
            or isinstance(segment, (str, bytes))
            or len(segment) != 2
        ):
            raise ValueError("bound action segment must have two endpoints")
        gt_start = int(float(segment[0]) / duration * video_frames)
        gt_end = int(float(segment[1]) / duration * video_frames)
        identity = (gt_start, gt_end, labels.index(label))
        if identity not in deduplicated:
            deduplicated.append(identity)

    reconstructed: list[list[float]] = []
    for gt_start, gt_end, _label in deduplicated:
        if not (gt_start < window_end and gt_end > window_start):
            continue
        original_length = max(float(gt_end - gt_start), 1.0e-6)
        truncated_start = max(float(gt_start), window_start)
        truncated_end = min(float(gt_end), window_end)
        completeness = (truncated_end - truncated_start) / original_length
        if completeness <= ioa_thresh:
            continue
        reconstructed.append(
            [
                (truncated_start - window_start) / snippet_stride,
                (truncated_end - window_start) / snippet_stride,
            ]
        )
    return reconstructed


@lru_cache(maxsize=8)
def _load_bound_annotation(
    annotation_path_text: str,
    annotation_sha256: str,
    class_map_path_text: str,
    class_map_sha256: str,
) -> tuple[Mapping[str, Any], tuple[str, ...]]:
    annotation_path = Path(annotation_path_text)
    class_map_path = Path(class_map_path_text)
    if not annotation_path.is_file() or sha256(annotation_path) != annotation_sha256:
        raise ValueError("bound annotation file is missing or changed")
    if not class_map_path.is_file() or sha256(class_map_path) != class_map_sha256:
        raise ValueError("bound class-map file is missing or changed")
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("bound annotation root must be an object")
    labels = tuple(
        line.rstrip("\n")
        for line in class_map_path.read_text(encoding="utf-8").splitlines()
    )
    if not labels:
        raise ValueError("bound class map is empty")
    return payload, labels


def axis_from_record(row: Mapping[str, Any]) -> PhysicalAxis:
    raw = row.get("physical_axis")
    if not isinstance(raw, Mapping):
        raise ValueError("physical_axis is required")
    axis = PhysicalAxis.from_source_frames(
        raw.get("source_frames", []),
        decoder_fps=float(raw.get("decoder_fps", 0.0)),
        annotation_fps=float(raw.get("annotation_fps", 0.0)),
    )
    dense = tuple(int(value) for value in raw.get("dense_ordinals", []))
    if dense != axis.dense_ordinals:
        raise ValueError("dense_ordinals must be a contiguous zero-based valid prefix")
    seconds = tuple(float(value) for value in raw.get("seconds", []))
    if len(seconds) != axis.valid_len:
        raise ValueError("seconds coordinate length mismatch")
    if any(abs(left - right) > 1.0e-8 for left, right in zip(seconds, axis.seconds)):
        raise ValueError("seconds coordinates must equal source_frames / decoder_fps")
    if int(row.get("valid_len", -1)) != axis.valid_len:
        raise ValueError("valid_len does not match physical axis")
    return axis


def allocation_metrics(
    positions: Sequence[int],
    gt_segments: Sequence[Sequence[float | int]],
    *,
    valid_len: int,
    radii: Sequence[int],
    short_action_max_length: float,
) -> dict[str, Any]:
    selected = tuple(int(value) for value in positions)
    segments = _canonical_gt_segments(gt_segments, valid_len)
    endpoints = tuple(value for segment in segments for value in segment)
    distances = [
        min(abs(float(position) - endpoint) for position in selected)
        for endpoint in endpoints
    ]
    metrics: dict[str, Any] = {
        "endpoint_count": len(endpoints),
        "mean_endpoint_distance": _mean(distances),
        "max_endpoint_distance": max(distances) if distances else None,
        "uniform_overlap_placeholder": None,
    }
    for radius in radii:
        hit = [distance <= int(radius) + 1.0e-9 for distance in distances]
        both = [
            hit[2 * segment_index] and hit[2 * segment_index + 1]
            for segment_index in range(len(segments))
        ]
        metrics[f"endpoint_recall_r{radius}"] = _mean(float(value) for value in hit)
        metrics[f"both_boundary_recall_r{radius}"] = _mean(float(value) for value in both)

    short_segments = [
        segment
        for segment in segments
        if segment[1] - segment[0] <= float(short_action_max_length) + 1.0e-9
    ]
    short_supported = [
        any(start - 1.0e-9 <= position <= end + 1.0e-9 for position in selected)
        for start, end in short_segments
    ]
    metrics["short_action_count"] = len(short_segments)
    metrics["short_action_support"] = _mean(float(value) for value in short_supported)
    background = [
        not any(start - 1.0e-9 <= position <= end + 1.0e-9 for start, end in segments)
        for position in selected
    ]
    metrics["selected_background_fraction"] = _mean(float(value) for value in background)
    return metrics


def diagnose_record(
    row: Mapping[str, Any],
    *,
    score_key: str,
    cap_policy: str,
    cap_value: float | None,
    gt_families: str,
    objective_spec: GroundTruthObjectiveSpec,
    quantization_scale: int,
    gt_time_limit_seconds: float | None,
    compute_upper_envelopes: bool,
) -> dict[str, Any]:
    validate_input_record(row, context=str(row.get("sample_id")))
    axis = axis_from_record(row)
    requested_budget = int(row["requested_budget"])
    raw_scores = row.get("scores")
    if not isinstance(raw_scores, Mapping) or score_key not in raw_scores:
        raise ValueError(f"{row.get('sample_id')}: score key {score_key} is unavailable")
    scores = [float(value) for value in raw_scores[score_key]]
    if len(scores) != axis.valid_len or any(not math.isfinite(value) for value in scores):
        raise ValueError(f"{row.get('sample_id')}: invalid deploy-visible score vector")
    cap = resolve_physical_cap(
        axis,
        requested_budget=requested_budget,
        policy=cap_policy,
        value=cap_value,
    )
    gt_segments = row.get("gt_segments", [])
    if not isinstance(gt_segments, Sequence):
        raise ValueError("gt_segments must be a sequence")

    family_a = select_family_a(axis, requested_budget=requested_budget, cap=cap)
    family_b = select_family_b(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
    )
    family_c = select_family_c(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
    )
    family_d, d_solve = select_family_d(
        axis,
        scores,
        requested_budget=requested_budget,
        cap=cap,
        quantization_scale=quantization_scale,
    )
    families: list[dict[str, Any]] = []
    for selection in (family_a, family_b, family_c, family_d):
        payload = selection.to_dict()
        payload["family_key"] = (
            "D_deploy_score"
            if selection.family == "D_global_exact_k_physical_gap"
            else selection.family
        )
        payload["allocation_metrics"] = allocation_metrics(
            selection.positions,
            gt_segments,
            valid_len=axis.valid_len,
            radii=objective_spec.boundary_radii,
            short_action_max_length=objective_spec.short_action_max_length,
        )
        payload["allocation_metrics"]["uniform_overlap"] = len(
            set(selection.positions) & set(family_a.positions)
        ) / len(family_a.positions)
        if selection.family == "D_global_exact_k_physical_gap":
            payload["additive_solver"] = d_solve.to_dict()
        families.append(payload)

    if gt_families in {"d", "both"}:
        d_gt = solve_ground_truth_lexicographic(
            axis,
            gt_segments,
            requested_budget=requested_budget,
            cap=cap,
            objective_spec=objective_spec,
            compute_upper_envelopes=compute_upper_envelopes,
            time_limit_seconds=gt_time_limit_seconds,
        )
        validate_physical_selection(
            axis,
            d_gt.positions,
            requested_budget=requested_budget,
            cap=cap,
        )
        families.append(
            _gt_family_payload(
                family_key="D_privileged_gt_ceiling",
                axis=axis,
                positions=d_gt.positions,
                uniform_positions=family_a.positions,
                gt_segments=gt_segments,
                objective_spec=objective_spec,
                solver=d_gt.to_dict(),
                cap_compliant=True,
            )
        )

    if gt_families in {"e", "both"}:
        e_gt = solve_ground_truth_lexicographic(
            axis,
            gt_segments,
            requested_budget=requested_budget,
            cap=None,
            objective_spec=objective_spec,
            compute_upper_envelopes=compute_upper_envelopes,
            time_limit_seconds=gt_time_limit_seconds,
        )
        report = physical_gap_report(axis, e_gt.positions)
        cap_compliant = True
        if cap.max_source_frame_interval is not None:
            cap_compliant = (
                report.source_frame_max_interval <= cap.max_source_frame_interval + 1.0e-9
            )
        if cap.max_seconds_interval is not None:
            cap_compliant = cap_compliant and (
                report.seconds_max_interval <= cap.max_seconds_interval + 1.0e-9
            )
        families.append(
            _gt_family_payload(
                family_key="E_privileged_unrestricted_gt",
                axis=axis,
                positions=e_gt.positions,
                uniform_positions=family_a.positions,
                gt_segments=gt_segments,
                objective_spec=objective_spec,
                solver=e_gt.to_dict(),
                cap_compliant=cap_compliant,
            )
        )

    output = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "sample_id": row["sample_id"],
        "video_id": row["video_id"],
        "split": row["split"],
        "valid_len": axis.valid_len,
        "requested_budget": requested_budget,
        "score_key": score_key,
        "cap": cap.to_dict(),
        "coarse_signal_metrics": coarse_signal_metrics(
            raw_scores,
            gt_segments,
            valid_len=axis.valid_len,
            transition_radius=max(1, min(objective_spec.boundary_radii[-1], 4)),
        ),
        "families": families,
        "input_record_sha256": row["record_sha256"],
        "contract": {
            "offline_full_window": True,
            "deploy_score_uses_gt": False,
            "gt_families_privileged_only": True,
            "detector_mAP_oracle_claim": False,
            "exact_language_requires_optimal": True,
        },
    }
    output["record_sha256"] = canonical_sha256(output)
    return output


def run_diagnostic(
    *,
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    summary_json: str | Path,
    score_key: str,
    cap_policy: str,
    cap_value: float | None,
    gt_families: str,
    objective_spec: GroundTruthObjectiveSpec,
    quantization_scale: int,
    gt_time_limit_seconds: float | None,
    compute_upper_envelopes: bool,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).resolve()
    output_path = Path(output_jsonl).resolve()
    summary_path = Path(summary_json).resolve()
    if output_path.exists() or summary_path.exists():
        raise FileExistsError("allocation ceiling diagnostic never overwrites existing artifacts")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = read_input_records(input_path)

    outputs: list[dict[str, Any]] = []
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary.exists():
        raise FileExistsError(f"stale partial artifact exists: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for index, row in enumerate(inputs):
                result = diagnose_record(
                    row,
                    score_key=score_key,
                    cap_policy=cap_policy,
                    cap_value=cap_value,
                    gt_families=gt_families,
                    objective_spec=objective_spec,
                    quantization_scale=quantization_scale,
                    gt_time_limit_seconds=gt_time_limit_seconds,
                    compute_upper_envelopes=compute_upper_envelopes,
                )
                outputs.append(result)
                handle.write(
                    json.dumps(result, sort_keys=True, ensure_ascii=True, allow_nan=False)
                    + "\n"
                )
                if index % 20 == 0:
                    print(
                        json.dumps({"sample": index, "sample_id": result["sample_id"]}, sort_keys=True),
                        flush=True,
                    )
        temporary.replace(output_path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise

    summary = summarize_outputs(
        outputs,
        input_path=input_path,
        output_path=output_path,
        score_key=score_key,
        cap_policy=cap_policy,
        cap_value=cap_value,
        gt_families=gt_families,
        objective_spec=objective_spec,
        quantization_scale=quantization_scale,
        gt_time_limit_seconds=gt_time_limit_seconds,
        compute_upper_envelopes=compute_upper_envelopes,
    )
    write_json_exclusive(summary_path, summary)
    return summary


def summarize_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    input_path: Path,
    output_path: Path,
    score_key: str,
    cap_policy: str,
    cap_value: float | None,
    gt_families: str,
    objective_spec: GroundTruthObjectiveSpec,
    quantization_scale: int,
    gt_time_limit_seconds: float | None,
    compute_upper_envelopes: bool,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in outputs:
        for family in row["families"]:
            grouped[str(family["family_key"])].append(family)
    family_summary: dict[str, Any] = {}
    for family_key, rows in sorted(grouped.items()):
        metric_keys = sorted(
            {
                key
                for row in rows
                for key, value in row["allocation_metrics"].items()
                if isinstance(value, (int, float)) and value is not None
            }
        )
        family_summary[family_key] = {
            "sample_count": len(rows),
            "exact_fraction": _mean(float(row["exact"]) for row in rows),
            "cap_compliance_fraction": _mean(
                float(row["physical_cap_compliant"]) for row in rows
            ),
            "mean_dense_max_unselected_hole": _mean(
                row["gap_report"]["dense_max_unselected_hole"] for row in rows
            ),
            "mean_source_frame_max_interval": _mean(
                row["gap_report"]["source_frame_max_interval"] for row in rows
            ),
            "mean_seconds_max_interval": _mean(
                row["gap_report"]["seconds_max_interval"] for row in rows
            ),
            "mean_metrics": {
                key: _mean(row["allocation_metrics"].get(key) for row in rows)
                for key in metric_keys
            },
        }
    coarse_keys = sorted(
        {
            key
            for row in outputs
            for key, value in row["coarse_signal_metrics"].items()
            if isinstance(value, (int, float)) and value is not None
        }
    )
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "diagnostic_role": "allocation_family_geometry_and_recoverability_not_detector_oracle",
        "sample_count": len(outputs),
        "input_jsonl": str(input_path),
        "input_jsonl_sha256": sha256(input_path),
        "output_jsonl": str(output_path),
        "output_jsonl_sha256": sha256(output_path),
        "score_key": score_key,
        "cap_policy": cap_policy,
        "cap_value": cap_value,
        "gt_families": gt_families,
        "objective_spec": {
            "boundary_radii": objective_spec.boundary_radii,
            "short_action_max_length": objective_spec.short_action_max_length,
            "distance_scale": objective_spec.distance_scale,
            "lex_block_size": objective_spec.lex_block_size,
        },
        "quantization_scale": int(quantization_scale),
        "gt_solver_options": {
            "backend": "scipy.optimize.milp_highs",
            "presolve": True,
            "mip_rel_gap": 0.0,
            "time_limit_seconds": gt_time_limit_seconds,
            "compute_upper_envelopes": bool(compute_upper_envelopes),
        },
        "families": family_summary,
        "mean_coarse_signal_metrics": {
            key: _mean(row["coarse_signal_metrics"].get(key) for row in outputs)
            for key in coarse_keys
        },
        "contract": {
            "deploy_score_uses_gt": False,
            "gt_families_deployable": False,
            "exact_status_required": "OPTIMAL",
            "detector_mAP_evaluated": False,
        },
    }


def _gt_family_payload(
    *,
    family_key: str,
    axis: PhysicalAxis,
    positions: Sequence[int],
    uniform_positions: Sequence[int],
    gt_segments: Sequence[Sequence[float | int]],
    objective_spec: GroundTruthObjectiveSpec,
    solver: Mapping[str, Any],
    cap_compliant: bool,
) -> dict[str, Any]:
    metrics = allocation_metrics(
        positions,
        gt_segments,
        valid_len=axis.valid_len,
        radii=objective_spec.boundary_radii,
        short_action_max_length=objective_spec.short_action_max_length,
    )
    metrics["uniform_overlap"] = len(set(positions) & set(uniform_positions)) / len(
        uniform_positions
    )
    return {
        "family": family_key,
        "family_key": family_key,
        "positions": tuple(int(value) for value in positions),
        "budget": len(positions),
        "score_sum": None,
        "exact": solver.get("solver_status") == "OPTIMAL" and solver.get("exact") is True,
        "deployable": False,
        "privileged": True,
        "solver_status": solver.get("solver_status"),
        "physical_cap_compliant": bool(cap_compliant),
        "gap_report": physical_gap_report(axis, positions).to_dict(),
        "scaffold_positions": (),
        "residual_positions": (),
        "allocation_metrics": metrics,
        "gt_solver": dict(solver),
    }


def coarse_signal_metrics(
    scores: Mapping[str, Sequence[float | int]],
    gt_segments: Sequence[Sequence[float | int]],
    *,
    valid_len: int,
    transition_radius: int,
) -> dict[str, Any]:
    segments = _canonical_gt_segments(gt_segments, valid_len)
    action_target = [
        int(any(start - 1.0e-9 <= position <= end + 1.0e-9 for start, end in segments))
        for position in range(valid_len)
    ]
    endpoints = tuple(value for segment in segments for value in segment)
    transition_target = [
        int(any(abs(float(position) - endpoint) <= transition_radius for endpoint in endpoints))
        for position in range(valid_len)
    ]
    p_action = [float(value) for value in scores["p_action"]]
    transition_policy = [float(value) for value in scores["transition_policy_scores"]]
    raw_transition = [float(value) for value in scores["raw_transition_scores"]]
    abs_delta = [float(value) for value in scores["abs_delta_p_action"]]
    metrics = {
        "action_positive_fraction": _mean(action_target),
        "action_brier": _mean(
            (prediction - target) ** 2
            for prediction, target in zip(p_action, action_target)
        ),
        "transition_positive_fraction": _mean(transition_target),
        "transition_target_radius": int(transition_radius),
    }
    for prefix, values, target in (
        ("action_p", p_action, action_target),
        ("transition_policy", transition_policy, transition_target),
        ("raw_transition", raw_transition, transition_target),
        ("abs_delta", abs_delta, transition_target),
    ):
        binary = _binary_ranking_metrics(values, target)
        metrics[f"{prefix}_roc_auc"] = binary["roc_auc"]
        metrics[f"{prefix}_average_precision"] = binary["average_precision"]
        metrics[f"{prefix}_best_f1"] = binary["best_f1"]
    return metrics


def _binary_ranking_metrics(
    scores: Sequence[float | int],
    targets: Sequence[int],
) -> dict[str, float | None]:
    values = [float(value) for value in scores]
    labels = [int(value) for value in targets]
    if len(values) != len(labels) or not values:
        raise ValueError("binary metric scores and targets must be non-empty and aligned")
    if any(not math.isfinite(value) for value in values) or any(value not in (0, 1) for value in labels):
        raise ValueError("binary metric inputs are invalid")
    positives = sum(labels)
    negatives = len(labels) - positives
    roc_auc = None
    if positives and negatives:
        ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
        rank_sum = 0.0
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
                end += 1
            average_rank = (cursor + 1 + end) / 2.0
            rank_sum += average_rank * sum(labels[ordered[index]] for index in range(cursor, end))
            cursor = end
        roc_auc = (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)

    average_precision = None
    best_f1 = None
    if positives:
        ordered = sorted(range(len(values)), key=lambda index: (-values[index], index))
        true_positive = 0
        previous_recall = 0.0
        area = 0.0
        best = 0.0
        cursor = 0
        while cursor < len(ordered):
            end = cursor + 1
            while end < len(ordered) and values[ordered[end]] == values[ordered[cursor]]:
                end += 1
            true_positive += sum(labels[ordered[index]] for index in range(cursor, end))
            predicted_positive = end
            precision = true_positive / predicted_positive
            recall = true_positive / positives
            area += (recall - previous_recall) * precision
            denominator = precision + recall
            if denominator > 0:
                best = max(best, 2.0 * precision * recall / denominator)
            previous_recall = recall
            cursor = end
        average_precision = area
        best_f1 = best
    return {
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "best_f1": best_f1,
    }


def _canonical_gt_segments(
    gt_segments: Sequence[Sequence[float | int]],
    valid_len: int,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    upper = float(valid_len - 1)
    for index, segment in enumerate(gt_segments):
        if len(segment) != 2:
            raise AllocationContractError("each GT segment must contain two endpoints")
        start = float(segment[0])
        end = float(segment[1])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0.0
            or end > upper
            or end < start
        ):
            raise AllocationContractError(
                f"GT segment {index} is outside dense valid prefix [0,{upper}]"
            )
        result.append((start, end))
    return tuple(result)


def _mean(values) -> float | None:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    return None if not finite else sum(finite) / len(finite)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose exact DUCA allocation-family ceilings and recoverability."
    )
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument(
        "--score-key",
        default="transition_policy_scores",
        choices=[
            "p_action",
            "actionness_logits",
            "transition_policy_scores",
            "raw_transition_scores",
            "abs_delta_p_action",
            "uncertainty",
        ],
    )
    parser.add_argument(
        "--cap-policy",
        default="uniform_reference",
        choices=["uniform_reference", "explicit_frames", "explicit_seconds"],
    )
    parser.add_argument("--cap-value", type=float)
    parser.add_argument("--gt-families", choices=["none", "d", "e", "both"], default="both")
    parser.add_argument("--boundary-radii", type=int, nargs="+", default=[0, 1, 2, 4])
    parser.add_argument("--short-action-max-length", type=float, default=16.0)
    parser.add_argument("--distance-scale", type=int, default=1000)
    parser.add_argument("--lex-block-size", type=int, default=20)
    parser.add_argument("--quantization-scale", type=int, default=1_000_000)
    parser.add_argument("--gt-time-limit-seconds", type=float)
    parser.add_argument("--compute-upper-envelopes", action="store_true")
    args = parser.parse_args(argv)
    objective_spec = GroundTruthObjectiveSpec(
        boundary_radii=tuple(args.boundary_radii),
        short_action_max_length=args.short_action_max_length,
        distance_scale=args.distance_scale,
        lex_block_size=args.lex_block_size,
    )
    summary = run_diagnostic(
        input_jsonl=args.input_jsonl,
        output_jsonl=args.output_jsonl,
        summary_json=args.summary_json,
        score_key=args.score_key,
        cap_policy=args.cap_policy,
        cap_value=args.cap_value,
        gt_families=args.gt_families,
        objective_spec=objective_spec,
        quantization_scale=args.quantization_scale,
        gt_time_limit_seconds=args.gt_time_limit_seconds,
        compute_upper_envelopes=args.compute_upper_envelopes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
