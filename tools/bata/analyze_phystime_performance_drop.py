from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _as_float_array(value, *, columns=None):
    array = np.asarray(value, dtype=np.float64)
    if columns is not None:
        array = array.reshape(-1, columns)
    return array


def build_query_embedding_features(
    *, centers_sec, widths_sec, duration_sec, num_fourier_bands=4
):
    centers = _as_float_array(centers_sec).reshape(-1)
    widths = _as_float_array(widths_sec).reshape(-1)
    durations = _as_float_array(duration_sec).reshape(-1)
    if durations.size == 1 and centers.size != 1:
        durations = np.full(centers.size, durations.item(), dtype=np.float64)
    if centers.size != widths.size or centers.size != durations.size:
        raise ValueError("query center, width, and duration counts must match")
    durations = np.maximum(durations, np.finfo(np.float64).eps)
    normalized_center = centers / durations
    normalized_width = widths / durations
    frequencies = 2.0 ** np.arange(int(num_fourier_bands), dtype=np.float64)
    phase = normalized_center[:, None] * frequencies[None, :] * (2.0 * math.pi)
    return np.concatenate(
        (
            centers[:, None],
            normalized_center[:, None],
            np.log1p(durations)[:, None],
            widths[:, None],
            normalized_width[:, None],
            np.sin(phase),
            np.cos(phase),
        ),
        axis=-1,
    )


def summarize_linear_input_scale(*, features, weight, bias, feature_names):
    features = _as_float_array(features)
    weight = _as_float_array(weight)
    bias = _as_float_array(bias).reshape(-1)
    feature_names = tuple(str(name) for name in feature_names)
    if features.ndim != 2 or weight.ndim != 2:
        raise ValueError("linear input audit expects two-dimensional features and weights")
    if features.shape[1] != weight.shape[1] or len(feature_names) != features.shape[1]:
        raise ValueError("feature and first-layer input dimensions must match")
    if bias.size != weight.shape[0]:
        raise ValueError("first-layer bias dimension must match its output dimension")

    column_norms = np.linalg.norm(weight, axis=0)
    mean_abs_values = np.mean(np.abs(features), axis=0)
    contribution = mean_abs_values * column_norms
    contribution_total = float(contribution.sum())
    preactivation = features @ weight.T + bias[None, :]
    dominant_index = int(np.argmax(contribution))
    return {
        "sample_count": int(features.shape[0]),
        "dominant_feature": feature_names[dominant_index],
        "feature_mean_abs": {
            name: float(value) for name, value in zip(feature_names, mean_abs_values)
        },
        "weight_column_l2": {
            name: float(value) for name, value in zip(feature_names, column_norms)
        },
        "mean_abs_contribution": {
            name: float(value) for name, value in zip(feature_names, contribution)
        },
        "contribution_share": {
            name: float(value / contribution_total) if contribution_total else 0.0
            for name, value in zip(feature_names, contribution)
        },
        "preactivation_abs": _distribution(np.abs(preactivation).reshape(-1)),
        "preactivation_abs_gt_10_fraction": float((np.abs(preactivation) > 10.0).mean()),
        "preactivation_abs_gt_50_fraction": float((np.abs(preactivation) > 50.0).mean()),
    }


def summarize_attention_rows(*, weights, mass, logits=None):
    weights = _as_float_array(weights)
    mass = _as_float_array(mass)
    if weights.ndim != 2 or mass.shape != weights.shape:
        raise ValueError("attention weights and mass must have matching [Q,K] shapes")
    if logits is not None:
        logits = _as_float_array(logits)
        if logits.shape != weights.shape:
            raise ValueError("attention logits must match the weight shape")
    covered = mass > 0
    valid_rows = covered.any(axis=1)
    row_weights = weights[valid_rows]
    row_covered = covered[valid_rows]
    if row_weights.shape[0] == 0:
        raise ValueError("attention audit requires at least one covered query")

    effective_counts = []
    normalized_entropies = []
    max_weights = []
    covered_counts = []
    logit_spans = []
    for row_index, (weight_row, covered_row) in enumerate(zip(row_weights, row_covered)):
        values = weight_row[covered_row]
        values = values / max(float(values.sum()), np.finfo(np.float64).eps)
        count = int(values.size)
        covered_counts.append(count)
        max_weights.append(float(values.max()))
        effective_counts.append(float(1.0 / np.maximum(np.square(values).sum(), 1.0e-12)))
        entropy = float(-(values * np.log(np.maximum(values, 1.0e-12))).sum())
        normalized_entropies.append(entropy / math.log(count) if count > 1 else 1.0)
        if logits is not None:
            logit_values = logits[valid_rows][row_index][covered_row]
            logit_spans.append(float(logit_values.max() - logit_values.min()))
    report = {
        "query_count": int(row_weights.shape[0]),
        "covered_observation_count": _distribution(covered_counts),
        "effective_observation_count": _distribution(effective_counts),
        "effective_observation_fraction": _distribution(
            np.asarray(effective_counts) / np.asarray(covered_counts)
        ),
        "normalized_entropy": _distribution(normalized_entropies),
        "max_attention_weight": _distribution(max_weights),
    }
    if logits is not None:
        report["covered_logit_span"] = _distribution(logit_spans)
    return report


def build_physical_query_levels(
    *,
    duration_sec,
    domain_start_sec,
    domain_end_sec,
    support_intervals_sec,
    base_spacing_sec,
    num_levels,
):
    duration_sec = float(duration_sec)
    domain_start_sec = float(domain_start_sec)
    domain_end_sec = float(domain_end_sec)
    base_spacing_sec = float(base_spacing_sec)
    supports = _as_float_array(support_intervals_sec, columns=2)
    if not (0 <= domain_start_sec < domain_end_sec <= duration_sec):
        raise ValueError("physical query domain must lie inside the video duration")
    if base_spacing_sec <= 0 or int(num_levels) <= 0:
        raise ValueError("physical query spacing and level count must be positive")

    levels = []
    for level_index in range(int(num_levels)):
        spacing = base_spacing_sec * (2**level_index)
        start_index = math.floor(domain_start_sec / spacing)
        end_index = math.ceil(domain_end_sec / spacing)
        cell_indices = np.arange(start_index, end_index, dtype=np.int64)
        left = np.maximum(cell_indices.astype(np.float64) * spacing, 0.0)
        right = np.minimum(left + spacing, duration_sec)
        valid = right > left
        intervals = np.stack((left[valid], right[valid]), axis=-1)
        centers = intervals.mean(axis=-1)
        widths = intervals[:, 1] - intervals[:, 0]

        if supports.size:
            overlap_left = np.maximum(intervals[:, None, 0], supports[None, :, 0])
            overlap_right = np.minimum(intervals[:, None, 1], supports[None, :, 1])
            overlap = np.maximum(overlap_right - overlap_left, 0.0)
            coverage = overlap.sum(axis=1)
            observation_count = (overlap > 0).sum(axis=1)
        else:
            overlap = np.zeros((intervals.shape[0], 0), dtype=np.float64)
            coverage = np.zeros(intervals.shape[0], dtype=np.float64)
            observation_count = np.zeros(intervals.shape[0], dtype=np.int64)
        covered = coverage > 0
        total_width = float(widths.sum())
        levels.append(
            {
                "level": level_index,
                "spacing_sec": spacing,
                "intervals_sec": intervals,
                "centers": centers,
                "widths": widths,
                "coverage_sec": coverage,
                "covered_mask": covered,
                "observation_count": observation_count,
                "query_count": int(intervals.shape[0]),
                "covered_query_count": int(covered.sum()),
                "coverage_fraction": float(coverage.sum() / total_width) if total_width > 0 else 0.0,
                "overlap_mass": overlap,
            }
        )
    return levels


def build_actionformer_levels(
    *,
    sequence_length,
    strides=(1, 2, 4, 8, 16, 32),
    regression_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
):
    sequence_length = int(sequence_length)
    if sequence_length <= 0 or len(strides) != len(regression_ranges):
        raise ValueError("invalid ActionFormer point specification")
    levels = []
    for level_index, (stride, regression_range) in enumerate(zip(strides, regression_ranges)):
        stride = float(stride)
        count = int(math.ceil(sequence_length / stride))
        centers = np.arange(count, dtype=np.float64) * stride
        levels.append(
            {
                "level": level_index,
                "centers": centers,
                "lower_ranges": np.full(count, float(regression_range[0])),
                "upper_ranges": np.full(count, float(regression_range[1])),
                "strides": np.full(count, stride),
                "valid_mask": np.ones(count, dtype=bool),
            }
        )
    return levels


def _interp_selected_to_physical(coords, selected_positions, dense_valid_len):
    selected_positions = _as_float_array(selected_positions).reshape(-1)
    if selected_positions.size == 0:
        raise ValueError("physical-grid mapping requires selected positions")
    xp = np.concatenate((np.arange(selected_positions.size, dtype=np.float64), [float(selected_positions.size)]))
    fp = np.concatenate((selected_positions, [float(dense_valid_len)]))
    return np.interp(np.clip(coords, 0.0, float(selected_positions.size)), xp, fp)


def map_actionformer_levels_to_physical_grid(levels, *, selected_positions, dense_valid_len):
    selected_positions = _as_float_array(selected_positions).reshape(-1)
    mapped = []
    for level in levels:
        centers = _as_float_array(level["centers"]).reshape(-1)
        nominal_stride = _as_float_array(level["strides"]).reshape(-1)
        physical_centers = _interp_selected_to_physical(centers, selected_positions, dense_valid_len)
        previous = _interp_selected_to_physical(
            np.maximum(centers - nominal_stride, 0.0), selected_positions, dense_valid_len
        )
        following = _interp_selected_to_physical(
            centers + nominal_stride, selected_positions, dense_valid_len
        )
        physical_stride = np.maximum((following - previous) * 0.5, 1.0e-6)
        range_scale = physical_stride / np.maximum(nominal_stride, 1.0e-6)
        mapped.append(
            {
                **level,
                "centers": physical_centers,
                "lower_ranges": _as_float_array(level["lower_ranges"]) * range_scale,
                "upper_ranges": _as_float_array(level["upper_ranges"]) * range_scale,
                "strides": physical_stride,
            }
        )
    return mapped


def warp_segments_to_selected_axis(gt_segments, selected_positions, *, dense_valid_len):
    segments = _as_float_array(gt_segments, columns=2)
    selected_positions = _as_float_array(selected_positions).reshape(-1)
    if selected_positions.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    xp = np.concatenate((selected_positions, [float(dense_valid_len)]))
    fp = np.concatenate((np.arange(selected_positions.size, dtype=np.float64), [float(selected_positions.size)]))
    clipped = np.clip(segments, 0.0, float(dense_valid_len))
    return np.stack(
        (
            np.interp(clipped[:, 0], xp, fp),
            np.interp(clipped[:, 1], xp, fp),
        ),
        axis=-1,
    )


def concatenate_point_levels(levels, *, valid_key="valid_mask"):
    points = []
    masks = []
    for level in levels:
        points.append(
            np.stack(
                (
                    level["centers"],
                    level["lower_ranges"],
                    level["upper_ranges"],
                    level["strides"],
                ),
                axis=-1,
            )
        )
        masks.append(np.asarray(level[valid_key], dtype=bool))
    return np.concatenate(points, axis=0), np.concatenate(masks, axis=0)


def summarize_target_assignment(*, points, valid_mask, gt_segments, center_sample_radius):
    points = _as_float_array(points, columns=4)
    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    segments = _as_float_array(gt_segments, columns=2)
    if points.shape[0] != valid_mask.size:
        raise ValueError("point and validity counts must match")
    if segments.size == 0:
        return {
            "gt_count": 0,
            "positive_location_count": 0,
            "gt_without_eligible_location_count": 0,
            "eligible_location_count_per_gt": [],
            "assigned_location_count_per_gt": [],
            "multi_min_gt_location_count": 0,
            "max_min_gt_multiplicity": 0,
        }

    centers = points[:, 0, None]
    left = centers - segments[None, :, 0]
    right = segments[None, :, 1] - centers
    distances = np.stack((left, right), axis=-1)
    segment_centers = 0.5 * (segments[:, 0] + segments[:, 1])
    radius = points[:, 3, None] * float(center_sample_radius)
    center_left = centers - np.maximum(segment_centers[None, :] - radius, segments[None, :, 0])
    center_right = np.minimum(segment_centers[None, :] + radius, segments[None, :, 1]) - centers
    inside = np.minimum(center_left, center_right) > 0
    max_distance = distances.max(axis=-1)
    in_range = (max_distance >= points[:, 1, None]) & (max_distance <= points[:, 2, None])
    eligible = inside & in_range & valid_mask[:, None]

    lengths = np.broadcast_to(segments[:, 1] - segments[:, 0], eligible.shape).copy()
    lengths[~eligible] = np.inf
    min_lengths = lengths.min(axis=1)
    chosen = lengths.argmin(axis=1)
    positive = np.isfinite(min_lengths) & valid_mask
    min_mask = (lengths <= (min_lengths[:, None] + 1.0e-3)) & np.isfinite(lengths)
    min_multiplicity = min_mask.sum(axis=1)
    assigned = np.zeros(segments.shape[0], dtype=np.int64)
    if positive.any():
        assigned += np.bincount(chosen[positive], minlength=segments.shape[0])
    eligible_per_gt = eligible.sum(axis=0).astype(np.int64)
    return {
        "gt_count": int(segments.shape[0]),
        "positive_location_count": int(positive.sum()),
        "gt_without_eligible_location_count": int((eligible_per_gt == 0).sum()),
        "eligible_location_count_per_gt": eligible_per_gt.tolist(),
        "assigned_location_count_per_gt": assigned.tolist(),
        "multi_min_gt_location_count": int(((min_multiplicity > 1) & positive).sum()),
        "max_min_gt_multiplicity": int(min_multiplicity.max()) if min_multiplicity.size else 0,
    }


def summarize_window_geometry(
    *,
    dense_valid_len,
    selected_positions,
    fps,
    snippet_stride,
    window_start_frame,
    duration_sec,
    gt_segments_dense,
    actionformer_sequence_length=384,
    actionformer_strides=(1, 2, 4, 8, 16, 32),
    actionformer_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
    phystime_base_spacing_sec=0.5,
    phystime_ranges=((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 1.0e8)),
    center_sample_radius=1.5,
):
    dense_valid_len = int(dense_valid_len)
    selected_positions = _as_float_array(selected_positions).reshape(-1)
    gt_segments_dense = _as_float_array(gt_segments_dense, columns=2)
    fps = float(fps)
    snippet_stride = float(snippet_stride)
    window_start_frame = float(window_start_frame)
    duration_sec = float(duration_sec)
    if dense_valid_len <= 0 or selected_positions.size <= 0:
        raise ValueError("window diagnostics require non-empty dense and selected axes")
    if fps <= 0 or snippet_stride <= 0 or duration_sec <= 0:
        raise ValueError("fps, snippet stride, and duration must be positive")
    if np.any(np.diff(selected_positions) <= 0):
        raise ValueError("selected positions must be strictly increasing")
    if selected_positions[0] < 0 or selected_positions[-1] >= dense_valid_len:
        raise ValueError("selected positions exceed the dense window")
    if len(phystime_ranges) <= 0:
        raise ValueError("PhysTime regression ranges cannot be empty")

    selected_frames = window_start_frame + selected_positions * snippet_stride
    timestamps_sec = selected_frames / fps
    half_width_sec = 0.5 * snippet_stride / fps
    supports = np.stack(
        (
            np.maximum(timestamps_sec - half_width_sec, 0.0),
            np.minimum(timestamps_sec + half_width_sec, duration_sec),
        ),
        axis=-1,
    )
    domain_start_sec = max(window_start_frame / fps - half_width_sec, 0.0)
    domain_last_center_frame = window_start_frame + (dense_valid_len - 1) * snippet_stride
    domain_end_sec = min(domain_last_center_frame / fps + half_width_sec, duration_sec)
    domain_span_sec = domain_end_sec - domain_start_sec

    edge_positions = np.concatenate(([0.0], selected_positions, [float(dense_valid_len)]))
    all_gaps = np.diff(edge_positions)
    interior_gaps = np.diff(selected_positions)

    actionformer_levels = build_actionformer_levels(
        sequence_length=actionformer_sequence_length,
        strides=actionformer_strides,
        regression_ranges=actionformer_ranges,
    )
    for level, stride in zip(actionformer_levels, actionformer_strides):
        valid_count = min(
            level["centers"].size,
            int(math.ceil(selected_positions.size / float(stride))),
        )
        level["valid_mask"] = np.arange(level["centers"].size) < valid_count

    selected_points, selected_mask = concatenate_point_levels(actionformer_levels)
    selected_gt = warp_segments_to_selected_axis(
        gt_segments_dense, selected_positions, dense_valid_len=dense_valid_len
    )
    selected_assignment = summarize_target_assignment(
        points=selected_points,
        valid_mask=selected_mask,
        gt_segments=selected_gt,
        center_sample_radius=center_sample_radius,
    )

    physical_levels = map_actionformer_levels_to_physical_grid(
        actionformer_levels,
        selected_positions=selected_positions,
        dense_valid_len=float(dense_valid_len),
    )
    physical_points, physical_mask = concatenate_point_levels(physical_levels)
    physical_assignment = summarize_target_assignment(
        points=physical_points,
        valid_mask=physical_mask,
        gt_segments=gt_segments_dense,
        center_sample_radius=center_sample_radius,
    )

    physical_query_levels = build_physical_query_levels(
        duration_sec=duration_sec,
        domain_start_sec=domain_start_sec,
        domain_end_sec=domain_end_sec,
        support_intervals_sec=supports,
        base_spacing_sec=phystime_base_spacing_sec,
        num_levels=len(phystime_ranges),
    )
    phystime_point_levels = []
    for query_level, regression_range in zip(physical_query_levels, phystime_ranges):
        count = query_level["centers"].size
        phystime_point_levels.append(
            {
                "centers": query_level["centers"],
                "lower_ranges": np.full(count, float(regression_range[0])),
                "upper_ranges": np.full(count, float(regression_range[1])),
                "strides": query_level["widths"],
                "valid_mask": query_level["covered_mask"],
            }
        )
    phystime_points, phystime_mask = concatenate_point_levels(phystime_point_levels)
    gt_segments_sec = (gt_segments_dense * snippet_stride + window_start_frame) / fps
    phystime_assignment = summarize_target_assignment(
        points=phystime_points,
        valid_mask=phystime_mask,
        gt_segments=gt_segments_sec,
        center_sample_radius=center_sample_radius,
    )

    support_sec = float((supports[:, 1] - supports[:, 0]).sum())
    return {
        "dense_valid_len": dense_valid_len,
        "selected_count": int(selected_positions.size),
        "domain_start_sec": domain_start_sec,
        "domain_end_sec": domain_end_sec,
        "domain_span_sec": domain_span_sec,
        "duration_sec": duration_sec,
        "gt_durations_sec": (gt_segments_sec[:, 1] - gt_segments_sec[:, 0]).tolist(),
        "sampling": {
            "mean_gap_dense": float(interior_gaps.mean()) if interior_gaps.size else 0.0,
            "max_gap_dense": float(interior_gaps.max()) if interior_gaps.size else 0.0,
            "max_gap_dense_with_edges": float(all_gaps.max()) if all_gaps.size else 0.0,
            "max_gap_sec_with_edges": (
                float(all_gaps.max() * snippet_stride / fps) if all_gaps.size else 0.0
            ),
            "support_sec": support_sec,
            "support_fraction": float(support_sec / domain_span_sec) if domain_span_sec > 0 else 0.0,
        },
        "candidate_count": {
            "selected_axis": int(selected_mask.sum()),
            "physical_grid": int(physical_mask.sum()),
            "phystime": int(phystime_mask.sum()),
        },
        "assignment": {
            "selected_axis": selected_assignment,
            "physical_grid": physical_assignment,
            "phystime": phystime_assignment,
        },
        "phystime_levels": physical_query_levels,
    }


def _distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def aggregate_window_reports(reports):
    reports = list(reports)
    if not reports:
        raise ValueError("at least one window report is required")
    methods = ("selected_axis", "physical_grid", "phystime")
    candidate_summary = {
        method: _distribution([report["candidate_count"][method] for report in reports])
        for method in methods
    }
    selected_total = sum(report["candidate_count"]["selected_axis"] for report in reports)
    phystime_total = sum(report["candidate_count"]["phystime"] for report in reports)
    candidate_summary["phystime_to_selected_ratio"] = (
        float(phystime_total / selected_total) if selected_total else 0.0
    )

    assignment_summary = {}
    for method in methods:
        gt_count = sum(report["assignment"][method]["gt_count"] for report in reports)
        no_eligible = sum(
            report["assignment"][method]["gt_without_eligible_location_count"]
            for report in reports
        )
        assignment_summary[method] = {
            "gt_count": int(gt_count),
            "positive_location_count": int(
                sum(report["assignment"][method]["positive_location_count"] for report in reports)
            ),
            "gt_without_eligible_location_count": int(no_eligible),
            "gt_without_eligible_location_fraction": float(no_eligible / gt_count) if gt_count else 0.0,
            "multi_min_gt_location_count": int(
                sum(
                    report["assignment"][method]["multi_min_gt_location_count"]
                    for report in reports
                )
            ),
            "eligible_locations_per_gt": _distribution(
                [
                    value
                    for report in reports
                    for value in report["assignment"][method]["eligible_location_count_per_gt"]
                ]
            ),
            "assigned_locations_per_gt": _distribution(
                [
                    value
                    for report in reports
                    for value in report["assignment"][method]["assigned_location_count_per_gt"]
                ]
            ),
        }

    level_count = len(reports[0]["phystime_levels"])
    if any(len(report["phystime_levels"]) != level_count for report in reports):
        raise ValueError("all window reports must use the same PhysTime level count")
    level_summary = []
    for level_index in range(level_count):
        levels = [report["phystime_levels"][level_index] for report in reports]
        query_count = sum(level["query_count"] for level in levels)
        covered_count = sum(level["covered_query_count"] for level in levels)
        coverage_sec = sum(float(level["coverage_sec"].sum()) for level in levels)
        query_width_sec = sum(float(level["widths"].sum()) for level in levels)
        covered_observation_counts = [
            int(value)
            for level in levels
            for value in level["observation_count"][level["covered_mask"]]
        ]
        level_summary.append(
            {
                "level": level_index,
                "spacing_sec": float(levels[0]["spacing_sec"]),
                "query_count": int(query_count),
                "covered_query_count": int(covered_count),
                "covered_query_fraction": float(covered_count / query_count) if query_count else 0.0,
                "evidence_mass_fraction": float(coverage_sec / query_width_sec) if query_width_sec else 0.0,
                "observations_per_covered_query": _distribution(covered_observation_counts),
            }
        )

    duration_edges = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))
    duration_bins = []
    for lower, upper in zip(duration_edges[:-1], duration_edges[1:]):
        bin_row = {
            "lower_sec": lower,
            "upper_sec": None if math.isinf(upper) else upper,
        }
        durations = []
        method_eligible = {method: [] for method in methods}
        for report in reports:
            for gt_index, duration in enumerate(report["gt_durations_sec"]):
                if lower <= duration < upper:
                    durations.append(duration)
                    for method in methods:
                        method_eligible[method].append(
                            report["assignment"][method]["eligible_location_count_per_gt"][gt_index]
                        )
        bin_row["gt_count"] = len(durations)
        for method in methods:
            no_eligible = sum(value == 0 for value in method_eligible[method])
            bin_row[f"{method}_no_eligible_fraction"] = (
                float(no_eligible / len(durations)) if durations else 0.0
            )
            bin_row[f"{method}_eligible_locations"] = _distribution(method_eligible[method])
        duration_bins.append(bin_row)

    return {
        "window_count": len(reports),
        "gt_count": int(sum(len(report["gt_durations_sec"]) for report in reports)),
        "selected_count": _distribution([report["selected_count"] for report in reports]),
        "sampling": {
            "max_gap_dense_with_edges": _distribution(
                [report["sampling"]["max_gap_dense_with_edges"] for report in reports]
            ),
            "max_gap_sec_with_edges": _distribution(
                [report["sampling"]["max_gap_sec_with_edges"] for report in reports]
            ),
            "support_fraction": _distribution(
                [report["sampling"]["support_fraction"] for report in reports]
            ),
        },
        "candidate_count": candidate_summary,
        "assignment": assignment_summary,
        "phystime_levels": level_summary,
        "duration_bins": duration_bins,
    }


def analyze_dataset_rows(
    rows,
    *,
    load_frames,
    snippet_stride,
    target_len,
    actionformer_sequence_length=384,
    actionformer_strides=(1, 2, 4, 8, 16, 32),
    actionformer_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
    phystime_base_spacing_sec=0.5,
    phystime_ranges=((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 1.0e8)),
    max_windows=None,
    include_window_reports=False,
):
    snippet_stride = int(snippet_stride)
    target_len = int(target_len)
    if snippet_stride <= 0 or target_len <= 0:
        raise ValueError("dataset snippet stride and target length must be positive")
    reports = []
    selection_digest = hashlib.sha256()
    limit = len(rows) if max_windows is None else min(len(rows), int(max_windows))
    for row_index, row in enumerate(rows[:limit]):
        video_name, video_info, video_anno, window_centers = row
        window_centers = np.asarray(window_centers, dtype=np.int64).reshape(-1)
        if window_centers.size == 0:
            raise ValueError(f"window {row_index} has no dense positions")
        sample_key = (
            f"{video_name}|random_fixed|{int(window_centers[0])}|"
            f"{int(window_centers[-1])}|{window_centers.size}|{target_len}"
        )
        selected_positions = np.asarray(
            load_frames._select_random_fixed_positions(
                int(window_centers.size), target_len, sample_key
            ),
            dtype=np.int64,
        )
        selection_digest.update(str(video_name).encode("utf-8"))
        selection_digest.update(np.asarray(selected_positions, dtype=np.int64).tobytes())

        gt_segments = np.asarray(video_anno.get("gt_segments", []), dtype=np.float64).reshape(-1, 2)
        gt_segments_dense = (
            gt_segments - float(window_centers[0])
        ) / float(snippet_stride)
        reports.append(
            summarize_window_geometry(
                dense_valid_len=int(window_centers.size),
                selected_positions=selected_positions,
                fps=float(video_info["frame"]) / float(video_info["duration"]),
                snippet_stride=snippet_stride,
                window_start_frame=float(window_centers[0]),
                duration_sec=float(video_info["duration"]),
                gt_segments_dense=gt_segments_dense,
                actionformer_sequence_length=actionformer_sequence_length,
                actionformer_strides=actionformer_strides,
                actionformer_ranges=actionformer_ranges,
                phystime_base_spacing_sec=phystime_base_spacing_sec,
                phystime_ranges=phystime_ranges,
            )
        )
    result = {
        "summary": aggregate_window_reports(reports),
        "selection_sha256": selection_digest.hexdigest(),
        "analyzed_window_count": len(reports),
        "available_window_count": len(rows),
    }
    if include_window_reports:
        result["_window_reports"] = reports
    return result


def analyze_training_rows(
    rows,
    *,
    load_frames,
    snippet_stride,
    target_len,
    source_len,
    samples_per_video=5,
    seed=42,
    actionformer_sequence_length=384,
    actionformer_strides=(1, 2, 4, 8, 16, 32),
    actionformer_ranges=((0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 10000)),
    phystime_base_spacing_sec=0.5,
    phystime_ranges=((0.0, 2.0), (2.0, 4.0), (4.0, 8.0), (8.0, 16.0), (16.0, 32.0), (32.0, 1.0e8)),
):
    snippet_stride = int(snippet_stride)
    target_len = int(target_len)
    source_len = int(source_len)
    samples_per_video = int(samples_per_video)
    if min(snippet_stride, target_len, source_len, samples_per_video) <= 0:
        raise ValueError("training diagnostics require positive sampling settings")
    reports = []
    selection_digest = hashlib.sha256()
    for row_index, row in enumerate(rows):
        video_name, video_info, video_anno = row
        dense_frames = np.arange(
            0, int(video_info["frame"]), snippet_stride, dtype=np.int64
        )
        gt_segments_dense = (
            np.asarray(video_anno.get("gt_segments", []), dtype=np.float64).reshape(-1, 2)
            / float(snippet_stride)
        )
        gt_labels = np.asarray(video_anno.get("gt_labels", []), dtype=np.int64).reshape(-1)
        for repetition in range(samples_per_video):
            random.seed(int(seed) + row_index * samples_per_video + repetition)
            dense_window, window_gt, _window_labels = load_frames.random_trunc(
                dense_frames,
                trunc_len=source_len,
                gt_segments=gt_segments_dense,
                gt_labels=gt_labels,
            )
            dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
            window_gt = np.asarray(window_gt, dtype=np.float64).reshape(-1, 2)
            sample_key = (
                f"{video_name}|random_fixed|{int(dense_window[0])}|"
                f"{int(dense_window[-1])}|{dense_window.size}|{target_len}"
            )
            selected_positions = np.asarray(
                load_frames._select_random_fixed_positions(
                    int(dense_window.size), target_len, sample_key
                ),
                dtype=np.int64,
            )
            selection_digest.update(str(video_name).encode("utf-8"))
            selection_digest.update(selected_positions.tobytes())
            reports.append(
                summarize_window_geometry(
                    dense_valid_len=int(dense_window.size),
                    selected_positions=selected_positions,
                    fps=float(video_info["frame"]) / float(video_info["duration"]),
                    snippet_stride=snippet_stride,
                    window_start_frame=float(dense_window[0]),
                    duration_sec=float(video_info["duration"]),
                    gt_segments_dense=window_gt,
                    actionformer_sequence_length=actionformer_sequence_length,
                    actionformer_strides=actionformer_strides,
                    actionformer_ranges=actionformer_ranges,
                    phystime_base_spacing_sec=phystime_base_spacing_sec,
                    phystime_ranges=phystime_ranges,
                )
            )
    return {
        "summary": aggregate_window_reports(reports),
        "selection_sha256": selection_digest.hexdigest(),
        "analyzed_window_count": len(reports),
        "available_video_count": len(rows),
        "samples_per_video": samples_per_video,
        "seed": int(seed),
    }


def _state_value_by_suffix(state_dict, suffix):
    matches = [value for key, value in state_dict.items() if str(key).endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one checkpoint tensor ending with {suffix!r}, found {len(matches)}")
    return matches[0]


def audit_query_embedding_checkpoint(checkpoint_path, window_reports):
    import torch

    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    state_dict = checkpoint.get("state_dict_ema", checkpoint.get("state_dict"))
    if not isinstance(state_dict, dict):
        raise ValueError("checkpoint has neither state_dict_ema nor state_dict")
    feature_names = (
        "absolute_center_sec",
        "normalized_center",
        "log1p_duration_sec",
        "absolute_width_sec",
        "normalized_width",
        "sin_1",
        "sin_2",
        "sin_4",
        "sin_8",
        "cos_1",
        "cos_2",
        "cos_4",
        "cos_8",
    )
    level_count = len(window_reports[0]["phystime_levels"])
    levels = []
    for level_index in range(level_count):
        centers = []
        widths = []
        durations = []
        for report in window_reports:
            level = report["phystime_levels"][level_index]
            mask = level["covered_mask"]
            centers.extend(level["centers"][mask].tolist())
            widths.extend(level["widths"][mask].tolist())
            durations.extend([report["duration_sec"]] * int(mask.sum()))
        features = build_query_embedding_features(
            centers_sec=np.asarray(centers),
            widths_sec=np.asarray(widths),
            duration_sec=np.asarray(durations),
            num_fourier_bands=4,
        )
        prefix = f"projection.level_attentions.{level_index}.query_embedding.net.0"
        weight = _state_value_by_suffix(state_dict, f"{prefix}.weight").detach().cpu().numpy()
        bias = _state_value_by_suffix(state_dict, f"{prefix}.bias").detach().cpu().numpy()
        row = summarize_linear_input_scale(
            features=features,
            weight=weight,
            bias=bias,
            feature_names=feature_names,
        )
        row.update(
            {
                "level": level_index,
                "spacing_sec": float(window_reports[0]["phystime_levels"][level_index]["spacing_sec"]),
            }
        )
        levels.append(row)
    return {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "state_dict_source": "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict",
        "levels": levels,
    }


def _find_load_frames_transform(dataset):
    for transform in dataset.pipeline.transforms:
        if transform.__class__.__name__ == "LoadFrames":
            return transform
    raise RuntimeError("dataset pipeline does not contain LoadFrames")


def run_dataset_diagnostics(
    config_path,
    *,
    split="val",
    max_windows=None,
    phystime_checkpoint=None,
    training_samples_per_video=5,
):
    from mmengine.config import Config

    from opentad.datasets import build_dataset

    config_path = Path(config_path).resolve()
    cfg = Config.fromfile(str(config_path))
    if split not in cfg.dataset:
        raise KeyError(f"config has no dataset split: {split}")
    dataset = build_dataset(cfg.dataset[split])
    load_frames = _find_load_frames_transform(dataset)
    common = {
        "load_frames": load_frames,
        "snippet_stride": dataset.snippet_stride,
        "target_len": int(cfg.window_size),
        "actionformer_sequence_length": int(cfg.window_size),
        "phystime_base_spacing_sec": float(cfg.model.projection.base_spacing_sec),
        "phystime_ranges": tuple(
            tuple(item) for item in cfg.model.rpn_head.regression_ranges_sec
        ),
    }
    if split == "train":
        if phystime_checkpoint is not None:
            raise ValueError("checkpoint query audit is only supported on sliding-window splits")
        result = analyze_training_rows(
            dataset.data_list,
            source_len=int(load_frames.source_len or cfg.dense_window_size),
            samples_per_video=training_samples_per_video,
            **common,
        )
    else:
        result = analyze_dataset_rows(
            dataset.data_list,
            max_windows=max_windows,
            include_window_reports=phystime_checkpoint is not None,
            **common,
        )
    window_reports = result.pop("_window_reports", None)
    if phystime_checkpoint is not None:
        result["query_embedding_checkpoint_audit"] = audit_query_embedding_checkpoint(
            phystime_checkpoint, window_reports
        )
    result.update(
        {
            "schema_version": "phystime_performance_geometry_diagnostic_v1",
            "config": str(config_path),
            "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "split": split,
            "snippet_stride": int(dataset.snippet_stride),
            "target_len": int(cfg.window_size),
            "dense_window_size": int(cfg.dense_window_size),
            "phystime_base_spacing_sec": float(cfg.model.projection.base_spacing_sec),
        }
    )
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Diagnose PhysTime performance loss from matched sparse-window geometry."
    )
    parser.add_argument("--config", required=True, help="Resolved PhysTime-AdaTAD config")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--training-samples-per-video", type=int, default=5)
    parser.add_argument("--phystime-checkpoint", default=None)
    parser.add_argument("--output", required=True, help="JSON output path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_dataset_diagnostics(
        args.config,
        split=args.split,
        max_windows=args.max_windows,
        phystime_checkpoint=args.phystime_checkpoint,
        training_samples_per_video=args.training_samples_per_video,
    )
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
