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

from tools.bata.analyze_phystime_performance_drop import (  # noqa: E402
    _distribution,
    build_actionformer_levels,
    concatenate_point_levels,
    summarize_target_assignment,
)


def _as_array(value, *, dtype=np.float64):
    return np.asarray(value, dtype=dtype).reshape(-1)


def _map_rank_to_axis(coords, positions, *, domain_start, domain_end, eps=1.0e-6):
    coords = _as_array(coords)
    positions = _as_array(positions)
    if positions.size <= 0:
        raise ValueError("G1a axis mapping requires at least one position")
    if positions.size > 1 and np.any(np.diff(positions) <= 0):
        raise ValueError("G1a axis positions must be strictly increasing")
    count = int(positions.size)
    xp = np.concatenate(([-0.5], np.arange(count, dtype=np.float64), [float(count) - 0.5]))
    fp = np.concatenate(([float(domain_start)], positions, [float(domain_end)]))
    flat = np.clip(coords, -0.5, float(count) - 0.5)
    right_idx = np.searchsorted(xp, flat, side="right")
    right_idx = np.clip(right_idx, 1, xp.size - 1)
    left_idx = right_idx - 1
    x0 = xp[left_idx]
    x1 = xp[right_idx]
    y0 = fp[left_idx]
    y1 = fp[right_idx]
    weight = (flat - x0) / np.maximum(x1 - x0, float(eps))
    return y0 + weight * (y1 - y0)


def _native_axis_positions(
    *,
    selected_positions,
    fps,
    snippet_stride,
    dense_origin_frame,
    dense_valid_len,
    duration_sec,
    tubelet_size=2,
):
    selected_positions = _as_array(selected_positions)
    if selected_positions.size <= 0:
        raise ValueError("G1a diagnostic requires selected positions")
    timestamps = (float(dense_origin_frame) + selected_positions * float(snippet_stride)) / float(fps)
    native_valid_count = int(math.ceil(selected_positions.size / float(tubelet_size)))
    token_timestamps = np.zeros(native_valid_count, dtype=np.float64)
    for token_idx in range(native_valid_count):
        start = token_idx * int(tubelet_size)
        end = min(start + int(tubelet_size), selected_positions.size)
        token_timestamps[token_idx] = timestamps[start:end].mean()
    domain_start = max(float(dense_origin_frame) / float(fps), 0.0)
    domain_end = min(
        (float(dense_origin_frame) + float(dense_valid_len) * float(snippet_stride)) / float(fps),
        float(duration_sec),
    )
    if domain_end <= domain_start:
        raise ValueError("G1a diagnostic received an empty physical domain")
    uniform_step = (domain_end - domain_start) / float(native_valid_count)
    uniform_positions = domain_start + (np.arange(native_valid_count, dtype=np.float64) + 0.5) * uniform_step
    return {
        "native_valid_count": native_valid_count,
        "domain_start": float(domain_start),
        "domain_end": float(domain_end),
        "uniform_rank_seconds": uniform_positions,
        "physical_time_seconds": token_timestamps,
    }


def _levels_on_seconds_axis(base_levels, *, axis_positions, domain_start, domain_end):
    mapped = []
    for level in base_levels:
        centers = _as_array(level["centers"])
        nominal_stride = _as_array(level["strides"])
        physical_center = _map_rank_to_axis(
            centers,
            axis_positions,
            domain_start=domain_start,
            domain_end=domain_end,
        )
        physical_left = _map_rank_to_axis(
            centers - 0.5 * nominal_stride,
            axis_positions,
            domain_start=domain_start,
            domain_end=domain_end,
        )
        physical_right = _map_rank_to_axis(
            centers + 0.5 * nominal_stride,
            axis_positions,
            domain_start=domain_start,
            domain_end=domain_end,
        )
        physical_stride = np.maximum(physical_right - physical_left, 1.0e-6)
        range_scale = physical_stride / np.maximum(nominal_stride, 1.0e-6)
        mapped.append(
            {
                **level,
                "centers": physical_center,
                "lower_ranges": _as_array(level["lower_ranges"]) * range_scale,
                "upper_ranges": _as_array(level["upper_ranges"]) * range_scale,
                "strides": physical_stride,
            }
        )
    return mapped


def summarize_g1a_window(
    *,
    dense_valid_len,
    selected_positions,
    fps,
    snippet_stride,
    dense_origin_frame,
    duration_sec,
    gt_segments_sec,
    sequence_length,
    strides,
    regression_ranges,
    center_sample_radius,
    tubelet_size=2,
):
    axis = _native_axis_positions(
        selected_positions=selected_positions,
        fps=fps,
        snippet_stride=snippet_stride,
        dense_origin_frame=dense_origin_frame,
        dense_valid_len=dense_valid_len,
        duration_sec=duration_sec,
        tubelet_size=tubelet_size,
    )
    base_levels = build_actionformer_levels(
        sequence_length=sequence_length,
        strides=strides,
        regression_ranges=regression_ranges,
    )
    for level, stride in zip(base_levels, strides):
        valid_count = min(level["centers"].size, int(math.ceil(axis["native_valid_count"] / float(stride))))
        level["valid_mask"] = np.arange(level["centers"].size) < valid_count

    reports = {}
    for name in ("uniform_rank_seconds", "physical_time_seconds"):
        levels = _levels_on_seconds_axis(
            base_levels,
            axis_positions=axis[name],
            domain_start=axis["domain_start"],
            domain_end=axis["domain_end"],
        )
        points, mask = concatenate_point_levels(levels)
        reports[name] = {
            "candidate_count": int(mask.sum()),
            "stride_sec": _distribution(points[mask, 3]),
            "range_upper_sec": _distribution(points[mask, 2]),
            "assignment": summarize_target_assignment(
                points=points,
                valid_mask=mask,
                gt_segments=gt_segments_sec,
                center_sample_radius=center_sample_radius,
            ),
        }

    delta = axis["physical_time_seconds"] - axis["uniform_rank_seconds"]
    return {
        "native_valid_count": int(axis["native_valid_count"]),
        "domain_duration_sec": float(axis["domain_end"] - axis["domain_start"]),
        "axis_delta_sec": _distribution(delta),
        "axis_abs_delta_sec": _distribution(np.abs(delta)),
        "uniform_rank_seconds": reports["uniform_rank_seconds"],
        "physical_time_seconds": reports["physical_time_seconds"],
        "gt_durations_sec": (
            np.asarray(gt_segments_sec, dtype=np.float64).reshape(-1, 2)[:, 1]
            - np.asarray(gt_segments_sec, dtype=np.float64).reshape(-1, 2)[:, 0]
        ).tolist()
        if np.asarray(gt_segments_sec).size
        else [],
    }


def aggregate_g1a_reports(reports):
    reports = list(reports)
    if not reports:
        raise ValueError("G1a diagnostics require at least one report")
    methods = ("uniform_rank_seconds", "physical_time_seconds")
    summary = {
        "window_count": len(reports),
        "gt_count": int(sum(len(report["gt_durations_sec"]) for report in reports)),
        "native_valid_count": _distribution([report["native_valid_count"] for report in reports]),
        "domain_duration_sec": _distribution([report["domain_duration_sec"] for report in reports]),
        "axis_abs_delta_sec": _distribution(
            [
                value
                for report in reports
                for value in np.atleast_1d(report["axis_abs_delta_sec"]["mean"])
            ]
        ),
    }
    for method in methods:
        gt_count = sum(report[method]["assignment"]["gt_count"] for report in reports)
        no_eligible = sum(
            report[method]["assignment"]["gt_without_eligible_location_count"]
            for report in reports
        )
        summary[method] = {
            "candidate_count": _distribution([report[method]["candidate_count"] for report in reports]),
            "stride_sec": _distribution(
                [report[method]["stride_sec"]["mean"] for report in reports]
            ),
            "range_upper_sec": _distribution(
                [report[method]["range_upper_sec"]["mean"] for report in reports]
            ),
            "assignment": {
                "gt_count": int(gt_count),
                "positive_location_count": int(
                    sum(report[method]["assignment"]["positive_location_count"] for report in reports)
                ),
                "gt_without_eligible_location_count": int(no_eligible),
                "gt_without_eligible_location_fraction": float(no_eligible / gt_count) if gt_count else 0.0,
                "eligible_locations_per_gt": _distribution(
                    [
                        value
                        for report in reports
                        for value in report[method]["assignment"]["eligible_location_count_per_gt"]
                    ]
                ),
                "assigned_locations_per_gt": _distribution(
                    [
                        value
                        for report in reports
                        for value in report[method]["assignment"]["assigned_location_count_per_gt"]
                    ]
                ),
            },
        }

    duration_edges = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))
    bins = []
    for lower, upper in zip(duration_edges[:-1], duration_edges[1:]):
        row = {"lower_sec": lower, "upper_sec": None if math.isinf(upper) else upper, "gt_count": 0}
        values = {method: [] for method in methods}
        for report in reports:
            durations = report["gt_durations_sec"]
            for gt_idx, duration in enumerate(durations):
                if lower <= duration < upper:
                    row["gt_count"] += 1
                    for method in methods:
                        values[method].append(
                            report[method]["assignment"]["eligible_location_count_per_gt"][gt_idx]
                        )
        for method in methods:
            row[f"{method}_eligible_locations"] = _distribution(values[method])
            row[f"{method}_no_eligible_fraction"] = (
                float(sum(v == 0 for v in values[method]) / len(values[method]))
                if values[method]
                else 0.0
            )
        bins.append(row)
    summary["duration_bins"] = bins
    return summary


def _find_load_frames_transform(dataset):
    for transform in dataset.pipeline.transforms:
        if transform.__class__.__name__ == "LoadFrames":
            return transform
    raise RuntimeError("dataset pipeline does not contain LoadFrames")


def _config_ranges(cfg):
    prior = cfg.model.rpn_head.prior_generator
    return {
        "strides": tuple(int(value) for value in prior.strides),
        "regression_ranges": tuple(tuple(float(v) for v in row) for row in prior.regression_range),
        "center_sample_radius": float(cfg.model.rpn_head.center_sample_radius),
        "sequence_length": int(cfg.native_token_count),
        "tubelet_size": int(cfg.model.native_temporal_geometry.tubelet_size),
    }


def run_g1a_geometry_diagnostics(
    config_path,
    *,
    split="val",
    max_windows=None,
    training_samples_per_video=5,
    seed=42,
):
    from mmengine.config import Config

    from opentad.datasets import build_dataset

    config_path = Path(config_path).resolve()
    cfg = Config.fromfile(str(config_path), lazy_import=False)
    dataset = build_dataset(cfg.dataset[split])
    load_frames = _find_load_frames_transform(dataset)
    settings = _config_ranges(cfg)
    reports = []
    digest = hashlib.sha256()
    if split == "train":
        limit = len(dataset.data_list)
        for row_index, row in enumerate(dataset.data_list[:limit]):
            video_name, video_info, video_anno = row
            dense_frames = np.arange(0, int(video_info["frame"]), int(dataset.snippet_stride), dtype=np.int64)
            gt_segments_dense = (
                np.asarray(video_anno.get("gt_segments", []), dtype=np.float64).reshape(-1, 2)
                / float(dataset.snippet_stride)
            )
            gt_labels = np.asarray(video_anno.get("gt_labels", []), dtype=np.int64).reshape(-1)
            for repetition in range(int(training_samples_per_video)):
                random.seed(int(seed) + row_index * int(training_samples_per_video) + repetition)
                dense_window, window_gt, _labels = load_frames.random_trunc(
                    dense_frames,
                    trunc_len=int(load_frames.source_len or cfg.dense_window_size),
                    gt_segments=gt_segments_dense,
                    gt_labels=gt_labels,
                )
                dense_window = np.asarray(dense_window, dtype=np.int64).reshape(-1)
                sample_key = (
                    f"{video_name}|random_fixed|{int(dense_window[0])}|"
                    f"{int(dense_window[-1])}|{dense_window.size}|{int(cfg.raw_observation_count)}"
                )
                selected = np.asarray(
                    load_frames._select_random_fixed_positions(
                        int(dense_window.size), int(cfg.raw_observation_count), sample_key
                    ),
                    dtype=np.int64,
                )
                digest.update(str(video_name).encode("utf-8"))
                digest.update(selected.tobytes())
                fps = float(video_info["frame"]) / float(video_info["duration"])
                dense_origin = float(dense_window[0])
                gt_seconds = (np.asarray(window_gt, dtype=np.float64).reshape(-1, 2) * float(dataset.snippet_stride) + dense_origin) / fps
                reports.append(
                    summarize_g1a_window(
                        dense_valid_len=int(dense_window.size),
                        selected_positions=selected,
                        fps=fps,
                        snippet_stride=float(dataset.snippet_stride),
                        dense_origin_frame=dense_origin,
                        duration_sec=float(video_info["duration"]),
                        gt_segments_sec=gt_seconds,
                        **settings,
                    )
                )
    else:
        limit = len(dataset.data_list) if max_windows is None else min(len(dataset.data_list), int(max_windows))
        for row in dataset.data_list[:limit]:
            video_name, video_info, video_anno, window_centers = row
            window_centers = np.asarray(window_centers, dtype=np.int64).reshape(-1)
            sample_key = (
                f"{video_name}|random_fixed|{int(window_centers[0])}|"
                f"{int(window_centers[-1])}|{window_centers.size}|{int(cfg.raw_observation_count)}"
            )
            selected = np.asarray(
                load_frames._select_random_fixed_positions(
                    int(window_centers.size), int(cfg.raw_observation_count), sample_key
                ),
                dtype=np.int64,
            )
            digest.update(str(video_name).encode("utf-8"))
            digest.update(selected.tobytes())
            fps = float(video_info["frame"]) / float(video_info["duration"])
            gt_seconds = np.asarray(video_anno.get("gt_segments", []), dtype=np.float64).reshape(-1, 2) / fps
            reports.append(
                summarize_g1a_window(
                    dense_valid_len=int(window_centers.size),
                    selected_positions=selected,
                    fps=fps,
                    snippet_stride=float(dataset.snippet_stride),
                    dense_origin_frame=float(window_centers[0]),
                    duration_sec=float(video_info["duration"]),
                    gt_segments_sec=gt_seconds,
                    **settings,
                )
            )
    return {
        "schema_version": "phystime_g1a_geometry_diagnostic_v1",
        "config": str(config_path),
        "split": split,
        "selection_sha256": digest.hexdigest(),
        "analyzed_window_count": len(reports),
        "summary": aggregate_g1a_reports(reports),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Diagnose G1a uniform-rank vs physical-time ActionFormer geometry.")
    parser.add_argument("--config", default=str(ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py"))
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--training-samples-per-video", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    report = run_g1a_geometry_diagnostics(
        args.config,
        split=args.split,
        max_windows=args.max_windows,
        training_samples_per_video=args.training_samples_per_video,
        seed=args.seed,
    )
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
