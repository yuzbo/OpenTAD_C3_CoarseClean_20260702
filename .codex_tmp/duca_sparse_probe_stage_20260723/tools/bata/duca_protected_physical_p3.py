from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import spearmanr


DURATION_STRATA = ("short", "medium", "long")
BOUNDARY_STRATA = ("near", "mid", "far")
WINDOWS_PER_STRATUM = 16
MAX_WINDOWS_PER_VIDEO = 4
SWAPS_PER_WINDOW = 12
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 20260720
MIN_PADDED_WINDOWS_PER_STRATUM = 4
SHORT_ACTION_SECONDS = 2.0
MEDIUM_ACTION_SECONDS = 8.0
NEAR_BOUNDARY_SECONDS = 0.5
MID_BOUNDARY_SECONDS = 2.0


def _sha_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def duration_stratum(shortest_duration_seconds: float) -> str:
    value = float(shortest_duration_seconds)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("P3 action duration must be finite and positive")
    if value <= SHORT_ACTION_SECONDS:
        return "short"
    if value <= MEDIUM_ACTION_SECONDS:
        return "medium"
    return "long"


def boundary_distance_stratum(distance_seconds: float) -> str:
    value = float(distance_seconds)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("P3 boundary distance must be finite and non-negative")
    if value <= NEAR_BOUNDARY_SECONDS:
        return "near"
    if value <= MID_BOUNDARY_SECONDS:
        return "mid"
    return "far"


def stratified_window_manifest(dataset) -> list[dict[str, Any]]:
    candidates: dict[str, list[dict[str, Any]]] = {
        key: [] for key in DURATION_STRATA
    }
    snippet_stride = float(dataset.snippet_stride)
    if not math.isfinite(snippet_stride) or snippet_stride <= 0.0:
        raise ValueError("P3 dataset snippet stride must be positive")
    for dataset_index, item in enumerate(dataset.data_list):
        video_name, video_info, _annotation, centers = item
        valid_len = int(np.asarray(centers).reshape(-1).size)
        if valid_len <= 384:
            continue
        center_row = np.asarray(centers, dtype=np.float64).reshape(-1)
        window_start_frame = float(center_row[0])
        window_end_frame = float(center_row[-1])
        duration = float(video_info["duration"])
        total_frames = int(video_info["frame"])
        avg_fps = float(total_frames) / duration
        if not math.isfinite(avg_fps) or avg_fps <= 0.0:
            raise ValueError("P3 video FPS must be finite and positive")
        contained_segments = []
        boundary_candidates = []
        boundary_frames = []
        boundary_seconds = []
        for action in video_info.get("annotations", []):
            if action.get("label") == "Ambiguous":
                continue
            start_seconds, end_seconds = sorted(
                (float(action["segment"][0]), float(action["segment"][1]))
            )
            # Match the THUMOS loader's integer-frame target construction
            # exactly, while retaining original annotation seconds for
            # boundary-distance evidence.
            start_frame = int(start_seconds / duration * total_frames)
            end_frame = int(end_seconds / duration * total_frames)
            start_candidate = (
                start_frame - window_start_frame - float(dataset.offset_frames)
            ) / snippet_stride
            end_candidate = (
                end_frame - window_start_frame - float(dataset.offset_frames)
            ) / snippet_stride
            if 0.0 <= start_candidate <= float(valid_len - 1):
                boundary_candidates.append(start_candidate)
                boundary_frames.append(float(start_frame))
                boundary_seconds.append(start_seconds)
            if 0.0 <= end_candidate <= float(valid_len - 1):
                boundary_candidates.append(end_candidate)
                boundary_frames.append(float(end_frame))
                boundary_seconds.append(end_seconds)
            if (
                start_frame >= window_start_frame
                and end_frame <= window_end_frame
            ):
                contained_segments.append(
                    {
                        "candidate": [start_candidate, end_candidate],
                        "source_frames": [start_frame, end_frame],
                        "seconds": [start_seconds, end_seconds],
                        "duration_seconds": end_seconds - start_seconds,
                    }
                )
        if not contained_segments or not boundary_candidates:
            continue
        shortest_seconds = float(
            min(row["duration_seconds"] for row in contained_segments)
        )
        stratum = duration_stratum(shortest_seconds)
        window_start = int(round(window_start_frame))
        row = {
            "dataset_index": int(dataset_index),
            "video_id": str(video_name),
            "window_start": window_start,
            "shortest_action_duration_seconds": shortest_seconds,
            "duration_stratum": stratum,
            "valid_len": valid_len,
            "window_kind": "full" if valid_len == 768 else "padded",
            "fully_contained_segments": contained_segments,
            "true_boundary_candidate_positions": boundary_candidates,
            "true_boundary_source_frames": boundary_frames,
            "true_boundary_seconds": boundary_seconds,
            "boundary_source": "original_uncropped_annotation",
        }
        row["selection_sha256"] = _sha_key(
            f"20260720|{row['video_id']}|{window_start}"
        )
        candidates[stratum].append(row)

    selected_by_stratum: dict[str, list[dict[str, Any]]] = {
        key: [] for key in DURATION_STRATA
    }
    video_counts: Counter[str] = Counter()
    ordered_by_stratum = {
        stratum: sorted(
            candidates[stratum],
            key=lambda row: (
                row["selection_sha256"],
                row["video_id"],
                row["window_start"],
                row["dataset_index"],
            ),
        )
        for stratum in DURATION_STRATA
    }
    for stratum in DURATION_STRATA:
        for row in ordered_by_stratum[stratum]:
            if row["window_kind"] != "padded":
                continue
            if video_counts[row["video_id"]] >= MAX_WINDOWS_PER_VIDEO:
                continue
            selected_by_stratum[stratum].append(row)
            video_counts[row["video_id"]] += 1
            if (
                len(selected_by_stratum[stratum])
                == MIN_PADDED_WINDOWS_PER_STRATUM
            ):
                break
        if (
            len(selected_by_stratum[stratum])
            != MIN_PADDED_WINDOWS_PER_STRATUM
        ):
            raise RuntimeError(
                f"P3 stratum {stratum} lacks the required padded windows"
            )
    for stratum in DURATION_STRATA:
        chosen = selected_by_stratum[stratum]
        chosen_keys = {
            (row["video_id"], row["window_start"]) for row in chosen
        }
        for row in ordered_by_stratum[stratum]:
            if (row["video_id"], row["window_start"]) in chosen_keys:
                continue
            if video_counts[row["video_id"]] >= MAX_WINDOWS_PER_VIDEO:
                continue
            chosen.append(row)
            video_counts[row["video_id"]] += 1
            if len(chosen) == WINDOWS_PER_STRATUM:
                break
        if len(chosen) != WINDOWS_PER_STRATUM:
            raise RuntimeError(
                f"P3 stratum {stratum} has only {len(chosen)} admissible windows"
            )
    selected = [
        row
        for stratum in DURATION_STRATA
        for row in selected_by_stratum[stratum]
    ]
    if len(selected) != 48 or max(video_counts.values(), default=0) > 4:
        raise RuntimeError("P3 window population violates the frozen contract")
    for stratum in DURATION_STRATA:
        padded = sum(
            row["window_kind"] == "padded"
            for row in selected_by_stratum[stratum]
        )
        if padded < MIN_PADDED_WINDOWS_PER_STRATUM:
            raise RuntimeError("P3 padded-window quota drifted")
    return selected


def legal_single_swaps(
    selected_positions: Iterable[int],
    physical_seconds: Iterable[float],
    valid_len: int,
    max_gap_seconds: float,
) -> list[tuple[int, int]]:
    selected = np.asarray(list(selected_positions), dtype=np.int64)
    physical = np.asarray(list(physical_seconds), dtype=np.float64)
    valid_len = int(valid_len)
    cap = float(max_gap_seconds)
    if selected.ndim != 1 or selected.size == 0:
        raise ValueError("P3 selected positions must be one nonempty row")
    if physical.ndim != 1 or physical.size < valid_len:
        raise ValueError("P3 physical axis is shorter than valid_len")
    if np.any(np.diff(selected) <= 0):
        raise ValueError("P3 selected positions must be unique and ordered")
    if selected[0] < 0 or selected[-1] >= valid_len:
        raise ValueError("P3 selected positions are outside the valid axis")
    if not math.isfinite(cap) or cap < 0.0:
        raise ValueError("P3 max-gap cap is invalid")
    active_axis = physical[:valid_len]
    if np.any(~np.isfinite(active_axis)) or np.any(np.diff(active_axis) <= 0):
        raise ValueError("P3 physical axis must be finite and increasing")

    selected_set = set(int(value) for value in selected)
    unselected = np.asarray(
        [index for index in range(valid_len) if index not in selected_set],
        dtype=np.int64,
    )
    tolerance = max(1.0e-9, 8.0 * np.finfo(np.float64).eps)
    legal = []
    for rank, removed in enumerate(selected):
        if rank == 0:
            left_time = active_axis[0]
            right_time = active_axis[selected[1]]
        elif rank == selected.size - 1:
            left_time = active_axis[selected[-2]]
            right_time = active_axis[-1]
        else:
            left_time = active_axis[selected[rank - 1]]
            right_time = active_axis[selected[rank + 1]]
        merged_gap = float(right_time - left_time)
        if merged_gap <= cap + tolerance:
            incoming = unselected
        else:
            incoming_time = active_axis[unselected]
            admissible = (
                (incoming_time >= left_time)
                & (incoming_time <= right_time)
                & ((incoming_time - left_time) <= cap + tolerance)
                & ((right_time - incoming_time) <= cap + tolerance)
            )
            incoming = unselected[admissible]
        legal.extend((int(removed), int(value)) for value in incoming)
    return legal


def deterministic_quartile_swaps(
    legal_swaps: Iterable[tuple[int, int]],
    score_gradient: Iterable[float],
    *,
    video_id: str,
    window_start: int,
) -> list[dict[str, Any]]:
    gradient = np.asarray(list(score_gradient), dtype=np.float64)
    rows = []
    for removed, incoming in legal_swaps:
        predicted = float(gradient[incoming] - gradient[removed])
        rows.append(
            {
                "removed": int(removed),
                "incoming": int(incoming),
                "predicted_delta": predicted,
            }
        )
    if len(rows) < SWAPS_PER_WINDOW:
        raise RuntimeError("P3 window has fewer than 12 legal swaps")
    rows.sort(
        key=lambda row: (
            row["predicted_delta"],
            row["removed"],
            row["incoming"],
        )
    )
    quartiles = np.array_split(np.arange(len(rows)), 4)
    sampled = []
    for quartile_index, indices in enumerate(quartiles):
        bucket = [rows[int(index)] for index in indices]
        for row in bucket:
            row["sampling_sha256"] = _sha_key(
                f"{video_id}|{int(window_start)}|{row['removed']}|"
                f"{row['incoming']}|20260720"
            )
        chosen = sorted(
            bucket,
            key=lambda row: (
                row["sampling_sha256"],
                row["removed"],
                row["incoming"],
            ),
        )[:3]
        if len(chosen) != 3:
            raise RuntimeError("P3 quartile has fewer than three swaps")
        for row in chosen:
            row["quartile"] = int(quartile_index)
            row["predicted_best_quartile"] = quartile_index == 0
            sampled.append(row)
    if len(sampled) != SWAPS_PER_WINDOW:
        raise RuntimeError("P3 deterministic sampling did not yield 12 swaps")
    return sampled


def _window_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["video_id"]), int(row["window_start"])


def _effective(row: Mapping[str, Any]) -> bool:
    if row.get("excluded_reason") not in (None, ""):
        return False
    predicted = float(row["predicted_delta"])
    actual = float(row["actual_delta"])
    return math.isfinite(predicted) and math.isfinite(actual)


def _sign_agreement(rows: list[Mapping[str, Any]]) -> float:
    if not rows:
        return float("nan")
    values = []
    for row in rows:
        predicted = float(row["predicted_delta"])
        actual = float(row["actual_delta"])
        if abs(predicted) <= 1.0e-10 or abs(actual) <= 1.0e-8:
            values.append(0.5)
        else:
            values.append(float(np.sign(predicted) == np.sign(actual)))
    return float(np.mean(values))


def _window_spearman(rows: list[Mapping[str, Any]]) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_window_key(row)].append(row)
    output = {}
    for key, items in grouped.items():
        predicted = np.asarray(
            [float(row["predicted_delta"]) for row in items],
            dtype=np.float64,
        )
        actual = np.asarray(
            [float(row["actual_delta"]) for row in items],
            dtype=np.float64,
        )
        if len(items) < 2 or np.ptp(predicted) == 0.0 or np.ptp(actual) == 0.0:
            output[key] = float("nan")
        else:
            output[key] = float(spearmanr(predicted, actual).statistic)
    return output


def _best_quartile_window_delta(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if bool(row["predicted_best_quartile"]):
            grouped[_window_key(row)].append(float(row["actual_delta"]))
    return {
        key: float(np.median(values)) for key, values in grouped.items() if values
    }


def _best_quartile_boundary_gain(
    rows: list[Mapping[str, Any]],
) -> dict[tuple[str, int], float]:
    grouped: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        if bool(row["predicted_best_quartile"]):
            grouped[_window_key(row)].append(
                float(row["boundary_distance_gain_seconds"])
            )
    return {
        key: float(np.median(values)) for key, values in grouped.items() if values
    }


def _cluster_bootstrap(
    rows: list[Mapping[str, Any]],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, list[float]]:
    by_video: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_video[str(row["video_id"])].append(row)
    videos = sorted(by_video)
    if not videos:
        raise RuntimeError("P3 bootstrap has no video clusters")
    rng = np.random.default_rng(int(seed))
    sign_values = []
    spearman_values = []
    best_values = []
    boundary_gain_values = []
    for _index in range(int(replicates)):
        sampled_videos = rng.choice(videos, size=len(videos), replace=True)
        sampled_rows = []
        sampled_window_spearman = []
        sampled_best = []
        sampled_boundary_gain = []
        for video in sampled_videos:
            cluster_rows = by_video[str(video)]
            sampled_rows.extend(cluster_rows)
            window_spearman = _window_spearman(cluster_rows)
            sampled_window_spearman.extend(window_spearman.values())
            best = _best_quartile_window_delta(cluster_rows)
            sampled_best.extend(best.values())
            boundary_gain = _best_quartile_boundary_gain(cluster_rows)
            sampled_boundary_gain.extend(boundary_gain.values())
        sign_values.append(_sign_agreement(sampled_rows))
        finite_spearman = [
            value for value in sampled_window_spearman if math.isfinite(value)
        ]
        spearman_values.append(
            float(np.median(finite_spearman))
            if finite_spearman
            else float("nan")
        )
        best_values.append(
            float(np.median(sampled_best)) if sampled_best else float("nan")
        )
        boundary_gain_values.append(
            float(np.median(sampled_boundary_gain))
            if sampled_boundary_gain
            else float("nan")
        )
    return {
        "sign_agreement": sign_values,
        "window_spearman_median": spearman_values,
        "predicted_best_actual_delta_median": best_values,
        "predicted_best_boundary_gain_seconds_median": (
            boundary_gain_values
        ),
    }


def _percentile_ci(values: Iterable[float]) -> list[float]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or np.any(~np.isfinite(array)):
        return [float("nan"), float("nan")]
    return [
        float(np.percentile(array, 2.5)),
        float(np.percentile(array, 97.5)),
    ]


def aggregate_p3_rows(
    rows: list[Mapping[str, Any]],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if len(rows) != 576:
        raise RuntimeError(f"P3 requires exactly 576 rows, got {len(rows)}")
    effective = [row for row in rows if _effective(row)]
    window_spearman = _window_spearman(effective)
    finite_window_spearman = [
        value for value in window_spearman.values() if math.isfinite(value)
    ]
    best_delta = _best_quartile_window_delta(effective)
    best_boundary_gain = _best_quartile_boundary_gain(effective)
    bootstrap = _cluster_bootstrap(
        effective,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    sign_point = _sign_agreement(effective)
    spearman_point = (
        float(np.median(finite_window_spearman))
        if finite_window_spearman
        else float("nan")
    )
    best_point = (
        float(np.median(list(best_delta.values())))
        if best_delta
        else float("nan")
    )
    best_boundary_gain_point = (
        float(np.median(list(best_boundary_gain.values())))
        if best_boundary_gain
        else float("nan")
    )
    duration = {}
    for stratum in DURATION_STRATA:
        stratum_rows = [
            row for row in effective if row["duration_stratum"] == stratum
        ]
        correlations = [
            value
            for value in _window_spearman(stratum_rows).values()
            if math.isfinite(value)
        ]
        duration[stratum] = {
            "effective_count": len(stratum_rows),
            "sign_agreement": _sign_agreement(stratum_rows),
            "window_spearman_median": (
                float(np.median(correlations))
                if correlations
                else float("nan")
            ),
        }
    boundary_counts = Counter(
        str(row["boundary_distance_stratum"]) for row in effective
    )
    unique_windows = {
        (
            str(row["duration_stratum"]),
            str(row["video_id"]),
            int(row["window_start"]),
        ): str(row.get("window_kind", ""))
        for row in rows
    }
    padded_windows = Counter(
        stratum
        for (stratum, _video_id, _window_start), kind in unique_windows.items()
        if kind == "padded"
    )
    violations = sum(int(row.get("physical_violation_count", 0)) for row in rows)
    restoration_mismatches = sum(
        int(row.get("restoration_mismatch", False)) for row in rows
    )
    repeated_base_max_error = max(
        (float(row.get("repeated_base_loss_abs_error", 0.0)) for row in rows),
        default=float("inf"),
    )
    finite_all = all(
        math.isfinite(float(row["predicted_delta"]))
        and math.isfinite(float(row["actual_delta"]))
        for row in rows
    )
    sign_ci = _percentile_ci(bootstrap["sign_agreement"])
    spearman_ci = _percentile_ci(bootstrap["window_spearman_median"])
    best_ci = _percentile_ci(
        bootstrap["predicted_best_actual_delta_median"]
    )
    best_boundary_gain_ci = _percentile_ci(
        bootstrap["predicted_best_boundary_gain_seconds_median"]
    )
    checks = {
        "hard_forward_equality": all(
            bool(row.get("hard_forward_equal", False)) for row in rows
        ),
        "physical_violation_count_zero": violations == 0,
        "effective_count_at_least_512": len(effective) >= 512,
        "duration_count_at_least_128": all(
            duration[key]["effective_count"] >= 128 for key in DURATION_STRATA
        ),
        "boundary_count_at_least_96": all(
            boundary_counts[key] >= 96 for key in BOUNDARY_STRATA
        ),
        "padded_window_quota_per_duration_stratum": all(
            padded_windows[key] >= MIN_PADDED_WINDOWS_PER_STRATUM
            for key in DURATION_STRATA
        ),
        "original_annotation_boundary_source": all(
            row.get("boundary_source")
            == "original_uncropped_annotation"
            for row in rows
        ),
        "sign_point_at_least_060": sign_point >= 0.60,
        "sign_ci_lower_above_050": sign_ci[0] > 0.50,
        "spearman_point_at_least_020": spearman_point >= 0.20,
        "spearman_ci_lower_above_zero": spearman_ci[0] > 0.0,
        "predicted_best_median_below_zero": best_point < 0.0,
        "predicted_best_ci_upper_below_zero": best_ci[1] < 0.0,
        "predicted_best_boundary_gain_above_zero": (
            best_boundary_gain_point > 0.0
        ),
        "predicted_best_boundary_gain_ci_lower_above_zero": (
            best_boundary_gain_ci[0] > 0.0
        ),
        "duration_strata_not_reversed": all(
            duration[key]["sign_agreement"] >= 0.55
            and duration[key]["window_spearman_median"] >= 0.0
            for key in DURATION_STRATA
        ),
        "finite": finite_all,
        "restoration_mismatch_zero": restoration_mismatches == 0,
        "repeated_base_error_at_most_1e6": repeated_base_max_error <= 1.0e-6,
    }
    return {
        "schema": "duca_protected_physical_p3_aggregate_v1",
        "ok": all(checks.values()),
        "preregistered_count": len(rows),
        "effective_count": len(effective),
        "excluded_count": len(rows) - len(effective),
        "sign_agreement": {"point": sign_point, "ci95": sign_ci},
        "window_spearman_median": {
            "point": spearman_point,
            "ci95": spearman_ci,
        },
        "predicted_best_actual_delta_median": {
            "point": best_point,
            "ci95": best_ci,
        },
        "predicted_best_boundary_gain_seconds_median": {
            "point": best_boundary_gain_point,
            "ci95": best_boundary_gain_ci,
        },
        "duration_strata": duration,
        "boundary_strata_effective_counts": {
            key: int(boundary_counts[key]) for key in BOUNDARY_STRATA
        },
        "padded_windows_per_duration_stratum": {
            key: int(padded_windows[key]) for key in DURATION_STRATA
        },
        "physical_violation_count": int(violations),
        "restoration_mismatch_count": int(restoration_mismatches),
        "repeated_base_loss_max_abs_error": repeated_base_max_error,
        "bootstrap": {
            "unit": "video_cluster",
            "replicates": int(bootstrap_replicates),
            "seed": int(bootstrap_seed),
            "ci": "two_sided_percentile_95",
        },
        "checks": checks,
    }


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "BOUNDARY_STRATA",
    "DURATION_STRATA",
    "MIN_PADDED_WINDOWS_PER_STRATUM",
    "aggregate_p3_rows",
    "boundary_distance_stratum",
    "deterministic_quartile_swaps",
    "duration_stratum",
    "legal_single_swaps",
    "stratified_window_manifest",
]
