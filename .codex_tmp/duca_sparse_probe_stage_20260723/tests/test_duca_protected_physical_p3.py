from __future__ import annotations

import numpy as np

from tools.bata.duca_protected_physical_p3 import (
    DURATION_STRATA,
    aggregate_p3_rows,
    deterministic_quartile_swaps,
    legal_single_swaps,
    stratified_window_manifest,
)


class _SyntheticP3Dataset:
    snippet_stride = 4
    offset_frames = 0

    def __init__(self):
        self.data_list = []
        durations = {"short": 1.0, "medium": 4.0, "long": 12.0}
        for stratum in DURATION_STRATA:
            for index in range(24):
                action_duration = durations[stratum]
                video = f"{stratum}_video_{index:02d}"
                valid_len = 640 if index % 2 else 768
                centers = np.arange(valid_len, dtype=np.int64) * 4
                video_info = {
                    "frame": 4000,
                    "duration": 160.0,
                    "annotations": [
                        {
                            "label": "action",
                            "segment": [2.0, 2.0 + action_duration],
                        }
                    ],
                }
                annotation = {
                    "gt_segments": np.asarray(
                        [[50.0, 50.0 + action_duration * 25.0]],
                        dtype=np.float32,
                    )
                }
                self.data_list.append(
                    [video, video_info, annotation, centers]
                )


def _brute_legal(selected, axis, cap):
    selected = list(selected)
    output = set()
    for removed in selected:
        for incoming in range(len(axis)):
            if incoming in selected:
                continue
            candidate = sorted(
                (set(selected) - {removed}) | {incoming}
            )
            intervals = [
                axis[candidate[0]] - axis[0],
                axis[-1] - axis[candidate[-1]],
            ]
            intervals.extend(
                axis[right] - axis[left]
                for left, right in zip(candidate[:-1], candidate[1:])
            )
            if max(intervals) <= cap + 1.0e-12:
                output.add((removed, incoming))
    return output


def test_p3_window_population_is_deterministic_and_stratified():
    dataset = _SyntheticP3Dataset()
    first = stratified_window_manifest(dataset)
    second = stratified_window_manifest(dataset)
    assert first == second
    assert len(first) == 48
    assert {
        stratum: sum(row["duration_stratum"] == stratum for row in first)
        for stratum in DURATION_STRATA
    } == {"short": 16, "medium": 16, "long": 16}
    assert max(
        sum(other["video_id"] == row["video_id"] for other in first)
        for row in first
    ) <= 4
    assert all(
        sum(
            row["duration_stratum"] == stratum
            and row["window_kind"] == "padded"
            for row in first
        )
        >= 4
        for stratum in DURATION_STRATA
    )
    assert all(
        row["boundary_source"] == "original_uncropped_annotation"
        for row in first
    )


def test_p3_window_population_records_full_and_padded_valid_lengths():
    dataset = _SyntheticP3Dataset()
    manifest = stratified_window_manifest(dataset)
    assert {row["window_kind"] for row in manifest} == {"full", "padded"}
    assert all(384 < row["valid_len"] <= 768 for row in manifest)
    assert all(
        row["window_kind"] == ("full" if row["valid_len"] == 768 else "padded")
        for row in manifest
    )


def test_legal_single_swaps_matches_brute_force_physical_cap():
    axis = np.asarray([0.0, 0.7, 1.4, 2.5, 3.1, 4.2, 5.0])
    selected = [0, 2, 4, 6]
    cap = 2.0
    observed = set(
        legal_single_swaps(selected, axis, len(axis), cap)
    )
    expected = _brute_legal(selected, axis, cap)
    assert observed == expected


def test_quartile_swap_sampling_is_deterministic_and_balanced():
    axis = np.arange(20, dtype=np.float64)
    selected = list(range(0, 20, 2))
    legal = legal_single_swaps(selected, axis, len(axis), 20.0)
    gradient = np.linspace(-1.0, 1.0, len(axis))
    first = deterministic_quartile_swaps(
        legal,
        gradient,
        video_id="video",
        window_start=0,
    )
    second = deterministic_quartile_swaps(
        legal,
        gradient,
        video_id="video",
        window_start=0,
    )
    assert first == second
    assert len(first) == 12
    assert [sum(row["quartile"] == q for row in first) for q in range(4)] == [
        3,
        3,
        3,
        3,
    ]


def _passing_rows():
    rows = []
    for window_index in range(48):
        duration = DURATION_STRATA[window_index // 16]
        for swap_index in range(12):
            predicted = float(swap_index - 5.5)
            rows.append(
                {
                    "video_id": f"video_{window_index:02d}",
                    "window_start": window_index * 10,
                    "duration_stratum": duration,
                    "window_kind": (
                        "padded" if window_index % 16 < 4 else "full"
                    ),
                    "boundary_source": "original_uncropped_annotation",
                    "boundary_distance_stratum": (
                        "near",
                        "mid",
                        "far",
                    )[swap_index % 3],
                    "predicted_delta": predicted,
                    "actual_delta": predicted * 0.1,
                    "predicted_best_quartile": swap_index < 3,
                    "boundary_distance_gain_seconds": (
                        0.5 if swap_index < 3 else -0.1
                    ),
                    "excluded_reason": None,
                    "physical_violation_count": 0,
                    "restoration_mismatch": False,
                    "repeated_base_loss_abs_error": 0.0,
                    "hard_forward_equal": True,
                }
            )
    return rows


def test_p3_aggregate_applies_all_preregistered_thresholds():
    report = aggregate_p3_rows(
        _passing_rows(),
        bootstrap_replicates=32,
        bootstrap_seed=20260720,
    )
    assert report["ok"] is True
    assert report["preregistered_count"] == 576
    assert report["effective_count"] == 576
    assert all(report["checks"].values())


def test_p3_aggregate_keeps_near_zero_rows_in_the_effective_population():
    rows = _passing_rows()
    rows[0]["predicted_delta"] = 0.0
    rows[0]["actual_delta"] = 0.0
    report = aggregate_p3_rows(
        rows,
        bootstrap_replicates=32,
        bootstrap_seed=20260720,
    )
    assert report["effective_count"] == 576
