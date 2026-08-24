from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from opentad.evaluations.builder import remove_duplicate_annotations
from tools.bata.duca_allocation_families import exact_uniform_positions
from tools.bata.bootstrap_duca_h65_official_map import (
    bootstrap_h65_official_map,
    exact_interval,
    seed_from_nonce,
)
from tools.bata.duca_p0_evaluation import EXPECTED_TIOU_THRESHOLDS, prediction_results, sha256_file
from tools.bata.duca_p0_training import atomic_write_json


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"expected a JSON object: {path}")
    return payload


def _uniform_positions(length: int, count: int) -> np.ndarray:
    _require(length > 0 and 0 < count <= length, "invalid exact-uniform dimensions")
    return np.asarray(exact_uniform_positions(length, count), dtype=np.int64)


def _record_distortion(record: Mapping[str, Any]) -> float:
    positions = np.asarray(record["selected_positions"], dtype=np.float64)
    length = int(record["dense_valid_len"])
    count = int(record["selected_valid_len"])
    _require(positions.shape == (count,) and count > 0, "identity record positions are malformed")
    canonical = _uniform_positions(length, count).astype(np.float64)
    # A trailing partial tubelet has no Clock support and must not fabricate a
    # time center; the unchanged RGB/mask padding contract still retains it.
    complete_count = count - (count % 2)
    if complete_count == 0:
        return 0.0
    actual_centers = positions[:complete_count].reshape(-1, 2).mean(axis=1)
    canonical_centers = canonical[:complete_count].reshape(-1, 2).mean(axis=1)
    return float(np.mean(np.abs(actual_centers - canonical_centers)) / max(length - 1, 1))


def _annotations(annotation: Mapping[str, Any], subset: str) -> list[Mapping[str, Any]]:
    database = annotation.get("database")
    _require(isinstance(database, Mapping), "annotation has no database mapping")
    rows = []
    for video in database.values():
        if isinstance(video, Mapping) and str(video.get("subset")) == subset:
            anns = video.get("annotations", [])
            _require(isinstance(anns, list), "video annotations must be a list")
            rows.extend(item for item in anns if isinstance(item, Mapping))
    return rows


def freeze_training_strata(
    *, training_identity_path: str | Path, annotation_path: str | Path
) -> dict[str, Any]:
    identity = _load(training_identity_path)
    annotation = _load(annotation_path)
    records = identity.get("records")
    _require(isinstance(records, list) and records, "training identity contains no records")
    by_video: dict[str, list[float]] = {}
    for record in records:
        _require(isinstance(record, Mapping), "identity record must be a mapping")
        video = str(record["video_name"])
        by_video.setdefault(video, []).append(_record_distortion(record))
    database = annotation["database"]
    training_videos = {
        str(video_id)
        for video_id, row in database.items()
        if isinstance(row, Mapping) and str(row.get("subset")) == "training"
    }
    _require(set(by_video) <= training_videos, "training identity contains a non-training video")
    video_distortion = {
        video: float(np.mean(values)) for video, values in sorted(by_video.items())
    }
    distortion_values = np.asarray(list(video_distortion.values()), dtype=np.float64)
    _require(distortion_values.size >= 4, "at least four training videos are required")

    durations = []
    for item in _annotations(annotation, "training"):
        segment = item.get("segment")
        if isinstance(segment, (list, tuple)) and len(segment) == 2:
            duration = float(segment[1]) - float(segment[0])
            if np.isfinite(duration) and duration > 0.0:
                durations.append(duration)
    _require(durations, "training annotation contains no positive action durations")
    q25, q50, q75 = np.quantile(
        distortion_values, (0.25, 0.50, 0.75), method="linear"
    )
    short_q25 = float(
        np.quantile(np.asarray(durations, dtype=np.float64), 0.25, method="linear")
    )
    return {
        "schema_version": "duca_h65_singleclock_training_strata_freeze_v1",
        "source_subset": "training",
        "validation_or_test_used": False,
        "training_identity_path": str(Path(training_identity_path).resolve()),
        "training_identity_sha256": sha256_file(training_identity_path),
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "distortion_definition": "per_video_mean_of_window_mean_abs_tubelet_center_displacement_divided_by_dense_len_minus_one",
        "distortion_quantile_method": "numpy_linear",
        "distortion_q25": float(q25),
        "distortion_q50": float(q50),
        "distortion_q75": float(q75),
        "training_video_distortion": video_distortion,
        "training_video_count": len(video_distortion),
        "short_action_definition": "training_ground_truth_duration_seconds_at_or_below_q25",
        "short_action_quantile_method": "numpy_linear",
        "short_action_duration_q25_seconds": short_q25,
        "training_action_instance_count": len(durations),
    }


def _validation_video_distortion(identity: Mapping[str, Any]) -> dict[str, float]:
    records = identity.get("records")
    _require(isinstance(records, list) and records, "validation identity contains no records")
    by_video: dict[str, list[float]] = {}
    for record in records:
        video = str(record["video_name"])
        by_video.setdefault(video, []).append(_record_distortion(record))
    return {video: float(np.mean(values)) for video, values in sorted(by_video.items())}


def _write_annotation(
    path: Path,
    annotation: Mapping[str, Any],
    *,
    videos: set[str] | None = None,
    max_duration: float | None = None,
) -> None:
    database = annotation["database"]
    output = {}
    for video_id, row in database.items():
        if not isinstance(row, Mapping) or str(row.get("subset")) != "validation":
            continue
        if videos is not None and str(video_id) not in videos:
            continue
        copied = dict(row)
        annotations = list(copied.get("annotations", []))
        if max_duration is not None:
            annotations = [
                item
                for item in annotations
                if isinstance(item, Mapping)
                and isinstance(item.get("segment"), (list, tuple))
                and len(item["segment"]) == 2
                and float(item["segment"][1]) - float(item["segment"][0]) <= max_duration
            ]
        copied["annotations"] = annotations
        output[str(video_id)] = copied
    _require(output, "stratified annotation is empty")
    path.write_text(json.dumps({"database": output}, sort_keys=True), encoding="utf-8")


def _write_predictions(path: Path, source: str | Path, videos: set[str] | None) -> None:
    results = prediction_results(source)
    if videos is not None:
        results = {video: rows for video, rows in results.items() if video in videos}
    _require(results, "stratified prediction is empty")
    path.write_text(json.dumps({"results": results}, sort_keys=True), encoding="utf-8")


def _delta_row(bootstrap: Mapping[str, Any], family: str, metric: str) -> dict[str, float]:
    row = bootstrap["comparisons"][family][metric]
    return {
        "point_pp": float(
            (
                bootstrap["point_estimates"][family][metric]
                - bootstrap["point_estimates"][bootstrap["baseline_family"]][metric]
            )
            * 100.0
        ),
        "ci_lower_pp": float(row["ci_lower_exact_rank"] * 100.0),
        "ci_upper_pp": float(row["ci_upper_exact_rank"] * 100.0),
        "delta_samples": row["delta_samples"],
    }


def evaluate_strata(
    *,
    frozen_path: str | Path,
    validation_identity_path: str | Path,
    annotation_path: str | Path,
    on_prediction_path: str | Path,
    gate_zero_prediction_path: str | Path,
    nonce: str,
    workers: int,
    evaluator_thread: int = 16,
    chunksize: int = 1,
) -> dict[str, Any]:
    frozen = _load(frozen_path)
    identity = _load(validation_identity_path)
    annotation = _load(annotation_path)
    _require(
        frozen.get("schema_version") == "duca_h65_singleclock_training_strata_freeze_v1"
        and frozen.get("validation_or_test_used") is False,
        "strata freeze is not training-only",
    )
    video_distortion = _validation_video_distortion(identity)
    low = {
        video for video, value in video_distortion.items()
        if value <= float(frozen["distortion_q25"])
    }
    high = {
        video for video, value in video_distortion.items()
        if value >= float(frozen["distortion_q75"])
    }
    _require(low and high and low.isdisjoint(high), "validation distortion strata are invalid")

    def cfg(path: Path) -> dict[str, Any]:
        return {
            "type": "mAP",
            "ground_truth_filename": str(path),
            "subset": "validation",
            "tiou_thresholds": EXPECTED_TIOU_THRESHOLDS,
            "top_k": None,
            "blocked_videos": None,
            "thread": int(evaluator_thread),
        }

    with tempfile.TemporaryDirectory(prefix="duca-h65-singleclock-strata-") as directory:
        root = Path(directory)
        predictions = {}
        for stratum, videos in (("all", None), ("low", low), ("high", high)):
            predictions[stratum] = {}
            for family, source in (("on", on_prediction_path), ("gate_zero", gate_zero_prediction_path)):
                path = root / f"{stratum}_{family}.json"
                _write_predictions(path, source, videos)
                predictions[stratum][family] = path
        short_annotation = root / "short_annotation.json"
        low_annotation = root / "low_annotation.json"
        high_annotation = root / "high_annotation.json"
        _write_annotation(
            short_annotation,
            annotation,
            max_duration=float(frozen["short_action_duration_q25_seconds"]),
        )
        _write_annotation(low_annotation, annotation, videos=low)
        _write_annotation(high_annotation, annotation, videos=high)
        short = bootstrap_h65_official_map(
            predictions["all"], cfg(short_annotation), baseline_family="gate_zero",
            nonce=nonce, namespace="SHORT_ACTION_Q25_V1", workers=workers,
            chunksize=chunksize,
        )
        low_result = bootstrap_h65_official_map(
            predictions["low"], cfg(low_annotation), baseline_family="gate_zero",
            nonce=nonce, namespace="DISTORTION_LOW_Q1_V1", workers=workers,
            chunksize=chunksize,
        )
        high_result = bootstrap_h65_official_map(
            predictions["high"], cfg(high_annotation), baseline_family="gate_zero",
            nonce=nonce, namespace="DISTORTION_HIGH_Q4_V1", workers=workers,
            chunksize=chunksize,
        )

    short_row = _delta_row(short, "on", "average_mAP")
    low_row = _delta_row(low_result, "on", "average_mAP")
    high_row = _delta_row(high_result, "on", "average_mAP")
    interaction = np.asarray(high_row.pop("delta_samples")) - np.asarray(low_row.pop("delta_samples"))
    interaction_lower, interaction_upper = exact_interval(interaction, lower_rank=250, upper_rank=9750)
    short_row.pop("delta_samples")
    return {
        "schema_version": "duca_h65_singleclock_strata_v1",
        "primary_checkpoint_state_key": "state_dict_ema",
        "training_freeze_path": str(Path(frozen_path).resolve()),
        "training_freeze_sha256": sha256_file(frozen_path),
        "validation_identity_path": str(Path(validation_identity_path).resolve()),
        "validation_identity_sha256": sha256_file(validation_identity_path),
        "validation_or_test_used_for_cutpoints": False,
        "short_action_delta_pp": short_row["point_pp"],
        "short_action": short_row,
        "distortion_low": low_row,
        "distortion_high": high_row,
        "distortion_interaction_point_pp": high_row["point_pp"] - low_row["point_pp"],
        "distortion_interaction_ci_lower_pp": float(interaction_lower * 100.0),
        "distortion_interaction_ci_upper_pp": float(interaction_upper * 100.0),
        "low_distortion_videos": sorted(low),
        "high_distortion_videos": sorted(high),
        "all_predictions_retained_as_potential_false_positives_for_short_action": True,
        "official_evaluator_reexecuted_per_resample": True,
        "samples": 10000,
        "nonce": nonce,
        "bootstrap_execution": {
            "workers": int(workers),
            "evaluator_thread": int(evaluator_thread),
            "chunksize": int(chunksize),
        },
    }


def _gap_cv(record: Mapping[str, Any]) -> float:
    positions = np.asarray(record["selected_positions"], dtype=np.float64)
    count = int(record["selected_valid_len"])
    _require(positions.shape == (count,), "identity record positions are malformed")
    complete_count = count - (count % 2)
    _require(complete_count >= 4, "gap-CV needs at least two complete tubelets")
    centers = positions[:complete_count].reshape(-1, 2).mean(axis=1)
    gaps = np.diff(centers)
    _require(gaps.size > 0 and bool(np.all(gaps > 0.0)), "tubelet centers must be strictly increasing")
    mean = float(np.mean(gaps, dtype=np.float64))
    _require(mean > 0.0, "mean physical tubelet gap must be positive")
    return float(np.std(gaps, ddof=0, dtype=np.float64) / mean)


def _window_records(
    identity: Mapping[str, Any], ledger: Mapping[str, Any], *, expected_subset: str
) -> list[dict[str, Any]]:
    _require(
        ledger.get("schema_version") == "duca_h65_physical_window_ledger_v1",
        "physical window ledger schema mismatch",
    )
    _require(ledger.get("subset") == expected_subset, "physical window ledger subset mismatch")
    identity_rows = identity.get("records")
    ledger_rows = ledger.get("records")
    _require(isinstance(identity_rows, list) and identity_rows, "identity contains no records")
    _require(isinstance(ledger_rows, list) and ledger_rows, "physical ledger contains no records")
    by_id = {str(row.get("sample_id")): row for row in ledger_rows if isinstance(row, Mapping)}
    _require(len(by_id) == len(ledger_rows), "physical ledger sample IDs are not unique")
    output = []
    for identity_row in identity_rows:
        _require(isinstance(identity_row, Mapping), "identity record must be a mapping")
        sample_id = str(identity_row.get("sample_id"))
        _require(sample_id in by_id, f"physical ledger misses identity sample {sample_id}")
        ledger_row = by_id[sample_id]
        _require(
            str(ledger_row.get("video_name")) == str(identity_row.get("video_name")),
            "identity/ledger video mismatch",
        )
        start = float(ledger_row["valid_start_seconds"])
        end = float(ledger_row["valid_end_seconds"])
        _require(np.isfinite(start) and np.isfinite(end) and end > start, "invalid physical support")
        output.append(
            {
                "sample_id": sample_id,
                "video_name": str(identity_row["video_name"]),
                "valid_start_seconds": start,
                "valid_end_seconds": end,
                "is_final_valid_window": bool(ledger_row.get("is_final_valid_window", False)),
                "gap_cv": _gap_cv(identity_row),
            }
        )
    _require(set(by_id) == {row["sample_id"] for row in output}, "ledger has records not bound to identity")
    return sorted(output, key=lambda row: (row["video_name"], row["valid_start_seconds"], row["sample_id"]))


def _video_annotations(
    annotation: Mapping[str, Any], *, subset: str
) -> dict[str, list[dict[str, Any]]]:
    database = annotation.get("database")
    _require(isinstance(database, Mapping), "annotation has no database mapping")
    output: dict[str, list[dict[str, Any]]] = {}
    for video_name, video in sorted(database.items()):
        if not isinstance(video, Mapping) or str(video.get("subset")) != subset:
            continue
        rows = []
        annotations = video.get("annotations", [])
        _require(isinstance(annotations, list), "video annotations must be a list")
        annotations = remove_duplicate_annotations(annotations)
        sortable = []
        for source_index, item in enumerate(annotations):
            if not isinstance(item, Mapping):
                continue
            segment = item.get("segment")
            if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                continue
            start, end = float(segment[0]), float(segment[1])
            label = str(item.get("label"))
            if np.isfinite(start) and np.isfinite(end) and end > start and label:
                sortable.append((start, end, label, source_index))
        for occurrence_index, (start, end, label, source_index) in enumerate(sorted(sortable)):
            rows.append(
                {
                    "video_name": str(video_name),
                    "label": label,
                    "start": start,
                    "end": end,
                    "canonical_occurrence_index": occurrence_index,
                    "source_index": source_index,
                }
            )
        output[str(video_name)] = rows
    return output


def _boundary_in_window(boundary: float, window: Mapping[str, Any]) -> bool:
    start = float(window["valid_start_seconds"])
    end = float(window["valid_end_seconds"])
    return bool(
        start <= boundary < end
        or (window["is_final_valid_window"] and boundary == end)
    )


def _add_boundary_density(
    windows: list[dict[str, Any]], annotations: Mapping[str, list[Mapping[str, Any]]]
) -> None:
    for window in windows:
        boundaries = 0
        for item in annotations.get(window["video_name"], []):
            boundaries += int(_boundary_in_window(float(item["start"]), window))
            boundaries += int(_boundary_in_window(float(item["end"]), window))
        duration = window["valid_end_seconds"] - window["valid_start_seconds"]
        window["boundary_density_per_second"] = float(boundaries / duration)


def freeze_boundary_risk_strata(
    *,
    training_identity_path: str | Path,
    training_window_ledger_path: str | Path,
    annotation_path: str | Path,
) -> dict[str, Any]:
    identity = _load(training_identity_path)
    ledger = _load(training_window_ledger_path)
    annotation = _load(annotation_path)
    annotations = _video_annotations(annotation, subset="training")
    windows = _window_records(identity, ledger, expected_subset="training")
    _add_boundary_density(windows, annotations)
    gap_values = np.asarray([row["gap_cv"] for row in windows], dtype=np.float64)
    density_values = np.asarray(
        [row["boundary_density_per_second"] for row in windows], dtype=np.float64
    )
    _require(gap_values.size >= 4, "at least four training windows are required")
    gap_q25, gap_q50, gap_q75 = np.quantile(
        gap_values, (0.25, 0.50, 0.75), method="linear"
    )
    bd_q25, bd_q50, bd_q75 = np.quantile(
        density_values, (0.25, 0.50, 0.75), method="linear"
    )
    return {
        "schema_version": "duca_h65_singleclock_boundary_risk_freeze_v1",
        "source_subset": "training",
        "validation_or_test_used": False,
        "training_identity_path": str(Path(training_identity_path).resolve()),
        "training_identity_sha256": sha256_file(training_identity_path),
        "training_window_ledger_path": str(Path(training_window_ledger_path).resolve()),
        "training_window_ledger_sha256": sha256_file(training_window_ledger_path),
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "gap_cv_definition": "population_std_over_mean_of_complete_physical_tubelet_center_gaps",
        "boundary_density_definition": "original_gt_start_end_boundaries_in_valid_half_open_window_per_valid_second",
        "quantile_method": "numpy_linear",
        "gap_cv_q25": float(gap_q25),
        "gap_cv_q50": float(gap_q50),
        "gap_cv_q75": float(gap_q75),
        "boundary_density_q25": float(bd_q25),
        "boundary_density_q50": float(bd_q50),
        "boundary_density_q75": float(bd_q75),
        "training_window_count": len(windows),
    }


def _iou(lhs: tuple[float, float], rhs: tuple[float, float]) -> float:
    intersection = max(0.0, min(lhs[1], rhs[1]) - max(lhs[0], rhs[0]))
    union = max(lhs[1], rhs[1]) - min(lhs[0], rhs[0])
    return 0.0 if union <= 0.0 else intersection / union


def _match_boundary_errors(
    predictions: Mapping[str, Any],
    annotations: Mapping[str, list[Mapping[str, Any]]],
) -> dict[str, dict[int, float]]:
    output: dict[str, dict[int, float]] = {}
    for video_name, gt_rows in annotations.items():
        output[video_name] = {}
        by_label: dict[str, list[Mapping[str, Any]]] = {}
        for row in gt_rows:
            by_label.setdefault(str(row["label"]), []).append(row)
        prediction_rows = predictions.get(video_name, [])
        _require(isinstance(prediction_rows, list), "prediction video rows must be a list")
        for label, label_gt in by_label.items():
            candidates = []
            for row_index, row in enumerate(prediction_rows):
                if not isinstance(row, Mapping) or str(row.get("label")) != label:
                    continue
                segment = row.get("segment")
                if not isinstance(segment, (list, tuple)) or len(segment) != 2:
                    continue
                start, end = float(segment[0]), float(segment[1])
                score = float(row.get("score", 0.0))
                if np.isfinite(start) and np.isfinite(end) and np.isfinite(score) and end > start:
                    candidates.append(((-score, start, end, row_index), (start, end)))
            candidates.sort(key=lambda item: item[0])
            unmatched = {
                int(row["canonical_occurrence_index"]): row for row in label_gt
            }
            matched: dict[int, tuple[float, float]] = {}
            for _, prediction_segment in candidates:
                ranked = sorted(
                    (
                        (-_iou(prediction_segment, (float(gt["start"]), float(gt["end"]))),
                         float(gt["start"]), float(gt["end"]), occurrence)
                        for occurrence, gt in unmatched.items()
                    )
                )
                if not ranked or -ranked[0][0] < 0.5:
                    continue
                occurrence = ranked[0][3]
                matched[occurrence] = prediction_segment
                unmatched.pop(occurrence)
            for gt in label_gt:
                occurrence = int(gt["canonical_occurrence_index"])
                prediction_segment = matched.get(occurrence)
                if prediction_segment is None:
                    output[video_name][occurrence] = 1.0
                    continue
                duration = float(gt["end"]) - float(gt["start"])
                start_error = min(1.0, abs(prediction_segment[0] - float(gt["start"])) / duration)
                end_error = min(1.0, abs(prediction_segment[1] - float(gt["end"])) / duration)
                output[video_name][occurrence] = float((start_error + end_error) / 2.0)
    return output


def _stratum_gt_keys(
    annotations: Mapping[str, list[Mapping[str, Any]]],
    windows: list[Mapping[str, Any]],
) -> set[tuple[str, int]]:
    by_video: dict[str, list[Mapping[str, Any]]] = {}
    for window in windows:
        by_video.setdefault(str(window["video_name"]), []).append(window)
    selected = set()
    for video_name, gt_rows in annotations.items():
        for gt in gt_rows:
            if any(
                _boundary_in_window(float(gt["start"]), window)
                or _boundary_in_window(float(gt["end"]), window)
                for window in by_video.get(video_name, [])
            ):
                selected.add((video_name, int(gt["canonical_occurrence_index"])))
    return selected


def _per_video_stratum_error(
    errors: Mapping[str, Mapping[int, float]], keys: set[tuple[str, int]]
) -> dict[str, float]:
    by_video: dict[str, list[float]] = {}
    for video_name, occurrence in sorted(keys):
        _require(
            video_name in errors and occurrence in errors[video_name],
            "boundary errors do not cover frozen stratum GT",
        )
        by_video.setdefault(video_name, []).append(float(errors[video_name][occurrence]))
    return {video: float(np.mean(values, dtype=np.float64)) for video, values in sorted(by_video.items())}


def _paired_video_error_delta(
    on: Mapping[str, float], off: Mapping[str, float], *, nonce: str, namespace: str
) -> dict[str, Any]:
    videos = sorted(set(on) & set(off))
    _require(videos and set(videos) == set(on) == set(off), "paired boundary videos do not align")
    on_values = np.asarray([on[video] for video in videos], dtype=np.float64)
    off_values = np.asarray([off[video] for video in videos], dtype=np.float64)
    point = float(np.mean(on_values) - np.mean(off_values))
    seed, seed_sha256 = seed_from_nonce(nonce, namespace)
    generator = np.random.Generator(np.random.PCG64(seed))
    indices = generator.integers(0, len(videos), size=(10000, len(videos)))
    samples = np.mean(on_values[indices], axis=1) - np.mean(off_values[indices], axis=1)
    lower, upper = exact_interval(samples, lower_rank=250, upper_rank=9750)
    return {
        "point": point,
        "ci_lower_exact_rank_report_only": float(lower),
        "ci_upper_exact_rank_report_only": float(upper),
        "video_count": len(videos),
        "rng": "numpy.random.PCG64",
        "seed_sha256": seed_sha256,
    }


def evaluate_boundary_risk_strata(
    *,
    frozen_path: str | Path,
    validation_identity_path: str | Path,
    validation_window_ledger_path: str | Path,
    annotation_path: str | Path,
    on_prediction_path: str | Path,
    off_prediction_path: str | Path,
    nonce: str,
) -> dict[str, Any]:
    frozen = _load(frozen_path)
    _require(
        frozen.get("schema_version") == "duca_h65_singleclock_boundary_risk_freeze_v1"
        and frozen.get("validation_or_test_used") is False,
        "boundary risk freeze is not training-only",
    )
    identity = _load(validation_identity_path)
    ledger = _load(validation_window_ledger_path)
    annotation = _load(annotation_path)
    annotations = _video_annotations(annotation, subset="validation")
    windows = _window_records(identity, ledger, expected_subset="validation")
    _add_boundary_density(windows, annotations)
    high_gap_windows = [row for row in windows if row["gap_cv"] >= float(frozen["gap_cv_q75"])]
    high_density_windows = [
        row
        for row in windows
        if row["boundary_density_per_second"]
        >= float(frozen["boundary_density_q75"])
    ]
    high_gap_keys = _stratum_gt_keys(annotations, high_gap_windows)
    high_density_keys = _stratum_gt_keys(annotations, high_density_windows)
    missing = []
    if not high_gap_keys:
        missing.append("no_validation_gt_in_high_gapcv_stratum")
    if not high_density_keys:
        missing.append("no_validation_gt_in_high_boundary_density_stratum")
    if missing:
        return {
            "schema_version": "duca_h65_singleclock_boundary_gate_v1",
            "status": "NOT_EVALUABLE_PREEXISTING_ARTIFACT_GAP",
            "comparison": "ema_on_minus_h65_off_ema",
            "missing_artifacts": missing,
            "used_for_decision": False,
            "boundary_mechanism_claim_supported": False,
        }
    on_errors = _match_boundary_errors(prediction_results(on_prediction_path), annotations)
    off_errors = _match_boundary_errors(prediction_results(off_prediction_path), annotations)
    high_gap = _paired_video_error_delta(
        _per_video_stratum_error(on_errors, high_gap_keys),
        _per_video_stratum_error(off_errors, high_gap_keys),
        nonce=nonce,
        namespace="H65_SINGLECLOCK_HIGH_GAPCV_BOUNDARY_V1",
    )
    high_density = _paired_video_error_delta(
        _per_video_stratum_error(on_errors, high_density_keys),
        _per_video_stratum_error(off_errors, high_density_keys),
        nonce=nonce,
        namespace="H65_SINGLECLOCK_HIGH_BOUNDARY_DENSITY_V1",
    )
    return {
        "schema_version": "duca_h65_singleclock_boundary_gate_v1",
        "status": "EVALUABLE",
        "comparison": "ema_on_minus_h65_off_ema",
        "high_gapcv_delta_point": high_gap["point"],
        "high_gapcv_pass": high_gap["point"] <= 0.0,
        "high_gapcv": high_gap,
        "high_boundary_density_delta_point": high_density["point"],
        "high_boundary_density_pass": high_density["point"] <= 0.0,
        "high_boundary_density": high_density,
        "bootstrap_samples": 10000,
        "bootstrap_cluster": "whole_video",
        "ci_role": "report_only",
        "training_freeze_path": str(Path(frozen_path).resolve()),
        "training_freeze_sha256": sha256_file(frozen_path),
        "validation_identity_path": str(Path(validation_identity_path).resolve()),
        "validation_identity_sha256": sha256_file(validation_identity_path),
        "validation_window_ledger_path": str(Path(validation_window_ledger_path).resolve()),
        "validation_window_ledger_sha256": sha256_file(validation_window_ledger_path),
        "annotation_path": str(Path(annotation_path).resolve()),
        "annotation_sha256": sha256_file(annotation_path),
        "on_prediction_path": str(Path(on_prediction_path).resolve()),
        "on_prediction_sha256": sha256_file(on_prediction_path),
        "off_prediction_path": str(Path(off_prediction_path).resolve()),
        "off_prediction_sha256": sha256_file(off_prediction_path),
        "validation_or_test_used_for_cutpoints": False,
        "nonce": nonce,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze/evaluate H65 SingleClock strata")
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--training-identity", required=True)
    freeze.add_argument("--annotation", required=True)
    freeze.add_argument("--output", required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--frozen", required=True)
    evaluate.add_argument("--validation-identity", required=True)
    evaluate.add_argument("--annotation", required=True)
    evaluate.add_argument("--on-prediction", required=True)
    evaluate.add_argument("--gate-zero-prediction", required=True)
    evaluate.add_argument("--nonce", required=True)
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--evaluator-thread", type=int, default=16)
    evaluate.add_argument("--chunksize", type=int, default=1)
    evaluate.add_argument("--output", required=True)
    boundary_freeze = sub.add_parser("boundary-freeze")
    boundary_freeze.add_argument("--training-identity", required=True)
    boundary_freeze.add_argument("--training-window-ledger", required=True)
    boundary_freeze.add_argument("--annotation", required=True)
    boundary_freeze.add_argument("--output", required=True)
    boundary_evaluate = sub.add_parser("boundary-evaluate")
    boundary_evaluate.add_argument("--frozen", required=True)
    boundary_evaluate.add_argument("--validation-identity", required=True)
    boundary_evaluate.add_argument("--validation-window-ledger", required=True)
    boundary_evaluate.add_argument("--annotation", required=True)
    boundary_evaluate.add_argument("--on-prediction", required=True)
    boundary_evaluate.add_argument("--off-prediction", required=True)
    boundary_evaluate.add_argument("--nonce", required=True)
    boundary_evaluate.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "freeze":
        payload = freeze_training_strata(
            training_identity_path=args.training_identity,
            annotation_path=args.annotation,
        )
    elif args.command == "evaluate":
        payload = evaluate_strata(
            frozen_path=args.frozen,
            validation_identity_path=args.validation_identity,
            annotation_path=args.annotation,
            on_prediction_path=args.on_prediction,
            gate_zero_prediction_path=args.gate_zero_prediction,
            nonce=args.nonce,
            workers=args.workers,
            evaluator_thread=args.evaluator_thread,
            chunksize=args.chunksize,
        )
    elif args.command == "boundary-freeze":
        payload = freeze_boundary_risk_strata(
            training_identity_path=args.training_identity,
            training_window_ledger_path=args.training_window_ledger,
            annotation_path=args.annotation,
        )
    else:
        payload = evaluate_boundary_risk_strata(
            frozen_path=args.frozen,
            validation_identity_path=args.validation_identity,
            validation_window_ledger_path=args.validation_window_ledger,
            annotation_path=args.annotation,
            on_prediction_path=args.on_prediction,
            off_prediction_path=args.off_prediction,
            nonce=args.nonce,
        )
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
