from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from tools.bata.duca_allocation_families import exact_uniform_positions
from tools.bata.bootstrap_duca_h65_official_map import bootstrap_h65_official_map, exact_interval
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
            "thread": 16,
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
        )
        low_result = bootstrap_h65_official_map(
            predictions["low"], cfg(low_annotation), baseline_family="gate_zero",
            nonce=nonce, namespace="DISTORTION_LOW_Q1_V1", workers=workers,
        )
        high_result = bootstrap_h65_official_map(
            predictions["high"], cfg(high_annotation), baseline_family="gate_zero",
            nonce=nonce, namespace="DISTORTION_HIGH_Q4_V1", workers=workers,
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
    evaluate.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "freeze":
        payload = freeze_training_strata(
            training_identity_path=args.training_identity,
            annotation_path=args.annotation,
        )
    else:
        payload = evaluate_strata(
            frozen_path=args.frozen,
            validation_identity_path=args.validation_identity,
            annotation_path=args.annotation,
            on_prediction_path=args.on_prediction,
            gate_zero_prediction_path=args.gate_zero_prediction,
            nonce=args.nonce,
            workers=args.workers,
        )
    atomic_write_json(args.output, payload)


if __name__ == "__main__":
    main()
