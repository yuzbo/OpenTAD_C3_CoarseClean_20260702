#!/usr/bin/env python3
"""Reconstruct the frozen TAR32 short-action and boundary diagnostics.

This tool is deliberately evaluation-only.  It consumes one frozen R1
prediction and one terminal TAR32 prediction, applies the existing OpenTAD AP
primitive to the canonical short-action subset, and reconstructs the
pre-registered median absolute boundary errors with the v004 matching rule.
It does not choose a route or authorize a cost run.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import statistics
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from opentad.evaluations.builder import remove_duplicate_annotations
from opentad.evaluations.mAP import compute_average_precision_detection, mAP


SHORT_ACTION_MAX_SECONDS = 5.0
BOUNDARY_MATCH_TIOU = 0.50
SHORT_ACTION_MAX_DECREASE_PP = 1.50
BOUNDARY_MAX_WORSENING_RATIO = 1.10
SHORT_ACTION_TIOU_THRESHOLDS = np.asarray([0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _finite_float(value: Any, *, field: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return number


def _segment(value: Any, *, field: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"{field} must be a two-value segment")
    return (
        _finite_float(value[0], field=f"{field}[0]"),
        _finite_float(value[1], field=f"{field}[1]"),
    )


def _validate_annotation(annotation: Mapping[str, Any]) -> None:
    database = annotation.get("database")
    if not isinstance(database, Mapping):
        raise ValueError("ground truth must contain a database object")
    for video_id, video in database.items():
        if not isinstance(video, Mapping):
            raise ValueError(f"database[{video_id!r}] must be an object")
        annotations = video.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"database[{video_id!r}].annotations must be a list")
        for index, row in enumerate(annotations):
            if not isinstance(row, Mapping) or not isinstance(row.get("label"), str):
                raise ValueError(f"annotation {video_id}:{index} must contain a string label")
            _segment(row.get("segment"), field=f"annotation {video_id}:{index}.segment")


def _validate_prediction(prediction: Mapping[str, Any]) -> None:
    results = prediction.get("results")
    if not isinstance(results, Mapping):
        raise ValueError("prediction must contain a results object")
    for video_id, rows in results.items():
        if not isinstance(rows, list):
            raise ValueError(f"results[{video_id!r}] must be a list")
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping) or not isinstance(row.get("label"), str):
                raise ValueError(f"prediction {video_id}:{index} must contain a string label")
            _segment(row.get("segment"), field=f"prediction {video_id}:{index}.segment")
            _finite_float(row.get("score"), field=f"prediction {video_id}:{index}.score")


def short_action_annotation(
    annotation: Mapping[str, Any], *, subset: str
) -> dict[str, Any]:
    """Return the canonical ``0 < duration <= 5s`` subset used by v004."""

    filtered = copy.deepcopy(dict(annotation))
    database = filtered["database"]
    for video in database.values():
        if video.get("subset") != subset:
            continue
        rows = remove_duplicate_annotations(video.get("annotations", []))
        video["annotations"] = [
            row
            for row in rows
            if row.get("label") != "Ambiguous"
            and 0.0
            < _finite_float(row["segment"][1], field="annotation end")
            - _finite_float(row["segment"][0], field="annotation start")
            <= SHORT_ACTION_MAX_SECONDS
        ]
    return filtered


def serial_short_action_map(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    subset: str,
) -> dict[str, Any]:
    """Use OpenTAD's official AP primitive without multiprocessing."""

    filtered = short_action_annotation(annotation, subset=subset)
    with tempfile.TemporaryDirectory(prefix="zoomtoken-tar32-short-") as directory:
        ground_truth_path = Path(directory) / "short_action_ground_truth.json"
        ground_truth_path.write_text(
            json.dumps(filtered, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
        evaluator = mAP(
            ground_truth_filename=str(ground_truth_path),
            prediction_filename=dict(prediction),
            subset=subset,
            tiou_thresholds=SHORT_ACTION_TIOU_THRESHOLDS,
            thread=1,
        )

    if not evaluator.activity_index:
        raise ValueError("short-action subset contains no evaluable activity class")

    ap = np.zeros((len(SHORT_ACTION_TIOU_THRESHOLDS), len(evaluator.activity_index)), dtype=float)
    for class_index in evaluator.activity_index.values():
        gt_rows = evaluator.ground_truth.loc[
            evaluator.ground_truth["label"] == class_index
        ].reset_index(drop=True)
        pred_rows = evaluator.prediction.loc[
            evaluator.prediction["label"] == class_index
        ].reset_index(drop=True)
        ap[:, class_index] = compute_average_precision_detection(
            gt_rows,
            pred_rows,
            tiou_thresholds=SHORT_ACTION_TIOU_THRESHOLDS,
        )

    map_values = ap.mean(axis=1)
    return {
        "definition_seconds": "0 < duration <= 5.0",
        "evaluator": "OpenTAD compute_average_precision_detection",
        "class_count": len(evaluator.activity_index),
        "ground_truth_count": int(len(evaluator.ground_truth)),
        "prediction_count": int(len(evaluator.prediction)),
        "average_mAP": float(map_values.mean()),
        "mAP_by_tiou": {
            f"{threshold:.1f}": float(value)
            for threshold, value in zip(SHORT_ACTION_TIOU_THRESHOLDS, map_values)
        },
    }


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0.0 else 0.0


def boundary_diagnostics(
    annotation: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    subset: str,
) -> dict[str, Any]:
    """Reconstruct median absolute endpoint errors under the frozen v004 match."""

    ground_truth: list[dict[str, Any]] = []
    for video_id, video in annotation["database"].items():
        if video.get("subset") != subset:
            continue
        for index, item in enumerate(video.get("annotations", [])):
            if item.get("label") == "Ambiguous":
                continue
            segment = _segment(item["segment"], field=f"annotation {video_id}:{index}.segment")
            duration = segment[1] - segment[0]
            if duration <= 0.0:
                continue
            ground_truth.append(
                {
                    "id": f"{video_id}:{index}",
                    "video": str(video_id),
                    "label": str(item["label"]),
                    "segment": segment,
                    "duration": duration,
                }
            )

    candidates: list[dict[str, Any]] = []
    for video_id, rows in prediction["results"].items():
        for index, item in enumerate(rows):
            candidates.append(
                {
                    "id": f"{video_id}:{index}",
                    "video": str(video_id),
                    "label": str(item["label"]),
                    "segment": _segment(
                        item["segment"], field=f"prediction {video_id}:{index}.segment"
                    ),
                    "score": _finite_float(
                        item["score"], field=f"prediction {video_id}:{index}.score"
                    ),
                }
            )

    matched_ground_truth: set[str] = set()
    matches: list[tuple[dict[str, Any], dict[str, Any], float]] = []
    for candidate in sorted(candidates, key=lambda row: (-row["score"], row["id"])):
        available = [
            (temporal_iou(candidate["segment"], gt["segment"]), gt)
            for gt in ground_truth
            if gt["id"] not in matched_ground_truth
            and gt["video"] == candidate["video"]
            and gt["label"] == candidate["label"]
        ]
        available = [row for row in available if row[0] >= BOUNDARY_MATCH_TIOU]
        if not available:
            continue
        overlap, gt = max(available, key=lambda row: (row[0], row[1]["id"]))
        matched_ground_truth.add(gt["id"])
        matches.append((candidate, gt, overlap))

    if not matches:
        raise ValueError("no boundary match at the frozen tIoU >= 0.50 rule")

    start_errors: list[float] = []
    end_errors: list[float] = []
    short_start_errors: list[float] = []
    short_end_errors: list[float] = []
    for candidate, gt, _overlap in matches:
        start_error = abs(candidate["segment"][0] - gt["segment"][0]) / gt["duration"]
        end_error = abs(candidate["segment"][1] - gt["segment"][1]) / gt["duration"]
        start_errors.append(start_error)
        end_errors.append(end_error)
        if gt["duration"] <= SHORT_ACTION_MAX_SECONDS:
            short_start_errors.append(start_error)
            short_end_errors.append(end_error)

    short_ground_truth_count = sum(
        0.0 < gt["duration"] <= SHORT_ACTION_MAX_SECONDS for gt in ground_truth
    )
    return {
        "matching": "score_greedy_same_class_tiou_at_least_0.50",
        "normalization": "absolute_endpoint_error_divided_by_ground_truth_duration",
        "ground_truth_count": len(ground_truth),
        "matched_count": len(matches),
        "median_abs_start_error_normalized": float(statistics.median(start_errors)),
        "median_abs_end_error_normalized": float(statistics.median(end_errors)),
        "mean_abs_start_error_normalized": float(statistics.fmean(start_errors)),
        "mean_abs_end_error_normalized": float(statistics.fmean(end_errors)),
        "short_action": {
            "definition_seconds": "0 < duration <= 5.0",
            "ground_truth_count": short_ground_truth_count,
            "matched_count": len(short_start_errors),
            "median_abs_start_error_normalized": (
                float(statistics.median(short_start_errors)) if short_start_errors else None
            ),
            "median_abs_end_error_normalized": (
                float(statistics.median(short_end_errors)) if short_end_errors else None
            ),
        },
    }


def compare_frozen_guards(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    reference_short = float(reference["short_action_mAP"]["average_mAP"])
    candidate_short = float(candidate["short_action_mAP"]["average_mAP"])
    short_decrease_pp = (reference_short - candidate_short) * 100.0

    comparison: dict[str, Any] = {
        "short_action_mAP_decrease_pp": short_decrease_pp,
        "short_action_max_decrease_pp": SHORT_ACTION_MAX_DECREASE_PP,
        "short_action_guard_passed": short_decrease_pp <= SHORT_ACTION_MAX_DECREASE_PP
        or math.isclose(
            short_decrease_pp,
            SHORT_ACTION_MAX_DECREASE_PP,
            rel_tol=0.0,
            abs_tol=1e-12,
        ),
        "boundary_max_worsening_ratio": BOUNDARY_MAX_WORSENING_RATIO,
    }
    for endpoint in ("start", "end"):
        key = f"median_abs_{endpoint}_error_normalized"
        reference_value = float(reference["boundary"][key])
        candidate_value = float(candidate["boundary"][key])
        if reference_value <= 0.0:
            raise ValueError(f"reference {key} must be positive for the relative frozen guard")
        ratio = candidate_value / reference_value
        comparison[f"boundary_{endpoint}_worsening_ratio"] = ratio
        comparison[f"boundary_{endpoint}_guard_passed"] = (
            ratio <= BOUNDARY_MAX_WORSENING_RATIO
            or math.isclose(
                ratio,
                BOUNDARY_MAX_WORSENING_RATIO,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )

    comparison["reconstructed_guards_passed"] = all(
        comparison[key]
        for key in (
            "short_action_guard_passed",
            "boundary_start_guard_passed",
            "boundary_end_guard_passed",
        )
    )
    return comparison


def evaluate_pair(
    annotation: Mapping[str, Any],
    reference_prediction: Mapping[str, Any],
    candidate_prediction: Mapping[str, Any],
    *,
    subset: str = "validation",
) -> dict[str, Any]:
    _validate_annotation(annotation)
    _validate_prediction(reference_prediction)
    _validate_prediction(candidate_prediction)

    arms = {}
    for name, prediction in (
        ("reference_r1", reference_prediction),
        ("candidate_tar32", candidate_prediction),
    ):
        arms[name] = {
            "short_action_mAP": serial_short_action_map(
                annotation, prediction, subset=subset
            ),
            "boundary": boundary_diagnostics(
                annotation, prediction, subset=subset
            ),
        }

    return {
        "schema": "zoomtoken-tar32-terminal-diagnostics-v1",
        "evidence_type": "reconstructed_from_frozen_predictions_and_canonical_annotation",
        "subset": subset,
        "arms": arms,
        "comparison": compare_frozen_guards(
            arms["reference_r1"], arms["candidate_tar32"]
        ),
        "scope": {
            "official_full_accuracy_metrics_included": False,
            "cost_authorized": False,
            "route_decision_authorized": False,
            "fresh_pro_review_required": True,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--reference-prediction", required=True, type=Path)
    parser.add_argument("--candidate-prediction", required=True, type=Path)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    annotation = _load_json(args.ground_truth)
    reference_prediction = _load_json(args.reference_prediction)
    candidate_prediction = _load_json(args.candidate_prediction)
    report = evaluate_pair(
        annotation,
        reference_prediction,
        candidate_prediction,
        subset=args.subset,
    )
    report["inputs"] = {
        "ground_truth": {
            "path": str(args.ground_truth),
            "sha256": _sha256_file(args.ground_truth),
        },
        "reference_prediction": {
            "path": str(args.reference_prediction),
            "sha256": _sha256_file(args.reference_prediction),
        },
        "candidate_prediction": {
            "path": str(args.candidate_prediction),
            "sha256": _sha256_file(args.candidate_prediction),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
