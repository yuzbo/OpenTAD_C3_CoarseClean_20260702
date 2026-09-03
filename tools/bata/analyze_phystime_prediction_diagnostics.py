from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "phystime_prediction_diagnostic_v1"


def distribution(values):
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


def temporal_iou(segment, candidates):
    segment = np.asarray(segment, dtype=np.float64).reshape(2)
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    if candidates.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    intersection = np.maximum(
        np.minimum(segment[1], candidates[:, 1]) - np.maximum(segment[0], candidates[:, 0]),
        0.0,
    )
    segment_length = max(float(segment[1] - segment[0]), 0.0)
    candidate_lengths = np.maximum(candidates[:, 1] - candidates[:, 0], 0.0)
    union = segment_length + candidate_lengths - intersection
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def _recall(best_ious, thresholds):
    best_ious = np.asarray(best_ious, dtype=np.float64)
    return {
        f"{threshold:.2f}": float((best_ious >= threshold).mean()) if best_ious.size else 0.0
        for threshold in thresholds
    }


def _evaluate_prediction_subset(predictions, ground_truth, thresholds, topk=None):
    class_agnostic_best = []
    class_aware_best = []
    best_localization_label_correct = []
    class_aware_start_error = []
    class_aware_end_error = []
    class_aware_normalized_error = []
    class_aware_match_iou = []
    gt_durations = []

    for video_name, annotations in ground_truth.items():
        rows = sorted(
            predictions.get(video_name, []), key=lambda row: float(row.get("score", 0.0)), reverse=True
        )
        if topk is not None:
            rows = rows[: int(topk)]
        candidate_segments = np.asarray(
            [row["segment"] for row in rows], dtype=np.float64
        ).reshape(-1, 2)
        candidate_labels = [str(row["label"]) for row in rows]

        for annotation in annotations:
            gt_segment = np.asarray(annotation["segment"], dtype=np.float64)
            gt_label = str(annotation["label"])
            duration = max(float(gt_segment[1] - gt_segment[0]), np.finfo(np.float64).eps)
            gt_durations.append(duration)
            ious = temporal_iou(gt_segment, candidate_segments)
            if ious.size:
                best_index = int(np.argmax(ious))
                class_agnostic_best.append(float(ious[best_index]))
                best_localization_label_correct.append(candidate_labels[best_index] == gt_label)
            else:
                class_agnostic_best.append(0.0)
                best_localization_label_correct.append(False)

            matching_indices = [index for index, label in enumerate(candidate_labels) if label == gt_label]
            if matching_indices:
                matching_ious = ious[matching_indices]
                local_index = int(np.argmax(matching_ious))
                prediction_index = matching_indices[local_index]
                class_aware_best.append(float(matching_ious[local_index]))
                prediction_segment = candidate_segments[prediction_index]
                class_aware_match_iou.append(float(matching_ious[local_index]))
                start_error = abs(float(prediction_segment[0] - gt_segment[0]))
                end_error = abs(float(prediction_segment[1] - gt_segment[1]))
                class_aware_start_error.append(start_error)
                class_aware_end_error.append(end_error)
                class_aware_normalized_error.append(0.5 * (start_error + end_error) / duration)
            else:
                class_aware_best.append(0.0)

    boundary_error = {
        "matched_gt_count": len(class_aware_start_error),
        "start_mae": float(np.mean(class_aware_start_error)) if class_aware_start_error else 0.0,
        "end_mae": float(np.mean(class_aware_end_error)) if class_aware_end_error else 0.0,
        "normalized_mean_error": (
            float(np.mean(class_aware_normalized_error)) if class_aware_normalized_error else 0.0
        ),
        "by_min_iou": {},
    }
    for threshold in thresholds:
        indices = [
            index for index, match_iou in enumerate(class_aware_match_iou) if match_iou >= threshold
        ]
        boundary_error["by_min_iou"][f"{threshold:.2f}"] = {
            "matched_gt_count": len(indices),
            "start_mae": (
                float(np.mean([class_aware_start_error[index] for index in indices]))
                if indices
                else 0.0
            ),
            "end_mae": (
                float(np.mean([class_aware_end_error[index] for index in indices]))
                if indices
                else 0.0
            ),
            "normalized_mean_error": (
                float(np.mean([class_aware_normalized_error[index] for index in indices]))
                if indices
                else 0.0
            ),
        }

    return {
        "class_agnostic_recall": _recall(class_agnostic_best, thresholds),
        "class_aware_recall": _recall(class_aware_best, thresholds),
        "class_agnostic_best_iou": distribution(class_agnostic_best),
        "class_aware_best_iou": distribution(class_aware_best),
        "best_localization_label_accuracy": (
            float(np.mean(best_localization_label_correct))
            if best_localization_label_correct
            else 0.0
        ),
        "best_class_aware_boundary_error_sec": boundary_error,
        "_class_agnostic_best": class_agnostic_best,
        "_class_aware_best": class_aware_best,
        "_gt_durations": gt_durations,
    }


def analyze_prediction_dict(
    predictions,
    ground_truth,
    *,
    tiou_thresholds=(0.3, 0.5, 0.7),
    topk_values=(50, 100, 200, 500),
):
    thresholds = tuple(float(value) for value in tiou_thresholds)
    all_result = _evaluate_prediction_subset(predictions, ground_truth, thresholds)
    internal_best_agnostic = all_result.pop("_class_agnostic_best")
    internal_best_aware = all_result.pop("_class_aware_best")
    durations = all_result.pop("_gt_durations")
    report = {
        "video_count": len(ground_truth),
        "gt_count": int(sum(len(rows) for rows in ground_truth.values())),
        "prediction_count": int(sum(len(rows) for rows in predictions.values())),
        "predictions_per_video": distribution(
            [len(predictions.get(video_name, [])) for video_name in ground_truth]
        ),
        "score": distribution(
            [float(row.get("score", 0.0)) for rows in predictions.values() for row in rows]
        ),
        "all_predictions": all_result,
        "best_localization_label_accuracy": all_result["best_localization_label_accuracy"],
        "best_class_aware_boundary_error_sec": all_result[
            "best_class_aware_boundary_error_sec"
        ],
        "topk": {},
    }
    for topk in topk_values:
        result = _evaluate_prediction_subset(predictions, ground_truth, thresholds, topk=topk)
        result.pop("_class_agnostic_best")
        result.pop("_class_aware_best")
        result.pop("_gt_durations")
        report["topk"][str(int(topk))] = result

    duration_edges = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, float("inf"))
    report["duration_bins"] = []
    for lower, upper in zip(duration_edges[:-1], duration_edges[1:]):
        indices = [index for index, duration in enumerate(durations) if lower <= duration < upper]
        report["duration_bins"].append(
            {
                "lower_sec": lower,
                "upper_sec": None if np.isinf(upper) else upper,
                "gt_count": len(indices),
                "class_agnostic_recall": _recall(
                    [internal_best_agnostic[index] for index in indices], thresholds
                ),
                "class_aware_recall": _recall(
                    [internal_best_aware[index] for index in indices], thresholds
                ),
            }
        )
    return report


def load_ground_truth(path, subset):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    database = payload["database"]
    result = {}
    for video_name, video_info in database.items():
        if str(video_info.get("subset")) != str(subset):
            continue
        annotations = [
            {"segment": annotation["segment"], "label": annotation["label"]}
            for annotation in video_info.get("annotations", [])
            if annotation.get("label") != "Ambiguous"
        ]
        if annotations:
            result[video_name] = annotations
    return result


def load_predictions(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload.get("results", payload)


def parse_prediction_argument(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must use NAME=PATH")
    name, path = value.split("=", 1)
    if not name or not path:
        raise argparse.ArgumentTypeError("prediction must use non-empty NAME=PATH")
    return name, path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Decompose TAD prediction localization and classification.")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--prediction", action="append", type=parse_prediction_argument, required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    ground_truth = load_ground_truth(args.ground_truth, args.subset)
    reports = {
        name: analyze_prediction_dict(load_predictions(path), ground_truth)
        for name, path in args.prediction
    }
    output = {
        "schema_version": SCHEMA_VERSION,
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "subset": args.subset,
        "methods": reports,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
