"""Frozen source-video boundary diagnostics for ZoomToken development results.

This tool consumes the same ``database``/``results`` JSON schema as OpenTAD's
THUMOS14 mAP evaluator.  It is intentionally offline: training and inference do
not import it, and it never selects a checkpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


GT_BINS = (
    "HIT_070",
    "START_LIMITED",
    "END_LIMITED",
    "EITHER_ENDPOINT_RESCUES",
    "JOINT_BOUNDARY_LIMITED",
    "CLASS_CONFUSION",
    "MISS_OR_SEVERE_LOCALIZATION",
)
FP_BINS = ("DUPLICATE_FP", "CLASS_CONFUSION_FP", "OTHER_FP")


def _load_json(value):
    if isinstance(value, dict):
        return value
    with Path(value).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _canonical_id(record, keys, fallback):
    # Standard OpenTAD JSON has no explicit IDs, so source-array position is its
    # canonical identity.  If an explicit ID is present, it must be well formed.
    for key in keys:
        if key in record:
            value = record[key]
            if isinstance(value, (str, int)) and str(value):
                return str(value)
            raise ValueError(f"malformed canonical ID in field {key!r}")
    return fallback


def _segment(record, *, ground_truth):
    value = record.get("segment")
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("segment must be a two-element list")
    start, end = float(value[0]), float(value[1])
    if not (math.isfinite(start) and math.isfinite(end)):
        raise ValueError("segment endpoints must be finite")
    if ground_truth and end <= start:
        raise ValueError("ground-truth duration must be positive")
    if not ground_truth and end < start:
        raise ValueError("prediction end must not precede start")
    return start, end


def temporal_iou(first, second):
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0.0 else 0.0


def _parse_ground_truth(payload, subset):
    if not isinstance(payload.get("database"), dict):
        raise ValueError("ground truth must contain a database mapping")
    records = []
    seen = {}
    for video_id, video in payload["database"].items():
        if subset is not None and video.get("subset") != subset:
            continue
        annotations = video.get("annotations", [])
        if not isinstance(annotations, list):
            raise ValueError(f"annotations for {video_id!r} must be a list")
        for index, annotation in enumerate(annotations):
            gt_id = _canonical_id(
                annotation,
                ("gt_id", "annotation_id", "id"),
                f"{video_id}:gt:{index}",
            )
            item = {
                "video_id": str(video_id),
                "label": str(annotation["label"]),
                "segment": _segment(annotation, ground_truth=True),
                "id": gt_id,
            }
            previous = seen.get(gt_id)
            if previous is not None:
                if previous != item:
                    raise ValueError(f"canonical GT ID {gt_id!r} is not unique")
                continue
            seen[gt_id] = item
            records.append(item)
    return records


def _parse_predictions(payload):
    if not isinstance(payload.get("results"), dict):
        raise ValueError("predictions must contain a results mapping")
    records = []
    seen = set()
    for video_id, predictions in payload["results"].items():
        if not isinstance(predictions, list):
            raise ValueError(f"predictions for {video_id!r} must be a list")
        for index, prediction in enumerate(predictions):
            prediction_id = _canonical_id(
                prediction,
                ("prediction_id", "id"),
                f"{video_id}:pred:{index}",
            )
            if prediction_id in seen:
                raise ValueError(f"canonical prediction ID {prediction_id!r} is not unique")
            seen.add(prediction_id)
            score = float(prediction["score"])
            if not math.isfinite(score):
                raise ValueError("prediction score must be finite")
            records.append(
                {
                    "video_id": str(video_id),
                    "label": str(prediction["label"]),
                    "segment": _segment(prediction, ground_truth=False),
                    "score": score,
                    "id": prediction_id,
                }
            )
    return records


def _mean_or_na(values, *, absolute=False):
    if not values:
        return "NA"
    numbers = [abs(value) for value in values] if absolute else values
    return sum(numbers) / len(numbers)


def _median_or_na(values):
    return statistics.median(values) if values else "NA"


def evaluate_frozen_diagnostics(ground_truth, predictions, *, subset="validation"):
    """Evaluate the accepted report-only boundary and high-IoU diagnostics."""

    gt_records = _parse_ground_truth(_load_json(ground_truth), subset)
    prediction_records = _parse_predictions(_load_json(predictions))

    gt_groups = defaultdict(list)
    prediction_groups = defaultdict(list)
    for item in gt_records:
        gt_groups[(item["video_id"], item["label"])].append(item)
    for item in prediction_records:
        prediction_groups[(item["video_id"], item["label"])].append(item)

    matched = []
    matched_gt_ids = set()
    matched_prediction_ids = set()
    for key, group_predictions in prediction_groups.items():
        group_gt = gt_groups.get(key, [])
        for prediction in sorted(
            group_predictions,
            key=lambda item: (-item["score"], item["segment"][0], item["segment"][1], item["id"]),
        ):
            candidates = []
            for gt in group_gt:
                if gt["id"] in matched_gt_ids:
                    continue
                overlap = temporal_iou(prediction["segment"], gt["segment"])
                if overlap >= 0.50:
                    candidates.append((overlap, gt))
            if not candidates:
                continue
            best_overlap = max(item[0] for item in candidates)
            best_gt = min(
                (gt for overlap, gt in candidates if overlap == best_overlap),
                key=lambda item: (item["segment"][0], item["segment"][1], item["id"]),
            )
            matched_gt_ids.add(best_gt["id"])
            matched_prediction_ids.add(prediction["id"])
            matched.append((prediction, best_gt, best_overlap))

    unmatched_gt = [item for item in gt_records if item["id"] not in matched_gt_ids]
    unmatched_predictions = [
        item for item in prediction_records if item["id"] not in matched_prediction_ids
    ]

    start_offsets, end_offsets = [], []
    matched_by_gt = {}
    for prediction, gt, overlap in matched:
        duration = gt["segment"][1] - gt["segment"][0]
        start_offsets.append((prediction["segment"][0] - gt["segment"][0]) / duration)
        end_offsets.append((prediction["segment"][1] - gt["segment"][1]) / duration)
        matched_by_gt[gt["id"]] = (prediction, overlap)

    gt_bins = {name: 0 for name in GT_BINS}
    for gt in gt_records:
        match = matched_by_gt.get(gt["id"])
        if match is None:
            class_confusion = any(
                prediction["video_id"] == gt["video_id"]
                and prediction["label"] != gt["label"]
                and temporal_iou(prediction["segment"], gt["segment"]) >= 0.50
                for prediction in prediction_records
            )
            gt_bins["CLASS_CONFUSION" if class_confusion else "MISS_OR_SEVERE_LOCALIZATION"] += 1
            continue
        prediction, overlap = match
        if overlap >= 0.70:
            gt_bins["HIT_070"] += 1
            continue
        start_corrected = temporal_iou(
            (gt["segment"][0], prediction["segment"][1]), gt["segment"]
        ) >= 0.70
        end_corrected = temporal_iou(
            (prediction["segment"][0], gt["segment"][1]), gt["segment"]
        ) >= 0.70
        if start_corrected and end_corrected:
            gt_bins["EITHER_ENDPOINT_RESCUES"] += 1
        elif start_corrected:
            gt_bins["START_LIMITED"] += 1
        elif end_corrected:
            gt_bins["END_LIMITED"] += 1
        else:
            gt_bins["JOINT_BOUNDARY_LIMITED"] += 1

    fp_bins = {name: 0 for name in FP_BINS}
    for prediction in unmatched_predictions:
        duplicate = any(
            gt["video_id"] == prediction["video_id"]
            and gt["label"] == prediction["label"]
            and temporal_iou(prediction["segment"], gt["segment"]) >= 0.50
            for gt in gt_records
        )
        class_confusion = any(
            gt["video_id"] == prediction["video_id"]
            and gt["label"] != prediction["label"]
            and temporal_iou(prediction["segment"], gt["segment"]) >= 0.50
            for gt in gt_records
        )
        if duplicate:
            fp_bins["DUPLICATE_FP"] += 1
        elif class_confusion:
            fp_bins["CLASS_CONFUSION_FP"] += 1
        else:
            fp_bins["OTHER_FP"] += 1

    short_gt = [item for item in gt_records if item["segment"][1] - item["segment"][0] <= 5.0]
    short_ids = {item["id"] for item in short_gt}
    short_matches = [item for item in matched if item[1]["id"] in short_ids]
    short_start = []
    short_end = []
    for prediction, gt, _ in short_matches:
        duration = gt["segment"][1] - gt["segment"][0]
        short_start.append((prediction["segment"][0] - gt["segment"][0]) / duration)
        short_end.append((prediction["segment"][1] - gt["segment"][1]) / duration)
    short_hit_count = sum(overlap >= 0.70 for _, _, overlap in short_matches)

    videos = sorted({item["video_id"] for item in gt_records})
    return {
        "schema_version": "zoomtoken_frozen_boundary_diagnostics_v001",
        "protocol_valid": True,
        "subset": subset,
        "source_video_count": len(videos),
        "ground_truth_count": len(gt_records),
        "prediction_count": len(prediction_records),
        "boundary": {
            "matched_count": len(matched),
            "unmatched_gt_count": len(unmatched_gt),
            "unmatched_prediction_count": len(unmatched_predictions),
            "mean_abs_start_norm": _mean_or_na(start_offsets, absolute=True),
            "mean_abs_end_norm": _mean_or_na(end_offsets, absolute=True),
            "median_signed_start_norm": _median_or_na(start_offsets),
            "median_signed_end_norm": _median_or_na(end_offsets),
        },
        "short_action_report_only": {
            "definition_seconds": "0 < duration <= 5.0",
            "short_gt_count": len(short_gt),
            "non_short_gt_count": len(gt_records) - len(short_gt),
            "tp_at_070_recall": short_hit_count / len(short_gt) if short_gt else "NA",
            "mean_abs_start_norm": _mean_or_na(short_start, absolute=True),
            "mean_abs_end_norm": _mean_or_na(short_end, absolute=True),
            "median_signed_start_norm": _median_or_na(short_start),
            "median_signed_end_norm": _median_or_na(short_end),
        },
        "high_iou_gt_bins": gt_bins,
        "unmatched_prediction_bins": fp_bins,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--subset", default="validation")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = evaluate_frozen_diagnostics(
        args.ground_truth, args.prediction, subset=args.subset
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
