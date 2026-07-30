#!/usr/bin/env python3
"""Analyze an official ActionFormer dense/sparse raw-prediction pair.

The raw files are post-Soft-NMS ``eval_results.pkl`` artifacts.  This command
therefore reports only diagnostics identifiable from retained detections and
states the pre-NMS limitations explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


BUILDER_PATH = Path(__file__).with_name("build_actionformer_official_record.py")
SPEC = importlib.util.spec_from_file_location(
    "actionformer_official_record_builder_for_diagnostics",
    BUILDER_PATH,
)
official = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(official)
protocol = official.protocol

SCHEMA_VERSION = "actionformer_matched_postnms_diagnostics_v1"
TIOU_THRESHOLDS = np.asarray((0.3, 0.4, 0.5, 0.6, 0.7), dtype=np.float64)
FIXED_TOPK = (1, 5, 10, 20, 50, 100, 200)
DURATION_BINS = (
    ("lt_1s", 0.0, 1.0),
    ("1_2s", 1.0, 2.0),
    ("2_4s", 2.0, 4.0),
    ("4_8s", 4.0, 8.0),
    ("8_16s", 8.0, 16.0),
    ("16_32s", 16.0, 32.0),
    ("ge_32s", 32.0, math.inf),
)


def require(condition, message):
    if not condition:
        raise protocol.ProtocolError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def segment_iou(segment, candidates):
    candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 2)
    if candidates.size == 0:
        return np.zeros((0,), dtype=np.float64)
    start = np.maximum(float(segment[0]), candidates[:, 0])
    end = np.minimum(float(segment[1]), candidates[:, 1])
    intersection = np.maximum(0.0, end - start)
    union = (
        max(0.0, float(segment[1]) - float(segment[0]))
        + np.maximum(0.0, candidates[:, 1] - candidates[:, 0])
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def load_ground_truth(annotation_path):
    annotation_path = Path(annotation_path).resolve()
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    database = payload.get("database")
    require(isinstance(database, dict) and database, "annotation database is empty")
    by_video = {}
    label_names = {}
    test_video_ids = []
    for video_id, record in sorted(database.items()):
        if str(record.get("subset", "")).lower() != "test":
            continue
        test_video_ids.append(video_id)
        entries = []
        for annotation in record.get("annotations", []):
            label_name = str(annotation.get("label", ""))
            if label_name.lower() == "ambiguous":
                continue
            label_id = int(annotation["label_id"])
            segment = np.asarray(annotation["segment"], dtype=np.float64)
            require(segment.shape == (2,), f"invalid GT segment in {video_id}")
            require(
                np.isfinite(segment).all() and segment[1] > segment[0],
                f"non-positive GT segment in {video_id}",
            )
            previous = label_names.setdefault(label_id, label_name)
            require(previous == label_name, f"label mapping drift for ID {label_id}")
            entries.append(
                {
                    "video_id": video_id,
                    "label_id": label_id,
                    "label_name": label_name,
                    "segment": segment,
                    "duration": float(segment[1] - segment[0]),
                }
            )
        by_video[video_id] = entries
    require(len(test_video_ids) == 212, "official test-video count drift")
    require(set(label_names) == set(range(20)), "official label-ID set drift")
    return by_video, label_names, test_video_ids


def group_predictions(raw_predictions):
    grouped = {}
    video_ids = np.asarray(raw_predictions["video-id"], dtype=object)
    starts = np.asarray(raw_predictions["t-start"], dtype=np.float64)
    ends = np.asarray(raw_predictions["t-end"], dtype=np.float64)
    labels = np.asarray(raw_predictions["label"], dtype=np.int64)
    scores = np.asarray(raw_predictions["score"], dtype=np.float64)
    for video_id in sorted(set(video_ids.tolist())):
        indices = np.flatnonzero(video_ids == video_id)
        grouped[video_id] = {
            "segments": np.stack((starts[indices], ends[indices]), axis=1),
            "labels": labels[indices],
            "scores": scores[indices],
        }
    return grouped


def duration_bin_name(duration):
    for name, lower, upper in DURATION_BINS:
        if lower <= duration < upper:
            return name
    raise AssertionError(f"duration outside bins: {duration}")


def new_accumulator():
    return {
        "gt_count": 0,
        "class_aware_best_iou_sum": 0.0,
        "class_agnostic_best_iou_sum": 0.0,
        "class_aware_recall_counts": np.zeros(len(TIOU_THRESHOLDS), dtype=np.int64),
        "class_agnostic_recall_counts": np.zeros(len(TIOU_THRESHOLDS), dtype=np.int64),
        "boundary_available_count": 0,
        "start_abs_error_sum": 0.0,
        "end_abs_error_sum": 0.0,
        "center_abs_error_sum": 0.0,
        "start_normalized_error_sum": 0.0,
        "end_normalized_error_sum": 0.0,
        "center_normalized_error_sum": 0.0,
    }


def update_accumulator(accumulator, row):
    accumulator["gt_count"] += 1
    accumulator["class_aware_best_iou_sum"] += row["class_aware_best_iou"]
    accumulator["class_agnostic_best_iou_sum"] += row["class_agnostic_best_iou"]
    accumulator["class_aware_recall_counts"] += (
        row["class_aware_best_iou"] >= TIOU_THRESHOLDS
    )
    accumulator["class_agnostic_recall_counts"] += (
        row["class_agnostic_best_iou"] >= TIOU_THRESHOLDS
    )
    if row["boundary_available"]:
        accumulator["boundary_available_count"] += 1
        for key in (
            "start_abs_error",
            "end_abs_error",
            "center_abs_error",
            "start_normalized_error",
            "end_normalized_error",
            "center_normalized_error",
        ):
            accumulator[f"{key}_sum"] += row[key]


def finalize_accumulator(accumulator):
    count = int(accumulator["gt_count"])
    boundary_count = int(accumulator["boundary_available_count"])
    if count == 0:
        return {
            "gt_count": 0,
            "class_aware_recall": {
                f"{threshold:.1f}": None for threshold in TIOU_THRESHOLDS
            },
            "class_agnostic_recall": {
                f"{threshold:.1f}": None for threshold in TIOU_THRESHOLDS
            },
            "class_aware_mean_best_iou": None,
            "class_agnostic_mean_best_iou": None,
            "boundary_oracle": {
                "available_count": 0,
                "start_abs_error_mean": None,
                "end_abs_error_mean": None,
                "center_abs_error_mean": None,
                "start_normalized_error_mean": None,
                "end_normalized_error_mean": None,
                "center_normalized_error_mean": None,
            },
        }
    boundary = {"available_count": boundary_count}
    for key in (
        "start_abs_error",
        "end_abs_error",
        "center_abs_error",
        "start_normalized_error",
        "end_normalized_error",
        "center_normalized_error",
    ):
        boundary[f"{key}_mean"] = (
            accumulator[f"{key}_sum"] / boundary_count
            if boundary_count
            else None
        )
    return {
        "gt_count": count,
        "class_aware_recall": {
            f"{threshold:.1f}": float(value) / count
            for threshold, value in zip(
                TIOU_THRESHOLDS,
                accumulator["class_aware_recall_counts"],
            )
        },
        "class_agnostic_recall": {
            f"{threshold:.1f}": float(value) / count
            for threshold, value in zip(
                TIOU_THRESHOLDS,
                accumulator["class_agnostic_recall_counts"],
            )
        },
        "class_aware_mean_best_iou": (
            accumulator["class_aware_best_iou_sum"] / count
        ),
        "class_agnostic_mean_best_iou": (
            accumulator["class_agnostic_best_iou_sum"] / count
        ),
        "boundary_oracle": boundary,
    }


def quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0, "cannot summarize empty values")
    levels = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        f"{level:.2f}": float(value)
        for level, value in zip(levels, np.quantile(values, levels))
    }


def summarize_retained_outputs(grouped_predictions, ground_truth):
    scores = []
    best_same_label_ious = []
    overlapping_pairs = {0.5: 0, 0.7: 0}
    total_same_label_pairs = 0
    for video_id, predictions in grouped_predictions.items():
        gt_entries = ground_truth[video_id]
        gt_by_label = defaultdict(list)
        for entry in gt_entries:
            gt_by_label[entry["label_id"]].append(entry["segment"])
        segments = predictions["segments"]
        labels = predictions["labels"]
        scores.extend(predictions["scores"].tolist())
        for segment, label_id in zip(segments, labels):
            candidates = gt_by_label.get(int(label_id), [])
            ious = segment_iou(segment, candidates)
            best_same_label_ious.append(float(ious.max()) if ious.size else 0.0)
        for label_id in np.unique(labels):
            label_segments = segments[labels == label_id]
            for left in range(len(label_segments)):
                if left + 1 >= len(label_segments):
                    continue
                ious = segment_iou(label_segments[left], label_segments[left + 1 :])
                total_same_label_pairs += int(ious.size)
                for threshold in overlapping_pairs:
                    overlapping_pairs[threshold] += int(np.sum(ious >= threshold))
    scores = np.asarray(scores, dtype=np.float64)
    best_same_label_ious = np.asarray(best_same_label_ious, dtype=np.float64)
    score_iou_bins = {}
    for name, lower, upper in (
        ("iou_0_0.1", 0.0, 0.1),
        ("iou_0.1_0.3", 0.1, 0.3),
        ("iou_0.3_0.5", 0.3, 0.5),
        ("iou_0.5_0.7", 0.5, 0.7),
        ("iou_ge_0.7", 0.7, math.inf),
    ):
        mask = (best_same_label_ious >= lower) & (best_same_label_ious < upper)
        score_iou_bins[name] = {
            "prediction_count": int(mask.sum()),
            "score_mean": float(scores[mask].mean()) if mask.any() else None,
            "score_median": float(np.median(scores[mask])) if mask.any() else None,
        }
    return {
        "prediction_count": int(scores.size),
        "score_quantiles": quantiles(scores),
        "best_same_label_gt_iou_quantiles": quantiles(best_same_label_ious),
        "retained_prediction_has_same_label_gt_iou": {
            f"{threshold:.1f}": float(np.mean(best_same_label_ious >= threshold))
            for threshold in TIOU_THRESHOLDS
        },
        "score_by_best_same_label_gt_iou_bin": score_iou_bins,
        "same_label_prediction_pair_count": total_same_label_pairs,
        "same_label_overlap_pair_fraction": {
            f"{threshold:.1f}": (
                float(count) / total_same_label_pairs
                if total_same_label_pairs
                else 0.0
            )
            for threshold, count in overlapping_pairs.items()
        },
    }


def summarize_arm(raw_predictions, ground_truth, label_names):
    grouped = group_predictions(raw_predictions)
    require(set(grouped) == set(ground_truth), "prediction video coverage drift")
    overall = new_accumulator()
    by_class = {label_id: new_accumulator() for label_id in label_names}
    by_duration = {name: new_accumulator() for name, _, _ in DURATION_BINS}
    video_accumulators = {}
    fixed_topk_counts = {
        topk: np.zeros(len(TIOU_THRESHOLDS), dtype=np.int64)
        for topk in FIXED_TOPK
    }
    total_gt = 0
    for video_id, gt_entries in ground_truth.items():
        predictions = grouped[video_id]
        segments = predictions["segments"]
        labels = predictions["labels"]
        scores = predictions["scores"]
        order = np.argsort(scores)[::-1]
        video_accumulator = new_accumulator()
        for gt in gt_entries:
            total_gt += 1
            all_ious = segment_iou(gt["segment"], segments)
            class_indices = np.flatnonzero(labels == gt["label_id"])
            class_ious = (
                all_ious[class_indices]
                if class_indices.size
                else np.zeros((0,), dtype=np.float64)
            )
            class_best_iou = float(class_ious.max()) if class_ious.size else 0.0
            agnostic_best_iou = float(all_ious.max()) if all_ious.size else 0.0
            row = {
                "class_aware_best_iou": class_best_iou,
                "class_agnostic_best_iou": agnostic_best_iou,
                "boundary_available": bool(class_ious.size),
            }
            if class_ious.size:
                best_prediction_index = int(class_indices[int(class_ious.argmax())])
                best_segment = segments[best_prediction_index]
                duration = gt["duration"]
                gt_center = float(gt["segment"].mean())
                pred_center = float(best_segment.mean())
                row.update(
                    {
                        "start_abs_error": abs(
                            float(best_segment[0] - gt["segment"][0])
                        ),
                        "end_abs_error": abs(
                            float(best_segment[1] - gt["segment"][1])
                        ),
                        "center_abs_error": abs(pred_center - gt_center),
                    }
                )
                row["start_normalized_error"] = row["start_abs_error"] / duration
                row["end_normalized_error"] = row["end_abs_error"] / duration
                row["center_normalized_error"] = row["center_abs_error"] / duration
            update_accumulator(overall, row)
            update_accumulator(by_class[gt["label_id"]], row)
            update_accumulator(by_duration[duration_bin_name(gt["duration"])], row)
            update_accumulator(video_accumulator, row)

            for topk in FIXED_TOPK:
                selected = order[: min(topk, len(order))]
                selected = selected[labels[selected] == gt["label_id"]]
                selected_ious = all_ious[selected]
                best_topk = (
                    float(selected_ious.max()) if selected_ious.size else 0.0
                )
                fixed_topk_counts[topk] += best_topk >= TIOU_THRESHOLDS
        video_accumulators[video_id] = video_accumulator

    failures = []
    for video_id, accumulator in video_accumulators.items():
        summary = finalize_accumulator(accumulator)
        failures.append(
            {
                "video_id": video_id,
                "gt_count": summary["gt_count"],
                "class_aware_mean_best_iou": summary[
                    "class_aware_mean_best_iou"
                ],
                "class_aware_recall@0.5": summary["class_aware_recall"]["0.5"],
                "class_aware_recall@0.7": summary["class_aware_recall"]["0.7"],
            }
        )
    failures.sort(
        key=lambda item: (
            item["class_aware_recall@0.7"],
            item["class_aware_recall@0.5"],
            item["class_aware_mean_best_iou"],
            item["video_id"],
        )
    )
    return {
        "overall": finalize_accumulator(overall),
        "per_class": {
            label_names[label_id]: {
                "label_id": label_id,
                **finalize_accumulator(by_class[label_id]),
            }
            for label_id in sorted(label_names)
        },
        "duration_bins": {
            name: finalize_accumulator(by_duration[name])
            for name, _, _ in DURATION_BINS
        },
        "fixed_topk_class_aware_recall": {
            str(topk): {
                f"{threshold:.1f}": float(value) / total_gt
                for threshold, value in zip(
                    TIOU_THRESHOLDS,
                    fixed_topk_counts[topk],
                )
            }
            for topk in FIXED_TOPK
        },
        "retained_output_diagnostics": summarize_retained_outputs(
            grouped,
            ground_truth,
        ),
        "worst_videos": failures[:30],
    }


def evaluate_official_with_details(repo, annotation_path, raw_predictions):
    previous_path = list(sys.path)
    previous_modules = {
        key: value
        for key, value in sys.modules.items()
        if key == "libs" or key.startswith("libs.")
    }
    for key in list(previous_modules):
        del sys.modules[key]
    sys.path.insert(0, str(repo))
    try:
        metrics_module = importlib.import_module("libs.utils.metrics")
        evaluator = metrics_module.ANETdetection(
            str(annotation_path),
            split="test",
            tiou_thresholds=TIOU_THRESHOLDS,
            top_k=(1, 5),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            mAP, average_mAP, mean_recall = evaluator.evaluate(
                raw_predictions,
                verbose=True,
            )
        ap = np.asarray(evaluator.ap, dtype=np.float64)
        recall = np.asarray(evaluator.recall, dtype=np.float64)
        activity_index = dict(evaluator.activity_index)
    except Exception as error:
        raise protocol.ProtocolError(
            f"official detailed evaluation failed: {error}"
        ) from error
    finally:
        sys.path[:] = previous_path
        for key in list(sys.modules):
            if key == "libs" or key.startswith("libs."):
                del sys.modules[key]
        sys.modules.update(previous_modules)
    require(ap.shape == (5, 20), "official per-class AP shape drift")
    require(recall.shape == (5, 2, 20), "official recall shape drift")
    return {
        "average_mAP": float(average_mAP),
        "mAP": {
            f"{threshold:.1f}": float(value)
            for threshold, value in zip(TIOU_THRESHOLDS, mAP)
        },
        "mean_recall_at_multiple_of_gt": {
            f"{threshold:.1f}": {
                "1x": float(mean_recall[index, 0]),
                "5x": float(mean_recall[index, 1]),
            }
            for index, threshold in enumerate(TIOU_THRESHOLDS)
        },
        "per_label_id_ap": {
            str(label_id): {
                f"{threshold:.1f}": float(ap[index, column])
                for index, threshold in enumerate(TIOU_THRESHOLDS)
            }
            for label_id, column in sorted(activity_index.items())
        },
    }


def subtract_mapping(sparse, dense):
    return {
        key: float(sparse[key]) - float(dense[key])
        for key in dense
        if dense[key] is not None and sparse[key] is not None
    }


def build_contrasts(dense, sparse):
    per_class_ap_delta = {}
    for class_name in dense["per_class"]:
        per_class_ap_delta[class_name] = subtract_mapping(
            sparse["per_class"][class_name]["official_ap"],
            dense["per_class"][class_name]["official_ap"],
        )
    duration_recall_delta = {}
    for name in dense["duration_bins"]:
        duration_recall_delta[name] = subtract_mapping(
            sparse["duration_bins"][name]["class_aware_recall"],
            dense["duration_bins"][name]["class_aware_recall"],
        )
    return {
        "official_mAP": subtract_mapping(
            sparse["official"]["mAP"],
            dense["official"]["mAP"],
        ),
        "average_mAP": (
            sparse["official"]["average_mAP"] - dense["official"]["average_mAP"]
        ),
        "overall_class_aware_recall": subtract_mapping(
            sparse["overall"]["class_aware_recall"],
            dense["overall"]["class_aware_recall"],
        ),
        "overall_class_agnostic_recall": subtract_mapping(
            sparse["overall"]["class_agnostic_recall"],
            dense["overall"]["class_agnostic_recall"],
        ),
        "per_class_official_ap": per_class_ap_delta,
        "duration_class_aware_recall": duration_recall_delta,
        "fixed_topk_class_aware_recall": {
            topk: subtract_mapping(
                sparse["fixed_topk_class_aware_recall"][topk],
                dense["fixed_topk_class_aware_recall"][topk],
            )
            for topk in dense["fixed_topk_class_aware_recall"]
        },
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--dense-raw", required=True)
    parser.add_argument("--expected-dense-raw-sha256", required=True)
    parser.add_argument("--sparse-raw", required=True)
    parser.add_argument("--expected-sparse-raw-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    require(not output.exists(), f"output already exists: {output}")
    annotation = Path(args.annotation).resolve()
    dense_raw_path = Path(args.dense_raw).resolve()
    sparse_raw_path = Path(args.sparse_raw).resolve()
    require(
        sha256_file(annotation) == args.expected_annotation_sha256,
        "annotation SHA-256 mismatch",
    )
    require(
        sha256_file(dense_raw_path) == args.expected_dense_raw_sha256,
        "dense raw-prediction SHA-256 mismatch",
    )
    require(
        sha256_file(sparse_raw_path) == args.expected_sparse_raw_sha256,
        "sparse raw-prediction SHA-256 mismatch",
    )
    repo, commit, tree = official.verify_official_source(args.official_repo)
    evaluator_manifest, evaluator_fingerprint = official.build_evaluator_manifest(
        repo,
        commit,
        tree,
    )
    ground_truth, label_names, test_video_ids = load_ground_truth(annotation)
    dense_raw, dense_count = official.load_and_validate_raw_predictions(
        dense_raw_path
    )
    sparse_raw, sparse_count = official.load_and_validate_raw_predictions(
        sparse_raw_path
    )
    require(dense_count == sparse_count == 42400, "raw prediction count drift")
    expected_video_ids = sorted(test_video_ids)
    for arm, raw in (("dense", dense_raw), ("sparse", sparse_raw)):
        require(
            sorted(set(raw["video-id"])) == expected_video_ids,
            f"{arm} raw prediction coverage drift",
        )

    summaries = {}
    for arm, raw in (("dense", dense_raw), ("sparse", sparse_raw)):
        summary = summarize_arm(raw, ground_truth, label_names)
        detailed = evaluate_official_with_details(repo, annotation, raw)
        summary["official"] = detailed
        for class_name, class_record in summary["per_class"].items():
            label_id = str(class_record["label_id"])
            class_record["official_ap"] = detailed["per_label_id_ap"][label_id]
        summaries[arm] = summary

    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "issues": [],
        "new_training": False,
        "official_source": {
            "root": str(repo),
            "commit": commit,
            "tree": tree,
            "evaluator_fingerprint_sha256": evaluator_fingerprint,
            "evaluator_manifest_sha256": protocol.canonical_sha256(
                evaluator_manifest
            ),
        },
        "annotation": {
            "path": str(annotation),
            "sha256": sha256_file(annotation),
            "official_test_video_count": len(test_video_ids),
            "ground_truth_instance_count": sum(
                len(entries) for entries in ground_truth.values()
            ),
        },
        "raw_predictions": {
            "dense": {
                "path": str(dense_raw_path),
                "sha256": sha256_file(dense_raw_path),
                "prediction_count": dense_count,
            },
            "sparse": {
                "path": str(sparse_raw_path),
                "sha256": sha256_file(sparse_raw_path),
                "prediction_count": sparse_count,
            },
        },
        "arms": summaries,
        "sparse_minus_dense": build_contrasts(
            summaries["dense"],
            summaries["sparse"],
        ),
        "identifiability": {
            "supported": [
                "official per-class post-NMS AP",
                "class-aware and class-agnostic retained-output GT recall",
                "fixed top-k retained-output recall",
                "duration-stratified retained-output recall",
                "oracle best-retained boundary error",
                "retained score/GT-IoU summaries",
                "same-label retained-output overlap density",
                "worst-video ranking",
            ],
            "not_supported_by_postnms_raw_files": [
                "pre-NMS proposal recall",
                "suppressed-proposal identities",
                "Soft-NMS threshold counterfactuals",
                "background-logit calibration",
                "selector assignment/support observability",
                "training gradient or EMA-normalizer causality",
            ],
        },
        "claim_boundary": (
            "postnms_diagnostic_only;single_seed;not_model_selection;"
            "not_pre_nms_or_training_causality"
        ),
        "paper_main_table_eligible": False,
        "paper_ready": False,
        "end_to_end_cost_claim_allowed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
