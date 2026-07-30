#!/usr/bin/env python
"""Analyze the completed DCSR G1 negative result without retraining.

This analyzer is intentionally restricted to the frozen validation holdout.
It combines the original paired predictions with two checkpoint
counterfactuals and reports proposal-retention diagnostics that cannot be
mistaken for an official THUMOS test result.
"""

import argparse
import glob
import hashlib
import json
import os
import pickle

import numpy as np

from libs.utils import ANETdetection
from libs.utils.metrics import remove_duplicate_annotations


SCHEMA_VERSION = "actionformer_dcsr_negative_diagnostics_v1"
TIOU_THRESHOLDS = np.linspace(0.3, 0.7, 5)
TOP_KS = (1, 5, 10, 20, 50, 100, 200)
DURATION_BINS = (
    ("lt_2s", 0.0, 2.0),
    ("2_4s", 2.0, 4.0),
    ("4_8s", 4.0, 8.0),
    ("8_16s", 8.0, 16.0),
    ("16_32s", 16.0, 32.0),
    ("ge_32s", 32.0, float("inf")),
)
SCORE_BIN_EDGES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.000001)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with open(path, "r", encoding="utf-8") as fid:
        return json.load(fid)


def _load_predictions(path):
    with open(path, "rb") as fid:
        raw = pickle.load(fid)
    expected = {"video-id", "t-start", "t-end", "label", "score"}
    if set(raw) != expected:
        raise ValueError("unexpected prediction schema: " + path)
    lengths = {len(value) for value in raw.values()}
    if len(lengths) != 1:
        raise ValueError("inconsistent prediction lengths: " + path)
    predictions = []
    for video_id, start, end, label, score in zip(
        raw["video-id"],
        raw["t-start"],
        raw["t-end"],
        raw["label"],
        raw["score"],
    ):
        item = {
            "video_id": str(video_id),
            "start": float(start),
            "end": float(end),
            "label": int(label),
            "score": float(score),
        }
        if (
            not np.isfinite(
                [item["start"], item["end"], item["score"]]
            ).all()
            or item["end"] <= item["start"]
            or item["score"] < 0.0
            or item["score"] > 1.0
        ):
            raise ValueError("invalid prediction value: " + path)
        predictions.append(item)
    return raw, predictions


def _load_ground_truth(annotation_path, manifest):
    annotation = _load_json(annotation_path)
    holdout_ids = frozenset(manifest["holdout_video_ids"])
    ground_truth = []
    class_names = {}
    for video_id in sorted(holdout_ids):
        record = annotation["database"].get(video_id)
        if record is None or record["subset"].lower() != "validation":
            raise ValueError("manifest contains a non-validation video")
        for item in remove_duplicate_annotations(record["annotations"]):
            label = int(item["label_id"])
            class_names[label] = str(item.get("label", label))
            ground_truth.append(
                {
                    "video_id": video_id,
                    "start": float(item["segment"][0]),
                    "end": float(item["segment"][1]),
                    "label": label,
                }
            )
    if not ground_truth:
        raise ValueError("holdout ground truth is empty")
    if frozenset(class_names) != frozenset(manifest["all_class_ids"]):
        raise ValueError("manifest/annotation class coverage mismatch")
    return ground_truth, class_names


def segment_iou(segment, candidates):
    """Return temporal IoU between one segment and an ``Nx2`` array."""
    candidates = np.asarray(candidates, dtype=np.float64)
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float64)
    intersection = np.maximum(
        0.0,
        np.minimum(segment[1], candidates[:, 1])
        - np.maximum(segment[0], candidates[:, 0]),
    )
    union = (
        (segment[1] - segment[0])
        + (candidates[:, 1] - candidates[:, 0])
        - intersection
    )
    return intersection / np.maximum(union, np.finfo(np.float64).eps)


def _group_predictions(predictions):
    grouped = {}
    for item in predictions:
        grouped.setdefault(item["video_id"], []).append(item)
    for items in grouped.values():
        items.sort(key=lambda item: item["score"], reverse=True)
    return grouped


def _best_match(gt, ranked_predictions, top_k, class_aware):
    if class_aware:
        ranked_predictions = [
            item
            for item in ranked_predictions
            if item["label"] == gt["label"]
        ]
    candidates = ranked_predictions[:top_k]
    if not candidates:
        return None, 0.0
    ious = segment_iou(
        (gt["start"], gt["end"]),
        [(item["start"], item["end"]) for item in candidates],
    )
    best_index = int(np.argmax(ious))
    return candidates[best_index], float(ious[best_index])


def _recall_by_topk(ground_truth, grouped):
    payload = {"class_aware": {}, "class_agnostic": {}}
    for mode, class_aware in (
        ("class_aware", True),
        ("class_agnostic", False),
    ):
        for top_k in TOP_KS:
            best_ious = [
                _best_match(
                    gt,
                    grouped.get(gt["video_id"], []),
                    top_k,
                    class_aware,
                )[1]
                for gt in ground_truth
            ]
            payload[mode][str(top_k)] = [
                float(np.mean(np.asarray(best_ious) >= threshold))
                for threshold in TIOU_THRESHOLDS
            ]
    return payload


def _duration_recall(ground_truth, grouped):
    output = {}
    for name, lower, upper in DURATION_BINS:
        subset = [
            gt
            for gt in ground_truth
            if lower <= gt["end"] - gt["start"] < upper
        ]
        best_ious = [
            _best_match(
                gt, grouped.get(gt["video_id"], []), 200, True
            )[1]
            for gt in subset
        ]
        output[name] = {
            "gt_count": len(subset),
            "class_aware_recall_at_200": [
                (
                    float(np.mean(np.asarray(best_ious) >= threshold))
                    if best_ious
                    else None
                )
                for threshold in TIOU_THRESHOLDS
            ],
        }
    return output


def _boundary_summary(ground_truth, grouped):
    start_errors = []
    end_errors = []
    normalized_start = []
    normalized_end = []
    best_ious = []
    for gt in ground_truth:
        match, best_iou = _best_match(
            gt, grouped.get(gt["video_id"], []), 200, True
        )
        best_ious.append(best_iou)
        if match is None:
            continue
        duration = max(gt["end"] - gt["start"], np.finfo(np.float64).eps)
        start_error = abs(match["start"] - gt["start"])
        end_error = abs(match["end"] - gt["end"])
        start_errors.append(start_error)
        end_errors.append(end_error)
        normalized_start.append(start_error / duration)
        normalized_end.append(end_error / duration)

    def stats(values):
        if not values:
            return {"count": 0, "mean": None, "median": None, "p90": None}
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(values),
            "mean": float(np.mean(array)),
            "median": float(np.median(array)),
            "p90": float(np.quantile(array, 0.9)),
        }

    return {
        "same_label_match_count": len(start_errors),
        "start_absolute_seconds": stats(start_errors),
        "end_absolute_seconds": stats(end_errors),
        "start_normalized_by_gt_duration": stats(normalized_start),
        "end_normalized_by_gt_duration": stats(normalized_end),
        "best_same_label_tiou": stats(best_ious),
    }


def _score_conditioned_tp_rate(ground_truth, predictions, threshold):
    gt_by_key = {}
    for index, gt in enumerate(ground_truth):
        gt_by_key.setdefault((gt["video_id"], gt["label"]), []).append(
            (index, gt)
        )
    matched = set()
    rows = []
    for prediction in sorted(
        predictions, key=lambda item: item["score"], reverse=True
    ):
        candidates = [
            (index, gt)
            for index, gt in gt_by_key.get(
                (prediction["video_id"], prediction["label"]), []
            )
            if index not in matched
        ]
        is_tp = False
        if candidates:
            ious = segment_iou(
                (prediction["start"], prediction["end"]),
                [(gt["start"], gt["end"]) for _, gt in candidates],
            )
            best_index = int(np.argmax(ious))
            if float(ious[best_index]) >= threshold:
                matched.add(candidates[best_index][0])
                is_tp = True
        rows.append((prediction["score"], is_tp))

    bins = []
    for lower, upper in zip(SCORE_BIN_EDGES[:-1], SCORE_BIN_EDGES[1:]):
        selected = [
            is_tp
            for score, is_tp in rows
            if lower <= score < upper
        ]
        bins.append(
            {
                "lower_inclusive": lower,
                "upper_exclusive": upper,
                "prediction_count": len(selected),
                "tp_rate": (
                    float(np.mean(selected)) if selected else None
                ),
            }
        )
    return {
        "tiou_threshold": threshold,
        "descriptive_only_not_probability_calibration": True,
        "bins": bins,
    }


def _retained_overlap(grouped):
    maxima = []
    for predictions in grouped.values():
        for index, prediction in enumerate(predictions):
            candidates = [
                other
                for other_index, other in enumerate(predictions)
                if other_index != index
                and other["label"] == prediction["label"]
            ]
            if not candidates:
                maxima.append(0.0)
                continue
            maxima.append(
                float(
                    np.max(
                        segment_iou(
                            (prediction["start"], prediction["end"]),
                            [
                                (item["start"], item["end"])
                                for item in candidates
                            ],
                        )
                    )
                )
            )
    array = np.asarray(maxima, dtype=np.float64)
    return {
        "prediction_count": len(maxima),
        "mean_max_same_label_tiou": (
            float(np.mean(array)) if len(array) else None
        ),
        "fraction_max_same_label_tiou_ge_0_5": (
            float(np.mean(array >= 0.5)) if len(array) else None
        ),
        "fraction_max_same_label_tiou_ge_0_7": (
            float(np.mean(array >= 0.7)) if len(array) else None
        ),
        "identifiability": "post_nms_retained_outputs_only",
    }


def _per_video_recall(ground_truth, grouped, threshold=0.7):
    by_video = {}
    for gt in ground_truth:
        by_video.setdefault(gt["video_id"], []).append(gt)
    output = {}
    for video_id, items in by_video.items():
        recalled = [
            _best_match(gt, grouped.get(video_id, []), 200, True)[1]
            >= threshold
            for gt in items
        ]
        output[video_id] = {
            "gt_count": len(items),
            "class_aware_recall_at_200_tiou_0_7": float(np.mean(recalled)),
        }
    return output


def summarize_arm(annotation_path, holdout_ids, ground_truth, predictions):
    grouped = _group_predictions(predictions)
    if frozenset(grouped) - holdout_ids:
        raise ValueError("predictions escaped the internal holdout")
    allowed_labels = frozenset(gt["label"] for gt in ground_truth)
    unexpected_labels = (
        frozenset(item["label"] for item in predictions) - allowed_labels
    )
    if unexpected_labels:
        raise ValueError("predictions contain labels outside the holdout")
    evaluator = ANETdetection(
        annotation_path,
        split="validation",
        tiou_thresholds=TIOU_THRESHOLDS,
        video_ids=holdout_ids,
        num_workers=1,
    )
    raw = {
        "video-id": np.asarray(
            [item["video_id"] for item in predictions]
        ),
        "t-start": np.asarray([item["start"] for item in predictions]),
        "t-end": np.asarray([item["end"] for item in predictions]),
        "label": np.asarray([item["label"] for item in predictions]),
        "score": np.asarray([item["score"] for item in predictions]),
    }
    mAP, average_mAP, _ = evaluator.evaluate(raw, verbose=False)
    scores = np.asarray([item["score"] for item in predictions])
    label_order = sorted(evaluator.activity_index)
    per_class_ap = {
        str(label): [float(value) for value in evaluator.ap[:, index]]
        for index, label in enumerate(label_order)
    }
    return {
        "prediction_count": len(predictions),
        "prediction_video_count": len(grouped),
        "score": {
            "mean": float(np.mean(scores)) if len(scores) else None,
            "quantiles": {
                str(q): (
                    float(np.quantile(scores, q)) if len(scores) else None
                )
                for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
            },
        },
        "official_holdout_evaluator": {
            "average_mAP": float(average_mAP),
            "mAP_at_0_3_to_0_7": [float(value) for value in mAP],
            "per_class_ap_at_0_3_to_0_7": per_class_ap,
        },
        "post_nms_recall": _recall_by_topk(ground_truth, grouped),
        "duration": _duration_recall(ground_truth, grouped),
        "boundary": _boundary_summary(ground_truth, grouped),
        "score_conditioned_tp_rate": {
            "tiou_0_5": _score_conditioned_tp_rate(
                ground_truth, predictions, 0.5
            ),
            "tiou_0_7": _score_conditioned_tp_rate(
                ground_truth, predictions, 0.7
            ),
        },
        "retained_overlap": _retained_overlap(grouped),
        "per_video": _per_video_recall(ground_truth, grouped),
    }


def _one_path(pattern):
    matches = glob.glob(pattern)
    if len(matches) != 1:
        raise ValueError(
            "expected exactly one path for pattern: {:s}".format(pattern)
        )
    return matches[0]


def _original_arm(source_root, seed, variant, aggregate):
    seed_root = os.path.join(source_root, "seed_{:d}".format(seed))
    attestation_path = os.path.join(
        seed_root, variant, "METRIC_ATTESTATION.json"
    )
    attestation = _load_json(attestation_path)
    if (
        attestation.get("validation_pass") is not True
        or attestation.get("source_split") != "validation"
        or attestation.get("paper_performance_row_allowed") is not False
        or attestation.get("test_gt_used") is not False
        or attestation.get("test_predictions_used") is not False
    ):
        raise ValueError("invalid source metric attestation")
    raw_path = _one_path(
        os.path.join(
            seed_root,
            "work",
            "ckpt",
            "*_{:s}".format(variant),
            "eval_results.pkl",
        )
    )
    if _sha256(raw_path) != attestation["raw_predictions_sha256"]:
        raise ValueError("source raw prediction SHA-256 mismatch")
    expected_pair_path = os.path.realpath(
        os.path.join(seed_root, "DCSR_G1_PAIR_COMPLETE.json")
    )
    pair_matches = [
        item
        for item in aggregate["pair_receipts"]
        if os.path.realpath(item["path"]) == expected_pair_path
    ]
    if len(pair_matches) != 1:
        raise ValueError("aggregate/source pair path binding mismatch")
    pair_record = pair_matches[0]
    if _sha256(pair_record["path"]) != pair_record["sha256"]:
        raise ValueError("source pair receipt SHA-256 mismatch")
    pair = _load_json(pair_record["path"])
    if (
        pair.get("validation_pass") is not True
        or pair.get("seed") != seed
        or pair.get("git_commit") != aggregate["git_commit"]
        or pair.get("git_tree") != aggregate["git_tree"]
        or pair.get("source_split") != "validation"
        or pair.get("test_gt_used") is not False
        or pair.get("test_predictions_used") is not False
    ):
        raise ValueError("source pair receipt content mismatch")
    return {
        "path": raw_path,
        "sha256": _sha256(raw_path),
        "attestation_path": attestation_path,
        "attestation_sha256": _sha256(attestation_path),
    }


def _counterfactual_arm(counterfactual_root, seed, arm, aggregate):
    receipt_path = os.path.join(
        counterfactual_root,
        "seed_{:d}".format(seed),
        arm,
        "COUNTERFACTUAL_RECEIPT.json",
    )
    receipt = _load_json(receipt_path)
    if (
        receipt.get("schema_version")
        != "actionformer_dcsr_counterfactual_v1"
        or receipt.get("validation_pass") is not True
        or receipt.get("diagnostic_only") is not True
        or receipt.get("paper_performance_row_allowed") is not False
        or receipt.get("test_gt_used") is not False
        or receipt.get("test_predictions_used") is not False
        or receipt.get("seed") != seed
        or receipt.get("source_training_commit")
        != aggregate["git_commit"]
        or receipt.get("source_training_tree") != aggregate["git_tree"]
        or receipt["counterfactual"]["arm"] != arm
    ):
        raise ValueError("invalid counterfactual receipt")
    raw_path = receipt["predictions"]["path"]
    expected_raw_path = os.path.realpath(
        os.path.join(
            counterfactual_root,
            "seed_{:d}".format(seed),
            arm,
            "eval_results.pkl",
        )
    )
    if os.path.realpath(raw_path) != expected_raw_path:
        raise ValueError("counterfactual prediction path escaped its arm root")
    if _sha256(raw_path) != receipt["predictions"]["sha256"]:
        raise ValueError("counterfactual raw prediction SHA-256 mismatch")
    return {
        "path": raw_path,
        "sha256": _sha256(raw_path),
        "receipt_path": receipt_path,
        "receipt_sha256": _sha256(receipt_path),
    }


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(array, axis=0)),
        "sample_std": (
            float(np.std(array, ddof=1)) if len(array) > 1 else None
        ),
    }


def _vector_mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": [float(value) for value in np.mean(array, axis=0)],
        "sample_std": (
            [
                float(value)
                for value in np.std(array, axis=0, ddof=1)
            ]
            if len(array) > 1
            else None
        ),
    }


def _aggregate(seed_payloads):
    arms = ("dense", "k384", "scaffold_only", "all_query_residual")
    arm_summary = {}
    for arm in arms:
        arm_summary[arm] = {
            "average_mAP": _mean_std(
                [
                    payload["arms"][arm]["summary"][
                        "official_holdout_evaluator"
                    ]["average_mAP"]
                    for payload in seed_payloads
                ]
            ),
            "mAP_at_0_3_to_0_7": _vector_mean_std(
                [
                    payload["arms"][arm]["summary"][
                        "official_holdout_evaluator"
                    ]["mAP_at_0_3_to_0_7"]
                    for payload in seed_payloads
                ]
            ),
            "class_aware_recall_at_200": _vector_mean_std(
                [
                    payload["arms"][arm]["summary"]["post_nms_recall"][
                        "class_aware"
                    ]["200"]
                    for payload in seed_payloads
                ]
            ),
        }

    contrasts = {}
    for left, right, name in (
        ("k384", "dense", "k384_minus_dense"),
        ("scaffold_only", "dense", "scaffold_only_minus_dense"),
        (
            "all_query_residual",
            "dense",
            "all_query_residual_minus_dense",
        ),
        ("k384", "scaffold_only", "k384_residual_value"),
        (
            "all_query_residual",
            "scaffold_only",
            "all_query_residual_value",
        ),
        ("k384", "all_query_residual", "k384_support_penalty"),
    ):
        avg_values = []
        tiou_values = []
        for payload in seed_payloads:
            left_metrics = payload["arms"][left]["summary"][
                "official_holdout_evaluator"
            ]
            right_metrics = payload["arms"][right]["summary"][
                "official_holdout_evaluator"
            ]
            avg_values.append(
                left_metrics["average_mAP"] - right_metrics["average_mAP"]
            )
            tiou_values.append(
                np.asarray(left_metrics["mAP_at_0_3_to_0_7"])
                - np.asarray(right_metrics["mAP_at_0_3_to_0_7"])
            )
        contrasts[name] = {
            "average_mAP": _mean_std(avg_values),
            "mAP_at_0_3_to_0_7": _vector_mean_std(tiou_values),
        }
    return {"arms": arm_summary, "contrasts": contrasts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--source-run-root", required=True)
    parser.add_argument("--counterfactual-run-root", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--expected-aggregate-sha256", required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if os.path.exists(args.output) or os.path.exists(args.output + ".tmp"):
        raise FileExistsError("refusing to overwrite negative diagnostics")
    if _sha256(args.annotation) != args.expected_annotation_sha256:
        raise ValueError("annotation SHA-256 mismatch")
    if _sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("manifest SHA-256 mismatch")
    if _sha256(args.aggregate) != args.expected_aggregate_sha256:
        raise ValueError("G1 aggregate SHA-256 mismatch")
    manifest = _load_json(args.manifest)
    aggregate = _load_json(args.aggregate)
    if (
        manifest.get("schema_version")
        != "actionformer_dcsr_internal_holdout_v1"
        or manifest.get("source_split") != "validation"
        or manifest.get("test_annotations_used") is not False
        or manifest.get("test_records_selected") is not False
        or aggregate.get("schema_version")
        != "actionformer_dcsr_g1_internal_aggregate_v1"
        or aggregate.get("validation_pass") is not True
        or aggregate.get("g1_gate_pass") is not False
        or aggregate.get("paper_performance_row_allowed") is not False
        or aggregate.get("test_gt_used") is not False
        or aggregate.get("test_predictions_used") is not False
        or sorted(args.seeds) != aggregate["development_seeds"]
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError("invalid frozen G1 negative-result contract")

    holdout_ids = frozenset(manifest["holdout_video_ids"])
    ground_truth, class_names = _load_ground_truth(args.annotation, manifest)
    seed_payloads = []
    for seed in args.seeds:
        sources = {
            "dense": _original_arm(
                args.source_run_root, seed, "dense", aggregate
            ),
            "k384": _original_arm(
                args.source_run_root, seed, "dcsr", aggregate
            ),
            "scaffold_only": _counterfactual_arm(
                args.counterfactual_run_root,
                seed,
                "scaffold_only",
                aggregate,
            ),
            "all_query_residual": _counterfactual_arm(
                args.counterfactual_run_root,
                seed,
                "all_query_residual",
                aggregate,
            ),
        }
        arms = {}
        for arm, source in sources.items():
            _, predictions = _load_predictions(source["path"])
            arms[arm] = {
                "source": source,
                "summary": summarize_arm(
                    args.annotation,
                    holdout_ids,
                    ground_truth,
                    predictions,
                ),
            }
        seed_payloads.append({"seed": seed, "arms": arms})

    payload = {
        "schema_version": SCHEMA_VERSION,
        "validation_pass": True,
        "diagnostic_only": True,
        "paper_performance_row_allowed": False,
        "efficiency_claim_allowed": False,
        "source_split": "validation",
        "holdout_only": True,
        "test_gt_used": False,
        "test_predictions_used": False,
        "source_training_commit": aggregate["git_commit"],
        "source_training_tree": aggregate["git_tree"],
        "source_g1_aggregate": {
            "path": os.path.realpath(args.aggregate),
            "sha256": _sha256(args.aggregate),
            "g1_gate_pass": False,
        },
        "annotation": {
            "path": os.path.realpath(args.annotation),
            "sha256": _sha256(args.annotation),
        },
        "manifest": {
            "path": os.path.realpath(args.manifest),
            "sha256": _sha256(args.manifest),
            "holdout_video_count": len(holdout_ids),
            "ground_truth_count": len(ground_truth),
        },
        "class_names": {
            str(label): name for label, name in sorted(class_names.items())
        },
        "seeds": seed_payloads,
        "aggregate_diagnostics": _aggregate(seed_payloads),
        "identifiability_limits": [
            "post_nms_predictions_do_not_identify_pre_nms_suppressed_proposals",
            "score_conditioned_tp_rate_is_not_probability_calibration",
            "counterfactuals_change_inference_support_not_training",
            "validation_holdout_diagnostics_are_not_official_test_results",
        ],
    }
    temporary_path = args.output + ".tmp"
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(
        json.dumps(
            payload["aggregate_diagnostics"]["contrasts"],
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
