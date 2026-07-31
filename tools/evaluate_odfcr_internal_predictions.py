#!/usr/bin/env python
"""Independently recompute ODF-CR metrics on the frozen holdout-v2."""

import argparse
import hashlib
import json
import os
import pickle
import re

import numpy as np

from libs.utils import ANETdetection
from tools.build_odfcr_internal_holdout_v2 import (
    EXPECTED_HOLDOUT_COUNT,
    _read_json,
    _sha256_file,
    validate_manifest_contract,
)


ARMS = ("d1_off", "d1_all", "d3_off", "d3_all")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact(path):
    return {
        "path": os.path.realpath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def _parse_logged_metrics(path):
    text = open(path, "r", encoding="utf-8").read()
    per_tiou = [
        float(value)
        for value in re.findall(
            r"\|tIoU = [0-9.]+: mAP =\s*([0-9.]+) \(%\)",
            text,
        )
    ]
    average = re.findall(r"Average mAP:\s*([0-9.]+) \(%\)", text)
    if len(per_tiou) != 5 or len(average) != 1:
        raise ValueError("could not parse the five official holdout metrics")
    return per_tiou, float(average[0])


def _load_and_validate_manifest(
    manifest_path,
    previous_manifest_path,
    annotation_path,
    expected_annotation_sha256,
):
    annotation_sha256 = _sha256_file(annotation_path)
    if annotation_sha256 != expected_annotation_sha256:
        raise ValueError("annotation SHA-256 mismatch during manifest validation")
    manifest = _read_json(manifest_path)
    previous = _read_json(previous_manifest_path)
    annotation = _read_json(annotation_path)
    previous_sha256 = _sha256(previous_manifest_path)
    _, holdout_ids, _ = validate_manifest_contract(
        manifest,
        previous,
        previous_sha256,
        annotation,
        annotation_sha256,
    )
    return manifest, holdout_ids


def _validate_predictions(predictions, holdout_ids):
    expected_keys = {"video-id", "t-start", "t-end", "label", "score"}
    if not isinstance(predictions, dict) or set(predictions) != expected_keys:
        raise ValueError("unexpected raw prediction schema")
    lengths = {len(value) for value in predictions.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("raw prediction arrays are empty or inconsistent")
    prediction_ids = frozenset(str(value) for value in predictions["video-id"])
    if (
        prediction_ids != holdout_ids
        or len(prediction_ids) != EXPECTED_HOLDOUT_COUNT
    ):
        raise ValueError("raw predictions do not exactly cover holdout-v2")
    starts = np.asarray(predictions["t-start"])
    ends = np.asarray(predictions["t-end"])
    scores = np.asarray(predictions["score"])
    labels = np.asarray(predictions["label"])
    if not all(np.isfinite(values).all() for values in (starts, ends, scores)):
        raise ValueError("raw predictions contain non-finite numeric values")
    if np.any(ends <= starts):
        raise ValueError("raw predictions contain a non-positive segment")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("raw prediction labels must have integer dtype")
    if np.any(labels < 0) or np.any(labels >= 20):
        raise ValueError("raw prediction labels fall outside THUMOS classes")
    return prediction_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--expected-previous-manifest-sha256", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-tree", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite metric attestation")
    if _sha256(args.annotation) != args.expected_annotation_sha256:
        raise ValueError("annotation SHA-256 mismatch")
    if _sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("holdout-v2 manifest SHA-256 mismatch")
    if (
        _sha256(args.previous_manifest)
        != args.expected_previous_manifest_sha256
    ):
        raise ValueError("previous manifest SHA-256 mismatch")
    if _sha256(args.config) != args.expected_config_sha256:
        raise ValueError("ODF-CR config SHA-256 mismatch")
    if _sha256(args.checkpoint) != args.expected_checkpoint_sha256:
        raise ValueError("ODF-CR checkpoint SHA-256 mismatch")
    _, holdout_ids = _load_and_validate_manifest(
        args.manifest,
        args.previous_manifest,
        args.annotation,
        args.expected_annotation_sha256,
    )

    with open(args.raw_predictions, "rb") as fid:
        predictions = pickle.load(fid)
    prediction_ids = _validate_predictions(predictions, holdout_ids)
    evaluator = ANETdetection(
        args.annotation,
        split="validation",
        tiou_thresholds=np.linspace(0.3, 0.7, 5),
        video_ids=holdout_ids,
    )
    mAP, average_mAP, _ = evaluator.evaluate(
        predictions, verbose=False
    )
    logged_tiou, logged_average = _parse_logged_metrics(args.eval_log)
    recomputed_percent = [round(float(value) * 100.0, 2) for value in mAP]
    if recomputed_percent != logged_tiou:
        raise ValueError("official/recomputed per-tIoU metrics mismatch")
    if round(float(average_mAP) * 100.0, 2) != logged_average:
        raise ValueError("official/recomputed average mAP mismatch")

    payload = {
        "schema_version": "actionformer_odfcr_internal_metric_v1",
        "validation_pass": True,
        "arm": args.arm,
        "seed": args.seed,
        "source_commit": args.source_commit,
        "source_tree": args.source_tree,
        "source_split": "validation",
        "holdout_only": True,
        "test_gt_used": False,
        "test_predictions_used": False,
        "model_selection_role": "internal_development_gate_only",
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "metrics": {
            "average_mAP": float(average_mAP),
            "mAP_at_0_3_to_0_7": [float(value) for value in mAP],
        },
        "annotation_sha256": _sha256(args.annotation),
        "manifest_sha256": _sha256(args.manifest),
        "previous_manifest_sha256": _sha256(args.previous_manifest),
        "config_sha256": _sha256(args.config),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "raw_predictions_sha256": _sha256(args.raw_predictions),
        "eval_log_sha256": _sha256(args.eval_log),
        "artifacts": {
            "annotation": _artifact(args.annotation),
            "manifest": _artifact(args.manifest),
            "previous_manifest": _artifact(args.previous_manifest),
            "config": _artifact(args.config),
            "checkpoint": _artifact(args.checkpoint),
            "raw_predictions": _artifact(args.raw_predictions),
            "eval_log": _artifact(args.eval_log),
        },
        "holdout_video_count": len(holdout_ids),
        "prediction_video_count_with_detections": len(prediction_ids),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = args.output + ".tmp"
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(json.dumps(payload["metrics"], sort_keys=True))


if __name__ == "__main__":
    main()
