#!/usr/bin/env python
"""Independently recompute DCSR development metrics on the frozen holdout."""

import argparse
import hashlib
import json
import os
import pickle
import re

import numpy as np

from libs.utils import ANETdetection


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite metric attestation")
    if _sha256(args.annotation) != args.expected_annotation_sha256:
        raise ValueError("annotation SHA-256 mismatch")
    if _sha256(args.manifest) != args.expected_manifest_sha256:
        raise ValueError("internal holdout manifest SHA-256 mismatch")
    manifest = json.loads(
        open(args.manifest, "r", encoding="utf-8").read()
    )
    if (
        manifest.get("schema_version")
        != "actionformer_dcsr_internal_holdout_v1"
        or manifest.get("source_split") != "validation"
        or manifest.get("test_annotations_used") is not False
        or manifest.get("test_records_selected") is not False
    ):
        raise ValueError("invalid internal holdout manifest contract")
    holdout_ids = frozenset(manifest["holdout_video_ids"])

    with open(args.raw_predictions, "rb") as fid:
        predictions = pickle.load(fid)
    prediction_ids = frozenset(predictions["video-id"])
    unexpected = prediction_ids - holdout_ids
    if unexpected:
        raise ValueError("raw predictions escaped the holdout")
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
        "schema_version": "actionformer_dcsr_internal_metric_v1",
        "validation_pass": True,
        "source_split": "validation",
        "holdout_only": True,
        "test_gt_used": False,
        "test_predictions_used": False,
        "model_selection_role": "internal_development_gate_only",
        "paper_performance_row_allowed": False,
        "metrics": {
            "average_mAP": float(average_mAP),
            "mAP_at_0_3_to_0_7": [float(value) for value in mAP],
        },
        "annotation_sha256": _sha256(args.annotation),
        "manifest_sha256": _sha256(args.manifest),
        "raw_predictions_sha256": _sha256(args.raw_predictions),
        "eval_log_sha256": _sha256(args.eval_log),
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
