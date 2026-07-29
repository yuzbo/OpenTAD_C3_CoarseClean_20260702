#!/usr/bin/env python3
"""Independently evaluate ActionFormer raw predictions with pinned official code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


BUILDER_PATH = Path(__file__).with_name("build_actionformer_official_record.py")
SPEC = importlib.util.spec_from_file_location(
    "actionformer_official_record_builder",
    BUILDER_PATH,
)
official = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(official)
protocol = official.protocol


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_prediction_video_coverage(raw_predictions, videos):
    evaluated = sorted(
        video_id for video_id, subset in videos if subset.lower() == "test"
    )
    predicted = sorted(set(raw_predictions["video-id"]))
    unexpected = sorted(set(predicted) - set(evaluated))
    missing = sorted(set(evaluated) - set(predicted))
    if unexpected:
        raise protocol.ProtocolError(
            f"raw predictions contain non-test videos: {unexpected[:8]}"
        )
    if missing:
        raise protocol.ProtocolError(
            f"raw predictions omit {len(missing)} official test videos"
        )
    if len(evaluated) != protocol.OFFICIAL_EVALUATED_VIDEO_COUNT:
        raise protocol.ProtocolError(
            "official evaluated-video count changed: "
            f"{len(evaluated)} != {protocol.OFFICIAL_EVALUATED_VIDEO_COUNT}"
        )
    return evaluated, predicted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--raw-predictions", required=True)
    parser.add_argument("--eval-log", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise protocol.ProtocolError(
            f"independent metric attestation already exists: {output}"
        )
    annotation = Path(args.annotation).resolve()
    raw_path = Path(args.raw_predictions).resolve()
    eval_log = Path(args.eval_log).resolve()
    for label, path in {
        "annotation": annotation,
        "raw_predictions": raw_path,
        "eval_log": eval_log,
    }.items():
        if not path.is_file():
            raise protocol.ProtocolError(f"missing {label}: {path}")
    annotation_sha256 = _sha256_file(annotation)
    if annotation_sha256 != args.expected_annotation_sha256:
        raise protocol.ProtocolError(
            "official annotation SHA-256 mismatch: "
            f"{annotation_sha256} != {args.expected_annotation_sha256}"
        )

    repo, commit, tree = official.verify_official_source(args.official_repo)
    evaluator_manifest, evaluator_fingerprint = official.build_evaluator_manifest(
        repo,
        commit,
        tree,
    )
    _, split_counts, videos = official.parse_annotation(annotation)
    if split_counts != protocol.OFFICIAL_ANNOTATION_SPLIT_COUNTS:
        raise protocol.ProtocolError(
            f"official annotation split counts changed: {split_counts}"
        )
    raw_predictions, prediction_count = official.load_and_validate_raw_predictions(
        raw_path
    )
    evaluated, predicted = _validate_prediction_video_coverage(
        raw_predictions,
        videos,
    )
    logged_metrics = protocol.parse_actionformer_eval_log(
        eval_log.read_text(encoding="utf-8", errors="strict")
    )
    recomputed_metrics = official.recompute_official_metrics(
        repo,
        annotation,
        raw_predictions,
    )
    maximum_delta = protocol._assert_metrics_close(
        logged_metrics,
        recomputed_metrics,
        atol=5.1e-5,
        label="candidate_log_vs_pinned_official_recompute",
        left_is_logged=True,
    )
    payload = {
        "schema_version": "actionformer_independent_metric_attestation_v1",
        "validation_pass": True,
        "issues": [],
        "official_evaluator": {
            "repository_root": str(repo),
            "commit": commit,
            "tree": tree,
            "clean": True,
            "fingerprint_sha256": evaluator_fingerprint,
            "manifest_sha256": protocol.canonical_sha256(evaluator_manifest),
        },
        "annotation": {
            "path": str(annotation),
            "sha256": annotation_sha256,
            "split_counts": split_counts,
        },
        "raw_predictions": {
            "path": str(raw_path),
            "sha256": _sha256_file(raw_path),
            "prediction_count": prediction_count,
            "prediction_video_count": len(predicted),
            "prediction_video_ids_sha256": protocol.canonical_sha256(predicted),
            "evaluated_video_count": len(evaluated),
            "evaluated_video_ids_sha256": protocol.canonical_sha256(evaluated),
            "complete_official_test_coverage": True,
        },
        "eval_log": {
            "path": str(eval_log),
            "sha256": _sha256_file(eval_log),
        },
        "logged_metrics": logged_metrics,
        "recomputed_metrics": recomputed_metrics,
        "max_abs_delta": maximum_delta,
        "paper_main_table_eligible": False,
        "matched_record_suite_required": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol.atomic_write_json(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
