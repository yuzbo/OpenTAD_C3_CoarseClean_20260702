#!/usr/bin/env python
"""Bind one seed's four-arm ODF-CR matrix into a fail-closed receipt."""

import argparse
import hashlib
import json
import math
import os
import subprocess

from tools.build_odfcr_internal_holdout_v2 import (
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


def _git(*args):
    return subprocess.check_output(["git"] + list(args), text=True).strip()


def _receipt(path):
    return {
        "path": os.path.realpath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def _validate_artifact_record(record, expected_path=None, expected_sha256=None):
    if not isinstance(record, dict) or set(record) != {
        "path",
        "sha256",
        "size_bytes",
    }:
        raise ValueError("malformed ODF-CR artifact record")
    path = record["path"]
    if (
        not isinstance(path, str)
        or not os.path.isfile(path)
        or os.path.realpath(path) != path
        or record["sha256"] != _sha256(path)
        or record["size_bytes"] != os.path.getsize(path)
    ):
        raise ValueError("ODF-CR artifact record failed on-disk validation")
    if expected_path is not None and path != os.path.realpath(expected_path):
        raise ValueError("ODF-CR artifact path mismatch")
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise ValueError("ODF-CR artifact SHA-256 mismatch")


def _validate_metrics(metrics):
    if not isinstance(metrics, dict) or set(metrics) != {
        "average_mAP",
        "mAP_at_0_3_to_0_7",
    }:
        raise ValueError("malformed ODF-CR metric record")
    values = [metrics["average_mAP"]] + list(
        metrics["mAP_at_0_3_to_0_7"]
    )
    if (
        len(values) != 6
        or any(not math.isfinite(float(value)) for value in values)
        or any(float(value) < 0.0 or float(value) > 1.0 for value in values)
    ):
        raise ValueError("ODF-CR metrics are non-finite or out of range")


def _delta_pp(candidate, control):
    candidate_tiou = candidate["mAP_at_0_3_to_0_7"]
    control_tiou = control["mAP_at_0_3_to_0_7"]
    if len(candidate_tiou) != 5 or len(control_tiou) != 5:
        raise ValueError("ODF-CR metrics require five tIoU values")
    return {
        "average_mAP": (
            float(candidate["average_mAP"])
            - float(control["average_mAP"])
        )
        * 100.0,
        "mAP_at_0_3_to_0_7": [
            (float(candidate_value) - float(control_value)) * 100.0
            for candidate_value, control_value in zip(
                candidate_tiou, control_tiou
            )
        ],
        "unit": "percentage_points",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--g0-receipt", required=True)
    parser.add_argument("--d1-off-metric", required=True)
    parser.add_argument("--d1-all-metric", required=True)
    parser.add_argument("--d3-off-metric", required=True)
    parser.add_argument("--d3-all-metric", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--previous-manifest", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--expected-annotation-sha256", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--expected-previous-manifest-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite ODF-CR matrix receipt")
    if _git("rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("ODF-CR matrix commit mismatch")
    if _git("rev-parse", "HEAD^{tree}") != args.expected_tree:
        raise ValueError("ODF-CR matrix tree mismatch")
    if _git("status", "--porcelain=v1"):
        raise ValueError("ODF-CR matrix requires a clean source tree")
    manifest_sha256 = _sha256(args.manifest)
    previous_manifest_sha256 = _sha256(args.previous_manifest)
    if manifest_sha256 != args.expected_manifest_sha256:
        raise ValueError("ODF-CR matrix manifest SHA-256 mismatch")
    if (
        previous_manifest_sha256 != args.expected_previous_manifest_sha256
    ):
        raise ValueError("ODF-CR matrix previous manifest SHA-256 mismatch")
    annotation_sha256 = _sha256_file(args.annotation)
    if annotation_sha256 != args.expected_annotation_sha256:
        raise ValueError("ODF-CR matrix annotation SHA-256 mismatch")
    _, holdout_ids, _ = validate_manifest_contract(
        _read_json(args.manifest),
        _read_json(args.previous_manifest),
        previous_manifest_sha256,
        _read_json(args.annotation),
        annotation_sha256,
    )

    g0 = json.loads(open(args.g0_receipt, encoding="utf-8").read())
    if (
        g0.get("schema_version")
        != "actionformer_odfcr_g0_equivalence_v1"
        or g0.get("gate_pass") is not True
        or int(g0.get("seed")) != args.seed
        or g0.get("git_commit") != args.expected_commit
        or g0.get("git_tree") != args.expected_tree
        or not isinstance(g0.get("checks"), dict)
        or not g0["checks"]
        or not all(value is True for value in g0["checks"].values())
        or "official" not in g0.get("config_sha256", {})
    ):
        raise ValueError("ODF-CR G0 exact-equivalence receipt failed")
    metric_paths = {
        "d1_off": args.d1_off_metric,
        "d1_all": args.d1_all_metric,
        "d3_off": args.d3_off_metric,
        "d3_all": args.d3_all_metric,
    }
    attestations = {
        arm: json.loads(open(path, encoding="utf-8").read())
        for arm, path in metric_paths.items()
    }
    for arm, attestation in attestations.items():
        if (
            attestation.get("schema_version")
            != "actionformer_odfcr_internal_metric_v1"
            or attestation.get("validation_pass") is not True
            or attestation.get("arm") != arm
            or int(attestation.get("seed")) != args.seed
            or attestation.get("source_commit") != args.expected_commit
            or attestation.get("source_tree") != args.expected_tree
            or attestation.get("manifest_sha256") != manifest_sha256
            or (
                attestation.get("previous_manifest_sha256")
                != previous_manifest_sha256
            )
            or attestation.get("paper_performance_row_allowed") is not False
            or attestation.get("official_test_authorized") is not False
            or attestation.get("source_split") != "validation"
            or attestation.get("holdout_only") is not True
            or attestation.get("test_gt_used") is not False
            or attestation.get("test_predictions_used") is not False
            or attestation.get("model_selection_role")
            != "internal_development_gate_only"
            or attestation.get("annotation_sha256")
            != args.expected_annotation_sha256
            or attestation.get("holdout_video_count") != len(holdout_ids)
            or attestation.get("prediction_video_count_with_detections")
            != len(holdout_ids)
            or not isinstance(attestation.get("artifacts"), dict)
            or set(attestation["artifacts"])
            != {
                "annotation",
                "manifest",
                "previous_manifest",
                "config",
                "checkpoint",
                "raw_predictions",
                "eval_log",
            }
        ):
            raise ValueError(
                "invalid ODF-CR metric attestation for {:s}".format(arm)
            )
        artifacts = attestation["artifacts"]
        for name in ("config", "checkpoint", "raw_predictions", "eval_log"):
            _validate_artifact_record(
                artifacts[name],
                expected_sha256=attestation[name + "_sha256"],
            )
        _validate_artifact_record(
            artifacts["annotation"],
            expected_path=args.annotation,
            expected_sha256=args.expected_annotation_sha256,
        )
        _validate_artifact_record(
            artifacts["manifest"],
            expected_path=args.manifest,
            expected_sha256=manifest_sha256,
        )
        _validate_artifact_record(
            artifacts["previous_manifest"],
            expected_path=args.previous_manifest,
            expected_sha256=previous_manifest_sha256,
        )
        _validate_metrics(attestation["metrics"])
    metrics = {
        arm: attestation["metrics"]
        for arm, attestation in attestations.items()
    }
    contrasts = {
        "d1_all_minus_d1_off": _delta_pp(
            metrics["d1_all"], metrics["d1_off"]
        ),
        "d3_all_minus_d3_off": _delta_pp(
            metrics["d3_all"], metrics["d3_off"]
        ),
        "d1_off_minus_d3_off": _delta_pp(
            metrics["d1_off"], metrics["d3_off"]
        ),
    }
    d1_residual = contrasts["d1_all_minus_d1_off"]
    d3_residual = contrasts["d3_all_minus_d3_off"]
    contrasts["depth_by_residual_interaction"] = {
        "average_mAP": (
            d3_residual["average_mAP"] - d1_residual["average_mAP"]
        ),
        "mAP_at_0_3_to_0_7": [
            d3 - d1
            for d3, d1 in zip(
                d3_residual["mAP_at_0_3_to_0_7"],
                d1_residual["mAP_at_0_3_to_0_7"],
            )
        ],
        "unit": "percentage_points",
    }
    payload = {
        "schema_version": "actionformer_odfcr_internal_matrix_v1",
        "validation_pass": True,
        "seed": args.seed,
        "git_commit": args.expected_commit,
        "git_tree": args.expected_tree,
        "annotation_sha256": args.expected_annotation_sha256,
        "g0_exact_equivalence_pass": True,
        "arm_metrics": metrics,
        "arm_artifact_bindings": {
            arm: {
                "config_sha256": attestation["config_sha256"],
                "checkpoint_sha256": attestation["checkpoint_sha256"],
                "raw_predictions_sha256": (
                    attestation["raw_predictions_sha256"]
                ),
                "eval_log_sha256": attestation["eval_log_sha256"],
            }
            for arm, attestation in attestations.items()
        },
        "contrasts": contrasts,
        "receipts": {
            "g0": _receipt(args.g0_receipt),
            "internal_holdout_manifest": _receipt(args.manifest),
            "previous_holdout_manifest": _receipt(args.previous_manifest),
            "arm_metrics": {
                arm: _receipt(path) for arm, path in metric_paths.items()
            },
        },
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "route_decision_allowed": False,
        "requires_three_seed_aggregate": True,
        "efficiency_claim_allowed": False,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = args.output + ".tmp"
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(json.dumps(contrasts, sort_keys=True))


if __name__ == "__main__":
    main()
