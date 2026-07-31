#!/usr/bin/env python
"""Aggregate the three frozen ODF-CR factorial seeds and apply G2."""

import argparse
import hashlib
import json
import math
import os
import statistics

from tools.build_odfcr_internal_holdout_v2 import (
    _read_json,
    _sha256_file,
    validate_manifest_contract,
)


EXPECTED_DEV_SEEDS = (2026073101, 2026073102, 2026073103)
ARMS = ("d1_off", "d1_all", "d3_off", "d3_all")
CONTRASTS = (
    "d1_all_minus_d1_off",
    "d3_all_minus_d3_off",
    "d1_off_minus_d3_off",
    "depth_by_residual_interaction",
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_file_receipt(receipt):
    if (
        not isinstance(receipt, dict)
        or not {"path", "sha256"} <= set(receipt)
        or not isinstance(receipt["path"], str)
    ):
        raise ValueError("malformed ODF-CR file receipt")
    path = receipt["path"]
    if (
        not os.path.isfile(path)
        or os.path.realpath(path) != path
        or _sha256(path) != receipt["sha256"]
    ):
        raise ValueError("ODF-CR file receipt failed on-disk validation")
    if (
        "size_bytes" in receipt
        and os.path.getsize(path) != receipt["size_bytes"]
    ):
        raise ValueError("ODF-CR file receipt size mismatch")
    return path


def _validate_metric_shape(metrics, *, bounded):
    if (
        not isinstance(metrics, dict)
        or set(metrics) != {"average_mAP", "mAP_at_0_3_to_0_7"}
        or len(metrics["mAP_at_0_3_to_0_7"]) != 5
    ):
        raise ValueError("malformed ODF-CR metric shape")
    values = [metrics["average_mAP"]] + list(
        metrics["mAP_at_0_3_to_0_7"]
    )
    if any(
        type(value) not in (int, float) or not math.isfinite(float(value))
        for value in values
    ):
        raise ValueError("non-finite or non-numeric ODF-CR metric")
    if bounded and any(float(value) < 0.0 or float(value) > 1.0 for value in values):
        raise ValueError("ODF-CR arm metric is outside [0, 1]")


def _validate_matrix(matrix):
    if (
        matrix.get("schema_version")
        != "actionformer_odfcr_internal_matrix_v1"
        or matrix.get("validation_pass") is not True
        or matrix.get("g0_exact_equivalence_pass") is not True
        or matrix.get("requires_three_seed_aggregate") is not True
        or matrix.get("route_decision_allowed") is not False
        or matrix.get("paper_performance_row_allowed") is not False
        or matrix.get("official_test_authorized") is not False
        or matrix.get("efficiency_claim_allowed") is not False
        or matrix.get("source_split") != "validation"
        or matrix.get("test_gt_used") is not False
        or matrix.get("test_predictions_used") is not False
        or set(matrix.get("arm_metrics", {})) != set(ARMS)
        or set(matrix.get("arm_artifact_bindings", {})) != set(ARMS)
        or set(matrix.get("contrasts", {})) != set(CONTRASTS)
        or set(matrix.get("receipts", {}))
        != {
            "g0",
            "internal_holdout_manifest",
            "previous_holdout_manifest",
            "arm_metrics",
        }
        or set(matrix["receipts"].get("arm_metrics", {})) != set(ARMS)
    ):
        raise ValueError("one or more ODF-CR matrices failed validation")
    seed = int(matrix["seed"])
    for arm in ARMS:
        _validate_metric_shape(matrix["arm_metrics"][arm], bounded=True)
        binding = matrix["arm_artifact_bindings"][arm]
        if set(binding) != {
            "config_sha256",
            "checkpoint_sha256",
            "raw_predictions_sha256",
            "eval_log_sha256",
        }:
            raise ValueError("ODF-CR arm artifact binding keys drifted")
        metric_path = _validate_file_receipt(
            matrix["receipts"]["arm_metrics"][arm]
        )
        attestation = json.loads(open(metric_path, encoding="utf-8").read())
        if (
            attestation.get("schema_version")
            != "actionformer_odfcr_internal_metric_v1"
            or attestation.get("validation_pass") is not True
            or attestation.get("arm") != arm
            or int(attestation.get("seed")) != seed
            or attestation.get("source_commit") != matrix["git_commit"]
            or attestation.get("source_tree") != matrix["git_tree"]
            or attestation.get("metrics") != matrix["arm_metrics"][arm]
            or any(
                attestation.get(name + "_sha256") != binding[name + "_sha256"]
                for name in (
                    "config",
                    "checkpoint",
                    "raw_predictions",
                    "eval_log",
                )
            )
        ):
            raise ValueError("ODF-CR arm metric receipt binding mismatch")
    for contrast_name in CONTRASTS:
        contrast = matrix["contrasts"][contrast_name]
        if contrast.get("unit") != "percentage_points":
            raise ValueError("ODF-CR contrast unit drifted")
        _validate_metric_shape(
            {
                "average_mAP": contrast["average_mAP"],
                "mAP_at_0_3_to_0_7": contrast["mAP_at_0_3_to_0_7"],
            },
            bounded=False,
        )

    g0_path = _validate_file_receipt(matrix["receipts"]["g0"])
    g0 = json.loads(open(g0_path, encoding="utf-8").read())
    if (
        g0.get("schema_version") != "actionformer_odfcr_g0_equivalence_v1"
        or g0.get("gate_pass") is not True
        or int(g0.get("seed")) != seed
        or g0.get("git_commit") != matrix["git_commit"]
        or g0.get("git_tree") != matrix["git_tree"]
        or not g0.get("checks")
        or not all(value is True for value in g0["checks"].values())
    ):
        raise ValueError("ODF-CR G0 receipt binding mismatch")
    manifest_path = _validate_file_receipt(
        matrix["receipts"]["internal_holdout_manifest"]
    )
    previous_path = _validate_file_receipt(
        matrix["receipts"]["previous_holdout_manifest"]
    )
    manifest = json.loads(open(manifest_path, encoding="utf-8").read())
    annotation_path = manifest.get("source_annotation_path")
    if (
        not isinstance(annotation_path, str)
        or not os.path.isfile(annotation_path)
        or matrix.get("annotation_sha256")
        != manifest.get("source_annotation_sha256")
    ):
        raise ValueError("ODF-CR matrix annotation binding mismatch")
    annotation_sha256 = _sha256_file(annotation_path)
    if annotation_sha256 != matrix["annotation_sha256"]:
        raise ValueError("ODF-CR matrix annotation SHA-256 mismatch")
    validate_manifest_contract(
        manifest,
        _read_json(previous_path),
        _sha256_file(previous_path),
        _read_json(annotation_path),
        annotation_sha256,
    )
    return g0["config_sha256"]


def _mean_metric(records):
    return {
        "average_mAP": statistics.fmean(
            record["average_mAP"] for record in records
        ),
        "mAP_at_0_3_to_0_7": [
            statistics.fmean(values)
            for values in zip(
                *[record["mAP_at_0_3_to_0_7"] for record in records]
            )
        ],
    }


def _aggregate_contrast(records):
    averages = [float(record["average_mAP"]) for record in records]
    tiou = [record["mAP_at_0_3_to_0_7"] for record in records]
    return {
        "mean": {
            "average_mAP": statistics.fmean(averages),
            "mAP_at_0_3_to_0_7": [
                statistics.fmean(values) for values in zip(*tiou)
            ],
        },
        "sample_standard_deviation": {
            "average_mAP": statistics.stdev(averages),
            "mAP_at_0_3_to_0_7": [
                statistics.stdev(values) for values in zip(*tiou)
            ],
        },
        "per_seed_average_mAP": averages,
        "unit": "percentage_points",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite ODF-CR aggregate")
    if len(args.matrix) != len(EXPECTED_DEV_SEEDS):
        raise ValueError("ODF-CR aggregate requires exactly three matrices")
    matrices = [
        json.loads(open(path, encoding="utf-8").read())
        for path in args.matrix
    ]
    seeds = tuple(sorted(int(matrix["seed"]) for matrix in matrices))
    if seeds != EXPECTED_DEV_SEEDS:
        raise ValueError("ODF-CR development seed set mismatch")
    matrices = sorted(matrices, key=lambda matrix: int(matrix["seed"]))
    g0_config_hashes = {
        json.dumps(_validate_matrix(matrix), sort_keys=True)
        for matrix in matrices
    }
    commits = {matrix["git_commit"] for matrix in matrices}
    trees = {matrix["git_tree"] for matrix in matrices}
    manifest_hashes = {
        matrix["receipts"]["internal_holdout_manifest"]["sha256"]
        for matrix in matrices
    }
    previous_manifest_hashes = {
        matrix["receipts"]["previous_holdout_manifest"]["sha256"]
        for matrix in matrices
    }
    if (
        len(commits) != 1
        or len(trees) != 1
        or len(manifest_hashes) != 1
        or len(previous_manifest_hashes) != 1
        or len(g0_config_hashes) != 1
    ):
        raise ValueError(
            "ODF-CR matrices do not share source and manifest identity"
        )

    arm_means = {
        arm: _mean_metric(
            [matrix["arm_metrics"][arm] for matrix in matrices]
        )
        for arm in ARMS
    }
    contrast_aggregates = {
        contrast: _aggregate_contrast(
            [matrix["contrasts"][contrast] for matrix in matrices]
        )
        for contrast in CONTRASTS
    }
    d3_utility = contrast_aggregates["d3_all_minus_d3_off"]
    d3_mean = d3_utility["mean"]
    positive_seed_count = sum(
        value > 0.0 for value in d3_utility["per_seed_average_mAP"]
    )
    gate_checks = {
        "three_frozen_development_seeds": True,
        "all_g0_exact_equivalence_pass": all(
            matrix["g0_exact_equivalence_pass"] for matrix in matrices
        ),
        "mean_d3_residual_average_delta_ge_plus_0_25_pp": (
            d3_mean["average_mAP"] >= 0.25
        ),
        "at_least_two_positive_seed_average_deltas": (
            positive_seed_count >= 2
        ),
        "mean_d3_residual_mAP_0_6_delta_ge_0_pp": (
            d3_mean["mAP_at_0_3_to_0_7"][3] >= 0.0
        ),
        "mean_d3_residual_mAP_0_7_delta_ge_0_pp": (
            d3_mean["mAP_at_0_3_to_0_7"][4] >= 0.0
        ),
    }
    residual_utility_gate_pass = all(gate_checks.values())
    payload = {
        "schema_version": "actionformer_odfcr_g2_internal_aggregate_v1",
        "validation_pass": True,
        "residual_utility_gate_pass": residual_utility_gate_pass,
        "gate_checks": gate_checks,
        "threshold_equality_passes": True,
        "delta_unit": "percentage_points",
        "paired_subtraction_before_aggregation": True,
        "development_seeds": list(EXPECTED_DEV_SEEDS),
        "training_replicates_on_one_fixed_holdout": True,
        "independent_validation_splits": False,
        "git_commit": next(iter(commits)),
        "git_tree": next(iter(trees)),
        "manifest_sha256": next(iter(manifest_hashes)),
        "previous_manifest_sha256": next(iter(previous_manifest_hashes)),
        "mean_arm_metrics": arm_means,
        "contrast_aggregates": contrast_aggregates,
        "positive_d3_residual_seed_count": positive_seed_count,
        "matrix_receipts": [
            {
                "path": os.path.realpath(path),
                "sha256": _sha256(path),
            }
            for path in args.matrix
        ],
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "efficiency_claim_allowed": False,
        "next_step_if_pass": (
            "frozen d3_all stratified_uniform K384 replay"
        ),
        "next_step_if_fail": (
            "record legal negative residual-utility result and analyze"
        ),
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    temporary_path = args.output + ".tmp"
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(payload, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    print(
        json.dumps(
            {
                "residual_utility_gate_pass": (
                    residual_utility_gate_pass
                ),
                "d3_all_minus_d3_off": d3_utility,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
