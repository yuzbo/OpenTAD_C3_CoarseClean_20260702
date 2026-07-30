#!/usr/bin/env python
"""Bind one paired DCSR development seed into a fail-closed receipt."""

import argparse
import hashlib
import json
import os
import subprocess


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args):
    return subprocess.check_output(
        ["git"] + list(args), text=True
    ).strip()


def _receipt(path):
    return {
        "path": os.path.realpath(path),
        "sha256": _sha256(path),
        "size_bytes": os.path.getsize(path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--g0-receipt", required=True)
    parser.add_argument("--dense-metric", required=True)
    parser.add_argument("--dcsr-metric", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite paired receipt")
    if _git("rev-parse", "HEAD") != args.expected_commit:
        raise ValueError("paired receipt commit mismatch")
    if _git("rev-parse", "HEAD^{tree}") != args.expected_tree:
        raise ValueError("paired receipt tree mismatch")
    if _git("status", "--porcelain=v1"):
        raise ValueError("paired receipt requires a clean source tree")

    g0 = json.loads(open(args.g0_receipt, encoding="utf-8").read())
    dense = json.loads(open(args.dense_metric, encoding="utf-8").read())
    dcsr = json.loads(open(args.dcsr_metric, encoding="utf-8").read())
    if not g0.get("gate_pass"):
        raise ValueError("G0 exact-equivalence gate did not pass")
    if not dense.get("validation_pass") or not dcsr.get("validation_pass"):
        raise ValueError("paired internal metric attestation failed")
    dense_values = dense["metrics"]["mAP_at_0_3_to_0_7"]
    dcsr_values = dcsr["metrics"]["mAP_at_0_3_to_0_7"]
    deltas = [
        float(candidate) - float(control)
        for candidate, control in zip(dcsr_values, dense_values)
    ]
    average_delta = (
        float(dcsr["metrics"]["average_mAP"])
        - float(dense["metrics"]["average_mAP"])
    )
    payload = {
        "schema_version": "actionformer_dcsr_internal_pair_v1",
        "validation_pass": True,
        "seed": args.seed,
        "git_commit": args.expected_commit,
        "git_tree": args.expected_tree,
        "g0_exact_equivalence_pass": True,
        "dense_metrics": dense["metrics"],
        "dcsr_metrics": dcsr["metrics"],
        "dcsr_minus_dense": {
            "average_mAP": average_delta,
            "mAP_at_0_3_to_0_7": deltas,
        },
        "descriptive_single_seed_g1_thresholds": {
            "average_delta_ge_minus_0_005": average_delta >= -0.005,
            "mAP_0_6_delta_ge_minus_0_01": deltas[3] >= -0.01,
            "mAP_0_7_delta_ge_minus_0_01": deltas[4] >= -0.01,
        },
        "receipts": {
            "g0": _receipt(args.g0_receipt),
            "dense_metric": _receipt(args.dense_metric),
            "dcsr_metric": _receipt(args.dcsr_metric),
            "internal_holdout_manifest": _receipt(args.manifest),
        },
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "paper_performance_row_allowed": False,
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
    print(
        json.dumps(
            payload["dcsr_minus_dense"], sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
