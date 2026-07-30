#!/usr/bin/env python
"""Aggregate the three frozen DCSR G1 development seeds."""

import argparse
import hashlib
import json
import os
import statistics


EXPECTED_DEV_SEEDS = (2026073001, 2026073002, 2026073003)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite aggregate receipt")
    if len(args.pair) != len(EXPECTED_DEV_SEEDS):
        raise ValueError("G1 aggregate requires exactly three pair receipts")
    pairs = [
        json.loads(open(path, encoding="utf-8").read())
        for path in args.pair
    ]
    seeds = tuple(sorted(int(pair["seed"]) for pair in pairs))
    if seeds != EXPECTED_DEV_SEEDS:
        raise ValueError("G1 development seed set mismatch")
    if any(not pair.get("validation_pass") for pair in pairs):
        raise ValueError("one or more G1 pair receipts failed")
    commits = {pair["git_commit"] for pair in pairs}
    trees = {pair["git_tree"] for pair in pairs}
    if len(commits) != 1 or len(trees) != 1:
        raise ValueError("G1 pairs do not share one exact source identity")

    average_deltas = [
        pair["dcsr_minus_dense"]["average_mAP"] for pair in pairs
    ]
    tiou_deltas = [
        pair["dcsr_minus_dense"]["mAP_at_0_3_to_0_7"]
        for pair in pairs
    ]
    mean_tiou_deltas = [
        statistics.fmean(values)
        for values in zip(*tiou_deltas)
    ]
    mean_average_delta = statistics.fmean(average_deltas)
    gate_checks = {
        "three_frozen_development_seeds": True,
        "mean_average_delta_ge_minus_0_005": (
            mean_average_delta >= -0.005
        ),
        "mean_mAP_0_6_delta_ge_minus_0_01": (
            mean_tiou_deltas[3] >= -0.01
        ),
        "mean_mAP_0_7_delta_ge_minus_0_01": (
            mean_tiou_deltas[4] >= -0.01
        ),
        "all_g0_exact_equivalence_pass": all(
            pair["g0_exact_equivalence_pass"] for pair in pairs
        ),
    }
    g1_gate_pass = all(gate_checks.values())
    payload = {
        "schema_version": "actionformer_dcsr_g1_internal_aggregate_v1",
        "validation_pass": True,
        "g1_gate_pass": g1_gate_pass,
        "gate_checks": gate_checks,
        "development_seeds": list(EXPECTED_DEV_SEEDS),
        "git_commit": next(iter(commits)),
        "git_tree": next(iter(trees)),
        "mean_dcsr_minus_dense": {
            "average_mAP": mean_average_delta,
            "mAP_at_0_3_to_0_7": mean_tiou_deltas,
        },
        "per_seed_average_deltas": average_deltas,
        "pair_receipts": [
            {
                "path": os.path.realpath(path),
                "sha256": _sha256(path),
            }
            for path in args.pair
        ],
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "paper_performance_row_allowed": False,
        "efficiency_claim_allowed": False,
        "next_step_if_pass": "G2 learned selector on internal holdout",
        "next_step_if_fail": "terminate SparseHead route",
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
                "g1_gate_pass": g1_gate_pass,
                "mean_dcsr_minus_dense": payload[
                    "mean_dcsr_minus_dense"
                ],
            },
            sort_keys=True,
        )
    )
if __name__ == "__main__":
    main()
