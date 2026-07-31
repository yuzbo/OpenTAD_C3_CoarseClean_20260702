#!/usr/bin/env python
"""Aggregate frozen ODF-CR K384 replays and apply the G3 support gate."""

import argparse
import hashlib
import json
import os
import statistics

from tools.aggregate_odfcr_internal_matrices import EXPECTED_DEV_SEEDS
from tools.build_odfcr_internal_holdout_v2 import EXPECTED_HOLDOUT_COUNT


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _read_canonical_json(path, label):
    canonical_path = os.path.realpath(path)
    if (
        not os.path.isabs(path)
        or canonical_path != path
        or not os.path.isfile(path)
        or os.path.getsize(path) <= 0
    ):
        raise ValueError("{} must be a non-empty canonical absolute file".format(label))
    return (
        json.loads(open(path, encoding="utf-8").read()),
        {
            "path": canonical_path,
            "sha256": _sha256(path),
            "size_bytes": os.path.getsize(path),
        },
    )


def _validate_allocation_ledger(selector, holdout_ids):
    records = selector.get("allocation_records")
    if (
        selector.get("allocation_video_count") != EXPECTED_HOLDOUT_COUNT
        or not isinstance(records, list)
        or len(records) != EXPECTED_HOLDOUT_COUNT
    ):
        raise ValueError("K384 allocation ledger count mismatch")
    validated = []
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "video_id",
            "valid_counts_per_level",
            "selected_indices_per_level",
            "selected_count",
            "allocation_sha256",
        }:
            raise ValueError("malformed K384 allocation record")
        video_id = record["video_id"]
        valid_counts = record["valid_counts_per_level"]
        selected_indices = record["selected_indices_per_level"]
        if (
            not isinstance(video_id, str)
            or video_id in seen
            or not isinstance(valid_counts, list)
            or not isinstance(selected_indices, list)
            or len(valid_counts) != len(selected_indices)
            or not valid_counts
            or any(type(value) is not int or value < 0 for value in valid_counts)
        ):
            raise ValueError("invalid K384 allocation record structure")
        selected_count = 0
        for valid_count, indices in zip(valid_counts, selected_indices):
            if (
                not isinstance(indices, list)
                or len(indices) != len(set(indices))
                or any(
                    type(index) is not int
                    or index < 0
                    or index >= valid_count
                    for index in indices
                )
            ):
                raise ValueError("invalid K384 allocation indices")
            selected_count += len(indices)
        if (
            record["selected_count"] != selected_count
            or selected_count != min(384, sum(valid_counts))
        ):
            raise ValueError("K384 allocation violates exact budget")
        unsigned = dict(record)
        observed_sha256 = unsigned.pop("allocation_sha256")
        if observed_sha256 != _canonical_sha256(unsigned):
            raise ValueError("K384 allocation record hash mismatch")
        seen.add(video_id)
        validated.append(record)
    if frozenset(seen) != holdout_ids:
        raise ValueError("K384 allocation IDs do not equal holdout-v2")
    canonical_records = sorted(validated, key=lambda record: record["video_id"])
    if selector.get("allocation_ledger_sha256") != _canonical_sha256(
        canonical_records
    ):
        raise ValueError("K384 allocation ledger hash mismatch")


def _delta_pp(candidate, control):
    return {
        "average_mAP": (
            float(candidate["average_mAP"])
            - float(control["average_mAP"])
        )
        * 100.0,
        "mAP_at_0_3_to_0_7": [
            (float(candidate_value) - float(control_value)) * 100.0
            for candidate_value, control_value in zip(
                candidate["mAP_at_0_3_to_0_7"],
                control["mAP_at_0_3_to_0_7"],
            )
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--g2-aggregate", required=True)
    parser.add_argument("--matrix", action="append", required=True)
    parser.add_argument("--replay", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite ODF-CR G3 aggregate")
    if len(args.matrix) != 3 or len(args.replay) != 3:
        raise ValueError("ODF-CR G3 requires three matrices and three replays")
    g2, g2_receipt = _read_canonical_json(args.g2_aggregate, "G2 aggregate")
    if (
        g2.get("schema_version")
        != "actionformer_odfcr_g2_internal_aggregate_v1"
        or g2.get("validation_pass") is not True
        or g2.get("residual_utility_gate_pass") is not True
    ):
        raise ValueError("ODF-CR G2 did not authorize K384 replay")
    matrix_records = []
    matrix_receipts = []
    for path in args.matrix:
        record, receipt = _read_canonical_json(path, "ODF-CR source matrix")
        matrix_records.append((path, record))
        matrix_receipts.append(receipt)
    replay_records = []
    replay_receipts = []
    for path in args.replay:
        record, receipt = _read_canonical_json(path, "ODF-CR K384 replay")
        replay_records.append((path, record))
        replay_receipts.append(receipt)
    matrices = {
        int(record["seed"]): (path, record)
        for path, record in matrix_records
    }
    replays = {
        int(record["seed"]): (path, record)
        for path, record in replay_records
    }
    if (
        tuple(sorted(matrices)) != EXPECTED_DEV_SEEDS
        or tuple(sorted(replays)) != EXPECTED_DEV_SEEDS
    ):
        raise ValueError("ODF-CR G3 seed set mismatch")
    matrix_sha256s = {_sha256(path) for path, _ in matrix_records}
    if (
        {receipt.get("sha256") for receipt in g2.get("matrix_receipts", [])}
        != matrix_sha256s
        or g2.get("development_seeds") != list(EXPECTED_DEV_SEEDS)
        or g2.get("paper_performance_row_allowed") is not False
        or g2.get("official_test_authorized") is not False
    ):
        raise ValueError("G2 aggregate is not bound to the supplied matrices")
    g2_sha256 = g2_receipt["sha256"]
    deltas = []
    allocation_hashes = {}
    for seed in EXPECTED_DEV_SEEDS:
        matrix_path, matrix = matrices[seed]
        _, replay = replays[seed]
        manifest_receipt = matrix["receipts"]["internal_holdout_manifest"]
        manifest_path = manifest_receipt["path"]
        if (
            not os.path.isfile(manifest_path)
            or _sha256(manifest_path) != manifest_receipt["sha256"]
        ):
            raise ValueError("K384 source manifest receipt is invalid")
        manifest = json.loads(open(manifest_path, encoding="utf-8").read())
        holdout_ids = frozenset(manifest.get("holdout_video_ids", ()))
        if (
            replay.get("schema_version")
            != "actionformer_odfcr_k384_replay_v1"
            or replay.get("validation_pass") is not True
            or replay.get("training_performed") is not False
            or int(replay.get("seed")) != seed
            or replay["checkpoint"]["sha256"]
            != matrix["arm_artifact_bindings"]["d3_all"][
                "checkpoint_sha256"
            ]
            or replay.get("git_commit") != matrix.get("git_commit")
            or replay.get("git_tree") != matrix.get("git_tree")
            or replay.get("manifest_sha256")
            != matrix["receipts"]["internal_holdout_manifest"]["sha256"]
            or replay.get("previous_manifest_sha256")
            != matrix["receipts"]["previous_holdout_manifest"]["sha256"]
            or replay.get("source_matrix_receipt_sha256")
            != _sha256(matrix_path)
            or replay.get("g2_aggregate_sha256") != g2_sha256
            or replay["config"]["sha256"]
            != matrix["arm_artifact_bindings"]["d3_all"]["config_sha256"]
            or replay["selector"].get("policy") != "stratified_uniform"
            or replay["selector"].get("budget") != 384
            or replay["selector"].get("hash_seed") != 2026073100
            or replay["predictions"].get("video_count_with_detections")
            != EXPECTED_HOLDOUT_COUNT
            or replay.get("paper_performance_row_allowed") is not False
            or replay.get("official_test_authorized") is not False
            or len(holdout_ids) != EXPECTED_HOLDOUT_COUNT
        ):
            raise ValueError(
                "invalid ODF-CR K384 replay for seed {:d}".format(seed)
            )
        _validate_allocation_ledger(replay["selector"], holdout_ids)
        deltas.append(
            _delta_pp(
                replay["metrics"],
                matrix["arm_metrics"]["d3_all"],
            )
        )
        allocation_hashes[str(seed)] = replay["selector"][
            "allocation_ledger_sha256"
        ]
    averages = [delta["average_mAP"] for delta in deltas]
    tiou = [delta["mAP_at_0_3_to_0_7"] for delta in deltas]
    mean = {
        "average_mAP": statistics.fmean(averages),
        "mAP_at_0_3_to_0_7": [
            statistics.fmean(values) for values in zip(*tiou)
        ],
    }
    sample_std = {
        "average_mAP": statistics.stdev(averages),
        "mAP_at_0_3_to_0_7": [
            statistics.stdev(values) for values in zip(*tiou)
        ],
    }
    gate_checks = {
        "mean_average_penalty_ge_minus_0_50_pp": (
            mean["average_mAP"] >= -0.50
        ),
        "mean_mAP_0_6_penalty_ge_minus_1_00_pp": (
            mean["mAP_at_0_3_to_0_7"][3] >= -1.00
        ),
        "mean_mAP_0_7_penalty_ge_minus_1_00_pp": (
            mean["mAP_at_0_3_to_0_7"][4] >= -1.00
        ),
    }
    support_gate_pass = all(gate_checks.values())
    payload = {
        "schema_version": "actionformer_odfcr_g3_internal_aggregate_v1",
        "validation_pass": True,
        "support_gate_pass": support_gate_pass,
        "gate_checks": gate_checks,
        "threshold_equality_passes": True,
        "delta_unit": "percentage_points",
        "development_seeds": list(EXPECTED_DEV_SEEDS),
        "k384_minus_all_query": {
            "mean": mean,
            "sample_standard_deviation": sample_std,
            "per_seed": deltas,
        },
        "allocation_ledger_sha256_by_seed": allocation_hashes,
        "g2_aggregate": g2_receipt,
        "matrix_receipts": matrix_receipts,
        "replay_receipts": replay_receipts,
        "source_split": "validation",
        "test_gt_used": False,
        "test_predictions_used": False,
        "paper_performance_row_allowed": False,
        "official_test_authorized": False,
        "efficiency_claim_allowed": False,
        "next_step_if_pass": "write separate official/cost preregistration",
        "next_step_if_fail": (
            "record support-density bottleneck and stop fixed K384"
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
                "support_gate_pass": support_gate_pass,
                "k384_minus_all_query": payload["k384_minus_all_query"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
