#!/usr/bin/env python3
"""Validate the official ActionFormer matched S0 pair and freeze its verdict."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


PAIR_SCHEMA = "actionformer_official_matched_pair_completion_v1"
ARM_SCHEMA = "actionformer_official_matched_arm_completion_v1"
ATTESTATION_SCHEMA = "actionformer_independent_metric_attestation_v1"
VERDICT_SCHEMA = "actionformer_sparsehead_s0_screening_verdict_v1"
METRIC_KEYS = (
    "average_mAP",
    "mAP@0.3",
    "mAP@0.4",
    "mAP@0.5",
    "mAP@0.6",
    "mAP@0.7",
)
SCREENING_THRESHOLDS = {
    "average_mAP": -0.01,
    "mAP@0.6": -0.015,
    "mAP@0.7": -0.015,
}


class ProtocolError(RuntimeError):
    """Raised when a frozen S0 evidence contract does not validate."""


def require(condition, message):
    if not condition:
        raise ProtocolError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    path = Path(path).resolve()
    require(path.is_file(), f"missing JSON artifact: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        raise ProtocolError(f"cannot parse JSON artifact {path}: {error}") from error
    require(isinstance(payload, dict), f"JSON artifact is not an object: {path}")
    return payload


def validate_receipt(record, label):
    require(isinstance(record, dict), f"{label} receipt is not an object")
    require(set(record) == {"path", "sha256", "size_bytes"}, f"{label} receipt key drift")
    path = Path(record["path"]).resolve()
    require(path.is_file(), f"{label} artifact is missing: {path}")
    require(
        path.stat().st_size == int(record["size_bytes"]),
        f"{label} size mismatch",
    )
    require(sha256_file(path) == record["sha256"], f"{label} SHA-256 mismatch")
    return path


def validate_metrics(metrics, label):
    require(isinstance(metrics, dict), f"{label} metrics are not an object")
    require(set(metrics) == set(METRIC_KEYS), f"{label} metric key drift")
    result = {}
    for key in METRIC_KEYS:
        value = float(metrics[key])
        require(math.isfinite(value), f"{label}.{key} is not finite")
        require(0.0 <= value <= 1.0, f"{label}.{key} is outside [0, 1]")
        result[key] = value
    expected_average = sum(result[key] for key in METRIC_KEYS[1:]) / 5.0
    require(
        abs(result["average_mAP"] - expected_average) <= 1e-12,
        f"{label}.average_mAP is not the mean of mAP@0.3--0.7",
    )
    return result


def validate_arm(pair, arm_name):
    arm_path = validate_receipt(pair["arms"][arm_name], f"{arm_name}.ARM_COMPLETE")
    arm = load_json(arm_path)
    require(arm.get("schema_version") == ARM_SCHEMA, f"{arm_name} ARM schema drift")
    require(arm.get("validation_pass") is True, f"{arm_name} ARM did not validate")
    require(arm.get("arm") == arm_name, f"{arm_name} ARM name mismatch")
    require(arm.get("seed") == 1234567891, f"{arm_name} seed mismatch")
    require(arm.get("new_training") is True, f"{arm_name} was not freshly trained")
    require(
        arm.get("paper_main_table_eligible") is False,
        f"{arm_name} illegally claims main-table eligibility",
    )
    require(
        arm.get("end_to_end_cost_claim_allowed") is False,
        f"{arm_name} illegally claims end-to-end cost",
    )
    schedule = arm.get("schedule")
    require(isinstance(schedule, dict), f"{arm_name} schedule is missing")
    expected_schedule = {
        "optimizer_epochs": 30,
        "warmup_epochs": 5,
        "executed_epochs": 35,
        "terminal_checkpoint_epoch": 35,
        "evaluated_weights": "state_dict_ema",
        "resume": False,
    }
    require(schedule == expected_schedule, f"{arm_name} schedule drift")

    receipts = arm.get("receipts")
    require(isinstance(receipts, dict), f"{arm_name} receipts are missing")
    required_receipts = {
        "config",
        "checkpoint",
        "raw_predictions",
        "train_log",
        "saveonly_log",
        "eval_log",
        "independent_metric_attestation",
    }
    require(set(receipts) == required_receipts, f"{arm_name} receipt set drift")
    resolved = {
        key: validate_receipt(value, f"{arm_name}.{key}")
        for key, value in receipts.items()
    }

    attestation = load_json(resolved["independent_metric_attestation"])
    require(
        attestation.get("schema_version") == ATTESTATION_SCHEMA,
        f"{arm_name} independent attestation schema drift",
    )
    require(
        attestation.get("validation_pass") is True,
        f"{arm_name} independent attestation did not validate",
    )
    require(attestation.get("issues") == [], f"{arm_name} independent issues are nonempty")
    require(
        attestation.get("paper_main_table_eligible") is False,
        f"{arm_name} independent attestation claim drift",
    )
    raw_record = attestation.get("raw_predictions")
    require(isinstance(raw_record, dict), f"{arm_name} raw attestation is missing")
    require(
        raw_record.get("complete_official_test_coverage") is True,
        f"{arm_name} does not cover the official test set",
    )
    require(
        raw_record.get("evaluated_video_count") == 212,
        f"{arm_name} evaluated-video count drift",
    )
    require(
        raw_record.get("prediction_video_count") == 212,
        f"{arm_name} prediction-video count drift",
    )
    require(
        raw_record.get("prediction_count") == 42400,
        f"{arm_name} prediction count drift",
    )
    require(
        raw_record.get("sha256") == receipts["raw_predictions"]["sha256"],
        f"{arm_name} raw prediction SHA binding mismatch",
    )
    native_metrics = validate_metrics(arm.get("metrics"), f"{arm_name}.ARM")
    independent_metrics = validate_metrics(
        attestation.get("recomputed_metrics"),
        f"{arm_name}.independent",
    )
    for key in METRIC_KEYS:
        require(
            abs(native_metrics[key] - independent_metrics[key]) <= 1e-12,
            f"{arm_name}.{key} differs from independent recomputation",
        )
    return arm, native_metrics


def build_verdict(pair_path, expected_pair_sha256):
    pair_path = Path(pair_path).resolve()
    require(pair_path.is_file(), f"missing matched pair: {pair_path}")
    observed_pair_sha256 = sha256_file(pair_path)
    require(
        observed_pair_sha256 == expected_pair_sha256,
        "matched-pair SHA-256 mismatch",
    )
    pair = load_json(pair_path)
    require(pair.get("schema_version") == PAIR_SCHEMA, "matched-pair schema drift")
    require(pair.get("validation_pass") is True, "matched pair did not validate")
    require(pair.get("issues") == [], "matched pair issues are nonempty")
    require(
        pair.get("experiment_role") == "matched_single_seed_screening",
        "matched-pair experiment role drift",
    )
    require(
        pair.get("paper_main_table_eligible") is False,
        "S0 cannot be paper-main-table eligible",
    )
    require(
        pair.get("model_route_terminal_claim_allowed") is False,
        "S0 cannot make a route-terminal claim",
    )
    require(
        pair.get("end_to_end_cost_claim_allowed") is False,
        "S0 cannot make an end-to-end cost claim",
    )
    comparability = pair.get("comparability")
    require(isinstance(comparability, dict) and comparability, "comparability is missing")
    require(all(value is True for value in comparability.values()), "comparability failed")
    require(set(pair.get("arms", {})) == {"dense", "sparse"}, "paired arm set drift")

    source = pair.get("source")
    require(isinstance(source, dict), "source evidence is missing")
    validate_receipt(source["source_diff_attestation"], "source_diff_attestation")
    for key, record in pair.get("data", {}).items():
        validate_receipt(record, f"data.{key}")
    environment = pair.get("environment")
    require(isinstance(environment, dict), "environment evidence is missing")
    require(
        environment.get("same_environment_for_both_arms") is True,
        "arm environment mismatch",
    )
    require(
        environment.get("official_softnms_7arg_probe") is True,
        "official Soft-NMS probe did not pass",
    )
    validate_receipt(
        environment["official_runtime_environment"],
        "official_runtime_environment",
    )
    validate_receipt(environment["official_nms_extension"], "official_nms_extension")

    dense_arm, dense_metrics = validate_arm(pair, "dense")
    sparse_arm, sparse_metrics = validate_arm(pair, "sparse")
    require(
        dense_arm.get("method_intervention") is None,
        "dense arm unexpectedly contains an intervention",
    )
    intervention = sparse_arm.get("method_intervention")
    require(isinstance(intervention, dict), "sparse intervention is missing")
    require(intervention.get("budget") == 384, "sparse budget drift")
    require(
        intervention.get("policy") == "stratified_uniform",
        "sparse selection policy drift",
    )
    require(
        intervention.get("training_loss_support") == "selected_native_grid_queries",
        "sparse training-loss support drift",
    )
    require(
        intervention.get("interpretation") == "method_intervention_not_execution_only",
        "sparse intervention interpretation drift",
    )

    pair_metrics = pair.get("metrics")
    require(isinstance(pair_metrics, dict), "pair metrics are missing")
    require(
        validate_metrics(pair_metrics.get("dense"), "pair.dense") == dense_metrics,
        "pair dense metrics differ from ARM",
    )
    require(
        validate_metrics(pair_metrics.get("sparse"), "pair.sparse") == sparse_metrics,
        "pair sparse metrics differ from ARM",
    )
    deltas = {
        key: sparse_metrics[key] - dense_metrics[key]
        for key in METRIC_KEYS
    }
    require(
        abs(
            float(pair_metrics.get("sparse_minus_dense_average_mAP"))
            - deltas["average_mAP"]
        )
        <= 1e-12,
        "pair average delta mismatch",
    )

    threshold_pass = {
        key: deltas[key] >= threshold
        for key, threshold in SCREENING_THRESHOLDS.items()
    }
    screening_pass = all(threshold_pass.values())
    reason_codes = [
        f"{key}_below_preregistered_bound"
        for key, passed in threshold_pass.items()
        if not passed
    ]
    if screening_pass:
        verdict = "GO_TO_PREREGISTERED_FIVE_SEED_STUDY"
        result_class = "legal_positive_screen"
    else:
        verdict = "KILL_CURRENT_K384_SELECTED_LOSS_INTERVENTION"
        result_class = "legal_negative_model_result"

    return {
        "schema_version": VERDICT_SCHEMA,
        "validation_pass": True,
        "issues": [],
        "pair_completion": {
            "path": str(pair_path),
            "sha256": observed_pair_sha256,
            "size_bytes": pair_path.stat().st_size,
        },
        "experiment_role": "matched_single_seed_screening",
        "result_class": result_class,
        "engineering_failure": False,
        "legal_model_result": True,
        "screening_pass": screening_pass,
        "verdict": verdict,
        "reason_codes": reason_codes,
        "metrics": {
            "dense": dense_metrics,
            "sparse": sparse_metrics,
            "sparse_minus_dense": deltas,
            "sparse_minus_dense_pp": {
                key: value * 100.0 for key, value in deltas.items()
            },
        },
        "preregistered_thresholds": {
            "sparse_minus_dense": SCREENING_THRESHOLDS,
            "sparse_minus_dense_pp": {
                key: value * 100.0
                for key, value in SCREENING_THRESHOLDS.items()
            },
            "pass": threshold_pass,
        },
        "continuation": {
            "five_seed_main_study_authorized": screening_pass,
            "detector_pipeline_cost_study_authorized": False,
            "no_retraining_2x2_attribution_authorized": True,
            "negative_result_analysis_required": not screening_pass,
        },
        "claim_boundary": (
            "official_comparable_single_seed_screening_only;"
            "paper_main_table_eligible=false;"
            "no_accuracy_preservation_or_efficiency_claim;"
            "negative_conclusion_applies_only_to_frozen_k384_selected_loss_intervention"
        ),
        "paper_main_table_eligible": False,
        "paper_ready": False,
        "model_route_terminal_claim_allowed": False,
        "end_to_end_cost_claim_allowed": False,
    }


def atomic_write_json(path, payload):
    path = Path(path).resolve()
    if path.exists():
        raise ProtocolError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-completion", required=True)
    parser.add_argument("--expected-pair-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    verdict = build_verdict(
        args.pair_completion,
        args.expected_pair_sha256,
    )
    atomic_write_json(args.output, verdict)
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
