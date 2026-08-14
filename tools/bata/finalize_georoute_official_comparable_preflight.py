#!/usr/bin/env python3
"""Seal the no-performance official-comparable GeoRoute preflight."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_DIAGNOSTIC_ARMS,
    AMP_FORMAL_PREFLIGHT_PROFILE,
    amp_protocol_spec,
    diagnostic_cell_relative_path,
    validate_amp_diagnostic_job_receipt,
)
from tools.bata.georoute_amp_diagnostic_stage_runner import (  # noqa: E402
    validate_amp_diagnostic_stage_result,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA,
    read_json,
    validate_protocol_manifest,
    validate_world2_kat_receipt,
)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _validate_deployment(
    deployment: Mapping[str, Any],
    *,
    expected_commit: str,
    finalizer_job_id: str,
) -> tuple[dict[str, Any], str]:
    spec = amp_protocol_spec(AMP_FORMAL_PREFLIGHT_PROFILE)
    deployment = dict(deployment)
    jobs = validate_amp_diagnostic_job_receipt(
        deployment.get("jobs"), expected_finalizer=finalizer_job_id
    )
    kat_job_id = str(deployment.get("jobs", {}).get("world2_kat", ""))
    if (
        deployment.get("schema_version") != spec["deployment_schema"]
        or deployment.get("status") != spec["deployment_status"]
        or deployment.get("study_id") != spec["study_id"]
        or deployment.get("protocol_profile") != spec["profile"]
        or deployment.get("runtime_commit") != expected_commit
        or tuple(deployment.get("arms", ())) != AMP_DIAGNOSTIC_ARMS
        or int(deployment.get("seed", -1)) != int(spec["seed"])
        or not kat_job_id.isdigit()
        or kat_job_id in {
            jobs["finalizer"],
            *jobs["stage"].values(),
        }
        or deployment.get("official_test_opened") is not False
        or deployment.get("performance_inference_allowed") is not False
        or deployment.get("paper_claim_allowed") is not False
        or not _self_hash_matches(
            deployment, field="deployment_sha256"
        )
    ):
        raise ValueError("official-comparable preflight deployment is invalid")
    return jobs, kat_job_id


def _classify(
    *,
    stage_results: Mapping[str, Mapping[str, Any]],
    kat_passed: bool,
) -> dict[str, Any]:
    spec = amp_protocol_spec(AMP_FORMAL_PREFLIGHT_PROFILE)
    summaries = {
        arm: stage_results[arm]["diagnostic_receipt"]["summary"]
        for arm in AMP_DIAGNOSTIC_ARMS
        if arm in stage_results
    }
    per_arm_pass: dict[str, bool] = {}
    for arm in AMP_DIAGNOSTIC_ARMS:
        result = stage_results.get(arm)
        summary = summaries.get(arm, {})
        receipt = (
            result.get("diagnostic_receipt", {})
            if isinstance(result, Mapping)
            else {}
        )
        per_arm_pass[arm] = bool(
            isinstance(result, Mapping)
            and result.get("status") == spec["stage_pass_status"]
            and receipt.get("status") == spec["receipt_pass_status"]
            and int(summary.get("batch_count", -1))
            == int(spec["max_batches"])
            and int(summary.get("failed_attempt_count", 10**9))
            <= int(spec["max_skipped_attempts"])
            and int(
                summary.get("max_consecutive_skipped_attempts", 10**9)
            )
            <= int(spec["max_consecutive_skips"])
            and float(summary.get("minimum_observed_scale", -1.0))
            >= float(spec["minimum_scale"])
            and summary.get("stable_tail_all_success") is True
            and int(summary.get("stable_tail_success_count", -1))
            >= int(spec["stable_tail_batches"])
            and int(summary.get("retry_attempt_count", -1)) == 0
            and int(summary.get("replay_attempt_count", -1)) == 0
            and int(summary.get("scheduler_advance_count", -1))
            == int(spec["max_batches"])
            and int(summary.get("ema_update_count", -1))
            == int(spec["max_batches"])
            and summary.get("all_forward_losses_finite") is True
            and int(receipt.get("successful_updates", -1))
            >= int(spec["minimum_successful_updates"])
            and receipt.get("official_test_opened") is False
            and receipt.get("paper_claim_allowed") is False
        )

    matched_data_order = bool(
        len(summaries) == len(AMP_DIAGNOSTIC_ARMS)
        and list(
            summaries[AMP_DIAGNOSTIC_ARMS[0]].get(
                "data_fingerprint_sha256_by_batch", []
            )
        )
        == list(
            summaries[AMP_DIAGNOSTIC_ARMS[1]].get(
                "data_fingerprint_sha256_by_batch", []
            )
        )
    )
    skip_counts = {
        arm: int(summary.get("failed_attempt_count", -1))
        for arm, summary in summaries.items()
    }
    final_scales = {}
    for arm, summary in summaries.items():
        value = summary.get("final_scale")
        try:
            final_scales[arm] = float(value)
        except (TypeError, ValueError):
            final_scales[arm] = math.nan
    finite_scales = bool(
        len(final_scales) == len(AMP_DIAGNOSTIC_ARMS)
        and all(
            math.isfinite(value) and value > 0
            for value in final_scales.values()
        )
    )
    final_scale_ratio = (
        max(final_scales.values()) / min(final_scales.values())
        if finite_scales
        else math.inf
    )
    cross_arm_skip_delta = (
        abs(
            skip_counts[AMP_DIAGNOSTIC_ARMS[0]]
            - skip_counts[AMP_DIAGNOSTIC_ARMS[1]]
        )
        if len(skip_counts) == len(AMP_DIAGNOSTIC_ARMS)
        else 10**9
    )
    passed = bool(
        all(per_arm_pass.values())
        and kat_passed
        and matched_data_order
        and cross_arm_skip_delta <= int(spec["max_cross_arm_skip_delta"])
        and final_scale_ratio <= float(spec["max_final_scale_ratio"])
    )
    return {
        "passed": passed,
        "decision": (
            "FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED"
            if passed
            else "OFFICIAL_COMPARABLE_PREFLIGHT_HOLD"
        ),
        "per_arm_pass": per_arm_pass,
        "world2_fp32_ddp_kat_passed": kat_passed,
        "matched_data_order": matched_data_order,
        "skip_counts": skip_counts,
        "cross_arm_skip_delta": cross_arm_skip_delta,
        "final_scales": final_scales,
        "final_scale_ratio": (
            final_scale_ratio if math.isfinite(final_scale_ratio) else None
        ),
    }


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    expected_commit = str(args.expected_commit).lower()
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    if not slurm_job_id.isdigit():
        raise RuntimeError(
            "official-comparable preflight finalizer requires Slurm"
        )
    deployment_path = run_root / "control" / "deployment.json"
    protocol_path = run_root / "control" / "protocol_manifest.json"
    deployment = read_json(deployment_path)
    jobs, kat_job_id = _validate_deployment(
        deployment,
        expected_commit=expected_commit,
        finalizer_job_id=slurm_job_id,
    )
    protocol = validate_protocol_manifest(read_json(protocol_path))
    if (
        protocol["runtime_commit"] != expected_commit
        or deployment.get("protocol_manifest_file_sha256")
        != sha256_file(protocol_path)
        or deployment.get("protocol_sha256")
        != protocol["protocol_sha256"]
    ):
        raise ValueError("preflight protocol manifest binding changed")

    stage_results: dict[str, dict[str, Any]] = {}
    stage_failures: dict[str, dict[str, Any]] = {}
    for arm in AMP_DIAGNOSTIC_ARMS:
        result_path = (
            run_root
            / diagnostic_cell_relative_path(
                arm=arm,
                protocol_profile=AMP_FORMAL_PREFLIGHT_PROFILE,
            )
            / "stage_result.json"
        )
        try:
            result = validate_amp_diagnostic_stage_result(
                read_json(result_path),
                expected_arm=arm,
                expected_commit=expected_commit,
                expected_job_id=jobs["stage"][arm],
                expected_profile=AMP_FORMAL_PREFLIGHT_PROFILE,
            )
        except BaseException as error:
            stage_failures[arm] = {
                "path": str(result_path),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
            }
            continue
        stage_results[arm] = result

    kat_path = run_root / "control" / "world2_fp32_ddp_kat.json"
    kat = None
    kat_failure = None
    try:
        kat = validate_world2_kat_receipt(
            read_json(kat_path),
            expected_commit=expected_commit,
            expected_slurm_job_id=kat_job_id,
        )
    except BaseException as error:
        kat_failure = {
            "path": str(kat_path),
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
        }

    classification = _classify(
        stage_results=stage_results,
        kat_passed=kat is not None,
    )
    passed = bool(classification["passed"])
    finalization: dict[str, Any] = {
        "schema_version": OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA,
        "status": (
            "PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY"
            if passed
            else "INCOMPLETE_OFFICIAL_COMPARABLE_PREFLIGHT"
        ),
        "decision": classification["decision"],
        "runtime_commit": expected_commit,
        "run_root": str(run_root),
        "slurm_job_id": slurm_job_id,
        "deployment_path": str(deployment_path),
        "deployment_file_sha256": sha256_file(deployment_path),
        "protocol_manifest_path": str(protocol_path),
        "protocol_manifest_file_sha256": sha256_file(protocol_path),
        "protocol_sha256": protocol["protocol_sha256"],
        "stage_results": {
            arm: {
                "path": str(
                    run_root
                    / diagnostic_cell_relative_path(
                        arm=arm,
                        protocol_profile=AMP_FORMAL_PREFLIGHT_PROFILE,
                    )
                    / "stage_result.json"
                ),
                "file_sha256": sha256_file(
                    run_root
                    / diagnostic_cell_relative_path(
                        arm=arm,
                        protocol_profile=AMP_FORMAL_PREFLIGHT_PROFILE,
                    )
                    / "stage_result.json"
                ),
                "stage_result_sha256": result["stage_result_sha256"],
                "status": result["status"],
            }
            for arm, result in stage_results.items()
        },
        "stage_failures": stage_failures,
        "world2_fp32_ddp_kat": (
            {
                "path": str(kat_path),
                "file_sha256": sha256_file(kat_path),
                "kat_sha256": kat["kat_sha256"],
                "status": kat["status"],
            }
            if kat is not None
            else None
        ),
        "world2_fp32_ddp_kat_failure": kat_failure,
        "classification": classification,
        "formal_development_matrix_authorized": passed,
        "official_protocol_freeze_authorized": False,
        "official_test_opened": False,
        "performance_metrics": {},
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    output = run_root / "control" / "finalization.json"
    if output.exists():
        raise FileExistsError("preflight finalization already exists")
    _atomic_write_json(output, finalization)
    print(json.dumps(finalization, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
