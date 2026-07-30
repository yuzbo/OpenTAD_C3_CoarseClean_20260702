#!/usr/bin/env python3
"""Submit the matched two-arm GeoRoute gradient-decomposition DAG."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.deploy_georoute_amp_diagnostic import (  # noqa: E402
    _atomic_write_json,
    _cancel_jobs,
    _clean_export,
    _full_hex,
    _git_output,
    _inside,
    _release_jobs,
    _require_submit_capacity,
    _sbatch,
)
from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_STABILITY_V2_DEPLOYMENT_SCHEMA,
    AMP_STABILITY_V2_FINALIZATION_SCHEMA,
    AMP_STABILITY_V2_PROFILE,
    AMP_STABILITY_V2_STUDY_ID,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_gradient_decomposition import (  # noqa: E402
    ARMS,
    DEPLOYMENT_SCHEMA,
    DEPLOYMENT_STATUS,
    FINALIZER_SUBMISSION_STATUS,
    KAT_PASS_STATUS,
    KAT_SCHEMA,
    PROFILE,
    SEED,
    STAGE_RELEASE_STATUS,
    STUDY_ID,
    validate_job_receipt,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
MATCHED_PARENT_INPUTS = (
    "GEOROUTE_MANIFEST",
    "GEOROUTE_DEVELOPMENT_ANNOTATION",
    "GEOROUTE_CLASS_MAP",
    "GEOROUTE_DEVELOPMENT_VIDEO_ROOT",
    "GEOROUTE_PRETRAINED",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _stability_v2_skipped_batch_indices(
    receipt: Mapping[str, Any],
) -> list[int]:
    summary = receipt.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("stability-v2 receipt lacks a summary")
    values = summary.get("skipped_batch_indices")
    if (
        not isinstance(values, list)
        or any(not isinstance(value, int) or value < 0 for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError(
            "stability-v2 receipt has invalid skipped_batch_indices"
        )
    return list(values)


def _validate_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
    expected_arm_receipt_file_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "gradient decomposition requires the sealed stability-v2 parent"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("stability-v2 parent finalization file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != AMP_STABILITY_V2_FINALIZATION_SCHEMA
        or parent.get("study_id") != AMP_STABILITY_V2_STUDY_ID
        or parent.get("protocol_profile") != AMP_STABILITY_V2_PROFILE
        or parent.get("status")
        != "INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2"
        or parent.get("decision") != "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD"
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("all_arms_passed") is not False
        or parent.get("stability_gate_passed") is not False
        or parent.get("official_protocol_freeze_authorized") is not False
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError("stability-v2 parent is not the sealed fail-closed HOLD")
    deployment_path = Path(str(parent.get("deployment_path", ""))).resolve()
    if (
        not deployment_path.is_file()
        or sha256_file(deployment_path) != parent.get("deployment_file_sha256")
    ):
        raise ValueError("stability-v2 parent deployment changed")
    parent_deployment = _read_json(deployment_path)
    if (
        parent_deployment.get("schema_version")
        != AMP_STABILITY_V2_DEPLOYMENT_SCHEMA
        or parent_deployment.get("study_id") != AMP_STABILITY_V2_STUDY_ID
        or parent_deployment.get("protocol_profile") != AMP_STABILITY_V2_PROFILE
        or parent_deployment.get("runtime_commit") != expected_runtime_commit
        or not isinstance(parent_deployment.get("input_receipts"), Mapping)
        or not _self_hash_matches(
            parent_deployment, field="deployment_sha256"
        )
    ):
        raise ValueError("stability-v2 parent deployment is invalid")

    receipts: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_record = parent.get("arms", {}).get(arm)
        if not isinstance(arm_record, Mapping):
            raise ValueError(f"stability-v2 parent lacks arm {arm}")
        receipt_path = Path(
            str(arm_record.get("diagnostic_receipt_path", ""))
        ).resolve()
        expected_hash = expected_arm_receipt_file_sha256[arm]
        if (
            not receipt_path.is_file()
            or sha256_file(receipt_path) != expected_hash
            or arm_record.get("diagnostic_receipt_file_sha256")
            != expected_hash
        ):
            raise ValueError(f"stability-v2 {arm} receipt changed")
        receipt = validate_amp_diagnostic_receipt(
            _read_json(receipt_path),
            expected_arm=arm,
            expected_commit=expected_runtime_commit,
            expected_profile=AMP_STABILITY_V2_PROFILE,
        )
        expected_status = (
            "FAIL_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION"
            if arm == ARMS[0]
            else "PASS_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION_ONLY"
        )
        if receipt.get("status") != expected_status:
            raise ValueError(
                f"stability-v2 {arm} receipt has the wrong terminal status"
            )
        receipts[arm] = receipt
    return parent, parent_deployment, receipts


def _validate_kat(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "gradient decomposition requires its sealed CUDA KAT receipt"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("gradient-decomposition CUDA KAT file hash mismatch")
    kat = _read_json(path)
    bucket_records = kat.get("bucket_records")
    analytic = kat.get("analytic_gradient_direction")
    if (
        kat.get("schema_version") != KAT_SCHEMA
        or kat.get("status") != KAT_PASS_STATUS
        or kat.get("runtime_commit") != expected_runtime_commit
        or not str(kat.get("slurm_job_id", "")).isdigit()
        or int(kat.get("world_size", -1)) != 1
        or kat.get("standard_hook_future_completed") is not True
        or kat.get("observer_left_original_bucket_bitwise_unchanged") is not True
        or not isinstance(bucket_records, list)
        or not bucket_records
        or not any(
            record.get("telemetry", {}).get("cast_introduced_nonfinite") is True
            for record in bucket_records
        )
        or not isinstance(analytic, Mapping)
        or analytic.get("close") is not True
        or analytic.get("direction_positive") is not True
        or kat.get("checkpoint_emitted") is not False
        or kat.get("prediction_emitted") is not False
        or kat.get("evaluator_invoked") is not False
        or kat.get("official_test_opened") is not False
        or kat.get("performance_inference_allowed") is not False
        or kat.get("paper_claim_allowed") is not False
        or not _self_hash_matches(kat, field="kat_sha256")
    ):
        raise ValueError("gradient-decomposition CUDA KAT receipt is invalid")
    return kat


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--parent-stability-v2-finalization", type=Path, required=True
    )
    parser.add_argument(
        "--expected-parent-finalization-file-sha256", required=True
    )
    parser.add_argument("--expected-parent-runtime-commit", required=True)
    parser.add_argument("--expected-parent-pl-receipt-file-sha256", required=True)
    parser.add_argument("--expected-parent-st-receipt-file-sha256", required=True)
    parser.add_argument("--kat-receipt", type=Path, required=True)
    parser.add_argument("--expected-kat-receipt-file-sha256", required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--official-reference-config", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-origin-ref", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("gradient-decomposition root leaves write boundary")
    if run_root.exists():
        raise FileExistsError(
            "gradient-decomposition namespace exists; refusing resume"
        )
    expected_commit = _full_hex(
        args.expected_commit, length=40, name="--expected-commit"
    )
    expected_parent_commit = _full_hex(
        args.expected_parent_runtime_commit,
        length=40,
        name="--expected-parent-runtime-commit",
    )
    expected_parent_file_sha256 = _full_hex(
        args.expected_parent_finalization_file_sha256,
        length=64,
        name="--expected-parent-finalization-file-sha256",
    )
    expected_arm_hashes = {
        ARMS[0]: _full_hex(
            args.expected_parent_pl_receipt_file_sha256,
            length=64,
            name="--expected-parent-pl-receipt-file-sha256",
        ),
        ARMS[1]: _full_hex(
            args.expected_parent_st_receipt_file_sha256,
            length=64,
            name="--expected-parent-st-receipt-file-sha256",
        ),
    }
    expected_kat_file_sha256 = _full_hex(
        args.expected_kat_receipt_file_sha256,
        length=64,
        name="--expected-kat-receipt-file-sha256",
    )
    expected_origin_ref = str(args.expected_origin_ref)
    if (
        not expected_origin_ref.startswith("refs/remotes/origin/")
        or any(
            character in expected_origin_ref
            for character in (" ", "\t", "\n", "\r", "\x00")
        )
    ):
        raise ValueError("gradient decomposition requires a full origin ref")
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError(
            "gradient-decomposition source differs from --expected-commit"
        )
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("gradient-decomposition deployment requires clean source")
    if (
        _git_output("rev-parse", "--verify", expected_origin_ref).lower()
        != expected_commit
    ):
        raise RuntimeError("gradient-decomposition origin ref differs from source")

    parent_path = args.parent_stability_v2_finalization.resolve()
    parent, parent_deployment, parent_receipts = _validate_parent(
        parent_path,
        expected_file_sha256=expected_parent_file_sha256,
        expected_runtime_commit=expected_parent_commit,
        expected_arm_receipt_file_sha256=expected_arm_hashes,
    )
    kat_path = args.kat_receipt.resolve()
    kat = _validate_kat(
        kat_path,
        expected_file_sha256=expected_kat_file_sha256,
        expected_runtime_commit=expected_commit,
    )
    inputs = {
        "GEOROUTE_SOURCE_CONFIG": args.source_config.resolve(),
        "GEOROUTE_MANIFEST": args.manifest.resolve(),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": (
            args.development_annotation.resolve()
        ),
        "GEOROUTE_CLASS_MAP": args.class_map.resolve(),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": (
            args.development_video_root.resolve()
        ),
        "GEOROUTE_PRETRAINED": args.pretrained.resolve(),
        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG": (
            args.official_reference_config.resolve()
        ),
    }
    for name, input_path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not input_path.is_dir() or any(
                part.lower()
                in {"test", "testing", "test_videos", "official_test"}
                for part in input_path.parts
            ):
                raise ValueError(
                    "gradient-decomposition development video root is invalid"
                )
        elif not input_path.is_file() or input_path.is_symlink():
            raise FileNotFoundError(input_path)
    for name in ("GEOROUTE_SOURCE_CONFIG", "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"):
        if not _inside(inputs[name], ROOT):
            raise ValueError(f"{name} must come from the exact runtime checkout")
    input_receipts = {
        name: {
            "path": str(input_path),
            "sha256": sha256_file(input_path) if input_path.is_file() else None,
        }
        for name, input_path in inputs.items()
    }
    parent_inputs = parent_deployment["input_receipts"]
    if any(
        input_receipts[name] != parent_inputs.get(name)
        for name in MATCHED_PARENT_INPUTS
    ):
        raise ValueError(
            "gradient-decomposition immutable inputs differ from stability-v2"
        )
    for name in (
        "GEOROUTE_SOURCE_CONFIG",
        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG",
    ):
        if input_receipts[name]["sha256"] != parent_inputs.get(name, {}).get(
            "sha256"
        ):
            raise ValueError(
                f"gradient-decomposition {name} content differs from stability-v2"
            )

    parent_evidence: dict[str, Any] = {
        "stability_v2_finalization": {
            "path": str(parent_path),
            "file_sha256": expected_parent_file_sha256,
            "finalization_sha256": parent["finalization_sha256"],
            "runtime_commit": parent["runtime_commit"],
            "decision": parent["decision"],
        },
        "stability_v2_receipts": {
            arm: {
                "path": parent["arms"][arm]["diagnostic_receipt_path"],
                "file_sha256": expected_arm_hashes[arm],
                "receipt_sha256": parent_receipts[arm]["receipt_sha256"],
                "skipped_batch_indices": _stability_v2_skipped_batch_indices(
                    parent_receipts[arm]
                ),
            }
            for arm in ARMS
        },
        "cuda_observer_kat": {
            "path": str(kat_path),
            "file_sha256": expected_kat_file_sha256,
            "kat_sha256": kat["kat_sha256"],
            "slurm_job_id": kat["slurm_job_id"],
            "status": kat["status"],
        },
        "transitive_parent_chain": {
            key: parent_deployment.get(key)
            for key in (
                "parent_pilot",
                "parent_diagnostic",
                "parent_stability_v1",
                "official_reference_binding",
            )
        },
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    parent_evidence_sha256 = canonical_sha256(parent_evidence)

    # Every admission gate precedes immutable namespace creation.
    capacity = _require_submit_capacity(additional_jobs=3)
    storage = storage_capacity_receipt(run_root, cell_count=2)
    stage_script = (
        ROOT / "scripts" / "run_georoute_gradient_decomposition_stage_slurm.sh"
    )
    control_script = (
        ROOT / "scripts" / "run_georoute_gradient_decomposition_control_slurm.sh"
    )
    for script in (stage_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_GRADIENT_DECOMPOSITION_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key)
        for key, value in base_values.items()
    }
    stage_exports = {
        arm: {
            **base_exports,
            "GEOROUTE_GRADIENT_DECOMPOSITION_ARM": arm,
        }
        for arm in ARMS
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_GRADIENT_DECOMPOSITION_ACTION": "finalize",
    }
    logs = run_root / "slurm"
    job_slugs = {ARMS[0]: "pl", ARMS[1]: "st"}
    for arm in ARMS:
        _sbatch(
            name=f"grgdec_{job_slugs[arm]}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            gpu=True,
            test_only=True,
        )
    _sbatch(
        name="grgdec_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        gpu=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("diagnosis", "control", "slurm"):
        (run_root / directory).mkdir()
    _atomic_write_json(
        run_root / "control" / "submit_capacity_preflight.json", capacity
    )
    _atomic_write_json(
        run_root / "control" / "deployment_storage_preflight.json", storage
    )

    submitted: list[str] = []
    try:
        stage_jobs: dict[str, str] = {}
        for arm in ARMS:
            job_id = _sbatch(
                name=f"grgdec_{job_slugs[arm]}",
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                gpu=True,
                hold=True,
            )
            stage_jobs[arm] = job_id
            submitted.append(job_id)
        finalizer_job = _sbatch(
            name="grgdec_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            gpu=False,
            dependency=list(stage_jobs.values()),
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        jobs = validate_job_receipt(
            {"stage": stage_jobs, "finalizer": finalizer_job}
        )
        deployment: dict[str, Any] = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": DEPLOYMENT_STATUS,
            "study_id": STUDY_ID,
            "profile": PROFILE,
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arms": list(ARMS),
            "seed": SEED,
            "jobs": jobs,
            "input_receipts": input_receipts,
            "expected_origin_ref": expected_origin_ref,
            "origin_ref_parity_verified": True,
            "parent_stability_v2": {
                "path": str(parent_path),
                "file_sha256": expected_parent_file_sha256,
                "finalization_sha256": parent["finalization_sha256"],
                "runtime_commit": parent["runtime_commit"],
                "decision": parent["decision"],
            },
            "parent_evidence": parent_evidence,
            "parent_evidence_sha256": parent_evidence_sha256,
            "matched_parent_inputs": {
                "exact_path_and_content_names": list(MATCHED_PARENT_INPUTS),
                "content_matched_runtime_config_names": [
                    "GEOROUTE_SOURCE_CONFIG",
                    "GEOROUTE_OFFICIAL_REFERENCE_CONFIG",
                ],
                "all_equal": True,
            },
            "rng_matching_policy": {
                "data_fingerprint_all_batches_equal": True,
                "cpu_rng_all_batches_equal": True,
                "cuda_rng_batch_zero_equal": True,
                "cuda_rng_after_batch_zero_equality_required": False,
                "reason": "PL_Gumbel_sampling_consumes_CUDA_RNG_ST_does_not",
            },
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "two_arms_parallel": True,
                "stages_held_until_immutable_receipts": True,
                "finalizer_afterany_both_stages": True,
                "resume_allowed": False,
            },
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "deployment.json"
        _atomic_write_json(deployment_path, deployment)

        submission: dict[str, Any] = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": FINALIZER_SUBMISSION_STATUS,
            "profile": PROFILE,
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": list(stage_jobs.values()),
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        submission_path = run_root / "control" / "finalizer_submission.json"
        _atomic_write_json(submission_path, submission)
        _release_jobs(list(stage_jobs.values()))
        release: dict[str, Any] = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": STAGE_RELEASE_STATUS,
            "profile": PROFILE,
            "runtime_commit": expected_commit,
            "stage_job_ids": list(stage_jobs.values()),
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_submission_file_sha256": sha256_file(submission_path),
        }
        release["receipt_sha256"] = canonical_sha256(release)
        _atomic_write_json(
            run_root / "control" / "stage_release.json", release
        )
    except BaseException:
        _cancel_jobs(submitted)
        raise
    print(
        json.dumps(
            {**deployment, "finalizer_job_id": finalizer_job},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
