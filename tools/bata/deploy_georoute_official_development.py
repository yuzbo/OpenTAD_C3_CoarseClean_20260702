#!/usr/bin/env python3
"""Submit the legacy formal matrix or the atomic ZoomToken P1 first screen."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.deploy_georoute_amp_diagnostic import (  # noqa: E402
    _cancel_jobs,
    _clean_export,
    _full_hex,
    _git_output,
    _release_jobs,
    _require_submit_capacity,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    FORMAL_DEVELOPMENT_ARM_ORDER,
    FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    P1_DEVELOPMENT_SEED,
    P1_FIRST_SCREEN_ARM_ORDER,
    P1_DO_CONFIG_RELATIVE_PATH,
    p1_arm_spec,
    p1_source_config_relative_path,
    _validate_preflight_parent,
    formal_arm_spec,
    read_json,
    validate_protocol_manifest,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (  # noqa: E402
    P1_COST_LEAF_SPECS,
    P1_STUDY_ID,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
P1_CANCEL_MAX_POLLS = 10
P1_CANCEL_POLL_INTERVAL_SECONDS = 1.0


def _inside(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
    except ValueError:
        return False
    return path != boundary


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sbatch(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    stage: bool,
    dependency: Sequence[str] | None = None,
    dependency_type: str = "afterany",
    test_only: bool = False,
    hold: bool = False,
) -> str:
    command = [
        "sbatch",
        "--parsable",
        "--job-name",
        name,
        "--output",
        str(logs / f"{name}.%j.out"),
        "--error",
        str(logs / f"{name}.%j.err"),
    ]
    if test_only:
        command.append("--test-only")
    if hold and not test_only:
        command.append("--hold")
    if dependency:
        if dependency_type not in {"afterany", "afterok"}:
            raise ValueError("unsupported Slurm dependency type")
        command.extend(
            [
                "--dependency",
                dependency_type + ":" + ":".join(map(str, dependency)),
            ]
        )
    if stage:
        command.extend(
            [
                "--gpus",
                "2",
                "--cpus-per-task",
                "10",
                "--mem",
                "192000M",
            ]
        )
    else:
        # N16R4's batch partition is GPU-backed; reserve the minimum control
        # resource while the finalizer performs no CUDA work.
        command.extend(["--gpus", "1", "--cpus-per-task", "1"])
    command.extend(
        [
            "--export",
            ",".join(
                [
                    "ALL",
                    *(
                        f"{key}={value}"
                        for key, value in sorted(exports.items())
                    ),
                ]
            ),
            str(script),
        ]
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {name}: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    if test_only:
        return "TEST_ONLY_PASS"
    job_id = completed.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"invalid Slurm job ID for {name}")
    return job_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("formal", "p1"), default="formal")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preflight-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-origin-ref", required=True)
    parser.add_argument(
        "--expected-preflight-finalization-file-sha256",
        required=True,
    )
    parser.add_argument("--p1-runtime-container-image", type=Path)
    parser.add_argument("--p1-runtime-dependency-lock", type=Path)
    return parser.parse_args()


def _p1_source_config(arm: str, *, official_source: Path) -> Path:
    if arm == "DO":
        expected = Path(P1_DO_CONFIG_RELATIVE_PATH)
        if tuple(official_source.parts[-len(expected.parts) :]) != tuple(expected.parts):
            raise ValueError("P1 DO must retain the tracked official recipe")
        return official_source
    return (ROOT / p1_source_config_relative_path(arm)).resolve()


def _deployment_source_config_sha256(*, mode: str, protocol: Mapping) -> str:
    bridge = protocol["current_source_bridge"]
    if mode == "p1":
        return str(bridge["official_config_sha256"])
    return str(bridge["georoute_source_config_sha256"])


def _p1_job_ids(job_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(map(str, job_ids))
    if any(not job_id.isdigit() for job_id in normalized):
        raise ValueError("P1 control requires numeric Slurm job IDs")
    if len(set(normalized)) != len(normalized):
        raise ValueError("P1 control requires unique Slurm job IDs")
    return normalized


def _p1_receipt_self_hash(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> str:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    if not isinstance(observed, str) or observed != canonical_sha256(unsigned):
        raise ValueError(f"P1 {field} is invalid")
    return observed


def _validate_p1_pre_release_receipts(
    deployment_path: Path,
    submission_path: Path,
    *,
    expected_commit: str,
    expected_submitted: Sequence[str],
    expected_deployment_sha256: str,
    expected_submission_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reread the already-defined P1 receipts before releasing held jobs."""

    if not deployment_path.is_file() or not submission_path.is_file():
        raise FileNotFoundError("P1 deployment receipts are missing before release")
    deployment = read_json(deployment_path)
    deployment_sha256 = _p1_receipt_self_hash(
        deployment,
        field="deployment_sha256",
    )
    if deployment_sha256 != expected_deployment_sha256:
        raise ValueError("P1 deployment receipt changed before release")

    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    cost_jobs = jobs.get("cost") if isinstance(jobs, Mapping) else None
    seed_key = str(P1_DEVELOPMENT_SEED)
    if (
        deployment.get("schema_version") != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or deployment.get("study_id") != P1_STUDY_ID
        or deployment.get("mode") != "p1"
        or deployment.get("status") != "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX"
        or deployment.get("runtime_commit") != expected_commit
        or deployment.get("arms") != list(P1_FIRST_SCREEN_ARM_ORDER)
        or deployment.get("seed") != P1_DEVELOPMENT_SEED
        or deployment.get("seeds") != [P1_DEVELOPMENT_SEED]
        or deployment.get("accuracy_cells") != 5
        or deployment.get("cost_leaves") != 8
        or not isinstance(jobs, Mapping)
        or not isinstance(stage_jobs, Mapping)
        or set(stage_jobs) != set(P1_FIRST_SCREEN_ARM_ORDER)
        or not isinstance(cost_jobs, Mapping)
        or set(cost_jobs) != set(P1_COST_LEAF_SPECS)
        or deployment.get("dependency_policy")
        != {
            "all_fifteen_jobs_held_until_receipts_immutable": True,
            "accuracy_afterany_runtime_preflight": True,
            "cost_afterany_runtime_preflight_and_source_stages": True,
            "finalizer_afterany_all_fourteen_predecessors": True,
            "release_all_fifteen_atomically": True,
            "resume_allowed": False,
            "retry_allowed": False,
            "requeue_allowed": False,
        }
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
    ):
        raise ValueError("P1 deployment receipt contract is invalid before release")

    runtime_preflight_job = str(jobs.get("runtime_preflight", ""))
    finalizer_job = str(jobs.get("finalizer", ""))
    ordered_stage_jobs: list[str] = []
    for arm in P1_FIRST_SCREEN_ARM_ORDER:
        arm_jobs = stage_jobs[arm]
        if not isinstance(arm_jobs, Mapping) or set(arm_jobs) != {seed_key}:
            raise ValueError(f"P1 deployment stage receipt is invalid for {arm}")
        ordered_stage_jobs.append(str(arm_jobs[seed_key]))
    ordered_cost_jobs = [str(cost_jobs[leaf_id]) for leaf_id in P1_COST_LEAF_SPECS]
    predecessor_ids = _p1_job_ids(
        [runtime_preflight_job, *ordered_stage_jobs, *ordered_cost_jobs]
    )
    submitted_ids = _p1_job_ids([*predecessor_ids, finalizer_job])
    if len(predecessor_ids) != 14 or len(submitted_ids) != 15:
        raise ValueError("P1 deployment receipt does not bind the 15-job DAG")
    if submitted_ids != _p1_job_ids(expected_submitted):
        raise ValueError("P1 deployment receipt job population changed before release")

    submission = read_json(submission_path)
    submission_sha256 = _p1_receipt_self_hash(
        submission,
        field="receipt_sha256",
    )
    if (
        submission_sha256 != expected_submission_sha256
        or submission.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or submission.get("study_id") != P1_STUDY_ID
        or submission.get("mode") != "p1"
        or submission.get("status") != "SUBMITTED_P1_FINALIZER_AFTERANY"
        or submission.get("runtime_commit") != expected_commit
        or submission.get("deployment_file_sha256") != sha256_file(deployment_path)
        or str(submission.get("finalizer_job_id", "")) != finalizer_job
        or submission.get("dependency_type") != "afterany"
        or tuple(map(str, submission.get("predecessor_job_ids", ())))
        != predecessor_ids
    ):
        raise ValueError("P1 finalizer-submission receipt is invalid before release")
    return deployment, submission


def _release_p1_jobs_checked(job_ids: Sequence[str]) -> None:
    normalized = _p1_job_ids(job_ids)
    completed = subprocess.run(
        ["scontrol", "release", *normalized],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to release held P1 jobs: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _active_p1_job_ids(job_ids: Sequence[str]) -> tuple[str, ...]:
    normalized = _p1_job_ids(job_ids)
    completed = subprocess.run(
        [
            "squeue",
            "--noheader",
            "--jobs",
            ",".join(normalized),
            "--format=%A",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to verify P1 cancellation: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    active = {line.strip() for line in completed.stdout.splitlines() if line.strip()}
    return tuple(job_id for job_id in normalized if job_id in active)


def _cancel_p1_jobs_and_verify(
    job_ids: Sequence[str],
    *,
    max_polls: int = P1_CANCEL_MAX_POLLS,
    poll_interval_seconds: float = P1_CANCEL_POLL_INTERVAL_SECONDS,
) -> None:
    normalized = _p1_job_ids(job_ids)
    if not normalized:
        return
    if max_polls <= 0 or poll_interval_seconds < 0:
        raise ValueError("P1 cancellation verification bound is invalid")
    completed = subprocess.run(
        ["scancel", *normalized],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    cancel_error = None
    if completed.returncode != 0:
        cancel_error = completed.stderr.strip() or completed.stdout.strip()

    survivors: tuple[str, ...] = normalized
    verification_error = None
    try:
        for poll_index in range(max_polls):
            survivors = _active_p1_job_ids(normalized)
            if not survivors:
                break
            if poll_index + 1 < max_polls and poll_interval_seconds:
                time.sleep(poll_interval_seconds)
    except BaseException as error:
        verification_error = str(error)

    failures = []
    if cancel_error is not None:
        failures.append(f"scancel failed: {cancel_error}")
    if verification_error is not None:
        failures.append(f"terminal verification failed: {verification_error}")
    elif survivors:
        failures.append("surviving job IDs: " + ",".join(survivors))
    if failures:
        raise RuntimeError("P1 no-survivor cleanup failed; " + "; ".join(failures))


def _release_p1_jobs_from_receipts(
    deployment_path: Path,
    submission_path: Path,
    *,
    expected_commit: str,
    expected_submitted: Sequence[str],
    expected_deployment_sha256: str,
    expected_submission_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate, release, or leave no surviving job from the P1 population."""

    try:
        receipts = _validate_p1_pre_release_receipts(
            deployment_path,
            submission_path,
            expected_commit=expected_commit,
            expected_submitted=expected_submitted,
            expected_deployment_sha256=expected_deployment_sha256,
            expected_submission_sha256=expected_submission_sha256,
        )
        _release_p1_jobs_checked(expected_submitted)
    except BaseException as release_error:
        try:
            _cancel_p1_jobs_and_verify(expected_submitted)
        except BaseException as cleanup_error:
            raise RuntimeError(
                "P1 release boundary failed and no-survivor cleanup failed: "
                f"{cleanup_error}"
            ) from release_error
        raise
    return receipts


def _deploy_p1(
    *,
    args: argparse.Namespace,
    run_root: Path,
    expected_commit: str,
    expected_origin_ref: str,
    preflight_path: Path,
    preflight_file_hash: str,
    preflight: Mapping[str, Any],
    protocol: Mapping[str, Any],
    inputs: Mapping[str, Path],
) -> dict[str, Any]:
    """Submit one held, immutable 15-job P1 graph through the existing entry."""

    if args.p1_runtime_container_image is None or args.p1_runtime_dependency_lock is None:
        raise ValueError("P1 mode requires the immutable container image and dependency lock")
    container_image = args.p1_runtime_container_image.resolve()
    dependency_lock = args.p1_runtime_dependency_lock.resolve()
    if not container_image.is_file() or not dependency_lock.is_file():
        raise FileNotFoundError("P1 runtime image or dependency lock is missing")
    source_configs = {
        arm: _p1_source_config(
            arm,
            official_source=inputs["GEOROUTE_SOURCE_CONFIG"],
        )
        for arm in P1_FIRST_SCREEN_ARM_ORDER
    }
    for arm, path in source_configs.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if arm != "DO":
            p1_arm_spec(arm)

    capacity = _require_submit_capacity(additional_jobs=15)
    storage = storage_capacity_receipt(run_root, cell_count=13)
    stage_script = (
        ROOT / "scripts" / "run_georoute_official_development_stage_slurm.sh"
    )
    control_script = (
        ROOT / "scripts" / "run_georoute_official_development_control_slurm.sh"
    )
    for script in (stage_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    runtime_preflight_path = run_root / "control" / "runtime_preflight.json"
    common_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_OFFICIAL_DEVELOPMENT_MODE": "p1",
        "GEOROUTE_P1_RUNTIME_CONTAINER_IMAGE": str(container_image),
        "GEOROUTE_P1_RUNTIME_DEPENDENCY_LOCK": str(dependency_lock),
        "GEOROUTE_P1_RUNTIME_PREFLIGHT": str(runtime_preflight_path),
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key) for key, value in common_values.items()
    }

    def task_exports(task: str, *, attestation_name: str) -> dict[str, str]:
        return {
            **base_exports,
            "GEOROUTE_OFFICIAL_DEVELOPMENT_TASK": task,
            "GEOROUTE_P1_RUNTIME_ATTESTATION": str(
                run_root / "control" / "runtime_attestations" / attestation_name
            ),
        }

    preflight_exports = task_exports(
        "preflight", attestation_name="preflight_unused.json"
    )
    stage_exports: dict[str, dict[str, str]] = {}
    for arm in P1_FIRST_SCREEN_ARM_ORDER:
        stage_exports[arm] = {
            **task_exports("accuracy", attestation_name=f"accuracy_{arm}.json"),
            "GEOROUTE_OFFICIAL_DEVELOPMENT_ARM": arm,
            "GEOROUTE_OFFICIAL_DEVELOPMENT_SEED": str(P1_DEVELOPMENT_SEED),
            "GEOROUTE_SOURCE_CONFIG": str(source_configs[arm]),
        }
    cost_exports: dict[str, dict[str, str]] = {}
    for leaf_id, spec in P1_COST_LEAF_SPECS.items():
        cost_exports[leaf_id] = {
            **task_exports("cost", attestation_name=f"cost_{leaf_id}.json"),
            "GEOROUTE_P1_COST_LEAF_ID": leaf_id,
            "GEOROUTE_P1_COST_COMPARATOR": spec["comparator"],
            "GEOROUTE_P1_COST_ORDER": spec["order"],
            "GEOROUTE_OFFICIAL_DEVELOPMENT_SEED": str(P1_DEVELOPMENT_SEED),
        }

    # Test every exact batch shape before any namespace is created.
    _sbatch(
        name="ztp1_preflight",
        script=stage_script,
        logs=run_root / "slurm",
        exports=preflight_exports,
        stage=True,
        test_only=True,
    )
    for arm in P1_FIRST_SCREEN_ARM_ORDER:
        _sbatch(
            name=f"ztp1_{arm.lower()}_{P1_DEVELOPMENT_SEED}",
            script=stage_script,
            logs=run_root / "slurm",
            exports=stage_exports[arm],
            stage=True,
            test_only=True,
        )
    for leaf_id in P1_COST_LEAF_SPECS:
        _sbatch(
            name=f"ztp1_cost_{leaf_id.lower()}",
            script=stage_script,
            logs=run_root / "slurm",
            exports=cost_exports[leaf_id],
            stage=True,
            test_only=True,
        )
    _sbatch(
        name="ztp1_finalize",
        script=control_script,
        logs=run_root / "slurm",
        exports=base_exports,
        stage=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("development", "cost", "control", "slurm"):
        (run_root / directory).mkdir()
    protocol_path = run_root / "control" / "protocol_manifest.json"
    _atomic_write_json(protocol_path, protocol)
    _atomic_write_json(
        run_root / "control" / "submit_capacity_preflight.json", capacity
    )
    _atomic_write_json(
        run_root / "control" / "deployment_storage_preflight.json", storage
    )

    submitted: list[str] = []
    release_boundary_entered = False
    release_succeeded = False
    try:
        runtime_preflight_job = _sbatch(
            name="ztp1_preflight",
            script=stage_script,
            logs=run_root / "slurm",
            exports=preflight_exports,
            stage=True,
            hold=True,
        )
        submitted.append(runtime_preflight_job)
        stage_jobs: dict[str, dict[str, str]] = {}
        for arm in P1_FIRST_SCREEN_ARM_ORDER:
            job_id = _sbatch(
                name=f"ztp1_{arm.lower()}_{P1_DEVELOPMENT_SEED}",
                script=stage_script,
                logs=run_root / "slurm",
                exports=stage_exports[arm],
                stage=True,
                dependency=(runtime_preflight_job,),
                dependency_type="afterany",
                hold=True,
            )
            stage_jobs[arm] = {str(P1_DEVELOPMENT_SEED): job_id}
            submitted.append(job_id)
        cost_jobs: dict[str, str] = {}
        for leaf_id, spec in P1_COST_LEAF_SPECS.items():
            parents = tuple(
                dict.fromkeys(
                    (
                        runtime_preflight_job,
                        stage_jobs["Q"][str(P1_DEVELOPMENT_SEED)],
                        stage_jobs[spec["comparator"]][str(P1_DEVELOPMENT_SEED)],
                    )
                )
            )
            job_id = _sbatch(
                name=f"ztp1_cost_{leaf_id.lower()}",
                script=stage_script,
                logs=run_root / "slurm",
                exports=cost_exports[leaf_id],
                stage=True,
                dependency=parents,
                dependency_type="afterany",
                hold=True,
            )
            cost_jobs[leaf_id] = job_id
            submitted.append(job_id)
        predecessor_ids = [
            runtime_preflight_job,
            *(
                stage_jobs[arm][str(P1_DEVELOPMENT_SEED)]
                for arm in P1_FIRST_SCREEN_ARM_ORDER
            ),
            *(cost_jobs[leaf_id] for leaf_id in P1_COST_LEAF_SPECS),
        ]
        finalizer_job = _sbatch(
            name="ztp1_finalize",
            script=control_script,
            logs=run_root / "slurm",
            exports=base_exports,
            stage=False,
            dependency=predecessor_ids,
            dependency_type="afterany",
            hold=True,
        )
        submitted.append(finalizer_job)

        deployment: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "study_id": P1_STUDY_ID,
            "mode": "p1",
            "status": "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX",
            "runtime_commit": expected_commit,
            "origin_ref": expected_origin_ref,
            "origin_ref_parity_verified": True,
            "run_root": str(run_root),
            "arms": list(P1_FIRST_SCREEN_ARM_ORDER),
            "seed": P1_DEVELOPMENT_SEED,
            "seeds": [P1_DEVELOPMENT_SEED],
            "arm_specs": {
                "DO": {
                    **formal_arm_spec("dense_native"),
                    "causal_role": "official_recipe_reproduction_report_only",
                },
                **{
                    arm: p1_arm_spec(arm)
                    for arm in P1_FIRST_SCREEN_ARM_ORDER
                    if arm != "DO"
                },
            },
            "source_configs": {
                arm: {"path": str(path), "sha256": sha256_file(path)}
                for arm, path in source_configs.items()
            },
            "accuracy_cells": 5,
            "cost_leaves": 8,
            "jobs": {
                "runtime_preflight": runtime_preflight_job,
                "stage": stage_jobs,
                "cost": cost_jobs,
                "finalizer": finalizer_job,
            },
            "runtime_attestation": {
                "preflight_path": str(runtime_preflight_path),
                "leaf_paths": {
                    arm: stage_exports[arm]["GEOROUTE_P1_RUNTIME_ATTESTATION"]
                    for arm in P1_FIRST_SCREEN_ARM_ORDER
                },
                "cost_leaf_paths": {
                    leaf_id: cost_exports[leaf_id]["GEOROUTE_P1_RUNTIME_ATTESTATION"]
                    for leaf_id in P1_COST_LEAF_SPECS
                },
                "container_image": str(container_image),
                "container_image_sha256": sha256_file(container_image),
                "dependency_lock": str(dependency_lock),
                "dependency_lock_sha256": sha256_file(dependency_lock),
                "expected_visible_gpu_count": 2,
                "before_numpy_model_cuda_data_checkpoint": True,
            },
            "cost_protocol": {
                "leaf_specs": P1_COST_LEAF_SPECS,
                "physical_windows": 136,
                "video_clusters": 40,
                "warmup_windows_before_each_pass": 136,
                "power_interval_ms": 20,
                "bootstrap_replicates": 10_000,
                "dn_only_controlling_denominator": True,
                "q_over_dn_upper_bound_limit": 0.85,
                "do_mandatory_report_only": True,
            },
            "preflight_finalization_path": str(preflight_path.resolve()),
            "preflight_finalization_file_sha256": preflight_file_hash,
            "preflight_finalization_sha256": preflight["finalization_sha256"],
            "protocol_manifest_path": str(protocol_path.resolve()),
            "protocol_manifest_file_sha256": sha256_file(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
            "input_receipts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in inputs.items()
            },
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "all_fifteen_jobs_held_until_receipts_immutable": True,
                "accuracy_afterany_runtime_preflight": True,
                "cost_afterany_runtime_preflight_and_source_stages": True,
                "finalizer_afterany_all_fourteen_predecessors": True,
                "release_all_fifteen_atomically": True,
                "resume_allowed": False,
                "retry_allowed": False,
                "requeue_allowed": False,
            },
            "development_gate_only": True,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "deployment.json"
        _atomic_write_json(deployment_path, deployment)
        submission: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "study_id": P1_STUDY_ID,
            "mode": "p1",
            "status": "SUBMITTED_P1_FINALIZER_AFTERANY",
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": predecessor_ids,
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        submission_path = run_root / "control" / "finalizer_submission.json"
        _atomic_write_json(submission_path, submission)
        release_boundary_entered = True
        _release_p1_jobs_from_receipts(
            deployment_path,
            submission_path,
            expected_commit=expected_commit,
            expected_submitted=submitted,
            expected_deployment_sha256=deployment["deployment_sha256"],
            expected_submission_sha256=submission["receipt_sha256"],
        )
        release_succeeded = True
        release: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "study_id": P1_STUDY_ID,
            "mode": "p1",
            "status": "RELEASED_ATOMIC_P1_FIFTEEN_JOB_DAG",
            "runtime_commit": expected_commit,
            "released_job_ids": submitted,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_submission_file_sha256": sha256_file(submission_path),
        }
        release["receipt_sha256"] = canonical_sha256(release)
        _atomic_write_json(run_root / "control" / "stage_release.json", release)
    except BaseException as deployment_error:
        if not release_boundary_entered or release_succeeded:
            try:
                _cancel_p1_jobs_and_verify(submitted)
            except BaseException as cleanup_error:
                raise RuntimeError(
                    "P1 deployment failed and no-survivor cleanup failed: "
                    f"{cleanup_error}"
                ) from deployment_error
        raise
    return {**deployment, "finalizer_job_id": finalizer_job}


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    preflight_root = args.preflight_root.resolve()
    boundary = BOUNDARY.resolve()
    if not _inside(run_root, boundary) or not _inside(
        preflight_root, boundary
    ):
        raise ValueError("formal deployment root leaves remote boundary")
    if run_root.exists():
        raise FileExistsError("formal namespace exists; refusing resume")
    expected_commit = _full_hex(
        args.expected_commit,
        length=40,
        name="--expected-commit",
    )
    preflight_file_hash = _full_hex(
        args.expected_preflight_finalization_file_sha256,
        length=64,
        name="--expected-preflight-finalization-file-sha256",
    )
    expected_origin_ref = str(args.expected_origin_ref)
    if (
        not expected_origin_ref.startswith("refs/remotes/origin/")
        or any(
            character in expected_origin_ref
            for character in (" ", "\t", "\n", "\r", "\x00")
        )
    ):
        raise ValueError("--expected-origin-ref must be a full origin ref")
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("formal deployment source commit changed")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("formal deployment requires a clean source")
    if (
        _git_output("rev-parse", "--verify", expected_origin_ref).lower()
        != expected_commit
    ):
        raise RuntimeError("formal deployment origin ref changed")

    preflight_path = preflight_root / "control" / "finalization.json"
    preflight = _validate_preflight_parent(
        preflight_path,
        expected_file_sha256=preflight_file_hash,
        expected_runtime_commit=expected_commit,
    )
    protocol_source = preflight_root / "control" / "protocol_manifest.json"
    if (
        not protocol_source.is_file()
        or sha256_file(protocol_source)
        != preflight.get("protocol_manifest_file_sha256")
    ):
        raise ValueError("preflight protocol manifest changed")
    protocol = validate_protocol_manifest(read_json(protocol_source))
    if (
        protocol.get("protocol_sha256") != preflight.get("protocol_sha256")
        or protocol.get("runtime_commit") != expected_commit
    ):
        raise ValueError("preflight protocol does not bind this source")

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
    }
    protocol_inputs = protocol["development_inputs"]
    expected_paths = {
        "GEOROUTE_MANIFEST": protocol_inputs["manifest_path"],
        "GEOROUTE_DEVELOPMENT_ANNOTATION": protocol_inputs[
            "development_annotation"
        ]["path"],
        "GEOROUTE_CLASS_MAP": protocol_inputs["class_map_path"],
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": protocol_inputs[
            "development_video_root"
        ],
        "GEOROUTE_PRETRAINED": protocol_inputs[
            "pretrained_checkpoint_path"
        ],
    }
    for name, path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
        if name in expected_paths and str(path) != str(
            Path(expected_paths[name]).resolve()
        ):
            raise ValueError(f"{name} differs from the frozen protocol")
    if (
        sha256_file(inputs["GEOROUTE_SOURCE_CONFIG"])
        != _deployment_source_config_sha256(mode=args.mode, protocol=protocol)
        or sha256_file(inputs["GEOROUTE_MANIFEST"])
        != protocol_inputs["manifest_file_sha256"]
        or sha256_file(inputs["GEOROUTE_DEVELOPMENT_ANNOTATION"])
        != protocol_inputs["development_annotation"]["sha256"]
        or sha256_file(inputs["GEOROUTE_CLASS_MAP"])
        != protocol_inputs["class_map_sha256"]
        or sha256_file(inputs["GEOROUTE_PRETRAINED"])
        != protocol_inputs["pretrained_checkpoint_sha256"]
    ):
        raise ValueError("formal deployment input hash changed")

    if args.mode == "p1":
        deployment = _deploy_p1(
            args=args,
            run_root=run_root,
            expected_commit=expected_commit,
            expected_origin_ref=expected_origin_ref,
            preflight_path=preflight_path,
            preflight_file_hash=preflight_file_hash,
            preflight=preflight,
            protocol=protocol,
            inputs=inputs,
        )
        print(json.dumps(deployment, indent=2, sort_keys=True))
        return 0

    capacity = _require_submit_capacity(additional_jobs=16)
    storage = storage_capacity_receipt(run_root, cell_count=15)
    stage_script = (
        ROOT / "scripts" / "run_georoute_official_development_stage_slurm.sh"
    )
    control_script = (
        ROOT
        / "scripts"
        / "run_georoute_official_development_control_slurm.sh"
    )
    for script in (stage_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)
    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_OFFICIAL_DEVELOPMENT_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key)
        for key, value in base_values.items()
    }
    stage_exports: dict[tuple[str, int], dict[str, str]] = {}
    for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
        formal_arm_spec(arm)
        for seed in FORMAL_DEVELOPMENT_SEEDS:
            stage_exports[(arm, seed)] = {
                **base_exports,
                "GEOROUTE_OFFICIAL_DEVELOPMENT_ARM": arm,
                "GEOROUTE_OFFICIAL_DEVELOPMENT_SEED": str(seed),
            }
            _sbatch(
                name=f"groff_{arm[:8]}_{seed}",
                script=stage_script,
                logs=run_root / "slurm",
                exports=stage_exports[(arm, seed)],
                stage=True,
                test_only=True,
            )
    _sbatch(
        name="groff_finalize",
        script=control_script,
        logs=run_root / "slurm",
        exports=base_exports,
        stage=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("development", "control", "slurm"):
        (run_root / directory).mkdir()
    protocol_path = run_root / "control" / "protocol_manifest.json"
    _atomic_write_json(protocol_path, protocol)
    _atomic_write_json(
        run_root / "control" / "submit_capacity_preflight.json",
        capacity,
    )
    _atomic_write_json(
        run_root / "control" / "deployment_storage_preflight.json",
        storage,
    )
    submitted: list[str] = []
    try:
        stage_jobs: dict[str, dict[str, str]] = {
            arm: {} for arm in FORMAL_DEVELOPMENT_ARM_ORDER
        }
        for arm in FORMAL_DEVELOPMENT_ARM_ORDER:
            for seed in FORMAL_DEVELOPMENT_SEEDS:
                job_id = _sbatch(
                    name=f"groff_{arm[:8]}_{seed}",
                    script=stage_script,
                    logs=run_root / "slurm",
                    exports=stage_exports[(arm, seed)],
                    stage=True,
                    hold=True,
                )
                stage_jobs[arm][str(seed)] = job_id
                submitted.append(job_id)
        predecessor_ids = [
            stage_jobs[arm][str(seed)]
            for arm in FORMAL_DEVELOPMENT_ARM_ORDER
            for seed in FORMAL_DEVELOPMENT_SEEDS
        ]
        finalizer_job = _sbatch(
            name="groff_finalize",
            script=control_script,
            logs=run_root / "slurm",
            exports=base_exports,
            stage=False,
            dependency=predecessor_ids,
        )
        submitted.append(finalizer_job)
        deployment: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "status": "SUBMITTED_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX",
            "runtime_commit": expected_commit,
            "origin_ref": expected_origin_ref,
            "origin_ref_parity_verified": True,
            "run_root": str(run_root),
            "arms": list(FORMAL_DEVELOPMENT_ARM_ORDER),
            "arm_specs": {
                arm: formal_arm_spec(arm)
                for arm in FORMAL_DEVELOPMENT_ARM_ORDER
            },
            "seeds": list(FORMAL_DEVELOPMENT_SEEDS),
            "cells": 15,
            "jobs": {
                "stage": stage_jobs,
                "finalizer": finalizer_job,
            },
            "preflight_finalization_path": str(preflight_path.resolve()),
            "preflight_finalization_file_sha256": preflight_file_hash,
            "preflight_finalization_sha256": preflight[
                "finalization_sha256"
            ],
            "protocol_manifest_path": str(protocol_path.resolve()),
            "protocol_manifest_file_sha256": sha256_file(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
            "input_receipts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in inputs.items()
            },
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "all_fifteen_cells_parallel": True,
                "cells_held_until_receipts_immutable": True,
                "finalizer_afterany_all_fifteen": True,
                "resume_allowed": False,
            },
            "development_gate_only": True,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "deployment.json"
        _atomic_write_json(deployment_path, deployment)
        submission: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "status": "SUBMITTED_DEVELOPMENT_FINALIZER_AFTERANY",
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": predecessor_ids,
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        submission_path = (
            run_root / "control" / "finalizer_submission.json"
        )
        _atomic_write_json(submission_path, submission)
        _release_jobs(predecessor_ids)
        release: dict[str, Any] = {
            "schema_version": FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
            "status": "RELEASED_ALL_FIFTEEN_DEVELOPMENT_CELLS",
            "runtime_commit": expected_commit,
            "released_job_ids": predecessor_ids,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_submission_file_sha256": sha256_file(submission_path),
        }
        release["receipt_sha256"] = canonical_sha256(release)
        _atomic_write_json(
            run_root / "control" / "stage_release.json",
            release,
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
