#!/usr/bin/env python3
"""Submit the independent six-arm GeoRoute estimator pilot DAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.finalize_georoute_estimator_pilot_p0 import (  # noqa: E402
    _validate_preexperiment_parent,
)
from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_ARM_ORDER,
    PILOT_ARMS,
    PILOT_DEPLOYMENT_SCHEMA,
    PILOT_EPOCHS,
    PILOT_K,
    PILOT_SEED,
    PILOT_STUDY_ID,
    pilot_arm_spec,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")
CONTROL_SLURM_ARGS = ("--gpus", "1", "--cpus-per-task", "1")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _git_output(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(args)} failed"
        )
    return completed.stdout.strip()


def _clean_export(value: str, *, name: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"{name} is unsafe for sbatch --export")
    return value


def _full_hex(value: str, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != length
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def _sbatch(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    gpu: bool,
    dependency: Sequence[str] | None = None,
    dependency_type: str = "afterok",
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
        if dependency_type not in {"afterok", "afterany"}:
            raise ValueError("unsupported estimator pilot dependency type")
        command.extend(
            [
                "--dependency",
                f"{dependency_type}:" + ":".join(map(str, dependency)),
            ]
        )
    command.extend(GPU_OUTER_SLURM_ARGS if gpu else CONTROL_SLURM_ARGS)
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
        raise RuntimeError(f"invalid sbatch job id for {name}: {completed.stdout!r}")
    return job_id


def _cancel_jobs(job_ids: Sequence[str]) -> None:
    if not job_ids:
        return
    subprocess.run(
        ["scancel", *map(str, job_ids)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def _release_jobs(job_ids: Sequence[str]) -> None:
    completed = subprocess.run(
        ["scontrol", "release", *map(str, job_ids)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to release held estimator pilot P0 jobs: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _require_submit_capacity(*, additional_jobs: int) -> dict[str, int]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("estimator pilot cannot determine Slurm user")
    active = subprocess.run(
        ["squeue", "-h", "-u", user],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode != 0:
        raise RuntimeError(
            f"estimator pilot cannot query squeue: {active.stderr.strip()}"
        )
    active_count = len(
        [line for line in active.stdout.splitlines() if line.strip()]
    )
    # N16R4 currently enforces a finite MaxSubmitJobs association.  Query it
    # before creating the immutable namespace so a matrix cannot be half-built.
    association = subprocess.run(
        [
            "sacctmgr",
            "-n",
            "-P",
            "show",
            "assoc",
            "where",
            f"user={user}",
            "format=MaxSubmitJobs",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if association.returncode != 0:
        raise RuntimeError(
            "estimator pilot cannot query MaxSubmitJobs: "
            + association.stderr.strip()
        )
    limits = [
        int(line.split("|", 1)[0])
        for line in association.stdout.splitlines()
        if line.split("|", 1)[0].strip().isdigit()
    ]
    if not limits:
        raise RuntimeError("estimator pilot cannot determine MaxSubmitJobs")
    limit = min(limits)
    if active_count + int(additional_jobs) > limit:
        raise RuntimeError(
            "estimator pilot refuses partial submission: "
            f"active={active_count}, required={additional_jobs}, limit={limit}"
        )
    return {
        "active_jobs": active_count,
        "additional_jobs": int(additional_jobs),
        "max_submit_jobs": limit,
        "headroom_after_submission": limit - active_count - int(additional_jobs),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--preexperiment-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--expected-preexperiment-runtime-commit",
        required=True,
    )
    parser.add_argument(
        "--expected-source-experiment-commit",
        required=True,
    )
    parser.add_argument(
        "--expected-preexperiment-finalization-sha256",
        required=True,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    parent_root = args.preexperiment_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary) or run_root == boundary:
        raise ValueError("estimator pilot root leaves the remote write boundary")
    if run_root.exists():
        raise FileExistsError("estimator pilot namespace exists; refusing resume")
    if not _inside(parent_root, boundary) or not parent_root.is_dir():
        raise ValueError("estimator pilot preexperiment parent is invalid")
    expected_commit = _full_hex(
        args.expected_commit,
        length=40,
        name="--expected-commit",
    )
    expected_parent_runtime_commit = _full_hex(
        args.expected_preexperiment_runtime_commit,
        length=40,
        name="--expected-preexperiment-runtime-commit",
    )
    expected_source_experiment_commit = _full_hex(
        args.expected_source_experiment_commit,
        length=40,
        name="--expected-source-experiment-commit",
    )
    expected_parent_finalization_sha256 = _full_hex(
        args.expected_preexperiment_finalization_sha256,
        length=64,
        name="--expected-preexperiment-finalization-sha256",
    )
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("estimator pilot source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("estimator pilot deployment requires clean source")

    parent_finalization_path = parent_root / "control" / "finalization.json"
    parent_finalization_file_sha256 = sha256_file(parent_finalization_path)
    parent = _validate_preexperiment_parent(
        path=parent_finalization_path,
        expected_path_sha256=parent_finalization_file_sha256,
        expected_runtime_commit=expected_parent_runtime_commit,
        expected_source_experiment_commit=expected_source_experiment_commit,
        expected_finalization_sha256=expected_parent_finalization_sha256,
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
    }
    for name, path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not path.is_dir() or any(
                component.lower()
                in {"test", "testing", "test_videos", "official_test"}
                for component in path.parts
            ):
                raise ValueError("estimator pilot development video root is invalid")
        elif not path.is_file():
            raise FileNotFoundError(path)

    capacity = _require_submit_capacity(additional_jobs=14)
    deployment_storage = storage_capacity_receipt(run_root, cell_count=6)
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("p0", "pilot", "control", "slurm"):
        (run_root / directory).mkdir()
    _atomic_write_json(
        run_root / "control" / "deployment_storage_preflight.json",
        deployment_storage,
    )
    _atomic_write_json(
        run_root / "control" / "submit_capacity_preflight.json",
        capacity,
    )

    p0_script = ROOT / "scripts" / "run_georoute_p0_slurm.sh"
    stage_script = (
        ROOT / "scripts" / "run_georoute_estimator_pilot_stage_slurm.sh"
    )
    control_script = (
        ROOT / "scripts" / "run_georoute_estimator_pilot_control_slurm.sh"
    )
    for path in (p0_script, stage_script, control_script):
        if not path.is_file():
            raise FileNotFoundError(path)

    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_PILOT_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key)
        for key, value in base_values.items()
    }
    p0_exports: dict[str, dict[str, str]] = {}
    stage_exports: dict[str, dict[str, str]] = {}
    for arm in PILOT_ARM_ORDER:
        spec = pilot_arm_spec(arm)
        p0_exports[arm] = {
            **base_exports,
            "GEOROUTE_P0_OUTPUT": str(run_root / "p0" / f"{arm}.json"),
            "GEOROUTE_P0_PILOT_ARM": arm,
            "GEOROUTE_P0_ROUTE_MODE": str(spec["route_mode"]),
            "GEOROUTE_P0_POLICY_ESTIMATOR": str(spec["policy_estimator"]),
            "GEOROUTE_P0_TOKENS_PER_TUBELET": str(PILOT_K),
            "GEOROUTE_P0_HEIGHT": "180",
            "GEOROUTE_P0_WIDTH": "320",
            "GEOROUTE_P0_CONTEXT_TOKENS": str(spec["context_tokens"]),
            "GEOROUTE_P0_ROI_FRACTION": str(spec["roi_fraction"]),
            "GEOROUTE_P0_GEOMETRY_SIDE_CHANNEL": str(
                spec["geometry_side_channel"]
            ).lower(),
            "GEOROUTE_P0_ABSOLUTE_POSITION_ENABLED": "true",
            "GEOROUTE_P0_ABSOLUTE_COORDINATES_ENABLED": str(
                spec["absolute_coordinates_enabled"]
            ).lower(),
            "GEOROUTE_P0_ROI_RELATIVE_COORDINATES_ENABLED": str(
                spec["roi_relative_coordinates_enabled"]
            ).lower(),
            "GEOROUTE_P0_GEOMETRY_PROJECTION_ENABLED": str(
                spec["geometry_projection_enabled"]
            ).lower(),
            "GEOROUTE_P0_POLICY_TEMPERATURE": str(
                spec["policy_temperature"]
            ),
            "GEOROUTE_P0_SCORE_FUNCTION_WEIGHT": str(
                spec["score_function_weight"]
            ),
            "GEOROUTE_P0_SCORE_FUNCTION_BASELINE_MOMENTUM": str(
                spec["score_function_baseline_momentum"]
            ),
        }
        stage_exports[arm] = {
            **base_exports,
            "GEOROUTE_PILOT_ARM": arm,
        }
    p0_finalizer_exports = {
        **base_exports,
        "GEOROUTE_PILOT_ACTION": "p0-finalize",
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_PILOT_ACTION": "finalize",
    }
    logs = run_root / "slurm"

    for arm in PILOT_ARM_ORDER:
        slug = PILOT_ARMS[arm]["slug"]
        _sbatch(
            name=f"grep0_{slug}",
            script=p0_script,
            logs=logs,
            exports=p0_exports[arm],
            gpu=True,
            test_only=True,
        )
        _sbatch(
            name=f"grep_{slug}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            gpu=True,
            test_only=True,
        )
    _sbatch(
        name="grep0_finalize",
        script=control_script,
        logs=logs,
        exports=p0_finalizer_exports,
        gpu=False,
        test_only=True,
    )
    _sbatch(
        name="grep_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        gpu=False,
        test_only=True,
    )

    submitted: list[str] = []
    jobs: dict[str, Any] = {"p0": {}, "stage": {}}
    try:
        for arm in PILOT_ARM_ORDER:
            job_id = _sbatch(
                name=f"grep0_{PILOT_ARMS[arm]['slug']}",
                script=p0_script,
                logs=logs,
                exports=p0_exports[arm],
                gpu=True,
                hold=True,
            )
            jobs["p0"][arm] = job_id
            submitted.append(job_id)
        jobs["p0_finalizer"] = _sbatch(
            name="grep0_finalize",
            script=control_script,
            logs=logs,
            exports=p0_finalizer_exports,
            gpu=False,
            dependency=list(jobs["p0"].values()),
            dependency_type="afterany",
        )
        submitted.append(jobs["p0_finalizer"])
        for arm in PILOT_ARM_ORDER:
            job_id = _sbatch(
                name=f"grep_{PILOT_ARMS[arm]['slug']}",
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                gpu=True,
                dependency=[jobs["p0_finalizer"]],
                dependency_type="afterany",
            )
            jobs["stage"][arm] = job_id
            submitted.append(job_id)

        deployment: dict[str, Any] = {
            "schema_version": PILOT_DEPLOYMENT_SCHEMA,
            "status": "SUBMITTED_SIX_ARM_EXPLORATORY_PILOT",
            "study_id": PILOT_STUDY_ID,
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arms": list(PILOT_ARM_ORDER),
            "arm_specs": {
                arm: pilot_arm_spec(arm)
                for arm in PILOT_ARM_ORDER
            },
            "seed": PILOT_SEED,
            "epochs": PILOT_EPOCHS,
            "token_budget": PILOT_K,
            "input_receipts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in inputs.items()
            },
            "preexperiment_parent": {
                "root": str(parent_root),
                "path": str(parent_finalization_path.resolve()),
                "file_sha256": parent_finalization_file_sha256,
                "finalization_sha256": parent["finalization_sha256"],
                "runtime_commit": parent["runtime_commit"],
                "source_experiment_commit": parent["source_experiment_commit"],
                "expected_runtime_commit": expected_parent_runtime_commit,
                "expected_source_experiment_commit": (
                    expected_source_experiment_commit
                ),
                "expected_finalization_sha256": (
                    expected_parent_finalization_sha256
                ),
                "decision": parent["decision"],
            },
            "storage_preflight": deployment_storage,
            "submit_capacity_preflight": capacity,
            "jobs": jobs,
            "dependency_policy": {
                "p0_six_parallel": True,
                "p0_finalizer_afterany_all_p0": True,
                "training_wrappers_afterany_p0_finalizer": True,
                "training_requires_valid_p0_suite_before_cell_creation": True,
                "exploratory_finalizer_afterany_all_leaves": True,
                "p0_held_until_receipts_written": True,
            },
            "training_jobs_submitted": True,
            "training_completed": False,
            "single_seed_exploratory": True,
            "old_selector_reused": False,
            "selector_emitted": False,
            "p2_p3_opened": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "deployment.json"
        _atomic_write_json(deployment_path, deployment)

        predecessor_ids = [
            *jobs["p0"].values(),
            jobs["p0_finalizer"],
            *jobs["stage"].values(),
        ]
        finalizer_job = _sbatch(
            name="grep_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            gpu=False,
            dependency=predecessor_ids,
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        submission: dict[str, Any] = {
            "schema_version": PILOT_DEPLOYMENT_SCHEMA,
            "status": "SUBMITTED_EXPLORATORY_FINALIZER_AFTERANY",
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": predecessor_ids,
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        _atomic_write_json(
            run_root / "control" / "finalizer_submission.json",
            submission,
        )
        _release_jobs(list(jobs["p0"].values()))
        release: dict[str, Any] = {
            "schema_version": PILOT_DEPLOYMENT_SCHEMA,
            "status": "RELEASED_P0_AFTER_IMMUTABLE_RECEIPTS",
            "runtime_commit": expected_commit,
            "p0_job_ids": list(jobs["p0"].values()),
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_submission_file_sha256": sha256_file(
                run_root / "control" / "finalizer_submission.json"
            ),
        }
        release["receipt_sha256"] = canonical_sha256(release)
        _atomic_write_json(
            run_root / "control" / "p0_release.json",
            release,
        )
    except Exception:
        _cancel_jobs(submitted)
        raise
    print(
        json.dumps(
            {
                **deployment,
                "finalizer_job_id": finalizer_job,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
