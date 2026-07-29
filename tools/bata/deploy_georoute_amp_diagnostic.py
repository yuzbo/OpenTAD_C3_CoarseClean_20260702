#!/usr/bin/env python3
"""Submit the two-arm, no-metric GeoRoute real-batch AMP diagnostic DAG."""

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

from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA,
    AMP_DIAGNOSTIC_FINALIZATION_SCHEMA,
    AMP_DIAGNOSTIC_PROFILE,
    AMP_DIAGNOSTIC_STUDY_ID,
    AMP_STABILITY_PROFILE,
    amp_protocol_spec,
    validate_amp_diagnostic_job_receipt,
)
from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_ARMS,
    PILOT_FINALIZATION_SCHEMA,
    PILOT_SEED,
    PILOT_STUDY_ID,
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
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _full_hex(value: str, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != length
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def _clean_export(value: str, *, name: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"{name} is unsafe for sbatch --export")
    return value


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


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
            raise ValueError("unsupported AMP diagnostic dependency type")
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
        raise RuntimeError(f"invalid sbatch job id for {name}")
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
            "failed to release held AMP diagnostic jobs: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )


def _require_submit_capacity(*, additional_jobs: int) -> dict[str, int]:
    user = os.environ.get("USER")
    if not user:
        raise RuntimeError("AMP diagnostic cannot determine Slurm user")
    active = subprocess.run(
        ["squeue", "-h", "-u", user],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if active.returncode != 0:
        raise RuntimeError(
            f"AMP diagnostic cannot query squeue: {active.stderr.strip()}"
        )
    active_count = len(
        [line for line in active.stdout.splitlines() if line.strip()]
    )
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
            "AMP diagnostic cannot query MaxSubmitJobs: "
            + association.stderr.strip()
        )
    limits = [
        int(line.split("|", 1)[0])
        for line in association.stdout.splitlines()
        if line.split("|", 1)[0].strip().isdigit()
    ]
    if not limits:
        raise RuntimeError("AMP diagnostic cannot determine MaxSubmitJobs")
    limit = min(limits)
    if active_count + int(additional_jobs) > limit:
        raise RuntimeError(
            "AMP diagnostic refuses partial submission: "
            f"active={active_count}, required={additional_jobs}, limit={limit}"
        )
    return {
        "active_jobs": active_count,
        "additional_jobs": int(additional_jobs),
        "max_submit_jobs": limit,
        "headroom_after_submission": (
            limit - active_count - int(additional_jobs)
        ),
    }


def _validate_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError("AMP diagnostic requires its sealed pilot parent")
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("AMP diagnostic parent file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != PILOT_FINALIZATION_SCHEMA
        or parent.get("study_id") != PILOT_STUDY_ID
        or parent.get("status") != "INCOMPLETE_EXPLORATORY_PILOT"
        or parent.get("decision")
        != "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE"
        or parent.get("runtime_commit") != expected_runtime_commit
        or "residual_pl_rep_off" not in parent.get("failures", {})
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError("AMP diagnostic parent is not the sealed failed pilot")
    return parent


def _validate_diagnostic_parent(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "AMP stability gate requires the sealed matched diagnostic"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("AMP stability diagnostic-parent file hash mismatch")
    parent = _read_json(path)
    if (
        parent.get("schema_version") != AMP_DIAGNOSTIC_FINALIZATION_SCHEMA
        or parent.get("study_id") != AMP_DIAGNOSTIC_STUDY_ID
        or parent.get("status") != "COMPLETE_NUMERICAL_DIAGNOSTIC_ONLY"
        or parent.get("decision")
        != "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("all_arms_passed") is not True
        or parent.get("repair_authorized") is not True
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError(
            "AMP stability parent is not the repair-authorizing diagnostic"
        )
    return parent


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--parent-pilot-finalization", type=Path, required=True)
    parser.add_argument("--expected-parent-file-sha256", required=True)
    parser.add_argument("--expected-parent-runtime-commit", required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--protocol-profile",
        choices=(AMP_DIAGNOSTIC_PROFILE, AMP_STABILITY_PROFILE),
        default=AMP_DIAGNOSTIC_PROFILE,
    )
    parser.add_argument("--parent-diagnostic-finalization", type=Path)
    parser.add_argument("--expected-diagnostic-file-sha256")
    parser.add_argument("--expected-diagnostic-runtime-commit")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = amp_protocol_spec(args.protocol_profile)
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("AMP diagnostic root leaves the remote write boundary")
    if run_root.exists():
        raise FileExistsError("AMP diagnostic namespace exists; refusing resume")
    expected_commit = _full_hex(
        args.expected_commit,
        length=40,
        name="--expected-commit",
    )
    expected_parent_file_sha256 = _full_hex(
        args.expected_parent_file_sha256,
        length=64,
        name="--expected-parent-file-sha256",
    )
    expected_parent_runtime_commit = _full_hex(
        args.expected_parent_runtime_commit,
        length=40,
        name="--expected-parent-runtime-commit",
    )
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("AMP diagnostic source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("AMP diagnostic deployment requires clean source")
    parent_path = args.parent_pilot_finalization.resolve()
    parent = _validate_parent(
        parent_path,
        expected_file_sha256=expected_parent_file_sha256,
        expected_runtime_commit=expected_parent_runtime_commit,
    )
    diagnostic_parent = None
    diagnostic_parent_path = None
    expected_diagnostic_file_sha256 = None
    if args.protocol_profile == AMP_STABILITY_PROFILE:
        if (
            args.parent_diagnostic_finalization is None
            or args.expected_diagnostic_file_sha256 is None
            or args.expected_diagnostic_runtime_commit is None
        ):
            raise ValueError(
                "AMP stability gate requires exact diagnostic-parent arguments"
            )
        diagnostic_parent_path = (
            args.parent_diagnostic_finalization.resolve()
        )
        expected_diagnostic_file_sha256 = _full_hex(
            args.expected_diagnostic_file_sha256,
            length=64,
            name="--expected-diagnostic-file-sha256",
        )
        expected_diagnostic_runtime_commit = _full_hex(
            args.expected_diagnostic_runtime_commit,
            length=40,
            name="--expected-diagnostic-runtime-commit",
        )
        diagnostic_parent = _validate_diagnostic_parent(
            diagnostic_parent_path,
            expected_file_sha256=expected_diagnostic_file_sha256,
            expected_runtime_commit=expected_diagnostic_runtime_commit,
        )
    elif any(
        value is not None
        for value in (
            args.parent_diagnostic_finalization,
            args.expected_diagnostic_file_sha256,
            args.expected_diagnostic_runtime_commit,
        )
    ):
        raise ValueError(
            "diagnostic-parent arguments are stability-gate only"
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
    for name, input_path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not input_path.is_dir() or any(
                part.lower()
                in {"test", "testing", "test_videos", "official_test"}
                for part in input_path.parts
            ):
                raise ValueError(
                    "AMP diagnostic development video root is invalid"
                )
        elif not input_path.is_file():
            raise FileNotFoundError(input_path)

    # All admission gates precede immutable namespace creation.
    capacity = _require_submit_capacity(additional_jobs=3)
    storage = storage_capacity_receipt(run_root, cell_count=2)
    stage_script = ROOT / "scripts" / "run_georoute_amp_diagnostic_stage_slurm.sh"
    control_script = (
        ROOT / "scripts" / "run_georoute_amp_diagnostic_control_slurm.sh"
    )
    for script in (stage_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_AMP_DIAGNOSTIC_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_AMP_PROTOCOL_PROFILE": args.protocol_profile,
        **{name: str(path) for name, path in inputs.items()},
    }
    base_exports = {
        key: _clean_export(value, name=key)
        for key, value in base_values.items()
    }
    stage_exports = {
        arm: {
            **base_exports,
            "GEOROUTE_AMP_DIAGNOSTIC_ARM": arm,
        }
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_AMP_DIAGNOSTIC_ACTION": "finalize",
    }
    logs = run_root / "slurm"
    for arm in AMP_DIAGNOSTIC_ARMS:
        _sbatch(
            name=f"{spec['job_prefix']}_{PILOT_ARMS[arm]['slug']}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            gpu=True,
            test_only=True,
        )
    _sbatch(
        name=f"{spec['job_prefix']}_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        gpu=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("diagnostic", "control", "slurm"):
        (run_root / directory).mkdir()
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
        stage_jobs: dict[str, str] = {}
        for arm in AMP_DIAGNOSTIC_ARMS:
            job_id = _sbatch(
                name=f"{spec['job_prefix']}_{PILOT_ARMS[arm]['slug']}",
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                gpu=True,
                hold=True,
            )
            stage_jobs[arm] = job_id
            submitted.append(job_id)
        finalizer_job = _sbatch(
            name=f"{spec['job_prefix']}_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            gpu=False,
            dependency=list(stage_jobs.values()),
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        jobs = validate_amp_diagnostic_job_receipt(
            {
                "stage": stage_jobs,
                "finalizer": finalizer_job,
            }
        )
        deployment: dict[str, Any] = {
            "schema_version": spec["deployment_schema"],
            "status": spec["deployment_status"],
            "study_id": spec["study_id"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arms": list(AMP_DIAGNOSTIC_ARMS),
            "seed": PILOT_SEED,
            "jobs": jobs,
            "input_receipts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in inputs.items()
            },
            "parent_pilot": {
                "path": str(parent_path),
                "file_sha256": expected_parent_file_sha256,
                "finalization_sha256": parent["finalization_sha256"],
                "runtime_commit": parent["runtime_commit"],
                "decision": parent["decision"],
            },
            "parent_diagnostic": (
                {
                    "path": str(diagnostic_parent_path),
                    "file_sha256": expected_diagnostic_file_sha256,
                    "finalization_sha256": diagnostic_parent[
                        "finalization_sha256"
                    ],
                    "runtime_commit": diagnostic_parent["runtime_commit"],
                    "decision": diagnostic_parent["decision"],
                }
                if diagnostic_parent is not None
                else None
            ),
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "two_diagnostic_arms_parallel": True,
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
            "schema_version": spec["deployment_schema"],
            "status": spec["finalizer_submission_status"],
            "protocol_profile": spec["profile"],
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
            "schema_version": spec["deployment_schema"],
            "status": spec["stage_release_status"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "stage_job_ids": list(stage_jobs.values()),
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
