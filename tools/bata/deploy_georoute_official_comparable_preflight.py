#!/usr/bin/env python3
"""Submit the fail-closed official-comparable GeoRoute resource preflight."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


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
    _sbatch,
)
from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_DIAGNOSTIC_ARMS,
    AMP_FORMAL_PREFLIGHT_PROFILE,
    AMP_REPAIR_FINALIZATION_SCHEMA,
    AMP_REPAIR_PROFILE,
    AMP_REPAIR_STUDY_ID,
    amp_protocol_spec,
    validate_amp_diagnostic_job_receipt,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    OFFICIAL_CONFIG_RELATIVE_PATH,
    OFFICIAL_CONFIG_SHA256,
    OFFICIAL_UPSTREAM_RELEASE_COMMIT,
    build_protocol_manifest,
    read_json,
)
from tools.bata.georoute_storage import (  # noqa: E402
    no_artifact_storage_capacity_receipt,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
OFFICIAL_UPSTREAM_TRACKING_REF = (
    "refs/remotes/origin/adatad-release-01c58b9"
)


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


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _repo_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _sbatch_world2(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
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
    # N16R4 binds 55 GB to each requested GPU and rejects every explicit
    # ``--mem`` override in its submit Lua policy.  The inner KAT step therefore
    # inherits this two-GPU allocation instead of making a second memory
    # request.
    command.extend(["--gpus", "2", "--cpus-per-task", "12"])
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


def _validate_repair_parent(
    path: Path,
    *,
    expected_file_sha256: str,
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "official-comparable preflight requires the sealed repair parent"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("repair-parent finalization file hash mismatch")
    parent = read_json(path)
    if (
        parent.get("schema_version") != AMP_REPAIR_FINALIZATION_SCHEMA
        or parent.get("study_id") != AMP_REPAIR_STUDY_ID
        or parent.get("protocol_profile") != AMP_REPAIR_PROFILE
        or parent.get("status")
        != "COMPLETE_DDP_FP16_CAST_REPAIR_GATE_ONLY"
        or parent.get("decision")
        != (
            "DDP_FP16_CAST_REPAIR_GATE_PASS_"
            "MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED"
        )
        or parent.get("all_arms_passed") is not True
        or parent.get("repair_gate_passed") is not True
        or parent.get("matched_formal_protocol_freeze_authorized")
        is not True
        or parent.get("official_protocol_freeze_authorized") is not False
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(parent, field="finalization_sha256")
    ):
        raise ValueError("repair parent did not authorize a matched formal gate")
    return parent


def _validate_upstream_snapshot(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not (path / ".git").exists():
        raise FileNotFoundError("official upstream snapshot is not a git clone")
    head = _repo_output(path, "rev-parse", "HEAD").lower()
    tracked = _repo_output(
        path, "rev-parse", "--verify", OFFICIAL_UPSTREAM_TRACKING_REF
    ).lower()
    status = _repo_output(
        path, "status", "--porcelain=v1", "--untracked-files=all"
    )
    origin = _repo_output(path, "remote", "get-url", "origin")
    config = path / OFFICIAL_CONFIG_RELATIVE_PATH
    if (
        head != OFFICIAL_UPSTREAM_RELEASE_COMMIT
        or tracked != OFFICIAL_UPSTREAM_RELEASE_COMMIT
        or status
        or origin != "https://github.com/sming256/OpenTAD.git"
        or not config.is_file()
        or sha256_file(config) != OFFICIAL_CONFIG_SHA256
    ):
        raise ValueError(
            "official upstream snapshot is not the clean pinned AdaTAD release"
        )
    return {
        "path": str(path),
        "origin": origin,
        "head": head,
        "origin_tracking_ref": OFFICIAL_UPSTREAM_TRACKING_REF,
        "origin_tracking_ref_commit": tracked,
        "clean_tree": True,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--official-reference-config", type=Path, required=True)
    parser.add_argument("--official-upstream-snapshot", type=Path, required=True)
    parser.add_argument("--repair-parent-finalization", type=Path, required=True)
    parser.add_argument(
        "--expected-repair-parent-file-sha256", required=True
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-origin-ref", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    spec = amp_protocol_spec(AMP_FORMAL_PREFLIGHT_PROFILE)
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("official-comparable preflight root leaves boundary")
    if run_root.exists():
        raise FileExistsError(
            "official-comparable preflight namespace exists; refusing resume"
        )
    expected_commit = _full_hex(
        args.expected_commit, length=40, name="--expected-commit"
    )
    expected_parent_hash = _full_hex(
        args.expected_repair_parent_file_sha256,
        length=64,
        name="--expected-repair-parent-file-sha256",
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
        raise RuntimeError("preflight source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("preflight deployment requires a clean source")
    if (
        _git_output("rev-parse", "--verify", expected_origin_ref).lower()
        != expected_commit
    ):
        raise RuntimeError("preflight origin ref differs from source")

    repair_parent_path = args.repair_parent_finalization.resolve()
    repair_parent = _validate_repair_parent(
        repair_parent_path,
        expected_file_sha256=expected_parent_hash,
    )
    upstream_snapshot = _validate_upstream_snapshot(
        args.official_upstream_snapshot
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
    for name, path in inputs.items():
        if name == "GEOROUTE_DEVELOPMENT_VIDEO_ROOT":
            if not path.is_dir():
                raise FileNotFoundError(path)
        elif not path.is_file():
            raise FileNotFoundError(path)
    for name in (
        "GEOROUTE_SOURCE_CONFIG",
        "GEOROUTE_OFFICIAL_REFERENCE_CONFIG",
    ):
        if not _inside(inputs[name], ROOT):
            raise ValueError(f"{name} must come from the runtime checkout")

    protocol = build_protocol_manifest(
        runtime_commit=expected_commit,
        runtime_origin_ref=expected_origin_ref,
        current_official_config_path=inputs[
            "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"
        ],
        georoute_source_config_path=inputs["GEOROUTE_SOURCE_CONFIG"],
        manifest_path=inputs["GEOROUTE_MANIFEST"],
        development_annotation_path=inputs[
            "GEOROUTE_DEVELOPMENT_ANNOTATION"
        ],
        class_map_path=inputs["GEOROUTE_CLASS_MAP"],
        development_video_root=inputs[
            "GEOROUTE_DEVELOPMENT_VIDEO_ROOT"
        ],
        pretrained_checkpoint_path=inputs["GEOROUTE_PRETRAINED"],
        repair_parent=repair_parent,
    )
    input_receipts = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }
        for name, path in inputs.items()
    }

    capacity = _require_submit_capacity(additional_jobs=4)
    storage = no_artifact_storage_capacity_receipt(
        run_root,
        leaf_count=len(AMP_DIAGNOSTIC_ARMS),
    )
    stage_script = ROOT / "scripts" / "run_georoute_amp_diagnostic_stage_slurm.sh"
    kat_script = (
        ROOT / "scripts" / "run_georoute_official_world2_ddp_kat_slurm.sh"
    )
    control_script = (
        ROOT
        / "scripts"
        / "run_georoute_official_comparable_preflight_control_slurm.sh"
    )
    for script in (stage_script, kat_script, control_script):
        if not script.is_file():
            raise FileNotFoundError(script)

    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_AMP_DIAGNOSTIC_RUN_ROOT": str(run_root),
        "GEOROUTE_OFFICIAL_PREFLIGHT_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_AMP_PROTOCOL_PROFILE": AMP_FORMAL_PREFLIGHT_PROFILE,
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
    logs = run_root / "slurm"

    # Validate all scheduler requests before creating the immutable namespace.
    for arm in AMP_DIAGNOSTIC_ARMS:
        _sbatch(
            name=f"{spec['job_prefix']}_{'pl' if arm.endswith('pl_rep_off') else 'st'}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            gpu=True,
            test_only=True,
        )
    _sbatch_world2(
        name=f"{spec['job_prefix']}_world2",
        script=kat_script,
        logs=logs,
        exports=base_exports,
        test_only=True,
    )
    _sbatch(
        name=f"{spec['job_prefix']}_finalize",
        script=control_script,
        logs=logs,
        exports=base_exports,
        gpu=False,
        test_only=True,
    )

    run_root.mkdir(parents=True, exist_ok=False)
    for directory in (str(spec["cell_directory"]), "control", "slurm"):
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
    try:
        stage_jobs: dict[str, str] = {}
        for arm in AMP_DIAGNOSTIC_ARMS:
            job_id = _sbatch(
                name=(
                    f"{spec['job_prefix']}_"
                    f"{'pl' if arm.endswith('pl_rep_off') else 'st'}"
                ),
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                gpu=True,
                hold=True,
            )
            stage_jobs[arm] = job_id
            submitted.append(job_id)
        kat_job = _sbatch_world2(
            name=f"{spec['job_prefix']}_world2",
            script=kat_script,
            logs=logs,
            exports=base_exports,
            hold=True,
        )
        submitted.append(kat_job)
        finalizer_job = _sbatch(
            name=f"{spec['job_prefix']}_finalize",
            script=control_script,
            logs=logs,
            exports=base_exports,
            gpu=False,
            dependency=[*stage_jobs.values(), kat_job],
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        normalized_jobs = validate_amp_diagnostic_job_receipt(
            {"stage": stage_jobs, "finalizer": finalizer_job}
        )
        jobs = {
            **normalized_jobs,
            "world2_kat": kat_job,
        }
        deployment: dict[str, Any] = {
            "schema_version": spec["deployment_schema"],
            "status": spec["deployment_status"],
            "study_id": spec["study_id"],
            "protocol_profile": spec["profile"],
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arms": list(AMP_DIAGNOSTIC_ARMS),
            "seed": int(spec["seed"]),
            "jobs": jobs,
            "input_receipts": input_receipts,
            "expected_origin_ref": expected_origin_ref,
            "origin_ref_parity_verified": True,
            "official_upstream_snapshot": upstream_snapshot,
            "parent_repair_gate": {
                "path": str(repair_parent_path),
                "file_sha256": expected_parent_hash,
                "runtime_commit": repair_parent["runtime_commit"],
                "finalization_sha256": repair_parent[
                    "finalization_sha256"
                ],
                "decision": repair_parent["decision"],
            },
            "official_reference_binding": {
                "bound": True,
                "path": input_receipts[
                    "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"
                ]["path"],
                "sha256": input_receipts[
                    "GEOROUTE_OFFICIAL_REFERENCE_CONFIG"
                ]["sha256"],
                "upstream_release_commit": OFFICIAL_UPSTREAM_RELEASE_COMMIT,
                "upstream_config_sha256": OFFICIAL_CONFIG_SHA256,
                "full_official_training_claimed": False,
            },
            "world2_fp32_ddp_kat": {
                "required": True,
                "job_id": kat_job,
                "world_size": 2,
                "default_fp32_reduction": True,
                "fp16_shadow_must_overflow": True,
            },
            "protocol_manifest_path": str(protocol_path),
            "protocol_manifest_file_sha256": sha256_file(protocol_path),
            "protocol_sha256": protocol["protocol_sha256"],
            "submit_capacity_preflight": capacity,
            "storage_preflight": storage,
            "dependency_policy": {
                "real_batch_arms_parallel": True,
                "world2_kat_parallel": True,
                "stages_held_until_deployment_is_immutable": True,
                "finalizer_afterany_all_three": True,
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
        submission = {
            "schema_version": spec["deployment_schema"],
            "status": spec["finalizer_submission_status"],
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "finalizer_job_id": finalizer_job,
            "dependency_type": "afterany",
            "predecessor_job_ids": [*stage_jobs.values(), kat_job],
        }
        submission["receipt_sha256"] = canonical_sha256(submission)
        submission_path = run_root / "control" / "finalizer_submission.json"
        _atomic_write_json(submission_path, submission)
        _release_jobs([*stage_jobs.values(), kat_job])
        release = {
            "schema_version": spec["deployment_schema"],
            "status": spec["stage_release_status"],
            "runtime_commit": expected_commit,
            "released_job_ids": [*stage_jobs.values(), kat_job],
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
