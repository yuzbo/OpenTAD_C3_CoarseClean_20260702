#!/usr/bin/env python3
"""Submit the atomic 5-arm x 3-seed GeoRoute development matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    _validate_preflight_parent,
    formal_arm_spec,
    read_json,
    validate_protocol_manifest,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


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
        command.extend(
            [
                "--dependency",
                "afterany:" + ":".join(map(str, dependency)),
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
    return parser.parse_args()


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
        != protocol["current_source_bridge"][
            "georoute_source_config_sha256"
        ]
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
