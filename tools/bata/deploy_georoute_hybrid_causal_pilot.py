#!/usr/bin/env python3
"""Submit the complete nine-arm Hybrid causal exploratory screen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_hybrid_causal_contract import (  # noqa: E402
    HYBRID_CAUSAL_ARM_ORDER,
    HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
    HYBRID_CAUSAL_EPOCHS,
    HYBRID_CAUSAL_K,
    HYBRID_CAUSAL_P0_SUITE_SCHEMA,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STUDY_ID,
    hybrid_causal_arm_spec,
)
from tools.bata.georoute_hybrid_causal_deploy_common import (  # noqa: E402
    cancel_jobs,
    full_hex,
    git_output,
    release_jobs,
    require_submit_capacity,
    sbatch,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _self_hash_matches(payload: dict[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected_commit = full_hex(
        args.expected_commit,
        length=40,
        name="expected_commit",
    )
    if git_output(ROOT, "rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("Hybrid causal pilot source commit mismatch")
    if git_output(ROOT, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Hybrid causal pilot deployer requires a clean source")
    if git_output(ROOT, "rev-parse", "@{u}").lower() != expected_commit:
        raise RuntimeError("Hybrid causal pilot requires HEAD/upstream parity")

    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or not run_root.is_dir():
        raise ValueError("Hybrid causal pilot requires its bounded P0 run root")
    deployment_path = run_root / "control" / "pilot_deployment_receipt.json"
    if deployment_path.exists() or (run_root / "pilot").exists():
        raise FileExistsError("Hybrid causal pilot namespace already exists")
    p0_suite_path = run_root / "control" / "hybrid_causal_p0_suite.json"
    storage_profile_path = run_root / "control" / "georoute_storage_profile.json"
    if any(path.is_symlink() or not path.is_file() for path in (p0_suite_path, storage_profile_path)):
        raise FileNotFoundError("Hybrid causal pilot requires sealed P0/storage receipts")
    p0_suite = _read_json(p0_suite_path)
    if (
        p0_suite.get("schema_version") != HYBRID_CAUSAL_P0_SUITE_SCHEMA
        or p0_suite.get("status") != "PASS_MECHANICAL_ONLY"
        or p0_suite.get("runtime_commit") != expected_commit
        or p0_suite.get("performance_training_authorized") is not True
        or p0_suite.get("performance_outputs_emitted") is not False
        or not _self_hash_matches(p0_suite, field="suite_sha256")
    ):
        raise ValueError("Hybrid causal P0 suite does not authorize training")

    inputs = {
        "source_config": args.source_config.resolve(),
        "manifest": args.manifest.resolve(),
        "development_annotation": args.development_annotation.resolve(),
        "class_map": args.class_map.resolve(),
        "development_video_root": args.development_video_root.resolve(),
        "pretrained": args.pretrained.resolve(),
    }
    if any(
        not path.is_file()
        for name, path in inputs.items()
        if name != "development_video_root"
    ) or not inputs["development_video_root"].is_dir():
        raise FileNotFoundError("Hybrid causal pilot input artifact is missing")
    if any(
        component.lower() in {"test", "testing", "test_videos", "official_test"}
        for component in inputs["development_video_root"].parts
    ):
        raise ValueError("Hybrid causal pilot video root may not name an official test path")

    capacity = require_submit_capacity(root=ROOT, additional_jobs=10)
    storage_profile = _read_json(storage_profile_path)
    storage_preflight = storage_capacity_receipt(
        run_root,
        cell_count=9,
        storage_profile=storage_profile,
        expected_commit=expected_commit,
    )
    stage_script = ROOT / "scripts" / "run_georoute_hybrid_causal_stage_slurm.sh"
    control_script = ROOT / "scripts" / "run_georoute_hybrid_causal_control_slurm.sh"
    if not stage_script.is_file() or not control_script.is_file():
        raise FileNotFoundError("Hybrid causal pilot Slurm wrapper is missing")
    logs = run_root / "slurm"
    logs.mkdir(exist_ok=True)
    base_exports = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT.resolve()),
        "GEOROUTE_HYBRID_CAUSAL_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_SOURCE_CONFIG": str(inputs["source_config"]),
        "GEOROUTE_MANIFEST": str(inputs["manifest"]),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": str(inputs["development_annotation"]),
        "GEOROUTE_CLASS_MAP": str(inputs["class_map"]),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": str(inputs["development_video_root"]),
        "GEOROUTE_PRETRAINED": str(inputs["pretrained"]),
    }
    stage_exports = {
        arm: {
            **base_exports,
            "GEOROUTE_HYBRID_CAUSAL_ARM": arm,
        }
        for arm in HYBRID_CAUSAL_ARM_ORDER
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_HYBRID_CAUSAL_ACTION": "finalize",
    }
    for arm in HYBRID_CAUSAL_ARM_ORDER:
        sbatch(
            root=ROOT,
            name=f"ghc_{hybrid_causal_arm_spec(arm)['slug']}",
            script=stage_script,
            logs=logs,
            exports=stage_exports[arm],
            resource="stage2",
            test_only=True,
        )
    sbatch(
        root=ROOT,
        name="ghc_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        resource="control",
        test_only=True,
    )

    submitted: list[str] = []
    try:
        stage_jobs: dict[str, str] = {}
        for arm in HYBRID_CAUSAL_ARM_ORDER:
            job_id = sbatch(
                root=ROOT,
                name=f"ghc_{hybrid_causal_arm_spec(arm)['slug']}",
                script=stage_script,
                logs=logs,
                exports=stage_exports[arm],
                resource="stage2",
                hold=True,
            )
            stage_jobs[arm] = job_id
            submitted.append(job_id)
        finalizer_job = sbatch(
            root=ROOT,
            name="ghc_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            resource="control",
            dependency=list(stage_jobs.values()),
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        deployment: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
            "phase": "pilot",
            "status": "SUBMITTED_NINE_ARM_EXPLORATORY_SCREEN",
            "study_id": HYBRID_CAUSAL_STUDY_ID,
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "arm_order": list(HYBRID_CAUSAL_ARM_ORDER),
            "arm_specs": {
                arm: hybrid_causal_arm_spec(arm)
                for arm in HYBRID_CAUSAL_ARM_ORDER
            },
            "seed": HYBRID_CAUSAL_SEED,
            "epochs": HYBRID_CAUSAL_EPOCHS,
            "token_budget": HYBRID_CAUSAL_K,
            "input_receipts": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for name, path in inputs.items()
            },
            "p0_suite": {
                "path": str(p0_suite_path.resolve()),
                "file_sha256": sha256_file(p0_suite_path),
                "suite_sha256": str(p0_suite["suite_sha256"]),
            },
            "storage_profile": {
                "path": str(storage_profile_path.resolve()),
                "file_sha256": sha256_file(storage_profile_path),
            },
            "storage_preflight": storage_preflight,
            "submit_capacity_preflight": capacity,
            "jobs": {
                "stages": stage_jobs,
                "finalizer": finalizer_job,
            },
            "dependencies": {
                "finalizer": {
                    "type": "afterany",
                    "job_ids": list(stage_jobs.values()),
                }
            },
            "all_nine_held_until_immutable_receipt": True,
            "all_nine_required_before_interpretation": True,
            "partial_survivor_inference_allowed": False,
            "single_seed_screen_only": True,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        _atomic_write_json(deployment_path, deployment)
        release_jobs(ROOT, list(stage_jobs.values()))
        release: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
            "phase": "pilot_release",
            "status": "RELEASED_ALL_NINE_AFTER_IMMUTABLE_RECEIPT",
            "runtime_commit": expected_commit,
            "job_ids": list(stage_jobs.values()),
            "deployment_file_sha256": sha256_file(deployment_path),
        }
        release["release_sha256"] = canonical_sha256(release)
        _atomic_write_json(run_root / "control" / "pilot_release.json", release)
    except Exception:
        cancel_jobs(ROOT, submitted)
        raise
    print(json.dumps(deployment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
