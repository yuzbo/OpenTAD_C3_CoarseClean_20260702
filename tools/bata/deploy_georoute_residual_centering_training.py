#!/usr/bin/env python3
"""Precheck and atomically deploy the residual-centering training DAG."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_SEED,
    require_clean_dynamic_floor_m2_checkout,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_hybrid_causal_deploy_common import (  # noqa: E402
    cancel_jobs,
    release_jobs,
    require_submit_capacity,
    sbatch,
)
from tools.bata.georoute_residual_centering_training_contract import (  # noqa: E402
    RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_STUDY_ID,
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _safe_path(path: Path, *, file: bool) -> Path:
    path = path.resolve()
    if (file and not path.is_file()) or (not file and not path.is_dir()):
        raise FileNotFoundError(path)
    if "," in str(path) or any(ord(character) < 32 for character in str(path)):
        raise ValueError(f"unsafe Slurm export path: {path}")
    return path


def _run_precheck(
    script: Path, *, exports: Mapping[str, str], label: str
) -> dict[str, Any]:
    env = dict(os.environ)
    env.update({key: str(value) for key, value in exports.items()})
    env["PRECHECK_ONLY"] = "1"
    completed = subprocess.run(
        ["bash", str(script)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"residual-centering {label} precheck failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return {
        "label": label,
        "status": "PASS",
        "stdout_sha256": canonical_sha256({"stdout": completed.stdout}),
    }


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=expected_commit, root=ROOT
    )
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or run_root.exists():
        raise ValueError(
            "residual-centering deployment requires one fresh bounded run root"
        )
    source_config = _safe_path(args.source_config, file=True)
    manifest = _safe_path(args.manifest, file=True)
    annotation = _safe_path(args.development_annotation, file=True)
    class_map = _safe_path(args.class_map, file=True)
    video_root = _safe_path(args.development_video_root, file=False)
    pretrained = _safe_path(args.pretrained, file=True)
    if any(
        component.lower() in {"test", "testing", "official_test", "test_videos"}
        for component in video_root.parts
    ):
        raise ValueError("residual-centering deployment forbids a test video root")

    scripts = {
        "stage": _safe_path(
            ROOT
            / "scripts"
            / "run_georoute_residual_centering_training_stage_slurm.sh",
            file=True,
        ),
        "finalizer": _safe_path(
            ROOT
            / "scripts"
            / "run_georoute_residual_centering_training_finalizer_slurm.sh",
            file=True,
        ),
    }
    storage = storage_capacity_receipt(run_root, cell_count=2)
    capacity = require_submit_capacity(root=ROOT, additional_jobs=3)
    common_exports = {
        "GEOROUTE_BASE": str(BOUNDARY.resolve()),
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "SCNR_RESIDUAL_CENTERING_TRAINING_SOURCE_CONFIG": str(source_config),
        "GEOROUTE_DEVELOPMENT_MANIFEST": str(manifest),
        "GEOROUTE_DEVELOPMENT_ANNOTATION": str(annotation),
        "GEOROUTE_CLASS_MAP": str(class_map),
        "GEOROUTE_DEVELOPMENT_VIDEO_ROOT": str(video_root),
        "GEOROUTE_PRETRAINED": str(pretrained),
    }
    prechecks = [
        _run_precheck(
            scripts["stage"],
            exports={
                **common_exports,
                "SCNR_RESIDUAL_CENTERING_TRAINING_VARIANT": variant,
            },
            label=f"stage:{variant}",
        )
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    ]
    prechecks.append(
        _run_precheck(
            scripts["finalizer"],
            exports={
                **common_exports,
                "SCNR_RESIDUAL_CENTERING_NONE_JOB_ID": "1",
                "SCNR_RESIDUAL_CENTERING_CENTER_JOB_ID": "2",
            },
            label="finalizer",
        )
    )
    if args.precheck_only:
        return {
            "schema_version": "scnr_residual_centering_deploy_precheck_v1",
            "status": "PASS_RESIDUAL_CENTERING_DEPLOY_PRECHECK",
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "storage": storage,
            "submit_capacity": capacity,
            "prechecks": prechecks,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }

    run_root.mkdir(parents=True, exist_ok=False)
    control = run_root / "control"
    logs = run_root / "logs"
    control.mkdir()
    logs.mkdir()
    _atomic_write_json(control / "storage_preflight.json", storage)
    submitted: list[str] = []
    try:
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
            sbatch(
                root=ROOT,
                name=(
                    "scnr_rc_none"
                    if variant == "none_control"
                    else "scnr_rc_center"
                ),
                script=scripts["stage"],
                logs=logs,
                exports={
                    **common_exports,
                    "SCNR_RESIDUAL_CENTERING_TRAINING_VARIANT": variant,
                },
                resource="model1",
                test_only=True,
            )
        sbatch(
            root=ROOT,
            name="scnr_rc_finalize",
            script=scripts["finalizer"],
            logs=logs,
            exports={
                **common_exports,
                "SCNR_RESIDUAL_CENTERING_NONE_JOB_ID": "1",
                "SCNR_RESIDUAL_CENTERING_CENTER_JOB_ID": "2",
            },
            resource="control",
            test_only=True,
        )

        jobs: dict[str, str] = {}
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
            job_id = sbatch(
                root=ROOT,
                name=(
                    "scnr_rc_none"
                    if variant == "none_control"
                    else "scnr_rc_center"
                ),
                script=scripts["stage"],
                logs=logs,
                exports={
                    **common_exports,
                    "SCNR_RESIDUAL_CENTERING_TRAINING_VARIANT": variant,
                },
                resource="model1",
                hold=True,
            )
            jobs[variant] = job_id
            submitted.append(job_id)
        finalizer_exports = {
            **common_exports,
            "SCNR_RESIDUAL_CENTERING_NONE_JOB_ID": jobs["none_control"],
            "SCNR_RESIDUAL_CENTERING_CENTER_JOB_ID": jobs[
                "residual_window_center"
            ],
        }
        finalizer = sbatch(
            root=ROOT,
            name="scnr_rc_finalize",
            script=scripts["finalizer"],
            logs=logs,
            exports=finalizer_exports,
            resource="control",
            dependency=[jobs[variant] for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER],
            dependency_type="afterany",
            hold=True,
        )
        submitted.append(finalizer)
        jobs["finalizer"] = finalizer
        deployment: dict[str, Any] = {
            "schema_version": RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA,
            "status": "DEPLOYED_RESIDUAL_CENTERING_MATCHED_TRAINING_DAG",
            "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "seed": DYNAMIC_FLOOR_M2_SEED,
            "variants": list(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER),
            "jobs": jobs,
            "dependencies": {
                "finalizer": {
                    "type": "afterany",
                    "predecessors": [
                        jobs[variant]
                        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
                    ],
                }
            },
            "all_jobs_held_until_immutable_receipt": True,
            "finalizer_gpu_allocation_is_scheduling_overhead": True,
            "partial_survivor_inference_allowed": False,
            "input_receipts": {
                "source_config": {
                    "path": str(source_config),
                    "sha256": sha256_file(source_config),
                },
                "manifest": {
                    "path": str(manifest),
                    "sha256": sha256_file(manifest),
                },
                "development_annotation": {
                    "path": str(annotation),
                    "sha256": sha256_file(annotation),
                },
                "class_map": {
                    "path": str(class_map),
                    "sha256": sha256_file(class_map),
                },
                "pretrained": {
                    "path": str(pretrained),
                    "sha256": sha256_file(pretrained),
                },
                "development_video_root": str(video_root),
            },
            "storage": storage,
            "submit_capacity": capacity,
            "prechecks": prechecks,
            "paired_cost_submitted": False,
            "additional_seeds_opened": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        _atomic_write_json(control / "deployment.json", deployment)
        release_jobs(ROOT, submitted)
        return deployment
    except Exception:
        cancel_jobs(ROOT, submitted)
        raise


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
    parser.add_argument("--precheck-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = deploy(_parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
