#!/usr/bin/env python3
"""Finalize the two-arm residual-centering matched-training accuracy screen."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_dynamic_floor_m2_stage_runner import (  # noqa: E402
    BOUNDARY,
    _inside,
    _require_exact_clean_source,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_residual_centering_training_contract import (  # noqa: E402
    RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA,
    RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER,
    finalize_residual_centering_training,
    residual_centering_training_cell_relative_path,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402


def _read_stage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"stage result is not a JSON object: {path}")
    return payload


def _validate_deployment(
    deployment: dict[str, Any],
    *,
    run_root: Path,
    expected_commit: str,
    expected_jobs: dict[str, str],
    finalizer_job_id: str,
) -> dict[str, Any]:
    unsigned = dict(deployment)
    observed_hash = unsigned.pop("deployment_sha256", None)
    jobs = deployment.get("jobs")
    dependency = deployment.get("dependencies", {}).get("finalizer")
    expected_predecessors = [
        expected_jobs[variant]
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    ]
    if (
        deployment.get("schema_version")
        != RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != "DEPLOYED_RESIDUAL_CENTERING_MATCHED_TRAINING_DAG"
        or deployment.get("runtime_commit") != expected_commit
        or deployment.get("run_root") != str(run_root)
        or deployment.get("variants")
        != list(RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER)
        or not isinstance(jobs, dict)
        or {variant: str(jobs.get(variant)) for variant in expected_jobs}
        != expected_jobs
        or str(jobs.get("finalizer")) != finalizer_job_id
        or not isinstance(dependency, dict)
        or dependency.get("type") != "afterany"
        or dependency.get("predecessors") != expected_predecessors
        or deployment.get("all_jobs_held_until_immutable_receipt") is not True
        or deployment.get("partial_survivor_inference_allowed") is not False
        or deployment.get("paired_cost_submitted") is not False
        or deployment.get("additional_seeds_opened") is not False
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
        or observed_hash != canonical_sha256(unsigned)
    ):
        raise ValueError("residual-centering deployment receipt is invalid")
    return deployment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--none-job-id", required=True)
    parser.add_argument("--center-job-id", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    expected_commit = str(args.expected_commit).lower()
    _require_exact_clean_source(expected_commit)
    finalizer_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [
        item
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item
    ]
    if not finalizer_job_id.isdigit() or len(visible) != 1:
        raise RuntimeError(
            "residual-centering finalizer requires one Slurm scheduling GPU"
        )
    expected_jobs = {
        "none_control": str(args.none_job_id),
        "residual_window_center": str(args.center_job_id),
    }
    if any(not value.isdigit() for value in expected_jobs.values()):
        raise ValueError("residual-centering stage job IDs must be numeric")
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or not run_root.is_dir():
        raise ValueError("residual-centering finalizer run root is invalid")
    deployment_path = args.deployment.resolve()
    if (
        not _inside(deployment_path, run_root)
        or not deployment_path.is_file()
        or deployment_path.is_symlink()
    ):
        raise ValueError("residual-centering deployment receipt path is invalid")
    deployment = _validate_deployment(
        _read_stage(deployment_path),
        run_root=run_root,
        expected_commit=expected_commit,
        expected_jobs=expected_jobs,
        finalizer_job_id=finalizer_job_id,
    )
    output = (
        args.output.resolve()
        if args.output is not None
        else (run_root / "finalization" / "finalization.json").resolve()
    )
    if not _inside(output, run_root) or output.exists():
        raise ValueError(
            "residual-centering finalization output must be fresh inside run root"
        )

    stages: dict[str, dict[str, Any]] = {}
    stage_receipts: dict[str, Any] = {}
    for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
        stage_path = (
            run_root
            / residual_centering_training_cell_relative_path(variant=variant)
            / "stage_result.json"
        ).resolve()
        if stage_path.is_file() and not stage_path.is_symlink():
            try:
                stages[variant] = _read_stage(stage_path)
                stage_receipts[variant] = {
                    "path": str(stage_path),
                    "sha256": sha256_file(stage_path),
                    "present": True,
                }
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                stages[variant] = {}
                stage_receipts[variant] = {
                    "path": str(stage_path),
                    "present": True,
                    "read_error": f"{type(error).__name__}:{error}",
                }
        else:
            stage_receipts[variant] = {
                "path": str(stage_path),
                "present": False,
            }
    result = finalize_residual_centering_training(
        stages,
        expected_commit=expected_commit,
        expected_job_ids=expected_jobs,
    )
    result.pop("finalization_sha256")
    result.update(
        finalizer_slurm_job_id=finalizer_job_id,
        finalizer_gpu_allocation_is_scheduling_overhead=True,
        deployment_receipt={
            "path": str(deployment_path),
            "sha256": sha256_file(deployment_path),
            "deployment_sha256": deployment["deployment_sha256"],
        },
        stage_result_receipts=stage_receipts,
    )
    result["finalization_sha256"] = canonical_sha256(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
