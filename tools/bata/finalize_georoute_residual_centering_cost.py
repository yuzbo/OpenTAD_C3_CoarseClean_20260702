#!/usr/bin/env python3
"""Artifact-driven finalizer executed inside the paired-cost Slurm job."""

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

from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    require_clean_dynamic_floor_m2_checkout,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_residual_centering_cost_contract import (  # noqa: E402
    finalize_residual_centering_cost,
    validate_residual_centering_cost_deployment,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _read_optional(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    model_runtime_commit = str(args.model_runtime_commit).lower()
    execution_commit = str(args.execution_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=execution_commit, root=ROOT
    )
    run_root = args.run_root.resolve()
    training_root = args.training_run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or not run_root.is_dir():
        raise ValueError("residual-centering cost finalizer left write boundary")
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [
        item
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item
    ]
    if not job_id.isdigit() or len(visible) != 1:
        raise RuntimeError(
            "residual-centering cost finalizer requires its one-GPU Slurm job"
        )
    deployment_path = args.deployment.resolve()
    if deployment_path != (run_root / "control" / "deployment.json").resolve():
        raise ValueError("residual-centering cost deployment path changed")
    deployment = _read_optional(deployment_path)
    if deployment is None:
        raise FileNotFoundError(deployment_path)
    deployment = validate_residual_centering_cost_deployment(
        deployment,
        run_root=run_root,
        training_run_root=training_root,
        expected_model_runtime_commit=model_runtime_commit,
        expected_execution_commit=execution_commit,
        expected_job_id=job_id,
    )
    profile_path = (run_root / "cost" / "paired_cost_profile.json").resolve()
    profile = _read_optional(profile_path)
    result = finalize_residual_centering_cost(
        profile,
        expected_model_runtime_commit=model_runtime_commit,
        expected_execution_commit=execution_commit,
    )
    result.pop("finalization_sha256", None)
    result.update(
        finalizer_slurm_job_id=job_id,
        finalizer_in_same_paired_cost_job=True,
        deployment_receipt={
            "path": str(deployment_path),
            "sha256": sha256_file(deployment_path),
            "deployment_sha256": deployment["deployment_sha256"],
        },
        paired_cost_profile_receipt=(
            {
                "path": str(profile_path),
                "sha256": sha256_file(profile_path),
                "profile_sha256": profile["profile_sha256"],
            }
            if profile is not None
            else {"path": str(profile_path), "present": False}
        ),
    )
    result["finalization_sha256"] = canonical_sha256(result)
    output = (
        args.output.resolve()
        if args.output is not None
        else (run_root / "finalization" / "finalization.json").resolve()
    )
    if not _inside(output, run_root) or output.exists():
        raise ValueError("residual-centering cost finalization output must be fresh")
    output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(output, result)
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--deployment", type=Path, required=True)
    parser.add_argument("--model-runtime-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    result = finalize(_parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
