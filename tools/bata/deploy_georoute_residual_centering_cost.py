#!/usr/bin/env python3
"""Atomically deploy one immutable residual-centering paired-cost Slurm job."""

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
    require_clean_dynamic_floor_m2_checkout,
)
from tools.bata.georoute_experiment_contract import canonical_sha256  # noqa: E402
from tools.bata.georoute_hybrid_causal_deploy_common import (  # noqa: E402
    cancel_jobs,
    release_jobs,
    require_submit_capacity,
    sbatch,
)
from tools.bata.georoute_residual_centering_cost_contract import (  # noqa: E402
    RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA,
    RESIDUAL_CENTERING_COST_ORDER,
    RESIDUAL_CENTERING_COST_PAIRS,
    RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS,
    RESIDUAL_CENTERING_COST_STUDY_ID,
    validate_residual_centering_cost_deployment,
    validate_residual_centering_cost_source,
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


def _git(*args: str) -> str:
    completed = subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def validate_cost_execution_delta(
    *, model_runtime_commit: str, execution_commit: str
) -> dict[str, Any]:
    """Prove later cost-only code did not modify the trained model/config path."""

    model_runtime_commit = str(model_runtime_commit).lower()
    execution_commit = str(execution_commit).lower()
    for commit in (model_runtime_commit, execution_commit):
        _git("cat-file", "-e", f"{commit}^{{commit}}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", model_runtime_commit, execution_commit],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if ancestor.returncode != 0:
        raise ValueError("cost execution commit does not descend from model runtime")
    all_changed = tuple(
        line
        for line in _git(
            "diff", "--name-only", model_runtime_commit, execution_commit
        ).splitlines()
        if line
    )
    sensitive_changed = tuple(
        line
        for line in _git(
            "diff",
            "--name-only",
            model_runtime_commit,
            execution_commit,
            "--",
            *RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS,
        ).splitlines()
        if line
    )
    if sensitive_changed:
        raise ValueError(
            "cost execution changed trained model/config sources: "
            + ", ".join(sensitive_changed)
        )
    return {
        "model_runtime_commit": model_runtime_commit,
        "execution_commit": execution_commit,
        "model_runtime_is_ancestor": True,
        "source_model_or_config_changed": False,
        "sensitive_runtime_paths": list(
            RESIDUAL_CENTERING_COST_SENSITIVE_RUNTIME_PATHS
        ),
        "changed_files": list(all_changed),
        "changed_files_sha256": canonical_sha256(list(all_changed)),
    }


def _run_precheck(script: Path, *, exports: Mapping[str, str]) -> dict[str, Any]:
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
            "residual-centering cost precheck failed: "
            + (completed.stderr.strip() or completed.stdout.strip())
        )
    return {
        "status": "PASS",
        "stdout_sha256": canonical_sha256({"stdout": completed.stdout}),
    }


def deploy(args: argparse.Namespace) -> dict[str, Any]:
    model_runtime_commit = str(args.model_runtime_commit).lower()
    execution_commit = str(args.execution_commit).lower()
    require_clean_dynamic_floor_m2_checkout(
        expected_commit=execution_commit, root=ROOT
    )
    execution_delta = validate_cost_execution_delta(
        model_runtime_commit=model_runtime_commit,
        execution_commit=execution_commit,
    )
    run_root = args.run_root.resolve()
    training_root = _safe_path(args.training_run_root, file=False)
    if not _inside(run_root, BOUNDARY.resolve()) or run_root.exists():
        raise ValueError("residual-centering cost requires a fresh bounded run root")
    source = validate_residual_centering_cost_source(
        training_root, expected_model_runtime_commit=model_runtime_commit
    )
    script = _safe_path(
        ROOT / "scripts" / "run_georoute_residual_centering_cost_slurm.sh",
        file=True,
    )
    storage = storage_capacity_receipt(run_root, cell_count=1)
    capacity = require_submit_capacity(root=ROOT, additional_jobs=1)
    common_exports = {
        "GEOROUTE_BASE": str(BOUNDARY.resolve()),
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "SCNR_RESIDUAL_CENTERING_COST_RUN_ROOT": str(run_root),
        "SCNR_RESIDUAL_CENTERING_TRAINING_RUN_ROOT": str(training_root),
        "SCNR_RESIDUAL_CENTERING_MODEL_RUNTIME_COMMIT": model_runtime_commit,
        "SCNR_RESIDUAL_CENTERING_COST_EXECUTION_COMMIT": execution_commit,
    }
    precheck = _run_precheck(script, exports=common_exports)
    if args.precheck_only:
        return {
            "schema_version": "scnr_residual_centering_paired_cost_precheck_v1",
            "status": "PASS_RESIDUAL_CENTERING_PAIRED_COST_PRECHECK",
            "model_runtime_commit": model_runtime_commit,
            "execution_commit": execution_commit,
            "run_root": str(run_root),
            "training_run_root": str(training_root),
            "execution_delta": execution_delta,
            "source_stage_result_receipts": source["stage_result_receipts"],
            "training_finalization_receipt": source[
                "training_finalization_receipt"
            ],
            "storage": storage,
            "submit_capacity": capacity,
            "precheck": precheck,
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
        sbatch(
            root=ROOT,
            name="scnr_rc_cost",
            script=script,
            logs=logs,
            exports=common_exports,
            resource="model1",
            test_only=True,
        )
        job_id = sbatch(
            root=ROOT,
            name="scnr_rc_cost",
            script=script,
            logs=logs,
            exports=common_exports,
            resource="model1",
            hold=True,
        )
        submitted.append(job_id)
        deployment: dict[str, Any] = {
            "schema_version": RESIDUAL_CENTERING_COST_DEPLOYMENT_SCHEMA,
            "status": "DEPLOYED_RESIDUAL_CENTERING_SINGLE_JOB_PAIRED_COST",
            "study_id": RESIDUAL_CENTERING_COST_STUDY_ID,
            "model_runtime_commit": model_runtime_commit,
            "execution_commit": execution_commit,
            "run_root": str(run_root),
            "training_run_root": str(training_root),
            "cost_order": list(RESIDUAL_CENTERING_COST_ORDER),
            "paired_pass_indices": [
                list(pair) for pair in RESIDUAL_CENTERING_COST_PAIRS
            ],
            "jobs": {"paired_cost": job_id},
            "single_slurm_job": True,
            "single_visible_gpu": True,
            "continuous_power_sidecar": True,
            "training_or_resume_allowed": False,
            "source_model_or_config_changed": False,
            "execution_delta": execution_delta,
            "stage_result_receipts": source["stage_result_receipts"],
            "training_finalization_receipt": source[
                "training_finalization_receipt"
            ],
            "all_jobs_held_until_immutable_receipt": True,
            "storage": storage,
            "submit_capacity": capacity,
            "precheck": precheck,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        _atomic_write_json(control / "deployment.json", deployment)
        validate_residual_centering_cost_deployment(
            deployment,
            run_root=run_root,
            training_run_root=training_root,
            expected_model_runtime_commit=model_runtime_commit,
            expected_execution_commit=execution_commit,
            expected_job_id=job_id,
        )
        release_jobs(ROOT, submitted)
        return deployment
    except Exception:
        cancel_jobs(ROOT, submitted)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--training-run-root", type=Path, required=True)
    parser.add_argument("--model-runtime-commit", required=True)
    parser.add_argument("--execution-commit", required=True)
    parser.add_argument("--precheck-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    result = deploy(_parse_args())
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
