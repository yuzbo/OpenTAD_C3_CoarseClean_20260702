#!/usr/bin/env python3
"""Submit the two-leaf, no-performance Hybrid causal P0 DAG."""

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
    HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
    HYBRID_CAUSAL_SEED,
    HYBRID_CAUSAL_STUDY_ID,
)
from tools.bata.georoute_hybrid_causal_deploy_common import (  # noqa: E402
    cancel_jobs,
    full_hex,
    git_output,
    release_jobs,
    require_submit_capacity,
    sbatch,
)
from tools.bata.georoute_stage_runner import _atomic_write_json  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
MAIN_ARM = "hybrid_ctx8_roi28_res28_pl_support_only"


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
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
    head = git_output(ROOT, "rev-parse", "HEAD").lower()
    if head != expected_commit:
        raise RuntimeError("Hybrid causal P0 source commit mismatch")
    if git_output(ROOT, "status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Hybrid causal P0 deployer requires a clean source")
    upstream = git_output(ROOT, "rev-parse", "@{u}").lower()
    if upstream != expected_commit:
        raise RuntimeError("Hybrid causal P0 requires HEAD/upstream parity")

    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()) or run_root.exists():
        raise ValueError("Hybrid causal P0 requires a fresh bounded run root")
    source_config = args.source_config.resolve()
    pretrained = args.pretrained.resolve()
    if not source_config.is_file() or not pretrained.is_file():
        raise FileNotFoundError("Hybrid causal P0 config or pretrained checkpoint is missing")
    capacity = require_submit_capacity(root=ROOT, additional_jobs=3)
    run_root.mkdir(parents=True, exist_ok=False)
    for directory in ("p0", "control", "slurm"):
        (run_root / directory).mkdir()

    model_script = ROOT / "scripts" / "run_georoute_p0_slurm.sh"
    world2_script = ROOT / "scripts" / "run_georoute_official_world2_ddp_kat_slurm.sh"
    control_script = ROOT / "scripts" / "run_georoute_hybrid_causal_control_slurm.sh"
    if any(not path.is_file() for path in (model_script, world2_script, control_script)):
        raise FileNotFoundError("Hybrid causal P0 Slurm wrapper is missing")
    base_exports = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT.resolve()),
        "GEOROUTE_HYBRID_CAUSAL_RUN_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
    }
    model_exports = {
        **base_exports,
        "GEOROUTE_SOURCE_CONFIG": str(source_config),
        "GEOROUTE_PRETRAINED": str(pretrained),
        "GEOROUTE_P0_OUTPUT": str(run_root / "p0" / "a7_hybrid_pl.json"),
        "GEOROUTE_P0_ROUTE_MODE": "structured_hybrid",
        "GEOROUTE_P0_POLICY_ESTIMATOR": "score_function",
        "GEOROUTE_P0_TOKENS_PER_TUBELET": "64",
        "GEOROUTE_P0_CONTEXT_TOKENS": "8",
        "GEOROUTE_P0_STRUCTURED_ROI_TOKENS": "28",
        "GEOROUTE_P0_STRUCTURED_RESIDUAL_TOKENS": "28",
        "GEOROUTE_P0_GEOMETRY_TEMPORAL_SHIFT_TUBELETS": "0",
        "GEOROUTE_P0_POLICY_TEMPERATURE": "0.7",
        "GEOROUTE_P0_SCORE_FUNCTION_WEIGHT": "1.0",
        "GEOROUTE_P0_SCORE_FUNCTION_BASELINE_MOMENTUM": "0.95",
        "GEOROUTE_P0_HEIGHT": "180",
        "GEOROUTE_P0_WIDTH": "320",
        "GEOROUTE_P0_SEED": str(HYBRID_CAUSAL_SEED),
        "GEOROUTE_P0_HYBRID_CAUSAL_ARM": MAIN_ARM,
        "GEOROUTE_P0_GEOMETRY_SIDE_CHANNEL": "0",
        "GEOROUTE_P0_ABSOLUTE_POSITION_ENABLED": "1",
        "GEOROUTE_P0_ABSOLUTE_COORDINATES_ENABLED": "0",
        "GEOROUTE_P0_ROI_RELATIVE_COORDINATES_ENABLED": "0",
        "GEOROUTE_P0_GEOMETRY_PROJECTION_ENABLED": "0",
    }
    world2_exports = {
        **base_exports,
        "GEOROUTE_OFFICIAL_PREFLIGHT_RUN_ROOT": str(run_root),
    }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_HYBRID_CAUSAL_ACTION": "p0-finalize",
    }
    logs = run_root / "slurm"
    sbatch(
        root=ROOT,
        name="ghc_p0_model",
        script=model_script,
        logs=logs,
        exports=model_exports,
        resource="model1",
        test_only=True,
    )
    sbatch(
        root=ROOT,
        name="ghc_p0_world2",
        script=world2_script,
        logs=logs,
        exports=world2_exports,
        resource="world2",
        test_only=True,
    )
    sbatch(
        root=ROOT,
        name="ghc_p0_finalize",
        script=control_script,
        logs=logs,
        exports=finalizer_exports,
        resource="control",
        test_only=True,
    )

    submitted: list[str] = []
    try:
        model_job = sbatch(
            root=ROOT,
            name="ghc_p0_model",
            script=model_script,
            logs=logs,
            exports=model_exports,
            resource="model1",
            hold=True,
        )
        submitted.append(model_job)
        world2_job = sbatch(
            root=ROOT,
            name="ghc_p0_world2",
            script=world2_script,
            logs=logs,
            exports=world2_exports,
            resource="world2",
            hold=True,
        )
        submitted.append(world2_job)
        finalizer_job = sbatch(
            root=ROOT,
            name="ghc_p0_finalize",
            script=control_script,
            logs=logs,
            exports=finalizer_exports,
            resource="control",
            dependency=[model_job, world2_job],
            dependency_type="afterany",
        )
        submitted.append(finalizer_job)
        deployment: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
            "phase": "p0",
            "status": "SUBMITTED_HELD_P0_DAG",
            "study_id": HYBRID_CAUSAL_STUDY_ID,
            "runtime_commit": expected_commit,
            "run_root": str(run_root),
            "seed": HYBRID_CAUSAL_SEED,
            "main_arm": MAIN_ARM,
            "input_receipts": {
                "source_config": {
                    "path": str(source_config),
                    "sha256": sha256_file(source_config),
                },
                "pretrained": {
                    "path": str(pretrained),
                    "sha256": sha256_file(pretrained),
                },
            },
            "submit_capacity_preflight": capacity,
            "jobs": {
                "structured_model": model_job,
                "world2_ddp": world2_job,
                "p0_finalizer": finalizer_job,
            },
            "dependencies": {
                "p0_finalizer": {
                    "type": "afterany",
                    "job_ids": [model_job, world2_job],
                }
            },
            "p0_predecessors_held_until_receipt_written": True,
            "performance_outputs_emitted": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        deployment["deployment_sha256"] = canonical_sha256(deployment)
        deployment_path = run_root / "control" / "p0_deployment_receipt.json"
        _atomic_write_json(deployment_path, deployment)
        release_jobs(ROOT, [model_job, world2_job])
        release: dict[str, Any] = {
            "schema_version": HYBRID_CAUSAL_DEPLOYMENT_SCHEMA,
            "phase": "p0_release",
            "status": "RELEASED_AFTER_IMMUTABLE_RECEIPT",
            "runtime_commit": expected_commit,
            "job_ids": [model_job, world2_job],
            "deployment_file_sha256": sha256_file(deployment_path),
        }
        release["release_sha256"] = canonical_sha256(release)
        _atomic_write_json(run_root / "control" / "p0_release.json", release)
    except Exception:
        cancel_jobs(ROOT, submitted)
        raise
    print(json.dumps(deployment, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
