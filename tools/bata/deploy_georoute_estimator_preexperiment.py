#!/usr/bin/env python3
"""Submit the result-blind GeoRoute decode/KAT/Phase-M diagnostic DAG."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)


DEPLOYMENT_SCHEMA = "georoute_estimator_preexperiment_deployment_v1"
GPU_OUTER_SLURM_ARGS = ("--gpus", "2", "--cpus-per-task", "8")
CONTROL_SLURM_ARGS = ("--gpus", "1", "--cpus-per-task", "4")
MINIMUM_FREE_BYTES = 5 * 1024**3
PHASE_M_SPECS = {
    "dense": {
        "config": "p1_dense_native_matched_k_seed3407.py",
        "cell": "p1/dense_native/matched_k/seed3407",
        "source_variant": "dense_native",
    },
    "fixed": {
        "config": "p1_fixed_lattice_matched_k_seed3407.py",
        "cell": "p1/fixed_lattice/matched_k/seed3407",
        "source_variant": "fixed_lattice",
    },
    "fixed_geometry": {
        "config": "p1_fixed_lattice_geometry_matched_k_seed3407.py",
        "cell": "p1/fixed_lattice_geometry/matched_k/seed3407",
        "source_variant": "fixed_lattice_geometry",
    },
    "random": {
        "config": "p1_random_matched_k_seed3407.py",
        "cell": "p1/random/matched_k/seed3407",
        "source_variant": "random",
    },
    "free": {
        "config": "p1_free_matched_k_seed3407.py",
        "cell": "p1/free/matched_k/seed3407",
        "source_variant": "free",
    },
    "hybrid": {
        "config": "p1_hybrid_matched_k_seed3407.py",
        "cell": "p1/hybrid/matched_k/seed3407",
        "source_variant": "hybrid",
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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


def _require_clean_value(value: str, *, name: str) -> str:
    if (
        not value
        or value != value.strip()
        or any(character in value for character in (",", "\n", "\r", "\x00"))
    ):
        raise ValueError(f"{name} is unsafe for sbatch --export")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _sbatch(
    *,
    name: str,
    script: Path,
    logs: Path,
    exports: Mapping[str, str],
    gpu: bool,
    dependency: list[str] | None = None,
    dependency_type: str = "afterok",
    test_only: bool = False,
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
    if dependency:
        if dependency_type not in {"afterok", "afterany"}:
            raise ValueError("unsupported dependency type")
        command.extend(
            [
                "--dependency",
                dependency_type + ":" + ":".join(dependency),
            ]
        )
    command.extend(
        GPU_OUTER_SLURM_ARGS if gpu else CONTROL_SLURM_ARGS
    )
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
        raise RuntimeError(
            f"invalid sbatch job id for {name}: {completed.stdout!r}"
        )
    return job_id


def _validate_source_cell(
    *,
    source_root: Path,
    label: str,
    spec: Mapping[str, str],
    source_commit: str,
) -> dict[str, str]:
    config = source_root / "control" / "bound_configs" / spec["config"]
    cell = source_root / spec["cell"]
    stage_result_path = cell / "stage_result.json"
    prediction = cell / "gpu1_id0" / "result_detection.json"
    for path in (config, stage_result_path, prediction):
        if not path.is_file():
            raise FileNotFoundError(path)
        if not _inside(path.resolve(), source_root):
            raise RuntimeError(f"source artifact leaves old run root for {label}")
    stage_result = _read_json(stage_result_path)
    if (
        stage_result.get("status") != "PASS_DEVELOPMENT_ONLY"
        or stage_result.get("runtime_commit") != source_commit
        or stage_result.get("variant") != spec["source_variant"]
        or stage_result.get("official_test_opened") is not False
        or stage_result.get("gt_for_route_used") is not False
        or stage_result.get("raw_prediction_cache_used") is not False
    ):
        raise RuntimeError(f"invalid source stage result for {label}")
    if sha256_file(config) != stage_result.get("config_sha256"):
        raise RuntimeError(f"source config hash mismatch for {label}")
    if sha256_file(prediction) != stage_result.get("prediction_sha256"):
        raise RuntimeError(f"source prediction hash mismatch for {label}")
    checkpoint_receipt = stage_result.get("checkpoint_receipt")
    if not isinstance(checkpoint_receipt, dict):
        raise RuntimeError(f"source checkpoint receipt missing for {label}")
    checkpoint = Path(str(checkpoint_receipt["path"])).resolve()
    if not checkpoint.is_file() or not _inside(checkpoint, source_root):
        raise RuntimeError(f"source checkpoint leaves old run root for {label}")
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != checkpoint_receipt.get("sha256"):
        raise RuntimeError(f"source checkpoint hash mismatch for {label}")
    return {
        "label": label,
        "source_variant": spec["source_variant"],
        "source_config": str(config.resolve()),
        "source_config_sha256": sha256_file(config),
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": checkpoint_sha256,
        "source_prediction": str(prediction.resolve()),
        "source_prediction_sha256": sha256_file(prediction),
        "source_stage_result": str(stage_result_path.resolve()),
        "source_stage_result_sha256": sha256_file(stage_result_path),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-experiment-commit", required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    source_root = args.source_run_root.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, write_boundary):
        raise ValueError("preexperiment root leaves remote write boundary")
    if not _inside(source_root, write_boundary) or not source_root.is_dir():
        raise ValueError("source run root is invalid")
    if run_root.exists():
        raise FileExistsError(
            "preexperiment namespace exists; refusing overwrite/resume"
        )
    expected_commit = args.expected_commit.lower()
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("deployment source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("deployment requires clean source")
    source_commit = args.source_experiment_commit.lower()
    if len(source_commit) != 40:
        raise ValueError("source experiment commit must be a full SHA")
    source_cells = {
        label: _validate_source_cell(
            source_root=source_root,
            label=label,
            spec=spec,
            source_commit=source_commit,
        )
        for label, spec in PHASE_M_SPECS.items()
    }
    census_config = Path(source_cells["dense"]["source_config"])
    storage = shutil.disk_usage(run_root.parent)
    if storage.free < MINIMUM_FREE_BYTES:
        raise RuntimeError(
            "preexperiment storage preflight failed: "
            f"free={storage.free}, required={MINIMUM_FREE_BYTES}"
        )

    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "control").mkdir()
    (run_root / "phase_m").mkdir()
    logs = run_root / "slurm"
    logs.mkdir()
    storage_receipt = {
        "free_bytes": int(storage.free),
        "required_bytes": MINIMUM_FREE_BYTES,
        "passed": True,
        "checkpoint_writes_planned": 0,
    }
    _atomic_write_json(
        run_root / "control" / "storage_preflight.json",
        storage_receipt,
    )

    script = (
        ROOT
        / "scripts"
        / "run_georoute_estimator_preexperiment_slurm.sh"
    )
    if not script.is_file():
        raise FileNotFoundError(script)
    base_values = {
        "GEOROUTE_SOURCE_ROOT": str(ROOT),
        "GEOROUTE_SOURCE_RUN_ROOT": str(source_root),
        "GEOROUTE_PREEXPERIMENT_ROOT": str(run_root),
        "GEOROUTE_EXPECTED_COMMIT": expected_commit,
        "GEOROUTE_SOURCE_EXPERIMENT_COMMIT": source_commit,
    }
    base_exports = {
        key: _require_clean_value(value, name=key)
        for key, value in base_values.items()
    }
    kat_exports = {
        **base_exports,
        "GEOROUTE_PREEXPERIMENT_ACTION": "kat",
    }
    census_exports = {
        **base_exports,
        "GEOROUTE_PREEXPERIMENT_ACTION": "decode-census",
        "GEOROUTE_CENSUS_BOUND_CONFIG": str(census_config),
        "GEOROUTE_CENSUS_BOUND_CONFIG_SHA256": sha256_file(
            census_config
        ),
        "GEOROUTE_CENSUS_PASSES": "2",
    }
    phase_exports = {}
    for label, cell in source_cells.items():
        phase_exports[label] = {
            **base_exports,
            "GEOROUTE_PREEXPERIMENT_ACTION": "phase-m",
            "GEOROUTE_PHASE_M_VARIANT": label,
            "GEOROUTE_PHASE_M_SEED": "3407",
            "GEOROUTE_PHASE_M_SOURCE_CONFIG": cell["source_config"],
            "GEOROUTE_PHASE_M_SOURCE_CONFIG_SHA256": cell[
                "source_config_sha256"
            ],
            "GEOROUTE_PHASE_M_SOURCE_CHECKPOINT": cell[
                "source_checkpoint"
            ],
            "GEOROUTE_PHASE_M_SOURCE_CHECKPOINT_SHA256": cell[
                "source_checkpoint_sha256"
            ],
            "GEOROUTE_PHASE_M_SOURCE_PREDICTION": cell[
                "source_prediction"
            ],
            "GEOROUTE_PHASE_M_SOURCE_PREDICTION_SHA256": cell[
                "source_prediction_sha256"
            ],
        }
    finalizer_exports = {
        **base_exports,
        "GEOROUTE_PREEXPERIMENT_ACTION": "finalize",
    }

    for name, exports, gpu in [
        ("gr_kat", kat_exports, False),
        ("gr_decode", census_exports, False),
        *[
            (f"gr_m_{label}", exports, True)
            for label, exports in phase_exports.items()
        ],
        ("gr_finalize", finalizer_exports, False),
    ]:
        _sbatch(
            name=name,
            script=script,
            logs=logs,
            exports=exports,
            gpu=gpu,
            test_only=True,
        )

    jobs: dict[str, Any] = {}
    jobs["kat"] = _sbatch(
        name="gr_kat",
        script=script,
        logs=logs,
        exports=kat_exports,
        gpu=False,
    )
    jobs["decode_census"] = _sbatch(
        name="gr_decode",
        script=script,
        logs=logs,
        exports=census_exports,
        gpu=False,
    )
    gate_jobs = [jobs["kat"], jobs["decode_census"]]
    jobs["phase_m"] = {}
    for label, exports in phase_exports.items():
        jobs["phase_m"][label] = _sbatch(
            name=f"gr_m_{label}",
            script=script,
            logs=logs,
            exports=exports,
            gpu=True,
            dependency=gate_jobs,
            dependency_type="afterok",
        )

    deployment: dict[str, Any] = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "status": "SUBMITTED_D_K_AND_CONDITIONAL_PARALLEL_M",
        "runtime_commit": expected_commit,
        "source_experiment_commit": source_commit,
        "source_run_root": str(source_root),
        "run_root": str(run_root),
        "phase_m_variants": list(PHASE_M_SPECS),
        "source_cells": source_cells,
        "storage_preflight": storage_receipt,
        "jobs": jobs,
        "dependency_policy": {
            "kat_and_decode_parallel": True,
            "phase_m_after_both_pass": True,
            "phase_m_parallel": True,
            "finalizer_after_any_terminal_state": True,
        },
        "training_launched": False,
        "old_selector_reused": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    deployment_path = run_root / "control" / "deployment.json"
    _atomic_write_json(deployment_path, deployment)

    all_predecessors = [
        jobs["kat"],
        jobs["decode_census"],
        *jobs["phase_m"].values(),
    ]
    try:
        finalizer_job = _sbatch(
            name="gr_finalize",
            script=script,
            logs=logs,
            exports=finalizer_exports,
            gpu=False,
            dependency=all_predecessors,
            dependency_type="afterany",
        )
    except Exception as error:
        failure: dict[str, Any] = {
            "schema_version": DEPLOYMENT_SCHEMA,
            "status": "FAIL_FINALIZER_SUBMISSION",
            "runtime_commit": expected_commit,
            "deployment_file_sha256": sha256_file(deployment_path),
            "dependency_type": "afterany",
            "predecessor_job_ids": all_predecessors,
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
        }
        failure["receipt_sha256"] = canonical_sha256(failure)
        _atomic_write_json(
            run_root
            / "control"
            / "finalizer_submission_failure.json",
            failure,
        )
        raise
    finalizer_submission = {
        "schema_version": DEPLOYMENT_SCHEMA,
        "status": "SUBMITTED_FINALIZER_AFTERANY",
        "runtime_commit": expected_commit,
        "deployment_file_sha256": sha256_file(deployment_path),
        "finalizer_job_id": finalizer_job,
        "dependency_type": "afterany",
        "predecessor_job_ids": all_predecessors,
    }
    finalizer_submission["receipt_sha256"] = canonical_sha256(
        finalizer_submission
    )
    _atomic_write_json(
        run_root / "control" / "finalizer_submission.json",
        finalizer_submission,
    )
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
