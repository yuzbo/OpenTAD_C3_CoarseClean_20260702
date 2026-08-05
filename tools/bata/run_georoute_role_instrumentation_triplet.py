#!/usr/bin/env python3
"""Run a same-GPU OFF-A/OFF-B/ON causal role-instrumentation triplet."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import traceback
from typing import Any, Mapping

from mmengine.config import Config

from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_stage_runner import _run_logged, build_torchrun_prefix
from tools.bata.run_georoute_phase_m_replay import (
    _atomic_write_json,
    _inside,
    _read_json,
)
from tools.bata.run_georoute_role_instrumentation_pair import (
    _build_pair_test_arguments,
    _configure_pair_mode,
    _utc_now,
    _validate_formal_telemetry,
    _validate_source,
    compare_prediction_artifacts,
)


TRIPLET_SCHEMA = "georoute_role_instrumentation_causal_triplet_v1"
PAIR_BINDING_SCHEMA = DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA
TRIPLET_ORDER = ("role_off_a", "role_off_b", "role_on")
MODE_SPECIFICATIONS = (
    ("role_off_a", False, "role_off"),
    ("role_off_b", False, "role_off"),
    ("role_on", True, "role_on"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cell-root", type=Path, required=True)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--source-bound-config", type=Path, required=True)
    parser.add_argument("--source-bound-config-sha256", required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-checkpoint-sha256", required=True)
    parser.add_argument("--source-prediction", type=Path, required=True)
    parser.add_argument("--source-prediction-sha256", required=True)
    parser.add_argument("--source-population-sha256", required=True)
    parser.add_argument("--source-dataset-count", type=int, required=True)
    parser.add_argument("--source-experiment-commit", required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def classify_triplet_comparisons(
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify causality without reading telemetry statistics or task metrics."""

    control_parity = bool(comparisons["role_off_a_vs_role_off_b"]["raw_sha256_parity"])
    treatment_parity = bool(
        control_parity
        and comparisons["role_off_a_vs_role_on"]["raw_sha256_parity"]
        and comparisons["role_off_b_vs_role_on"]["raw_sha256_parity"]
    )
    source_parity = bool(
        comparisons["source_vs_role_off_a"]["raw_sha256_parity"]
        and comparisons["source_vs_role_off_b"]["raw_sha256_parity"]
        and comparisons["source_vs_role_on"]["raw_sha256_parity"]
    )
    if not control_parity:
        status = "FAIL_BASELINE_REPLAY_NONDETERMINISM"
    elif not treatment_parity:
        status = "FAIL_ROLE_INSTRUMENTATION_NONNEUTRAL"
    elif source_parity:
        status = "PASS_TRIPLET_NEUTRALITY_AND_SOURCE_PARITY"
    else:
        status = "PASS_TRIPLET_NEUTRALITY_SOURCE_REPLAY_DRIFT_DIAGNOSTIC_ONLY"
    return {
        "status": status,
        "baseline_replay_determinism_supported": control_parity,
        "paired_role_instrumentation_neutrality_supported": treatment_parity,
        "source_prediction_raw_parity": source_parity,
        "role_calibration_analysis_allowed_under_frozen_contract": bool(
            treatment_parity and source_parity
        ),
    }


def _execute(args: argparse.Namespace, cell_root: Path) -> dict[str, Any]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not slurm_job_id or not visible_devices or "," in visible_devices:
        raise RuntimeError("causal triplet requires exactly one Slurm-visible GPU")
    runtime_commit, source_config, checkpoint, source_prediction = _validate_source(
        args
    )
    binding_common = {
        "schema_version": PAIR_BINDING_SCHEMA,
        "causal_triplet_schema_version": TRIPLET_SCHEMA,
        "variant": str(args.variant),
        "seed": int(args.seed),
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "runtime_commit": runtime_commit,
        "source_bound_config_sha256": args.source_bound_config_sha256.lower(),
        "source_checkpoint_sha256": args.source_checkpoint_sha256.lower(),
        "source_prediction_sha256": args.source_prediction_sha256.lower(),
        "source_population_sha256": args.source_population_sha256.lower(),
        "source_dataset_count": int(args.source_dataset_count),
        "same_slurm_job": True,
        "same_visible_gpu": True,
        "serial_execution": True,
        "instrumentation_only": True,
        "changes_route_or_execution": False,
        "fixed_role_quota_used": False,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
    }
    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    modes: dict[str, Any] = {}
    for mode, role_enabled, pair_mode in MODE_SPECIFICATIONS:
        work_dir = cell_root / mode / "gpu1_id0"
        bound_config = cell_root / "control" / f"{mode}_bound_config.py"
        binding = {
            **binding_common,
            "pair_mode": pair_mode,
            "role_calibration_telemetry_enabled": role_enabled,
        }
        cfg = Config.fromfile(str(source_config))
        _configure_pair_mode(
            cfg,
            work_dir=work_dir,
            role_calibration_enabled=role_enabled,
            binding=binding,
        )
        bound_config.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(bound_config))
        command_prefix, rendezvous = build_torchrun_prefix(
            phase="test",
            slurm_job_id=str(slurm_job_id),
            stage=f"role_triplet_{mode}",
            variant=str(args.variant),
            seed=int(args.seed),
        )
        started_at = _utc_now()
        log_path = cell_root / mode / "test.out"
        _run_logged(
            _build_pair_test_arguments(
                command_prefix=command_prefix,
                bound_config=bound_config,
                checkpoint=checkpoint,
                seed=int(args.seed),
            ),
            log_path=log_path,
            env=inherited,
        )
        ended_at = _utc_now()
        prediction = work_dir / "result_detection.json"
        telemetry_path = work_dir / "georoute_diagnostic_telemetry.json"
        profile_path = work_dir / "georoute_development_profile.json"
        for artifact in (prediction, telemetry_path):
            if not artifact.is_file():
                raise FileNotFoundError(artifact)
        if profile_path.exists():
            raise RuntimeError("causal triplet unexpectedly emitted a timed profile")
        telemetry_receipt = _validate_formal_telemetry(
            _read_json(telemetry_path),
            expected_binding=binding,
            expected_population_sha256=args.source_population_sha256,
            expected_dataset_count=int(args.source_dataset_count),
            role_calibration_enabled=role_enabled,
        )
        modes[mode] = {
            "pair_mode": pair_mode,
            "role_calibration_telemetry_enabled": role_enabled,
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "bound_config_path": str(bound_config),
            "bound_config_sha256": sha256_file(bound_config),
            "prediction_path": str(prediction),
            "prediction_sha256": sha256_file(prediction),
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "test_log_path": str(log_path),
            "test_log_sha256": sha256_file(log_path),
            "telemetry_receipt": telemetry_receipt,
            "rendezvous": rendezvous,
        }
    prediction_paths = {
        mode: Path(receipt["prediction_path"]) for mode, receipt in modes.items()
    }
    comparisons = {
        "source_vs_role_off_a": compare_prediction_artifacts(
            source_prediction, prediction_paths["role_off_a"]
        ),
        "source_vs_role_off_b": compare_prediction_artifacts(
            source_prediction, prediction_paths["role_off_b"]
        ),
        "source_vs_role_on": compare_prediction_artifacts(
            source_prediction, prediction_paths["role_on"]
        ),
        "role_off_a_vs_role_off_b": compare_prediction_artifacts(
            prediction_paths["role_off_a"], prediction_paths["role_off_b"]
        ),
        "role_off_a_vs_role_on": compare_prediction_artifacts(
            prediction_paths["role_off_a"], prediction_paths["role_on"]
        ),
        "role_off_b_vs_role_on": compare_prediction_artifacts(
            prediction_paths["role_off_b"], prediction_paths["role_on"]
        ),
    }
    comparison_path = cell_root / "prediction_integrity_comparison.json"
    _atomic_write_json(comparison_path, comparisons)
    verdict = classify_triplet_comparisons(comparisons)
    result: dict[str, Any] = {
        "schema_version": TRIPLET_SCHEMA,
        **verdict,
        "variant": str(args.variant),
        "seed": int(args.seed),
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "runtime_commit": runtime_commit,
        "source_population_sha256": args.source_population_sha256.lower(),
        "source_dataset_count": int(args.source_dataset_count),
        "slurm_job_id": str(slurm_job_id),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible_devices,
        "same_slurm_job": True,
        "same_visible_gpu": True,
        "serial_execution": True,
        "triplet_order": list(TRIPLET_ORDER),
        "source_artifacts": {
            "source_run_root": str(args.source_run_root.resolve()),
            "bound_config_path": str(source_config),
            "bound_config_sha256": args.source_bound_config_sha256.lower(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": args.source_checkpoint_sha256.lower(),
            "prediction_path": str(source_prediction),
            "prediction_sha256": args.source_prediction_sha256.lower(),
        },
        "modes": modes,
        "prediction_integrity_comparison_path": str(comparison_path),
        "prediction_integrity_comparison_sha256": sha256_file(comparison_path),
        "role_calibration_statistics_summarized": False,
        "development_metric_interpreted": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    args = _parse_args()
    cell_root = args.cell_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(cell_root, boundary) or cell_root == boundary:
        raise ValueError("causal triplet cell root leaves the remote write boundary")
    if cell_root.exists():
        raise FileExistsError("causal triplet namespace exists; refusing overwrite")
    cell_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _execute(args, cell_root)
    except Exception as error:
        trace = traceback.format_exc()
        failure = {
            "schema_version": TRIPLET_SCHEMA,
            "status": "FAIL_CAUSAL_TRIPLET_EXECUTION",
            "variant": str(args.variant),
            "seed": int(args.seed),
            "expected_runtime_commit": args.expected_commit.lower(),
            "source_experiment_commit": args.source_experiment_commit.lower(),
            "source_prediction_sha256": args.source_prediction_sha256.lower(),
            "source_population_sha256": args.source_population_sha256.lower(),
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
            "traceback_sha256": hashlib.sha256(
                trace.encode("utf-8", errors="replace")
            ).hexdigest(),
            "role_calibration_statistics_summarized": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(cell_root / "triplet_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "triplet_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0 if not result["status"].startswith("FAIL_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
