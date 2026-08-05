#!/usr/bin/env python3
"""Replay one completed GeoRoute cell with prediction-neutral telemetry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _run_logged,
    build_torchrun_prefix,
)
from tools.bata.analyze_georoute_dynamic_role_calibration import (  # noqa: E402
    summarize_dynamic_role_calibration_telemetry,
)
from tools.bata.georoute_dynamic_floor_m2_contract import (  # noqa: E402
    DYNAMIC_FLOOR_M2_SEED,
    validate_dynamic_floor_m2_config,
)


PHASE_M_SCHEMA = "georoute_phase_m_diagnostic_replay_v1"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


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


def _require_source(expected_commit: str) -> str:
    actual = _git_output("rev-parse", "HEAD").lower()
    if actual != expected_commit.lower():
        raise RuntimeError("Phase M source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("Phase M requires a clean source snapshot")
    return actual


def _require_file_hash(
    path: Path, *, expected_sha256: str, label: str
) -> Path:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed = sha256_file(resolved)
    if observed != expected_sha256.lower():
        raise RuntimeError(f"{label} SHA-256 mismatch")
    return resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


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
    parser.add_argument("--source-population-sha256")
    parser.add_argument("--source-dataset-count", type=int)
    parser.add_argument("--source-experiment-commit", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--role-calibration-telemetry", action="store_true")
    return parser.parse_args()


def _configure_replay_instrumentation(
    cfg: Config,
    *,
    replay_work: Path,
    role_calibration_telemetry_enabled: bool,
) -> None:
    """Enable result-blind replay diagnostics without altering the hard route."""

    cfg.work_dir = str(replay_work)
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = True
    cfg.model.backbone.custom.georoute_role_calibration_telemetry_enabled = bool(
        role_calibration_telemetry_enabled
    )
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.post_processing.save_dict = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False


def _build_replay_test_arguments(
    *,
    command_prefix: list[str],
    bound_config: Path,
    checkpoint: Path,
    seed: int,
    role_calibration_telemetry_enabled: bool,
) -> list[str]:
    """Preserve the source evaluation contract for frozen M2 replays."""

    arguments = [
        *command_prefix,
        "tools/test.py",
        str(bound_config),
        "--checkpoint",
        str(checkpoint),
        "--seed",
        str(seed),
        "--id",
        "0",
    ]
    if not role_calibration_telemetry_enabled:
        arguments.append("--not_eval")
    return arguments


def _execute(args: argparse.Namespace, cell_root: Path) -> dict[str, Any]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if not slurm_job_id:
        raise RuntimeError("Phase M replay must run inside Slurm")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible_devices or "," in visible_devices:
        raise RuntimeError("Phase M requires exactly one Slurm-visible GPU")
    runtime_commit = _require_source(args.expected_commit)
    source_run_root = args.source_run_root.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if (
        not source_run_root.is_dir()
        or not _inside(source_run_root, write_boundary)
    ):
        raise RuntimeError("Phase M source run root is invalid")
    source_config = _require_file_hash(
        args.source_bound_config,
        expected_sha256=args.source_bound_config_sha256,
        label="source bound config",
    )
    checkpoint = _require_file_hash(
        args.source_checkpoint,
        expected_sha256=args.source_checkpoint_sha256,
        label="source checkpoint",
    )
    source_prediction = _require_file_hash(
        args.source_prediction,
        expected_sha256=args.source_prediction_sha256,
        label="source prediction",
    )
    for label, artifact in (
        ("source bound config", source_config),
        ("source checkpoint", checkpoint),
        ("source prediction", source_prediction),
    ):
        if not _inside(artifact, source_run_root):
            raise RuntimeError(f"{label} leaves the source run root")

    cfg = Config.fromfile(str(source_config))
    protocol = cfg.get("georoute_protocol", {})
    if (
        bool(protocol.get("official_test_open_allowed", True))
        or bool(protocol.get("gt_for_route_allowed", True))
        or str(cfg.dataset.test.get("subset_name", "")) != "training"
        or not bool(cfg.dataset.test.get("test_mode", False))
        or str(cfg.evaluation.get("subset", "")) != "training"
    ):
        raise RuntimeError(
            "Phase M source config is not the no-GT development population"
        )
    if bool(cfg.inference.get("load_from_raw_predictions", False)):
        raise RuntimeError("Phase M forbids raw-prediction replay shortcuts")
    source_population_sha256 = args.source_population_sha256
    source_binding = None
    if args.role_calibration_telemetry:
        if int(args.seed) != DYNAMIC_FLOOR_M2_SEED:
            raise RuntimeError("role calibration replay requires the frozen M2 seed")
        validate_dynamic_floor_m2_config(
            cfg,
            arm=str(args.variant),
            phase="accuracy",
        )
        source_binding = dict(cfg.georoute_dynamic_floor_m2_binding)
        if (
            source_binding.get("runtime_commit")
            != args.source_experiment_commit.lower()
            or bool(
                cfg.model.backbone.custom.get(
                    "georoute_role_calibration_telemetry_enabled",
                    False,
                )
            )
            or not isinstance(source_population_sha256, str)
            or len(source_population_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in source_population_sha256.lower()
            )
            or not isinstance(args.source_dataset_count, int)
            or isinstance(args.source_dataset_count, bool)
            or args.source_dataset_count <= 0
        ):
            raise RuntimeError("role calibration source binding is invalid")

    replay_work = cell_root / "replay"
    bound_config = cell_root / "control" / "phase_m_bound_config.py"
    _configure_replay_instrumentation(
        cfg,
        replay_work=replay_work,
        role_calibration_telemetry_enabled=args.role_calibration_telemetry,
    )
    replay_binding = dict(
        schema_version=PHASE_M_SCHEMA,
        variant=str(args.variant),
        seed=int(args.seed),
        source_run_root=str(source_run_root),
        source_experiment_commit=args.source_experiment_commit.lower(),
        runtime_commit=runtime_commit,
        source_bound_config_sha256=args.source_bound_config_sha256.lower(),
        source_checkpoint_sha256=args.source_checkpoint_sha256.lower(),
        source_prediction_sha256=args.source_prediction_sha256.lower(),
        instrumentation_only=True,
        official_test_opened=False,
    )
    if args.role_calibration_telemetry:
        replay_binding.update(
            role_calibration_telemetry_enabled=True,
            source_population_sha256=source_population_sha256.lower(),
            source_dataset_count=int(args.source_dataset_count),
            fixed_role_quota_used=False,
            changes_route_or_execution=False,
        )
    cfg.georoute_phase_m_binding = replay_binding
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    command_prefix, rendezvous = build_torchrun_prefix(
        phase="test",
        slurm_job_id=str(slurm_job_id),
        stage="phase_m",
        variant=str(args.variant),
        seed=int(args.seed),
    )
    test_log = cell_root / "test.out"
    _run_logged(
        _build_replay_test_arguments(
            command_prefix=command_prefix,
            bound_config=bound_config,
            checkpoint=checkpoint,
            seed=int(args.seed),
            role_calibration_telemetry_enabled=(
                args.role_calibration_telemetry
            ),
        ),
        log_path=test_log,
        env=inherited,
    )

    effective_work = replay_work / "gpu1_id0"
    prediction = effective_work / "result_detection.json"
    telemetry_path = (
        effective_work / "georoute_diagnostic_telemetry.json"
    )
    profile_path = effective_work / "georoute_development_profile.json"
    for artifact in (prediction, telemetry_path, profile_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    prediction_sha256 = sha256_file(prediction)
    prediction_parity = (
        prediction_sha256 == args.source_prediction_sha256.lower()
    )
    telemetry = _read_json(telemetry_path)
    if (
        telemetry.get("schema_version")
        != "georoute_diagnostic_telemetry_v1"
        or telemetry.get("official_test_opened") is not False
        or telemetry.get("gt_for_route_used") is not False
        or telemetry.get("teacher_for_route_used") is not False
        or telemetry.get("raw_prediction_cache_used") is not False
    ):
        raise RuntimeError("Phase M telemetry violated its no-leak schema")
    dataset_count = int(telemetry.get("dataset_count", -1))
    record_count = int(telemetry.get("record_count", -2))
    population_complete = dataset_count > 0 and record_count == dataset_count
    observed_population_sha256 = str(telemetry.get("population_sha256", ""))
    if not prediction_parity:
        raise RuntimeError("Phase M prediction SHA-256 parity failed")
    if not population_complete:
        raise RuntimeError("Phase M telemetry population is incomplete")
    calibration_summary_path = None
    calibration_summary = None
    if args.role_calibration_telemetry:
        if (
            observed_population_sha256 != source_population_sha256.lower()
            or dataset_count != int(args.source_dataset_count)
            or dict(telemetry.get("phase_m_binding", {})) != replay_binding
        ):
            raise RuntimeError("role calibration population SHA-256 parity failed")
        profile = _read_json(profile_path)
        last_audit = profile.get("last_georoute_audit")
        if (
            not isinstance(last_audit, Mapping)
            or last_audit.get("role_calibration_telemetry_enabled") is not True
        ):
            raise RuntimeError("role calibration replay did not activate instrumentation")
        calibration_summary = summarize_dynamic_role_calibration_telemetry(
            telemetry_path
        )
        if (
            calibration_summary.get("population_sha256")
            != source_population_sha256.lower()
            or calibration_summary.get("interpretation_boundary", {}).get(
                "changes_route_or_execution"
            )
            is not False
        ):
            raise RuntimeError("role calibration summary violated its frozen boundary")
        calibration_summary_path = cell_root / "role_calibration_summary.json"
        _atomic_write_json(calibration_summary_path, calibration_summary)

    result: dict[str, Any] = {
        "schema_version": PHASE_M_SCHEMA,
        "status": "PASS_DIAGNOSTIC_ONLY",
        "variant": str(args.variant),
        "seed": int(args.seed),
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "runtime_commit": runtime_commit,
        "slurm_job_id": str(slurm_job_id),
        "source_artifacts": {
            "source_run_root": str(source_run_root),
            "bound_config_path": str(source_config),
            "bound_config_sha256": args.source_bound_config_sha256.lower(),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": args.source_checkpoint_sha256.lower(),
            "prediction_path": str(source_prediction),
            "prediction_sha256": args.source_prediction_sha256.lower(),
        },
        "replay_artifacts": {
            "bound_config_path": str(bound_config),
            "bound_config_sha256": sha256_file(bound_config),
            "prediction_path": str(prediction),
            "prediction_sha256": prediction_sha256,
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "profile_path": str(profile_path),
            "profile_sha256": sha256_file(profile_path),
            "test_log_path": str(test_log),
            "test_log_sha256": sha256_file(test_log),
        },
        "prediction_sha256_parity": prediction_parity,
        "telemetry_population_complete": population_complete,
        "dataset_count": dataset_count,
        "record_count": record_count,
        "population_sha256": observed_population_sha256,
        "rendezvous": rendezvous,
        "instrumentation_only": True,
        "development_metric_is_replay_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
        "old_selector_completed": False,
        "paper_claim_allowed": False,
    }
    if args.role_calibration_telemetry:
        result.update(
            role_calibration_telemetry_enabled=True,
            source_population_sha256=source_population_sha256.lower(),
            source_dataset_count=int(args.source_dataset_count),
            population_sha256_parity=True,
            fixed_role_quota_used=False,
            changes_route_or_execution=False,
            role_calibration_summary=dict(calibration_summary),
        )
        result["replay_artifacts"]["role_calibration_summary_path"] = str(
            calibration_summary_path
        )
        result["replay_artifacts"]["role_calibration_summary_sha256"] = (
            sha256_file(calibration_summary_path)
        )
    result["result_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    args = _parse_args()
    cell_root = args.cell_root.resolve()
    write_boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(cell_root, write_boundary) or cell_root == write_boundary:
        raise ValueError("Phase M cell root leaves the remote write boundary")
    if cell_root.exists():
        raise FileExistsError(
            "Phase M cell namespace already exists; refusing resume"
        )
    cell_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _execute(args, cell_root)
    except Exception as error:
        trace = traceback.format_exc()
        try:
            observed_runtime_commit = _git_output(
                "rev-parse", "HEAD"
            ).lower()
        except Exception:
            observed_runtime_commit = None
        failure = {
            "schema_version": PHASE_M_SCHEMA,
            "status": "FAIL_DIAGNOSTIC_REPLAY",
            "variant": str(args.variant),
            "seed": int(args.seed),
            "observed_runtime_commit": observed_runtime_commit,
            "expected_runtime_commit": args.expected_commit.lower(),
            "source_experiment_commit": (
                args.source_experiment_commit.lower()
            ),
            "source_run_root": str(args.source_run_root.resolve()),
            "source_bound_config_path": str(
                args.source_bound_config.resolve()
            ),
            "source_bound_config_sha256": (
                args.source_bound_config_sha256.lower()
            ),
            "source_checkpoint_path": str(
                args.source_checkpoint.resolve()
            ),
            "source_checkpoint_sha256": (
                args.source_checkpoint_sha256.lower()
            ),
            "source_prediction_path": str(
                args.source_prediction.resolve()
            ),
            "source_prediction_sha256": (
                args.source_prediction_sha256.lower()
            ),
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
            "traceback_sha256": hashlib.sha256(
                trace.encode("utf-8", errors="replace")
            ).hexdigest(),
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        if args.role_calibration_telemetry:
            failure.update(
                role_calibration_telemetry_enabled=True,
                source_population_sha256=(
                    args.source_population_sha256.lower()
                    if isinstance(args.source_population_sha256, str)
                    else None
                ),
                source_dataset_count=args.source_dataset_count,
                fixed_role_quota_used=False,
            )
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(cell_root / "phase_m_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "phase_m_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
