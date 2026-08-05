#!/usr/bin/env python3
"""Run a same-GPU role-telemetry OFF/ON prediction-neutrality pair."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
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
    DYNAMIC_FLOOR_M2_SEED,
    validate_dynamic_floor_m2_config,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_stage_runner import _run_logged, build_torchrun_prefix
from tools.bata.run_georoute_phase_m_replay import (
    _atomic_write_json,
    _git_output,
    _inside,
    _read_json,
    _require_file_hash,
)


PAIR_SCHEMA = DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA
FORMAL_TELEMETRY_SCHEMA = "georoute_formal_development_telemetry_v1"
PAIR_ORDER = ("role_off", "role_on")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _configure_pair_mode(
    cfg: Config,
    *,
    work_dir: Path,
    role_calibration_enabled: bool,
    binding: Mapping[str, Any],
) -> None:
    """Change only output/provenance and the role-calibration observation flag."""

    cfg.work_dir = str(work_dir)
    custom = cfg.model.backbone.custom
    custom.georoute_diagnostic_telemetry_enabled = True
    custom.georoute_role_calibration_telemetry_enabled = bool(role_calibration_enabled)
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_development_profile = dict(enabled=False)
    cfg.post_processing.save_dict = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.georoute_phase_m_binding = dict(binding)


def _build_pair_test_arguments(
    *,
    command_prefix: list[str],
    bound_config: Path,
    checkpoint: Path,
    seed: int,
) -> list[str]:
    """Use the same frozen M2 accuracy/evaluation path for OFF and ON."""

    return [
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


def _validate_formal_telemetry(
    telemetry: Mapping[str, Any],
    *,
    expected_binding: Mapping[str, Any],
    expected_population_sha256: str,
    expected_dataset_count: int,
    role_calibration_enabled: bool,
) -> dict[str, Any]:
    if (
        telemetry.get("schema_version") != FORMAL_TELEMETRY_SCHEMA
        or telemetry.get("development_only") is not True
        or telemetry.get("official_test_opened") is not False
        or telemetry.get("gt_for_route_used") is not False
        or telemetry.get("teacher_for_route_used") is not False
        or telemetry.get("oracle_used") is not False
        or telemetry.get("raw_prediction_cache_used") is not False
        or dict(telemetry.get("phase_m_binding", {})) != dict(expected_binding)
    ):
        raise RuntimeError("paired replay telemetry violated its no-leak binding")
    dataset_count = int(telemetry.get("dataset_count", -1))
    record_count = int(telemetry.get("record_count", -2))
    unique_count = int(telemetry.get("unique_dataset_count", -3))
    padding_count = int(telemetry.get("sampler_padding_count", -4))
    if (
        dataset_count != int(expected_dataset_count)
        or record_count != dataset_count
        or unique_count != dataset_count
        or padding_count != 0
        or telemetry.get("population_sha256") != expected_population_sha256.lower()
    ):
        raise RuntimeError("paired replay telemetry population parity failed")
    records = telemetry.get("records")
    if not isinstance(records, list) or len(records) != dataset_count:
        raise RuntimeError("paired replay telemetry records are incomplete")
    policy_presence = []
    for record in records:
        route = record.get("route") if isinstance(record, Mapping) else None
        if not isinstance(route, Mapping):
            raise RuntimeError("paired replay telemetry lacks route records")
        policy_presence.append("policy_calibration" in route)
    if any(value is not role_calibration_enabled for value in policy_presence):
        raise RuntimeError("paired replay role-calibration treatment is inconsistent")
    return {
        "dataset_count": dataset_count,
        "record_count": record_count,
        "unique_dataset_count": unique_count,
        "sampler_padding_count": padding_count,
        "population_sha256": expected_population_sha256.lower(),
        "role_calibration_records_present": role_calibration_enabled,
    }


def _load_prediction(path: Path) -> tuple[dict[str, Any], bytes]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict):
        raise RuntimeError(f"prediction artifact has an invalid schema: {path}")
    return payload, raw


def _prediction_identities(payload: Mapping[str, Any]) -> Counter[tuple[Any, ...]]:
    identities: Counter[tuple[Any, ...]] = Counter()
    for video_id, records in payload["results"].items():
        if not isinstance(video_id, str) or not isinstance(records, list):
            raise RuntimeError("prediction artifact contains malformed video records")
        for record in records:
            if not isinstance(record, Mapping):
                raise RuntimeError("prediction artifact contains a malformed record")
            segment = record.get("segment")
            label = record.get("label")
            score = record.get("score")
            if (
                not isinstance(segment, list)
                or len(segment) != 2
                or not all(isinstance(value, (int, float)) for value in segment)
                or not isinstance(label, str)
                or not isinstance(score, (int, float))
            ):
                raise RuntimeError("prediction artifact contains an invalid detection")
            identities[(video_id, label, float(segment[0]), float(segment[1]))] += 1
    return identities


def compare_prediction_artifacts(left: Path, right: Path) -> dict[str, Any]:
    """Compare prediction integrity without computing or reading task metrics."""

    left_payload, left_raw = _load_prediction(left)
    right_payload, right_raw = _load_prediction(right)
    left_results = left_payload["results"]
    right_results = right_payload["results"]
    left_ids = _prediction_identities(left_payload)
    right_ids = _prediction_identities(right_payload)
    left_record_count = sum(len(records) for records in left_results.values())
    right_record_count = sum(len(records) for records in right_results.values())
    identity_overlap = sum((left_ids & right_ids).values())
    return {
        "left_sha256": hashlib.sha256(left_raw).hexdigest(),
        "right_sha256": hashlib.sha256(right_raw).hexdigest(),
        "raw_sha256_parity": left_raw == right_raw,
        "canonical_sha256_left": canonical_sha256(left_payload),
        "canonical_sha256_right": canonical_sha256(right_payload),
        "json_semantic_parity": left_payload == right_payload,
        "video_keyset_parity": set(left_results) == set(right_results),
        "left_video_count": len(left_results),
        "right_video_count": len(right_results),
        "left_record_count": left_record_count,
        "right_record_count": right_record_count,
        "exact_candidate_identity_overlap": identity_overlap,
    }


def _validate_source(args: argparse.Namespace) -> tuple[str, Path, Path, Path]:
    runtime_commit = _git_output("rev-parse", "HEAD").lower()
    if runtime_commit != args.expected_commit.lower():
        raise RuntimeError("paired replay source differs from --expected-commit")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("paired replay requires a clean source snapshot")
    if int(args.seed) != DYNAMIC_FLOOR_M2_SEED:
        raise RuntimeError("paired replay requires the frozen M2 seed")
    if (
        not isinstance(args.source_dataset_count, int)
        or isinstance(args.source_dataset_count, bool)
        or args.source_dataset_count <= 0
    ):
        raise RuntimeError("paired replay source dataset count is invalid")
    population_sha = args.source_population_sha256.lower()
    if len(population_sha) != 64 or any(
        c not in "0123456789abcdef" for c in population_sha
    ):
        raise RuntimeError("paired replay source population SHA-256 is invalid")
    source_root = args.source_run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not source_root.is_dir() or not _inside(source_root, boundary):
        raise RuntimeError("paired replay source run root is invalid")
    config_path = _require_file_hash(
        args.source_bound_config,
        expected_sha256=args.source_bound_config_sha256,
        label="source bound config",
    )
    checkpoint = _require_file_hash(
        args.source_checkpoint,
        expected_sha256=args.source_checkpoint_sha256,
        label="source checkpoint",
    )
    prediction = _require_file_hash(
        args.source_prediction,
        expected_sha256=args.source_prediction_sha256,
        label="source prediction",
    )
    for label, artifact in (
        ("source bound config", config_path),
        ("source checkpoint", checkpoint),
        ("source prediction", prediction),
    ):
        if not _inside(artifact, source_root):
            raise RuntimeError(f"{label} leaves the source run root")
    cfg = Config.fromfile(str(config_path))
    validate_dynamic_floor_m2_config(
        cfg,
        arm=str(args.variant),
        phase="accuracy",
    )
    source_binding = dict(cfg.georoute_dynamic_floor_m2_binding)
    if (
        source_binding.get("runtime_commit") != args.source_experiment_commit.lower()
        or bool(
            cfg.model.backbone.custom.get(
                "georoute_role_calibration_telemetry_enabled", False
            )
        )
        or not bool(
            cfg.model.backbone.custom.get(
                "georoute_diagnostic_telemetry_enabled", False
            )
        )
        or bool(cfg.georoute_development_profile.get("enabled", True))
    ):
        raise RuntimeError("paired replay source instrumentation contract is invalid")
    return runtime_commit, config_path, checkpoint, prediction


def _execute(args: argparse.Namespace, cell_root: Path) -> dict[str, Any]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not slurm_job_id or not visible_devices or "," in visible_devices:
        raise RuntimeError("paired replay requires exactly one Slurm-visible GPU")
    runtime_commit, source_config, checkpoint, source_prediction = _validate_source(
        args
    )
    pair_common = {
        "schema_version": PAIR_SCHEMA,
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
    for role_enabled, mode in ((False, "role_off"), (True, "role_on")):
        work_dir = cell_root / mode / "gpu1_id0"
        bound_config = cell_root / "control" / f"{mode}_bound_config.py"
        binding = {
            **pair_common,
            "pair_mode": mode,
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
            stage=f"role_pair_{mode}",
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
            raise RuntimeError("paired replay unexpectedly emitted a timed profile")
        telemetry_receipt = _validate_formal_telemetry(
            _read_json(telemetry_path),
            expected_binding=binding,
            expected_population_sha256=args.source_population_sha256,
            expected_dataset_count=int(args.source_dataset_count),
            role_calibration_enabled=role_enabled,
        )
        modes[mode] = {
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
    source_vs_off = compare_prediction_artifacts(
        source_prediction, Path(modes["role_off"]["prediction_path"])
    )
    source_vs_on = compare_prediction_artifacts(
        source_prediction, Path(modes["role_on"]["prediction_path"])
    )
    off_vs_on = compare_prediction_artifacts(
        Path(modes["role_off"]["prediction_path"]),
        Path(modes["role_on"]["prediction_path"]),
    )
    comparisons = {
        "source_vs_role_off": source_vs_off,
        "source_vs_role_on": source_vs_on,
        "role_off_vs_role_on": off_vs_on,
    }
    comparison_path = cell_root / "prediction_integrity_comparison.json"
    _atomic_write_json(comparison_path, comparisons)
    if not off_vs_on["raw_sha256_parity"]:
        raise RuntimeError("role OFF/ON prediction SHA-256 parity failed")
    source_parity = bool(
        source_vs_off["raw_sha256_parity"] and source_vs_on["raw_sha256_parity"]
    )
    result: dict[str, Any] = {
        **pair_common,
        "status": (
            "PASS_PAIRED_NEUTRALITY_AND_SOURCE_PARITY"
            if source_parity
            else "PASS_PAIRED_NEUTRALITY_SOURCE_REPLAY_DRIFT_DIAGNOSTIC_ONLY"
        ),
        "slurm_job_id": str(slurm_job_id),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible_devices,
        "pair_order": list(PAIR_ORDER),
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
        "paired_role_instrumentation_neutrality_supported": True,
        "source_prediction_raw_parity": source_parity,
        "role_calibration_analysis_allowed_under_frozen_contract": source_parity,
        "role_calibration_statistics_summarized": False,
        "development_metric_interpreted": False,
        "paper_claim_allowed": False,
    }
    result["result_sha256"] = canonical_sha256(result)
    return result


def main() -> int:
    args = _parse_args()
    cell_root = args.cell_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(cell_root, boundary) or cell_root == boundary:
        raise ValueError("paired replay cell root leaves the remote write boundary")
    if cell_root.exists():
        raise FileExistsError("paired replay namespace exists; refusing overwrite")
    cell_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _execute(args, cell_root)
    except Exception as error:
        trace = traceback.format_exc()
        failure = {
            "schema_version": PAIR_SCHEMA,
            "status": "FAIL_PAIRED_ROLE_INSTRUMENTATION_NEUTRALITY",
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
        comparison_path = cell_root / "prediction_integrity_comparison.json"
        if comparison_path.is_file():
            failure["prediction_integrity_comparison_path"] = str(comparison_path)
            failure["prediction_integrity_comparison_sha256"] = sha256_file(
                comparison_path
            )
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(cell_root / "pair_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "pair_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
