#!/usr/bin/env python3
"""Run a frozen-checkpoint SCNR residual-window-centering mechanism probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import traceback
from typing import Any, Mapping

from mmengine.config import Config

from tools.bata.analyze_georoute_dynamic_role_calibration import (
    summarize_dynamic_role_calibration_telemetry,
)
from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA,
    summarize_dynamic_floor_m2_telemetry,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.georoute_stage_runner import _run_logged, build_torchrun_prefix
from tools.bata.run_georoute_phase_m_replay import (
    _atomic_write_json,
    _inside,
    _read_json,
)
from tools.bata.run_georoute_role_instrumentation_pair import (
    _configure_pair_mode,
    _utc_now,
    _validate_formal_telemetry,
    _validate_source,
    compare_prediction_artifacts,
)


PROBE_SCHEMA = "scnr_residual_window_centering_probe_v1"
PROBE_BINDING_SCHEMA = DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA
PROBE_ORDER = ("centered_a", "centered_b")
CENTERING_MEAN_ABS_TOLERANCE = 1e-4


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


def _configure_probe_mode(
    cfg: Config,
    *,
    work_dir: Path,
    binding: Mapping[str, Any],
) -> None:
    """Enable only the approved residual offset calibration and diagnostics."""

    _configure_pair_mode(
        cfg,
        work_dir=work_dir,
        role_calibration_enabled=True,
        binding=binding,
    )
    cfg.model.backbone.custom.georoute_branch_calibration_mode = (
        "residual_window_center"
    )


def _build_probe_test_arguments(
    *,
    command_prefix: list[str],
    bound_config: Path,
    checkpoint: Path,
    seed: int,
) -> list[str]:
    """Run full inference/telemetry while making metric evaluation impossible."""

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
        "--not_eval",
    ]


def _route_payload_sha256(telemetry: Mapping[str, Any]) -> str:
    records = telemetry.get("records")
    if not isinstance(records, list):
        raise RuntimeError("residual-centering telemetry records are missing")
    routes = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("route"), Mapping
        ):
            raise RuntimeError("residual-centering telemetry route is missing")
        routes.append(
            {
                "dataset_index": record.get("dataset_index"),
                "video_id": record.get("video_id"),
                "route": dict(record["route"]),
            }
        )
    return canonical_sha256(routes)


def summarize_residual_centering_branch_payload(
    telemetry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the centering transform receipt without consuming task metrics."""

    records = telemetry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("residual-centering branch telemetry is empty")
    before_values: list[float] = []
    after_values: list[float] = []
    valid_counts: list[int] = []
    for record in records:
        route = record.get("route") if isinstance(record, Mapping) else None
        branch = route.get("branch_calibration") if isinstance(route, Mapping) else None
        if not isinstance(branch, Mapping):
            raise ValueError("residual-centering branch receipt is missing")
        tubelet_count = route.get("tubelet_count")
        item_count = route.get("item_count")
        valid_count = branch.get("valid_candidate_count")
        before = branch.get("residual_valid_mean_before")
        after = branch.get("residual_valid_mean_after")
        if (
            branch.get("schema_version")
            != "scnr_dynamic_branch_calibration_window_v1"
            or branch.get("mode") != "residual_window_center"
            or branch.get("target") != "delta_residual"
            or branch.get("scope") != "complete_window_all_valid_candidates"
            or branch.get("changes_q_base") is not False
            or branch.get("changes_delta_roi") is not False
            or branch.get("changes_context_zero_modifier") is not False
            or branch.get("changes_budget_or_role_quota") is not False
            or branch.get("mean_detached") is not False
            or isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or isinstance(tubelet_count, bool)
            or not isinstance(tubelet_count, int)
            or isinstance(item_count, bool)
            or not isinstance(item_count, int)
            or int(valid_count) != int(tubelet_count) * int(item_count)
            or isinstance(before, bool)
            or not isinstance(before, (int, float))
            or isinstance(after, bool)
            or not isinstance(after, (int, float))
            or not math.isfinite(float(before))
            or not math.isfinite(float(after))
            or abs(float(after)) > CENTERING_MEAN_ABS_TOLERANCE
        ):
            raise ValueError("residual-centering branch receipt is invalid")
        valid_counts.append(int(valid_count))
        before_values.append(float(before))
        after_values.append(float(after))
    return {
        "schema_version": "scnr_residual_window_centering_branch_summary_v1",
        "record_count": len(records),
        "valid_candidate_count_min": min(valid_counts),
        "valid_candidate_count_max": max(valid_counts),
        "residual_valid_mean_before_min": min(before_values),
        "residual_valid_mean_before_max": max(before_values),
        "residual_valid_mean_after_max_abs": max(map(abs, after_values)),
        "mean_after_abs_tolerance": CENTERING_MEAN_ABS_TOLERANCE,
        "transform_receipts_valid": True,
        "metric_consumed": False,
    }


def classify_residual_centering_role_gate(
    role_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the pre-registered structural reachability gate to one M2 arm."""

    roles = role_summary.get("roles")
    valid = roles.get("valid", {}).get("counts") if isinstance(roles, Mapping) else None
    selected = (
        roles.get("selected", {}).get("counts")
        if isinstance(roles, Mapping)
        else None
    )
    role_order = ("context", "roi", "residual")
    if (
        not isinstance(valid, Mapping)
        or not isinstance(selected, Mapping)
        or set(valid) != set(role_order)
        or set(selected) != set(role_order)
        or any(
            isinstance(counts[role], bool)
            or not isinstance(counts[role], int)
            or counts[role] < 0
            for counts in (valid, selected)
            for role in role_order
        )
        or sum(valid.values()) <= 0
        or sum(selected.values()) <= 0
    ):
        raise ValueError("residual-centering role summary is invalid")
    conditions = {
        "valid_context_reachable": int(valid["context"]) > 0,
        "valid_roi_reachable": int(valid["roi"]) > 0,
        "selected_non_residual_reachable": (
            int(selected["context"]) + int(selected["roi"]) > 0
        ),
        "residual_not_all_valid_candidates": (
            int(valid["residual"]) < sum(map(int, valid.values()))
        ),
    }
    passed = all(conditions.values())
    return {
        "status": (
            "PASS_ARM_RESIDUAL_WINDOW_CENTERING_ROLE_REACHABILITY"
            if passed
            else "HOLD_ARM_RESIDUAL_WINDOW_CENTERING_ROLE_REACHABILITY_NO_TRAINING"
        ),
        "passed": passed,
        "conditions": conditions,
        "valid_role_counts": {role: int(valid[role]) for role in role_order},
        "selected_role_counts": {
            role: int(selected[role]) for role in role_order
        },
        "claim_scope": "structural_reachability_only",
        "performance_claim_allowed": False,
    }


def _execute(args: argparse.Namespace, cell_root: Path) -> dict[str, Any]:
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not slurm_job_id or not visible_devices or "," in visible_devices:
        raise RuntimeError(
            "residual-centering probe requires exactly one Slurm-visible GPU"
        )
    runtime_commit, source_config, checkpoint, source_prediction = _validate_source(
        args
    )
    binding_common = {
        "schema_version": PROBE_BINDING_SCHEMA,
        "probe_schema_version": PROBE_SCHEMA,
        "variant": str(args.variant),
        "seed": int(args.seed),
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "runtime_commit": runtime_commit,
        "source_bound_config_sha256": args.source_bound_config_sha256.lower(),
        "source_checkpoint_sha256": args.source_checkpoint_sha256.lower(),
        "source_prediction_sha256": args.source_prediction_sha256.lower(),
        "source_population_sha256": args.source_population_sha256.lower(),
        "source_dataset_count": int(args.source_dataset_count),
        "branch_calibration_mode": "residual_window_center",
        "branch_calibration_scope": "complete_window_all_valid_candidates",
        "role_calibration_telemetry_enabled": True,
        "mechanism_probe_only": True,
        "training_performed": False,
        "same_slurm_job": True,
        "same_visible_gpu": True,
        "serial_execution": True,
        "strict_deterministic_algorithms": True,
        "sdp_backend": "math",
        "tf32_enabled": False,
        "fixed_role_quota_used": False,
        "q_ctx_used": False,
        "changes_route_or_execution": True,
        "metric_evaluation_enabled": False,
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
    for mode in PROBE_ORDER:
        work_dir = cell_root / mode / "gpu1_id0"
        bound_config = cell_root / "control" / f"{mode}_bound_config.py"
        binding = {**binding_common, "probe_mode": mode}
        cfg = Config.fromfile(str(source_config))
        _configure_probe_mode(cfg, work_dir=work_dir, binding=binding)
        bound_config.parent.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(bound_config))
        command_prefix, rendezvous = build_torchrun_prefix(
            phase="test",
            slurm_job_id=str(slurm_job_id),
            stage=f"residual_centering_probe_{mode}",
            variant=str(args.variant),
            seed=int(args.seed),
        )
        started_at = _utc_now()
        log_path = cell_root / mode / "test.out"
        _run_logged(
            _build_probe_test_arguments(
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
            raise RuntimeError(
                "residual-centering probe unexpectedly emitted a timed profile"
            )
        telemetry = _read_json(telemetry_path)
        formal_receipt = _validate_formal_telemetry(
            telemetry,
            expected_binding=binding,
            expected_population_sha256=args.source_population_sha256,
            expected_dataset_count=int(args.source_dataset_count),
            role_calibration_enabled=True,
        )
        floor_summary = summarize_dynamic_floor_m2_telemetry(telemetry_path)
        role_summary = summarize_dynamic_role_calibration_telemetry(telemetry_path)
        branch_summary = summarize_residual_centering_branch_payload(telemetry)
        gate = classify_residual_centering_role_gate(role_summary)
        modes[mode] = {
            "started_at_utc": started_at,
            "ended_at_utc": ended_at,
            "bound_config_path": str(bound_config),
            "bound_config_sha256": sha256_file(bound_config),
            "prediction_path": str(prediction),
            "prediction_sha256": sha256_file(prediction),
            "telemetry_path": str(telemetry_path),
            "telemetry_sha256": sha256_file(telemetry_path),
            "route_payload_sha256": _route_payload_sha256(telemetry),
            "test_log_path": str(log_path),
            "test_log_sha256": sha256_file(log_path),
            "formal_telemetry_receipt": formal_receipt,
            "exact_budget_ragged_summary": floor_summary,
            "role_calibration_summary": role_summary,
            "branch_calibration_summary": branch_summary,
            "structural_gate": gate,
            "rendezvous": rendezvous,
        }

    prediction_comparison = compare_prediction_artifacts(
        Path(modes["centered_a"]["prediction_path"]),
        Path(modes["centered_b"]["prediction_path"]),
    )
    route_payload_parity = (
        modes["centered_a"]["route_payload_sha256"]
        == modes["centered_b"]["route_payload_sha256"]
    )
    if not prediction_comparison["raw_sha256_parity"] or not route_payload_parity:
        raise RuntimeError(
            "residual-centering duplicate replay failed exact prediction/route parity"
        )
    gate_parity = (
        modes["centered_a"]["structural_gate"]
        == modes["centered_b"]["structural_gate"]
    )
    if not gate_parity:
        raise RuntimeError("residual-centering duplicate structural summaries differ")
    arm_gate = dict(modes["centered_a"]["structural_gate"])
    arm_passed = bool(arm_gate["passed"])
    comparison_path = cell_root / "duplicate_integrity_comparison.json"
    comparison = {
        "prediction": prediction_comparison,
        "route_payload_sha256_parity": route_payload_parity,
        "structural_gate_parity": gate_parity,
    }
    _atomic_write_json(comparison_path, comparison)
    result: dict[str, Any] = {
        "schema_version": PROBE_SCHEMA,
        "status": (
            "PASS_ARM_RESIDUAL_WINDOW_CENTERING_ROLE_REACHABILITY"
            if arm_passed
            else "HOLD_ARM_RESIDUAL_WINDOW_CENTERING_ROLE_REACHABILITY_NO_TRAINING"
        ),
        "variant": str(args.variant),
        "seed": int(args.seed),
        "source_experiment_commit": args.source_experiment_commit.lower(),
        "runtime_commit": runtime_commit,
        "source_population_sha256": args.source_population_sha256.lower(),
        "source_dataset_count": int(args.source_dataset_count),
        "slurm_job_id": str(slurm_job_id),
        "hostname": socket.gethostname(),
        "cuda_visible_devices": visible_devices,
        "probe_order": list(PROBE_ORDER),
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
        "duplicate_integrity_comparison_path": str(comparison_path),
        "duplicate_integrity_comparison_sha256": sha256_file(comparison_path),
        "deterministic_duplicate_replay_supported": True,
        "arm_structural_gate": arm_gate,
        "training_protocol_design_authorized_for_arm": arm_passed,
        "training_performed": False,
        "development_metric_evaluated": False,
        "prediction_artifact_used_for_determinism_only": True,
        "geometry_floor_selected": False,
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
        raise ValueError(
            "residual-centering probe root leaves the remote write boundary"
        )
    if cell_root.exists():
        raise FileExistsError(
            "residual-centering probe namespace exists; refusing overwrite"
        )
    cell_root.mkdir(parents=True, exist_ok=False)
    try:
        result = _execute(args, cell_root)
    except Exception as error:
        trace = traceback.format_exc()
        failure = {
            "schema_version": PROBE_SCHEMA,
            "status": "FAIL_RESIDUAL_WINDOW_CENTERING_PROBE_EXECUTION",
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
            "development_metric_evaluated": False,
            "training_performed": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_write_json(cell_root / "probe_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "probe_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
