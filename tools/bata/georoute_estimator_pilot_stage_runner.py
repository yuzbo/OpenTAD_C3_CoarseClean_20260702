#!/usr/bin/env python3
"""Run one frozen development-only GeoRoute estimator pilot arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_estimator_pilot_contract import (  # noqa: E402
    PILOT_CONTRACT_SCHEMA,
    PILOT_ARM_ORDER,
    PILOT_EPOCHS,
    PILOT_K,
    PILOT_P0_SUITE_SCHEMA,
    PILOT_SEED,
    PILOT_STAGE_RESULT_SCHEMA,
    PILOT_STUDY_ID,
    REPRESENTATION_KEYS,
    bind_pilot_config,
    pilot_arm_spec,
    pilot_cell_relative_path,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import (  # noqa: E402
    _atomic_write_json,
    _read_json,
    _run_logged,
    _validate_rendezvous_receipt,
    build_torchrun_prefix,
    parse_official_style_map,
)
from tools.bata.georoute_storage import storage_capacity_receipt  # noqa: E402


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    commit = completed.stdout.strip().lower()
    if len(commit) != 40:
        raise RuntimeError("estimator pilot could not resolve a full runtime commit")
    return commit


def _self_hash_matches(
    payload: Mapping[str, Any],
    *,
    field: str,
) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _read_parent_p0_suite(
    path: Path,
    *,
    expected_commit: str,
) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError("estimator pilot requires its sealed P0 suite")
    payload = _read_json(path)
    if (
        payload.get("schema_version") != PILOT_P0_SUITE_SCHEMA
        or payload.get("status") != "PASS_MECHANICAL_ONLY"
        or payload.get("runtime_commit") != expected_commit
        or payload.get("study_id") != PILOT_STUDY_ID
        or tuple(payload.get("arms", [])) != PILOT_ARM_ORDER
        or not _self_hash_matches(payload, field="suite_sha256")
        or payload.get("official_test_opened") is not False
        or payload.get("training_completed") is not False
        or payload.get("paper_claim_allowed") is not False
    ):
        raise RuntimeError("estimator pilot P0 suite is invalid")
    return payload


def _finite_positive(
    payload: Mapping[str, Any],
    key: str,
) -> float:
    value = payload.get(key)
    if (
        not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"estimator pilot profile lacks finite positive {key}")
    return float(value)


def _pilot_profile(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    scope = payload.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("development_only") is not True
        or scope.get("evaluator_excluded") is not True
        or scope.get("paper_grade_end_to_end_claim_allowed") is not False
    ):
        raise ValueError("estimator pilot profile is not development-only")
    return {
        "sample_count": int(payload.get("sample_count", 0)),
        "steady_sample_count": int(payload.get("steady_sample_count", 0)),
        "loader_wait_p50_ms": _finite_positive(payload, "loader_wait_p50_ms"),
        "loader_wait_p95_ms": _finite_positive(payload, "loader_wait_p95_ms"),
        "model_and_postprocess_p50_ms": _finite_positive(
            payload,
            "model_and_postprocess_p50_ms",
        ),
        "model_and_postprocess_p95_ms": _finite_positive(
            payload,
            "model_and_postprocess_p95_ms",
        ),
        "window_wall_p50_ms": _finite_positive(payload, "window_wall_p50_ms"),
        "window_wall_p95_ms": _finite_positive(payload, "window_wall_p95_ms"),
        "peak_allocated_mb": _finite_positive(payload, "peak_allocated_mb"),
        "paper_grade_end_to_end_claim_allowed": False,
        "profile_file_sha256": sha256_file(path),
        "scope": dict(scope),
    }


def _nested_value(record: Mapping[str, Any], path: Sequence[str]) -> Any:
    value: Any = record
    for key in path:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _mean(
    records: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> float | None:
    values = [
        float(value)
        for record in records
        if isinstance((value := _nested_value(record, path)), (int, float))
        and math.isfinite(float(value))
    ]
    return sum(values) / len(values) if values else None


def _population_standard_deviation(
    records: Sequence[Mapping[str, Any]],
    path: Sequence[str],
) -> float | None:
    values = [
        float(value)
        for record in records
        if isinstance((value := _nested_value(record, path)), (int, float))
        and math.isfinite(float(value))
    ]
    return statistics.pstdev(values) if values else None


def summarize_pilot_telemetry(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    records = payload.get("records")
    dataset_count = int(payload.get("dataset_count", -1))
    record_count = int(payload.get("record_count", -2))
    if (
        payload.get("schema_version") != "georoute_diagnostic_telemetry_v1"
        or payload.get("development_only") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("gt_for_route_used") is not False
        or payload.get("teacher_for_route_used") is not False
        or payload.get("oracle_used") is not False
        or payload.get("raw_prediction_cache_used") is not False
        or dataset_count <= 0
        or record_count != dataset_count
        or not isinstance(records, list)
        or len(records) != dataset_count
    ):
        raise ValueError("estimator pilot telemetry population is invalid")
    routes = []
    population_rows = []
    population_descriptors = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("route"),
            Mapping,
        ):
            raise ValueError("estimator pilot telemetry contains an invalid row")
        dataset_index = record.get("dataset_index")
        video_id = record.get("video_id")
        center_count = record.get("window_center_count")
        center_first = record.get("window_center_first")
        center_last = record.get("window_center_last")
        descriptor_sha256 = record.get("window_descriptor_sha256")
        if (
            not isinstance(dataset_index, int)
            or isinstance(dataset_index, bool)
            or dataset_index < 0
            or not isinstance(video_id, str)
            or not video_id
            or not isinstance(center_count, int)
            or isinstance(center_count, bool)
            or center_count <= 0
            or not isinstance(center_first, (int, float))
            or not math.isfinite(float(center_first))
            or not isinstance(center_last, (int, float))
            or not math.isfinite(float(center_last))
            or not _is_sha256(descriptor_sha256)
        ):
            raise ValueError("estimator pilot telemetry has an invalid window descriptor")
        descriptor = {
            "dataset_index": dataset_index,
            "video_id": video_id,
            "window_center_count": center_count,
            "window_center_first": center_first,
            "window_center_last": center_last,
        }
        descriptor_bytes = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        if hashlib.sha256(descriptor_bytes).hexdigest() != descriptor_sha256:
            raise ValueError("estimator pilot window-descriptor hash changed")
        routes.append(record["route"])
        population_rows.append(
            {
                **descriptor,
                "window_descriptor_sha256": descriptor_sha256,
            }
        )
        population_descriptors.append(
            {
                "dataset_index": dataset_index,
                "video_id": video_id,
                "window_descriptor_sha256": descriptor_sha256,
            }
        )
    if len({row["dataset_index"] for row in population_descriptors}) != len(
        population_descriptors
    ):
        raise ValueError("estimator pilot telemetry repeats dataset indices")
    role_counts = {
        json.dumps(route.get("role_counts", {}), sort_keys=True)
        for route in routes
    }
    if len(role_counts) != 1:
        raise ValueError("estimator pilot role-count contract changed by window")
    selected_hashes = {
        str(route.get("selected_index_sha256", ""))
        for route in routes
    }
    if any(not _is_sha256(value) for value in selected_hashes):
        raise ValueError("estimator pilot telemetry lacks selected-index hashes")
    observed_population_sha256 = payload.get("population_sha256")
    population_bytes = json.dumps(
        population_rows,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    computed_population_sha256 = hashlib.sha256(population_bytes).hexdigest()
    if observed_population_sha256 != computed_population_sha256:
        raise ValueError("estimator pilot telemetry population hash changed")
    summary: dict[str, Any] = {
        "dataset_count": dataset_count,
        "record_count": record_count,
        "population_sha256": computed_population_sha256,
        "population_descriptor_sha256": canonical_sha256(
            {"records": population_descriptors}
        ),
        "unique_selected_route_hash_count": len(selected_hashes),
        "role_counts": json.loads(next(iter(role_counts))),
        "adjacent_jaccard_mean": _mean(
            routes,
            ("adjacent", "jaccard_mean"),
        ),
        "adjacent_jaccard_population_sd": _population_standard_deviation(
            routes,
            ("adjacent", "jaccard_mean"),
        ),
        "lineage_retention_mean": _mean(
            routes,
            ("adjacent", "lineage_retention_mean"),
        ),
        "selected_x_span_mean": _mean(
            routes,
            ("coordinates", "x_span_mean"),
        ),
        "selected_y_span_mean": _mean(
            routes,
            ("coordinates", "y_span_mean"),
        ),
        "geometry_area_mean": _mean(routes, ("geometry", "area_mean")),
        "geometry_center_step_l2_mean": _mean(
            routes,
            ("geometry", "center_step_l2_mean"),
        ),
        "geometry_extent_step_l2_mean": _mean(
            routes,
            ("geometry", "extent_step_l2_mean"),
        ),
        "residual_selected_mean": _mean(
            routes,
            ("scores", "residual", "selected_mean"),
        ),
        "residual_unselected_mean": _mean(
            routes,
            ("scores", "residual", "unselected_mean"),
        ),
        "roi_selected_mean": _mean(
            routes,
            ("scores", "roi", "selected_mean"),
        ),
        "roi_unselected_mean": _mean(
            routes,
            ("scores", "roi", "unselected_mean"),
        ),
        "surrogate_selected_mean": _mean(
            routes,
            ("surrogate", "selected_mean"),
        ),
        "surrogate_unselected_mean": _mean(
            routes,
            ("surrogate", "unselected_mean"),
        ),
        "surrogate_hard_soft_l1_mean": _mean(
            routes,
            ("surrogate", "hard_soft_l1_mean"),
        ),
        "telemetry_file_sha256": sha256_file(path),
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _expected_representation(spec: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "absolute_position_enabled": bool(spec["absolute_position_enabled"]),
        "geometry_side_channel": bool(spec["geometry_side_channel"]),
        "learned_geometry_enabled": bool(spec["learned_geometry_enabled"]),
        "learned_residual_enabled": bool(spec["learned_residual_enabled"]),
        **{key: bool(spec[key]) for key in REPRESENTATION_KEYS},
    }


def validate_pilot_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
) -> None:
    unsigned = dict(result)
    observed_hash = unsigned.pop("stage_result_sha256", None)
    if observed_hash != canonical_sha256(unsigned):
        raise ValueError("estimator pilot stage-result self-hash mismatch")
    arm = str(result.get("arm", ""))
    spec = pilot_arm_spec(arm)
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("estimator pilot result is bound to another arm")
    if (
        result.get("schema_version") != PILOT_STAGE_RESULT_SCHEMA
        or result.get("status") != "PASS_EXPLORATORY_DEVELOPMENT_ONLY"
        or result.get("study_id") != PILOT_STUDY_ID
        or int(result.get("seed", -1)) != PILOT_SEED
        or int(result.get("epochs", -1)) != PILOT_EPOCHS
        or int(result.get("token_budget", -1)) != PILOT_K
        or result.get("arm_spec") != spec
        or result.get("arm_spec_sha256") != canonical_sha256(spec)
        or result.get("single_seed_exploratory") is not True
        or result.get("old_selector_reused") is not False
        or result.get("selector_emitted") is not False
        or result.get("p2_p3_opened") is not False
        or result.get("official_test_opened") is not False
        or result.get("gt_for_route_used") is not False
        or result.get("teacher_for_route_used") is not False
        or result.get("raw_prediction_cache_used") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("estimator pilot stage-result contract is invalid")
    if expected_commit is not None and result.get("runtime_commit") != expected_commit:
        raise ValueError("estimator pilot stage-result commit mismatch")
    metrics = result.get("metrics")
    required_metrics = {
        "average_mAP",
        "mAP@0.3",
        "mAP@0.4",
        "mAP@0.5",
        "mAP@0.6",
        "mAP@0.7",
        "high_iou_composite",
    }
    if (
        not isinstance(metrics, Mapping)
        or set(metrics) != required_metrics
        or any(
            not isinstance(metrics[key], (int, float))
            or not math.isfinite(float(metrics[key]))
            for key in metrics
        )
    ):
        raise ValueError("estimator pilot metrics are incomplete or non-finite")
    profile = result.get("profile")
    required_profile_keys = (
        "loader_wait_p50_ms",
        "loader_wait_p95_ms",
        "model_and_postprocess_p50_ms",
        "model_and_postprocess_p95_ms",
        "window_wall_p50_ms",
        "window_wall_p95_ms",
        "peak_allocated_mb",
    )
    if (
        not isinstance(profile, Mapping)
        or profile.get("paper_grade_end_to_end_claim_allowed") is not False
        or int(profile.get("sample_count", 0)) <= 0
        or int(profile.get("steady_sample_count", 0)) <= 0
        or int(profile.get("steady_sample_count", 0))
        > int(profile.get("sample_count", 0))
        or any(
            not isinstance(profile.get(key), (int, float))
            or not math.isfinite(float(profile[key]))
            or float(profile[key]) <= 0.0
            for key in required_profile_keys
        )
        or not _is_sha256(profile.get("profile_file_sha256"))
    ):
        raise ValueError("estimator pilot cost profile is invalid")
    audit = result.get("routing_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("estimator pilot result lacks a routing audit")
    if (
        audit.get("route_mode") != spec["route_mode"]
        or audit.get("policy_estimator") != spec["policy_estimator"]
        or float(audit.get("policy_temperature", -1.0))
        != float(spec["policy_temperature"])
        or float(audit.get("score_function_weight", -1.0))
        != float(spec["score_function_weight"])
        or float(audit.get("score_function_baseline_momentum", -1.0))
        != float(spec["score_function_baseline_momentum"])
        or float(audit.get("geometry_smoothness_weight", -1.0))
        != float(spec["geometry_smoothness_weight"])
        or float(audit.get("area_prior_weight", -1.0))
        != float(spec["area_prior_weight"])
        or audit.get("pooling_mode") != spec["pooling_mode"]
        or audit.get("adapter_mode") != spec["adapter_mode"]
        or int(audit.get("target_k", -1)) != PILOT_K
        or int(audit.get("selected_unique_count_min", -1)) != PILOT_K
        or int(audit.get("selected_unique_count_max", -1)) != PILOT_K
        or int(audit.get("selected_duplicate_count", -1)) != 0
        or int(audit.get("heavy_backbone_forward_count", -1)) != 1
        or audit.get("uses_grid_sample") is not False
        or audit.get("uses_resized_local_crop") is not False
    ):
        raise ValueError("estimator pilot routing audit violates exact-K execution")
    if {
        key: audit.get(key)
        for key in _expected_representation(spec)
    } != _expected_representation(spec):
        raise ValueError("estimator pilot representation audit differs from its arm")
    checkpoint = result.get("checkpoint_receipt")
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("policy") != "final_only_atomic"
        or not isinstance(checkpoint.get("sha256"), str)
        or len(str(checkpoint.get("sha256"))) != 64
        or int(checkpoint.get("size_bytes", 0)) <= 0
    ):
        raise ValueError("estimator pilot final-checkpoint receipt is invalid")
    telemetry = result.get("telemetry_summary")
    telemetry_unsigned = dict(telemetry) if isinstance(telemetry, Mapping) else {}
    observed_telemetry_summary_hash = telemetry_unsigned.pop(
        "summary_sha256",
        None,
    )
    if (
        not isinstance(telemetry, Mapping)
        or int(telemetry.get("dataset_count", -1)) <= 0
        or int(telemetry.get("record_count", -2))
        != int(telemetry.get("dataset_count", -1))
        or not _is_sha256(telemetry.get("population_sha256"))
        or not _is_sha256(telemetry.get("population_descriptor_sha256"))
        or not _is_sha256(telemetry.get("telemetry_file_sha256"))
        or observed_telemetry_summary_hash != canonical_sha256(telemetry_unsigned)
    ):
        raise ValueError("estimator pilot telemetry summary is invalid")
    parent = result.get("parent_p0_suite")
    if (
        not isinstance(parent, Mapping)
        or not isinstance(parent.get("suite_sha256"), str)
        or len(str(parent.get("suite_sha256"))) != 64
        or not isinstance(parent.get("file_sha256"), str)
        or len(str(parent.get("file_sha256"))) != 64
    ):
        raise ValueError("estimator pilot result lacks its P0 parent")
    inputs = result.get("input_receipts")
    if (
        not isinstance(inputs, Mapping)
        or any(
            not _is_sha256(inputs.get(key))
            for key in (
                "source_config_sha256",
                "manifest_file_sha256",
                "development_annotation_sha256",
                "class_map_sha256",
                "pretrained_checkpoint_sha256",
            )
        )
        or not isinstance(inputs.get("development_video_root"), str)
        or not str(inputs.get("development_video_root"))
    ):
        raise ValueError("estimator pilot result lacks immutable input receipts")
    binding = result.get("binding")
    if not isinstance(binding, Mapping):
        raise ValueError("estimator pilot result lacks its full immutable binding")
    binding_unsigned = dict(binding)
    observed_binding_hash = binding_unsigned.pop("binding_sha256", None)
    annotation = binding.get("development_annotation")
    if (
        not _is_sha256(observed_binding_hash)
        or canonical_sha256(binding_unsigned) != observed_binding_hash
        or result.get("binding_sha256") != observed_binding_hash
        or binding.get("schema_version") != PILOT_CONTRACT_SCHEMA
        or binding.get("study_id") != PILOT_STUDY_ID
        or binding.get("arm") != arm
        or binding.get("arm_spec") != spec
        or binding.get("arm_spec_sha256") != canonical_sha256(spec)
        or int(binding.get("seed", -1)) != PILOT_SEED
        or int(binding.get("epochs", -1)) != PILOT_EPOCHS
        or binding.get("source_config_sha256")
        != inputs["source_config_sha256"]
        or binding.get("manifest_file_sha256")
        != inputs["manifest_file_sha256"]
        or not isinstance(annotation, Mapping)
        or annotation.get("sha256")
        != inputs["development_annotation_sha256"]
        or binding.get("class_map_sha256") != inputs["class_map_sha256"]
        or binding.get("pretrained_checkpoint_sha256")
        != inputs["pretrained_checkpoint_sha256"]
        or binding.get("development_video_root")
        != inputs["development_video_root"]
        or binding.get("single_seed_exploratory") is not True
        or binding.get("old_selector_reused") is not False
        or binding.get("selector_emitted") is not False
        or binding.get("p2_p3_opened") is not False
        or binding.get("official_test_opened") is not False
        or binding.get("paper_claim_allowed") is not False
    ):
        raise ValueError("estimator pilot immutable binding is invalid")
    validated_rendezvous = _validate_rendezvous_receipt(
        result.get("rendezvous", {}),
        stage="estimator_pilot",
        variant=arm,
        seed=PILOT_SEED,
    )
    if result.get("rendezvous") != validated_rendezvous:
        raise ValueError("estimator pilot rendezvous receipt is not canonical")


def build_pilot_stage_result(
    *,
    arm: str,
    binding: Mapping[str, Any],
    config_path: Path,
    checkpoint_path: Path,
    storage_receipt_path: Path,
    prediction_path: Path,
    profile_path: Path,
    telemetry_path: Path,
    test_log_path: Path,
    runtime_commit: str,
    rendezvous: Mapping[str, Any],
    parent_p0_suite_path: Path,
    parent_p0_suite: Mapping[str, Any],
) -> dict[str, Any]:
    spec = pilot_arm_spec(arm)
    metrics = parse_official_style_map(
        test_log_path.read_text(encoding="utf-8", errors="replace")
    )
    metrics["high_iou_composite"] = 0.5 * (
        float(metrics["mAP@0.6"]) + float(metrics["mAP@0.7"])
    )
    raw_profile = _read_json(profile_path)
    audit = raw_profile.get("last_georoute_audit")
    if not isinstance(audit, Mapping):
        raise ValueError("estimator pilot profile lacks the routing audit")
    result: dict[str, Any] = {
        "schema_version": PILOT_STAGE_RESULT_SCHEMA,
        "status": "PASS_EXPLORATORY_DEVELOPMENT_ONLY",
        "study_id": PILOT_STUDY_ID,
        "experiment_schema_version": PILOT_CONTRACT_SCHEMA,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": PILOT_SEED,
        "epochs": PILOT_EPOCHS,
        "token_budget": PILOT_K,
        "metrics": metrics,
        "profile": _pilot_profile(profile_path),
        "profile_path": str(profile_path.resolve()),
        "telemetry_summary": summarize_pilot_telemetry(telemetry_path),
        "routing_audit": dict(audit),
        "binding": dict(binding),
        "binding_sha256": str(binding["binding_sha256"]),
        "input_receipts": {
            "source_config_sha256": str(binding["source_config_sha256"]),
            "manifest_file_sha256": str(binding["manifest_file_sha256"]),
            "development_annotation_sha256": str(
                binding["development_annotation"]["sha256"]
            ),
            "class_map_sha256": str(binding["class_map_sha256"]),
            "pretrained_checkpoint_sha256": str(
                binding["pretrained_checkpoint_sha256"]
            ),
            "development_video_root": str(binding["development_video_root"]),
        },
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "checkpoint_receipt": {
            "path": str(checkpoint_path.resolve()),
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": int(checkpoint_path.stat().st_size),
            "policy": "final_only_atomic",
        },
        "storage_receipt": _read_json(storage_receipt_path),
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "telemetry_path": str(telemetry_path.resolve()),
        "test_log_path": str(test_log_path.resolve()),
        "test_log_sha256": sha256_file(test_log_path),
        "runtime_commit": runtime_commit,
        "rendezvous": _validate_rendezvous_receipt(
            rendezvous,
            stage="estimator_pilot",
            variant=arm,
            seed=PILOT_SEED,
        ),
        "parent_p0_suite": {
            "path": str(parent_p0_suite_path.resolve()),
            "file_sha256": sha256_file(parent_p0_suite_path),
            "suite_sha256": str(parent_p0_suite["suite_sha256"]),
        },
        "single_seed_exploratory": True,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    validate_pilot_stage_result(
        result,
        expected_arm=arm,
        expected_commit=runtime_commit,
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace, *, cell_root: Path) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    if _current_commit() != expected_commit:
        raise RuntimeError("estimator pilot source differs from the bound commit")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("estimator pilot requires a clean source snapshot")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not slurm_job_id:
        raise RuntimeError("estimator pilot must run inside a Slurm leaf")
    if not visible or "," in visible:
        raise RuntimeError("estimator pilot requires one Slurm-visible GPU")

    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary) or run_root == boundary:
        raise ValueError("estimator pilot run root leaves the write boundary")
    spec = pilot_arm_spec(args.arm)
    if spec["tokens_per_tubelet"] != PILOT_K:
        raise RuntimeError("estimator pilot arm no longer uses frozen K")
    parent_p0_suite_path = run_root / "control" / "pilot_p0_suite.json"
    parent_p0_suite = _read_parent_p0_suite(
        parent_p0_suite_path,
        expected_commit=expected_commit,
    )
    storage_profile_path = run_root / "control" / "georoute_storage_profile.json"
    storage_profile = _read_json(storage_profile_path)
    storage_receipt = storage_capacity_receipt(
        run_root,
        cell_count=1,
        storage_profile=storage_profile,
        expected_commit=expected_commit,
    )

    cell_root.mkdir(parents=True, exist_ok=False)
    storage_receipt_path = cell_root / "storage_preflight.json"
    _atomic_write_json(storage_receipt_path, storage_receipt)
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{PILOT_STUDY_ID}_{args.arm}_seed{PILOT_SEED}.py"
    )
    if bound_config.exists():
        raise FileExistsError("estimator pilot bound config already exists")
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    cfg = bind_pilot_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=PILOT_SEED,
        work_dir=cell_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
    )
    cfg.dump(str(bound_config))

    inherited = dict(os.environ)
    inherited["PYTHONNOUSERSITE"] = "1"
    inherited["PYTHONDONTWRITEBYTECODE"] = "1"
    train_log = cell_root / "train.out"
    test_log = cell_root / "test.out"
    train_prefix, train_rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=slurm_job_id,
        stage="estimator_pilot",
        variant=args.arm,
        seed=PILOT_SEED,
    )
    _run_logged(
        [
            *train_prefix,
            "tools/train.py",
            str(bound_config),
            "--seed",
            str(PILOT_SEED),
            "--id",
            "0",
        ],
        log_path=train_log,
        env=inherited,
    )
    effective_work = cell_root / "gpu1_id0"
    checkpoint = (
        effective_work / "checkpoint" / f"epoch_{PILOT_EPOCHS - 1}.pth"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"estimator pilot final EMA checkpoint is missing: {checkpoint}"
        )
    payloads = sorted(checkpoint.parent.glob("*.pth"))
    temporaries = sorted(checkpoint.parent.glob("*.tmp*"))
    if payloads != [checkpoint] or temporaries:
        raise RuntimeError(
            "estimator pilot requires exactly one complete final checkpoint: "
            f"payloads={payloads}, temporaries={temporaries}"
        )

    test_prefix, test_rendezvous = build_torchrun_prefix(
        phase="test",
        slurm_job_id=slurm_job_id,
        stage="estimator_pilot",
        variant=args.arm,
        seed=PILOT_SEED,
    )
    _run_logged(
        [
            *test_prefix,
            "tools/test.py",
            str(bound_config),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(PILOT_SEED),
            "--id",
            "0",
        ],
        log_path=test_log,
        env=inherited,
    )
    prediction = effective_work / "result_detection.json"
    profile = effective_work / "georoute_development_profile.json"
    telemetry = effective_work / "georoute_diagnostic_telemetry.json"
    for path in (prediction, profile, telemetry):
        if not path.is_file():
            raise FileNotFoundError(path)
    return build_pilot_stage_result(
        arm=args.arm,
        binding=cfg.georoute_estimator_pilot_binding,
        config_path=bound_config,
        checkpoint_path=checkpoint,
        storage_receipt_path=storage_receipt_path,
        prediction_path=prediction,
        profile_path=profile,
        telemetry_path=telemetry,
        test_log_path=test_log,
        runtime_commit=expected_commit,
        rendezvous={
            "train": train_rendezvous,
            "test": test_rendezvous,
        },
        parent_p0_suite_path=parent_p0_suite_path,
        parent_p0_suite=parent_p0_suite,
    )


def main() -> int:
    args = _parse_args()
    pilot_arm_spec(args.arm)
    run_root = args.run_root.resolve()
    cell_root = run_root / pilot_cell_relative_path(
        arm=args.arm,
        seed=PILOT_SEED,
    )
    if cell_root.exists():
        raise FileExistsError(
            "estimator pilot cell exists; refusing overwrite or resume"
        )
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
        if cell_root.is_dir():
            trace = traceback.format_exc()
            try:
                observed_runtime_commit = _current_commit()
            except Exception:
                observed_runtime_commit = None
            failure: dict[str, Any] = {
                "schema_version": PILOT_STAGE_RESULT_SCHEMA,
                "status": "FAIL_EXPLORATORY_PILOT_ARM",
                "study_id": PILOT_STUDY_ID,
                "arm": args.arm,
                "seed": PILOT_SEED,
                "expected_runtime_commit": str(args.expected_commit).lower(),
                "observed_runtime_commit": observed_runtime_commit,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    trace.encode("utf-8", errors="replace")
                ).hexdigest(),
                "official_test_opened": False,
                "p2_p3_opened": False,
                "paper_claim_allowed": False,
            }
            failure["failure_sha256"] = canonical_sha256(failure)
            _atomic_write_json(cell_root / "pilot_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
