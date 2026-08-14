#!/usr/bin/env python3
"""Run one frozen, two-rank GeoRoute development cell.

The runner evaluates only the Fit/Gate split carved from THUMOS ``training``.
Its metrics and component profiler are admissible for method selection only;
they are never an official-validation/test or paper-table result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_estimator_pilot_stage_runner import (  # noqa: E402
    _mean,
    _pilot_profile,
    _population_standard_deviation,
)
from tools.bata.georoute_experiment_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    FORMAL_DEVELOPMENT_ARM_ORDER,
    FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA,
    FORMAL_DEVELOPMENT_RESULT_SCHEMA,
    FORMAL_DEVELOPMENT_SEEDS,
    FORMAL_EPOCHS,
    FORMAL_NATIVE_TOKEN_COUNT,
    FORMAL_SOURCE_GRID_HW,
    FORMAL_TOKEN_BUDGET,
    FORMAL_WORLD_SIZE,
    P1_DEVELOPMENT_SEED,
    P1_MATCHED_RUNNER_ARM_ORDER,
    P1_WINDOW_TOKEN_BUDGET,
    bind_formal_development_config,
    development_arm_spec,
    development_seed_allowed,
    formal_cell_relative_path,
    read_json,
    validate_formal_checkpoint_sidecar,
    validate_formal_development_binding,
    validate_protocol_manifest,
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
    return path != root


def _current_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().lower()


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _q_expect(payload: Any, expected: Mapping[str, Any], message: str) -> None:
    if not isinstance(payload, Mapping) or any(
        payload.get(key) != value for key, value in expected.items()
    ):
        raise ValueError(message)


def _q_k_values(payload: Any) -> list[int]:
    values = payload.get("values") if isinstance(payload, Mapping) else payload
    if (
        not isinstance(values, list)
        or len(values) != P1_WINDOW_TOKEN_BUDGET // FORMAL_TOKEN_BUDGET
        or any(
            type(value) is not int
            or not 0 <= value <= FORMAL_NATIVE_TOKEN_COUNT
            for value in values
        )
        or sum(values) != P1_WINDOW_TOKEN_BUDGET
    ):
        raise ValueError("Q dynamic K_t does not sum to the global exact B")
    if isinstance(payload, Mapping):
        histogram = {str(value): values.count(value) for value in set(values)}
        _q_expect(
            payload,
            {
                "min": min(values),
                "max": max(values),
                "zero_count": values.count(0),
                "histogram": histogram,
            },
            "Q K_t summary is inconsistent",
        )
    return values


def _q_attention_pairs(clip_counts: Any, observed: Any) -> int:
    if (
        not isinstance(clip_counts, list)
        or not clip_counts
        or any(type(value) is not int or value < 0 for value in clip_counts)
        or sum(clip_counts) != P1_WINDOW_TOKEN_BUDGET
        or type(observed) is not int
        or observed != sum(value**2 for value in clip_counts)
    ):
        raise ValueError("Q ragged clip/attention ledger is inconsistent")
    return observed


def validate_p1_q_routing_audit(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the production Q audit without imposing legacy per-tubelet K."""

    audit = dict(audit)
    _q_expect(
        audit,
        {
            "routing_schema": "georoute_dynamic_global_routing_v2",
            "route_mode": "dynamic_scnr",
            "policy_estimator": "straight_through",
            "target_k": None,
            "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
            "window_budget_is_global": True,
            "fixed_per_tubelet_k": False,
            "k_t_allows_zero": True,
            "zero_carrier_mode": "masked_zero",
            "heavy_valid_mask_matches_k_t": True,
            "dynamic_roi_modifier_enabled": False,
            "dynamic_residual_modifier_enabled": False,
            "requested_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "unique_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "padded_heavy_tokens_per_window": 0,
            "executed_patch_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "heavy_backbone_forward_count": 1,
            "diagnostic_telemetry_enabled": True,
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        },
        "Q routing audit violates the global dynamic-ragged contract",
    )
    _q_expect(
        audit.get("branch_calibration"),
        {
            "mode": "none",
            "scope": "disabled",
            "changes_q_base": False,
            "changes_delta_roi": False,
            "changes_context_zero_modifier": False,
            "changes_budget_or_role_quota": False,
        },
        "Q branch calibration changed route or budget",
    )
    packed = audit.get("packed")
    _q_expect(
        packed,
        {
            "schema_version": "videomae_native_ragged_v1",
            "execution_mode": "true_clip_ragged_no_padding",
            "adapter_execution": "coordinate_lineage_true_ragged",
            "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
            "requested_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "unique_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "padded_heavy_tokens_per_window": 0,
            "executed_patch_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
            "heavy_backbone_forward_count": 1,
            "dense_adapter_forward_count": 0,
        },
        "Q packed execution is not true ragged exact-B",
    )
    k_rows = audit.get("k_per_tubelet")
    role_rows = audit.get("role_counts_per_window")
    clip_rows = packed.get("clip_token_counts")
    pair_rows = packed.get("attention_pairs_per_window")
    if not all(
        isinstance(rows, list) and len(rows) == len(k_rows or [])
        for rows in (k_rows, role_rows, clip_rows, pair_rows)
    ) or not k_rows:
        raise ValueError("Q audit ledgers are not window-aligned")
    for k_values, roles, clips, pairs in zip(k_rows, role_rows, clip_rows, pair_rows):
        _q_k_values(k_values)
        if (
            not isinstance(roles, list)
            or len(roles) != 3
            or any(type(value) is not int or value < 0 for value in roles)
            or sum(roles) != P1_WINDOW_TOKEN_BUDGET
        ):
            raise ValueError("Q roles do not partition the exact B")
        _q_attention_pairs(clips, pairs)
    observed_roles = {
        name: sum(row[index] for row in role_rows)
        for index, name in enumerate(("context", "roi", "residual"))
    }
    if audit.get("role_counts") != observed_roles:
        raise ValueError("Q aggregate route roles are inconsistent")
    return audit


def _summarize_p1_q_routes(
    routes: Sequence[Mapping[str, Any]],
    *,
    dataset_count: int,
    record_count: int,
    padding_count: int,
    population_sha256: str,
    population_descriptor_sha256: str,
    telemetry_file_sha256: str,
) -> dict[str, Any]:
    all_k: list[int] = []
    role_totals = {name: 0 for name in ("context", "roi", "residual")}
    attention_pairs: list[int] = []
    selected_hashes: set[str] = set()
    for route in routes:
        _q_expect(
            route,
            {
                "schema_version": "georoute_dynamic_diagnostic_window_telemetry_v1",
                "measurement_scope": "accuracy_replay_only_excluded_from_timed_cost",
                "batch_size": 1,
                "tubelet_count": P1_WINDOW_TOKEN_BUDGET // FORMAL_TOKEN_BUDGET,
                "item_count": FORMAL_NATIVE_TOKEN_COUNT,
                "source_grid_hw": list(FORMAL_SOURCE_GRID_HW),
                "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
                "target_k": None,
                "role_assignment_changes_execution": False,
                "gt_for_route_used": False,
                "teacher_used": False,
                "oracle_used": False,
                "official_test_opened": False,
                "paper_claim_allowed": False,
            },
            "Q diagnostic telemetry schema/isolation changed",
        )
        _q_expect(
            route.get("branch_calibration"),
            {
                "mode": "none",
                "scope": "disabled",
                "changes_q_base": False,
                "changes_delta_roi": False,
                "changes_context_zero_modifier": False,
                "changes_budget_or_role_quota": False,
            },
            "Q branch calibration changed route or budget",
        )
        k_values = _q_k_values(route.get("k_t"))
        roles = route.get("roles")
        role_rows = roles.get("per_tubelet_counts") if isinstance(roles, Mapping) else None
        if (
            not isinstance(roles, Mapping)
            or roles.get("order") != list(role_totals)
            or not isinstance(role_rows, list)
            or len(role_rows) != len(k_values)
            or any(
                not isinstance(row, list)
                or len(row) != 3
                or any(type(value) is not int or value < 0 for value in row)
                or sum(row) != k_value
                for row, k_value in zip(role_rows, k_values)
            )
        ):
            raise ValueError("Q roles do not partition dynamic K_t")
        observed_roles = {
            name: sum(row[index] for row in role_rows)
            for index, name in enumerate(role_totals)
        }
        if roles.get("aggregate_counts") != observed_roles:
            raise ValueError("Q aggregate role telemetry is inconsistent")
        for name, value in observed_roles.items():
            role_totals[name] += value

        ragged = route.get("ragged_execution")
        _q_expect(
            ragged,
            {
                "requested_physical_tokens": P1_WINDOW_TOKEN_BUDGET,
                "unique_physical_tokens": P1_WINDOW_TOKEN_BUDGET,
                "padded_heavy_tokens": 0,
                "executed_patch_tokens": P1_WINDOW_TOKEN_BUDGET,
            },
            "Q telemetry lost unique no-padding execution",
        )
        if (
            int(ragged.get("ragged_attention_bucket_call_count", 0)) <= 0
            or int(ragged.get("ragged_mlp_bucket_call_count", 0)) <= 0
        ):
            raise ValueError("Q telemetry did not attest true ragged execution")
        attention_pairs.append(
            _q_attention_pairs(
                ragged.get("clip_token_counts"), ragged.get("attention_pairs")
            )
        )
        route_hash = route.get("selected_physical_index_sha256")
        if not _is_sha256(route_hash):
            raise ValueError("Q telemetry lacks selected-route hashes")
        selected_hashes.add(route_hash)
        all_k.extend(k_values)

    histogram = {str(value): all_k.count(value) for value in set(all_k)}
    summary: dict[str, Any] = {
        "schema_version": "georoute_p1_q_telemetry_summary_v001",
        "dataset_count": dataset_count,
        "record_count": record_count,
        "sampler_padding_count": padding_count,
        "population_sha256": population_sha256,
        "population_descriptor_sha256": population_descriptor_sha256,
        "target_k": None,
        "window_token_budget": P1_WINDOW_TOKEN_BUDGET,
        "window_budget_is_global": True,
        "dynamic_k_t": True,
        "k_t": {
            "min": min(all_k),
            "max": max(all_k),
            "zero_count": all_k.count(0),
            "histogram": histogram,
        },
        "role_counts": role_totals,
        "unique_physical_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
        "padded_heavy_tokens_per_window": 0,
        "executed_patch_tokens_per_window": P1_WINDOW_TOKEN_BUDGET,
        "ragged_execution": "true_clip_ragged_no_padding",
        "ragged_attention_pairs_mean": sum(attention_pairs) / len(attention_pairs),
        "unique_selected_route_hash_count": len(selected_hashes),
        "telemetry_file_sha256": telemetry_file_sha256,
        "development_only": True,
        "paper_grade_cost_allowed": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def _validate_deployment(
    *,
    run_root: Path,
    expected_commit: str,
    arm: str,
    seed: int,
    slurm_job_id: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p1_cell = arm in P1_MATCHED_RUNNER_ARM_ORDER
    expected_arm_order = (
        P1_MATCHED_RUNNER_ARM_ORDER if p1_cell else FORMAL_DEVELOPMENT_ARM_ORDER
    )
    expected_seeds = (
        (P1_DEVELOPMENT_SEED,) if p1_cell else FORMAL_DEVELOPMENT_SEEDS
    )
    deployment_path = run_root / "control" / "deployment.json"
    deployment = read_json(deployment_path)
    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    arm_jobs = stage_jobs.get(arm) if isinstance(stage_jobs, Mapping) else None
    if (
        deployment.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != "SUBMITTED_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX"
        or deployment.get("runtime_commit") != expected_commit
        or Path(str(deployment.get("run_root", ""))).resolve() != run_root
        or tuple(deployment.get("arms", ())) != expected_arm_order
        or tuple(deployment.get("seeds", ())) != expected_seeds
        or not _self_hash_matches(deployment, field="deployment_sha256")
        or not isinstance(arm_jobs, Mapping)
        or str(arm_jobs.get(str(seed), "")) != slurm_job_id
        or deployment.get("official_test_opened") is not False
        or deployment.get("paper_claim_allowed") is not False
    ):
        raise ValueError("formal development deployment receipt is invalid")
    protocol_path = Path(
        str(deployment.get("protocol_manifest_path", ""))
    ).resolve()
    preflight_path = Path(
        str(deployment.get("preflight_finalization_path", ""))
    ).resolve()
    if (
        protocol_path
        != (run_root / "control" / "protocol_manifest.json").resolve()
        or not protocol_path.is_file()
        or sha256_file(protocol_path)
        != deployment.get("protocol_manifest_file_sha256")
        or preflight_path.is_symlink()
        or not preflight_path.is_file()
        or sha256_file(preflight_path)
        != deployment.get("preflight_finalization_file_sha256")
    ):
        raise ValueError("formal development parent artifact changed")
    protocol = validate_protocol_manifest(read_json(protocol_path))
    preflight = read_json(preflight_path)
    if (
        protocol.get("runtime_commit") != expected_commit
        or protocol.get("protocol_sha256")
        != deployment.get("protocol_sha256")
        or preflight.get("runtime_commit") != expected_commit
        or preflight.get("decision")
        != "FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED"
        or preflight.get("formal_development_matrix_authorized") is not True
        or not _self_hash_matches(preflight, field="finalization_sha256")
    ):
        raise ValueError("formal development parent did not authorize this cell")
    return deployment, protocol, preflight


def summarize_formal_telemetry(path: Path, *, arm: str) -> dict[str, Any]:
    arm_spec = development_arm_spec(arm)
    payload = _read_json(path)
    records = payload.get("records")
    dataset_count = int(payload.get("dataset_count", -1))
    record_count = int(payload.get("record_count", -1))
    padding_count = int(payload.get("sampler_padding_count", -1))
    if (
        payload.get("schema_version")
        != "georoute_formal_development_telemetry_v1"
        or payload.get("development_only") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("gt_for_route_used") is not False
        or payload.get("teacher_for_route_used") is not False
        or payload.get("oracle_used") is not False
        or payload.get("raw_prediction_cache_used") is not False
        or int(payload.get("world_size", -1)) != FORMAL_WORLD_SIZE
        or int(payload.get("local_batch_size", -1)) != 1
        or dataset_count <= 0
        or int(payload.get("unique_dataset_count", -1)) != dataset_count
        or padding_count not in {0, 1}
        or record_count != dataset_count + padding_count
        or not isinstance(records, list)
        or len(records) != record_count
    ):
        raise ValueError("formal development telemetry population is invalid")
    descriptors: list[dict[str, Any]] = []
    routes: list[Mapping[str, Any]] = []
    unique_indices = set()
    for record in records:
        route = record.get("route") if isinstance(record, Mapping) else None
        if not isinstance(route, Mapping):
            raise ValueError("formal telemetry row lacks a route")
        descriptor = {
            key: record.get(key)
            for key in (
                "dataset_index",
                "rank",
                "local_batch_index",
                "video_id",
                "window_center_count",
                "window_center_first",
                "window_center_last",
            )
        }
        if (
            not isinstance(descriptor["dataset_index"], int)
            or not 0 <= int(descriptor["dataset_index"]) < dataset_count
            or int(descriptor["rank"]) not in range(FORMAL_WORLD_SIZE)
            or not isinstance(descriptor["local_batch_index"], int)
            or int(descriptor["local_batch_index"]) < 0
            or not isinstance(descriptor["video_id"], str)
            or not descriptor["video_id"]
            or not isinstance(descriptor["window_center_count"], int)
            or int(descriptor["window_center_count"]) <= 0
            or not isinstance(descriptor["window_center_first"], (int, float))
            or not math.isfinite(float(descriptor["window_center_first"]))
            or not isinstance(descriptor["window_center_last"], (int, float))
            or not math.isfinite(float(descriptor["window_center_last"]))
        ):
            raise ValueError("formal telemetry descriptor is invalid")
        descriptor_bytes = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor_hash = record.get("window_descriptor_sha256")
        if (
            not _is_sha256(descriptor_hash)
            or hashlib.sha256(descriptor_bytes).hexdigest()
            != descriptor_hash
        ):
            raise ValueError("formal telemetry descriptor hash changed")
        descriptors.append(
            {**descriptor, "window_descriptor_sha256": descriptor_hash}
        )
        unique_indices.add(int(descriptor["dataset_index"]))
        routes.append(route)
    if unique_indices != set(range(dataset_count)):
        raise ValueError("formal telemetry omitted development Gate rows")
    population_sha256 = hashlib.sha256(
        json.dumps(
            descriptors,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    if payload.get("population_sha256") != population_sha256:
        raise ValueError("formal telemetry population hash changed")
    population_descriptor_sha256 = canonical_sha256({"records": descriptors})
    if arm == "Q":
        return _summarize_p1_q_routes(
            routes,
            dataset_count=dataset_count,
            record_count=record_count,
            padding_count=padding_count,
            population_sha256=population_sha256,
            population_descriptor_sha256=population_descriptor_sha256,
            telemetry_file_sha256=sha256_file(path),
        )

    expected_k = (
        FORMAL_NATIVE_TOKEN_COUNT
        if arm_spec["tokens_per_tubelet"] is None
        else int(arm_spec["tokens_per_tubelet"])
    )
    role_counts = {
        json.dumps(route.get("role_counts", {}), sort_keys=True)
        for route in routes
    }
    target_k = {int(route.get("target_k", -1)) for route in routes}
    if len(role_counts) != 1 or target_k != {expected_k}:
        raise ValueError("formal telemetry exact-K role contract changed")
    selected_hashes = {
        str(route.get("selected_index_sha256", "")) for route in routes
    }
    if any(not _is_sha256(value) for value in selected_hashes):
        raise ValueError("formal telemetry lacks selected-route hashes")
    summary: dict[str, Any] = {
        "dataset_count": dataset_count,
        "record_count": record_count,
        "sampler_padding_count": padding_count,
        "population_sha256": population_sha256,
        "population_descriptor_sha256": population_descriptor_sha256,
        "target_k": expected_k,
        "role_counts": json.loads(next(iter(role_counts))),
        "unique_selected_route_hash_count": len(selected_hashes),
        "adjacent_jaccard_mean": _mean(
            routes, ("adjacent", "jaccard_mean")
        ),
        "adjacent_jaccard_population_sd": _population_standard_deviation(
            routes, ("adjacent", "jaccard_mean")
        ),
        "lineage_retention_mean": _mean(
            routes, ("adjacent", "lineage_retention_mean")
        ),
        "selected_x_span_mean": _mean(
            routes, ("coordinates", "x_span_mean")
        ),
        "selected_y_span_mean": _mean(
            routes, ("coordinates", "y_span_mean")
        ),
        "residual_selected_mean": _mean(
            routes, ("scores", "residual", "selected_mean")
        ),
        "residual_unselected_mean": _mean(
            routes, ("scores", "residual", "unselected_mean")
        ),
        "telemetry_file_sha256": sha256_file(path),
        "development_only": True,
        "paper_grade_cost_allowed": False,
    }
    summary["summary_sha256"] = canonical_sha256(summary)
    return summary


def validate_formal_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_seed: int | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    result = dict(result)
    if not _self_hash_matches(result, field="stage_result_sha256"):
        raise ValueError("formal stage-result self-hash mismatch")
    arm = str(result.get("arm", ""))
    seed = int(result.get("seed", -1))
    binding = result.get("binding")
    metrics = result.get("metrics")
    profile = result.get("profile")
    telemetry = result.get("telemetry_summary")
    if (
        result.get("schema_version") != FORMAL_DEVELOPMENT_RESULT_SCHEMA
        or result.get("status")
        != "PASS_OFFICIAL_COMPARABLE_DEVELOPMENT_ONLY"
        or arm
        not in (*FORMAL_DEVELOPMENT_ARM_ORDER, *P1_MATCHED_RUNNER_ARM_ORDER)
        or not development_seed_allowed(arm=arm, seed=seed)
        or int(result.get("epochs", -1)) != FORMAL_EPOCHS
        or result.get("arm_spec") != development_arm_spec(arm)
        or not isinstance(binding, Mapping)
        or validate_formal_development_binding(binding, seed=seed)
        != dict(binding)
        or result.get("binding_sha256") != binding.get("binding_sha256")
        or not isinstance(metrics, Mapping)
        or set(metrics)
        != {
            "average_mAP",
            "mAP@0.3",
            "mAP@0.4",
            "mAP@0.5",
            "mAP@0.6",
            "mAP@0.7",
            "high_iou_composite",
        }
        or any(
            not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in metrics.values()
        )
        or not isinstance(profile, Mapping)
        or profile.get("paper_grade_end_to_end_claim_allowed") is not False
        or not isinstance(telemetry, Mapping)
        or telemetry.get("development_only") is not True
        or result.get("official_test_opened") is not False
        or result.get("paper_grade_result_record_emitted") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("formal development stage-result contract is invalid")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("formal stage result belongs to another arm")
    if expected_seed is not None and seed != int(expected_seed):
        raise ValueError("formal stage result belongs to another seed")
    if (
        expected_commit is not None
        and result.get("runtime_commit") != expected_commit
    ):
        raise ValueError("formal stage result belongs to another commit")
    if arm == "Q" and (
        telemetry.get("schema_version")
        != "georoute_p1_q_telemetry_summary_v001"
        or telemetry.get("target_k") is not None
        or int(telemetry.get("window_token_budget", -1))
        != P1_WINDOW_TOKEN_BUDGET
        or telemetry.get("window_budget_is_global") is not True
        or telemetry.get("dynamic_k_t") is not True
        or int(telemetry.get("unique_physical_tokens_per_window", -1))
        != P1_WINDOW_TOKEN_BUDGET
        or int(telemetry.get("padded_heavy_tokens_per_window", -1)) != 0
        or telemetry.get("ragged_execution")
        != "true_clip_ragged_no_padding"
    ):
        raise ValueError("Q stage result lost its dynamic global-ragged summary")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=(*FORMAL_DEVELOPMENT_ARM_ORDER, *P1_MATCHED_RUNNER_ARM_ORDER),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--development-annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--development-video-root", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(
    args: argparse.Namespace,
    *,
    cell_root: Path,
) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    if _current_commit() != expected_commit:
        raise RuntimeError("formal development source commit changed")
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("formal development requires a clean source snapshot")
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    visible = [
        value
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value
    ]
    if not slurm_job_id.isdigit() or len(visible) != FORMAL_WORLD_SIZE:
        raise RuntimeError(
            "formal development requires one two-GPU Slurm allocation"
        )
    run_root = args.run_root.resolve()
    boundary = Path("/data/run01/sczc063/yuzibo").resolve()
    if not _inside(run_root, boundary):
        raise ValueError("formal development run root leaves write boundary")
    deployment, _protocol, _preflight = _validate_deployment(
        run_root=run_root,
        expected_commit=expected_commit,
        arm=args.arm,
        seed=args.seed,
        slurm_job_id=slurm_job_id,
    )
    storage_receipt = storage_capacity_receipt(run_root, cell_count=1)
    cell_root.mkdir(parents=True, exist_ok=False)
    storage_receipt_path = cell_root / "storage_preflight.json"
    _atomic_write_json(storage_receipt_path, storage_receipt)
    bound_config = (
        run_root
        / "control"
        / "bound_configs"
        / f"{args.arm}_seed{args.seed}.py"
    )
    if bound_config.exists():
        raise FileExistsError("formal bound config already exists")
    bound_config.parent.mkdir(parents=True, exist_ok=True)
    cfg = bind_formal_development_config(
        source_config_path=args.source_config,
        arm=args.arm,
        seed=args.seed,
        work_dir=cell_root,
        manifest_path=args.manifest,
        development_annotation_path=args.development_annotation,
        class_map_path=args.class_map,
        development_video_root=args.development_video_root,
        pretrained_checkpoint_path=args.pretrained,
        runtime_commit=expected_commit,
        preflight_finalization_path=deployment[
            "preflight_finalization_path"
        ],
        expected_preflight_file_sha256=deployment[
            "preflight_finalization_file_sha256"
        ],
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
        stage="officialdev",
        variant=args.arm,
        seed=args.seed,
        nproc_per_node=FORMAL_WORLD_SIZE,
    )
    _run_logged(
        [
            *train_prefix,
            "tools/train.py",
            str(bound_config),
            "--seed",
            str(args.seed),
            "--id",
            "0",
        ],
        log_path=train_log,
        env=inherited,
    )
    checkpoint = (
        cell_root / "checkpoint" / f"epoch_{FORMAL_EPOCHS - 1}.pth"
    )
    sidecar_path = Path(str(checkpoint) + ".metadata.json")
    validate_formal_checkpoint_sidecar(
        checkpoint,
        binding=cfg.georoute_official_development_binding,
    )
    payloads = sorted(checkpoint.parent.glob("*.pth"))
    sidecars = sorted(checkpoint.parent.glob("*.metadata.json"))
    temporaries = sorted(checkpoint.parent.glob("*.tmp*"))
    if (
        payloads != [checkpoint]
        or sidecars != [sidecar_path]
        or temporaries
    ):
        raise RuntimeError(
            "formal final-only policy requires one checkpoint-sidecar pair"
        )
    test_prefix, test_rendezvous = build_torchrun_prefix(
        phase="test",
        slurm_job_id=slurm_job_id,
        stage="officialdev",
        variant=args.arm,
        seed=args.seed,
        nproc_per_node=FORMAL_WORLD_SIZE,
    )
    _run_logged(
        [
            *test_prefix,
            "tools/test.py",
            str(bound_config),
            "--checkpoint",
            str(checkpoint),
            "--seed",
            str(args.seed),
            "--id",
            "0",
        ],
        log_path=test_log,
        env=inherited,
    )
    evaluation_root = cell_root / "gpu2_id0"
    prediction = evaluation_root / "result_detection.json"
    profile_path = evaluation_root / "georoute_development_profile.json"
    telemetry_path = (
        evaluation_root / "georoute_diagnostic_telemetry.json"
    )
    for artifact in (prediction, profile_path, telemetry_path):
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
    metrics = parse_official_style_map(
        test_log.read_text(encoding="utf-8", errors="replace")
    )
    metrics["high_iou_composite"] = 0.5 * (
        float(metrics["mAP@0.6"]) + float(metrics["mAP@0.7"])
    )
    profile = _pilot_profile(profile_path)
    raw_profile = _read_json(profile_path)
    routing_audit = raw_profile.get("last_georoute_audit")
    spec = development_arm_spec(args.arm)
    if args.arm == "Q":
        if not isinstance(routing_audit, Mapping):
            raise ValueError("Q development route audit is missing")
        validate_p1_q_routing_audit(routing_audit)
    else:
        expected_k = (
            FORMAL_NATIVE_TOKEN_COUNT
            if spec["tokens_per_tubelet"] is None
            else int(spec["tokens_per_tubelet"])
        )
        if (
            not isinstance(routing_audit, Mapping)
            or routing_audit.get("route_mode") != spec["route_mode"]
            or routing_audit.get("policy_estimator")
            != spec["policy_estimator"]
            or int(routing_audit.get("target_k", -1)) != expected_k
            or int(routing_audit.get("selected_unique_count_min", -1))
            != expected_k
            or int(routing_audit.get("selected_unique_count_max", -1))
            != expected_k
            or int(routing_audit.get("selected_duplicate_count", -1)) != 0
            or tuple(routing_audit.get("source_grid_hw", ()))
            != FORMAL_SOURCE_GRID_HW
            or int(routing_audit.get("item_count", -1))
            != FORMAL_NATIVE_TOKEN_COUNT
            or int(routing_audit.get("heavy_backbone_forward_count", -1)) != 1
            or routing_audit.get("uses_grid_sample") is not False
            or routing_audit.get("uses_resized_local_crop") is not False
        ):
            raise ValueError("formal development route audit violates exact-K")
    sidecar = validate_formal_checkpoint_sidecar(
        checkpoint,
        binding=cfg.georoute_official_development_binding,
    )
    result: dict[str, Any] = {
        "schema_version": FORMAL_DEVELOPMENT_RESULT_SCHEMA,
        "status": "PASS_OFFICIAL_COMPARABLE_DEVELOPMENT_ONLY",
        "runtime_commit": expected_commit,
        "arm": args.arm,
        "arm_spec": spec,
        "seed": int(args.seed),
        "epochs": FORMAL_EPOCHS,
        "metrics": metrics,
        "profile": profile,
        "telemetry_summary": summarize_formal_telemetry(
            telemetry_path,
            arm=args.arm,
        ),
        "routing_audit": dict(routing_audit),
        "binding": dict(cfg.georoute_official_development_binding),
        "binding_sha256": cfg.georoute_official_development_binding[
            "binding_sha256"
        ],
        "config_path": str(bound_config.resolve()),
        "config_sha256": sha256_file(bound_config),
        "checkpoint_receipt": {
            "path": str(checkpoint.resolve()),
            "sha256": sha256_file(checkpoint),
            "size_bytes": int(checkpoint.stat().st_size),
            "sidecar_path": str(sidecar_path.resolve()),
            "sidecar_file_sha256": sha256_file(sidecar_path),
            "sidecar_sha256": sidecar["sidecar_sha256"],
            "policy": "final_epoch_ema_only_atomic",
        },
        "storage_receipt": read_json(storage_receipt_path),
        "prediction_path": str(prediction.resolve()),
        "prediction_sha256": sha256_file(prediction),
        "profile_path": str(profile_path.resolve()),
        "telemetry_path": str(telemetry_path.resolve()),
        "test_log_path": str(test_log.resolve()),
        "test_log_sha256": sha256_file(test_log),
        "rendezvous": _validate_rendezvous_receipt(
            {"train": train_rendezvous, "test": test_rendezvous},
            stage="officialdev",
            variant=args.arm,
            seed=args.seed,
            nproc_per_node=FORMAL_WORLD_SIZE,
        ),
        "development_gate_only": True,
        "cross_cell_performance_inference_allowed": False,
        "official_test_opened": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    result["stage_result_sha256"] = canonical_sha256(result)
    return validate_formal_stage_result(
        result,
        expected_arm=args.arm,
        expected_seed=args.seed,
        expected_commit=expected_commit,
    )


def main() -> int:
    args = _parse_args()
    development_arm_spec(args.arm)
    if not development_seed_allowed(arm=args.arm, seed=args.seed):
        raise ValueError("formal seed is outside the frozen set")
    run_root = args.run_root.resolve()
    cell_root = run_root / formal_cell_relative_path(
        arm=args.arm,
        seed=args.seed,
    )
    if cell_root.exists():
        raise FileExistsError(
            "formal development cell exists; refusing overwrite or resume"
        )
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
        if cell_root.is_dir():
            trace = traceback.format_exc()
            failure: dict[str, Any] = {
                "schema_version": FORMAL_DEVELOPMENT_RESULT_SCHEMA,
                "status": "FAIL_OFFICIAL_COMPARABLE_DEVELOPMENT_CELL",
                "arm": args.arm,
                "seed": int(args.seed),
                "expected_runtime_commit": str(args.expected_commit).lower(),
                "observed_runtime_commit": (
                    _current_commit() if (ROOT / ".git").exists() else None
                ),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    trace.encode("utf-8", errors="replace")
                ).hexdigest(),
                "performance_inference_allowed": False,
                "official_test_opened": False,
                "paper_claim_allowed": False,
            }
            failure["failure_sha256"] = canonical_sha256(failure)
            _atomic_write_json(cell_root / "stage_failure.json", failure)
        raise
    _atomic_write_json(cell_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
