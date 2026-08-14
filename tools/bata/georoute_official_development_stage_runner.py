#!/usr/bin/env python3
"""Run one frozen GeoRoute accuracy cell or ZoomToken P1 cost leaf.

The runner evaluates only the Fit/Gate split carved from THUMOS ``training``.
Its metrics and component profiler are admissible for method selection only;
they are never an official-validation/test or paper-table result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import subprocess
import sys
import time
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
    P1_FIRST_SCREEN_ARM_ORDER,
    P1_MATCHED_RUNNER_ARM_ORDER,
    P1_WINDOW_TOKEN_BUDGET,
    bind_formal_development_config,
    development_arm_spec,
    development_seed_allowed,
    formal_arm_spec,
    formal_cell_relative_path,
    read_json,
    validate_formal_checkpoint_sidecar,
    validate_formal_development_binding,
    validate_protocol_manifest,
)
from tools.bata.georoute_p1_runtime_attestor import (  # noqa: E402
    validate_runtime_attestation,
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
from tools.bata.zoomtoken_scnr_steady_cost_contract_v001 import (  # noqa: E402
    PHYSICAL_WINDOWS,
    POWER_INTERVAL_MS,
    P1_COST_LEAF_SPECS,
    P1_DENSE_PHYSICAL_TOKENS,
    P1_STUDY_ID,
    WARMUP_WINDOWS_PER_PASS,
    add_self_hash,
    p1_cost_leaf_relative_path,
    p1_cost_leaf_sequence,
    p1_cost_leaf_spec,
    validate_p1_cost_rows,
    validate_p1_cost_warmup_rows,
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _write_accuracy_storage_preflight(
    *,
    run_root: Path,
    cell_root: Path,
    arm: str,
    seed: int,
) -> Path:
    """Persist capacity evidence without pre-creating the bound work directory."""
    if cell_root.exists():
        raise FileExistsError("formal accuracy work_dir must be fresh before training")
    receipt_path = (
        run_root / "control" / "storage_preflights" / f"{arm}_seed{seed}.json"
    )
    if receipt_path.exists():
        raise FileExistsError("formal accuracy storage preflight already exists")
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        receipt_path,
        storage_capacity_receipt(run_root, cell_count=1),
    )
    if cell_root.exists():
        raise RuntimeError("storage preflight created the bound training work_dir")
    return receipt_path


def _current_commit() -> str:
    if os.environ.get("GEOROUTE_SOURCE_IDENTITY_VERIFIED") == "1":
        expected = os.environ.get("GEOROUTE_EXPECTED_COMMIT", "").strip().lower()
        if len(expected) != 40 or any(character not in "0123456789abcdef" for character in expected):
            raise RuntimeError("outer source identity receipt lacks a full commit")
        return expected
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout.strip().lower()


def _assert_clean_source_snapshot() -> None:
    if os.environ.get("GEOROUTE_SOURCE_IDENTITY_VERIFIED") == "1":
        _current_commit()
        return
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if status:
        raise RuntimeError("formal development requires a clean source snapshot")


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


def _p1_execution_arm(arm: str) -> str:
    return "dense_native" if arm == "DO" else arm


def _first_screen_arm_spec(arm: str) -> dict[str, Any]:
    if arm == "DO":
        return {
            **formal_arm_spec("dense_native"),
            "causal_role": "official_recipe_reproduction_report_only",
        }
    return development_arm_spec(arm)


def _p1_cell_relative_path(*, arm: str, seed: int) -> Path:
    if arm not in P1_FIRST_SCREEN_ARM_ORDER or int(seed) != P1_DEVELOPMENT_SEED:
        raise ValueError("P1 accuracy cell is outside the frozen first screen")
    return Path("development") / arm / f"seed{int(seed)}"


def _read_p1_runtime_attestation(
    deployment: Mapping[str, Any],
    *,
    arm: str | None = None,
    leaf_id: str | None = None,
) -> dict[str, Any]:
    runtime = deployment.get("runtime_attestation")
    if not isinstance(runtime, Mapping) or (arm is None) == (leaf_id is None):
        raise ValueError("P1 deployment lacks an unambiguous runtime attestation")
    preflight_path = Path(str(runtime.get("preflight_path", ""))).resolve()
    path_map_name = "leaf_paths" if arm is not None else "cost_leaf_paths"
    path_key = str(arm if arm is not None else leaf_id)
    path_map = runtime.get(path_map_name)
    if not isinstance(path_map, Mapping):
        raise ValueError("P1 deployment lacks leaf attestation paths")
    leaf_path = Path(str(path_map.get(path_key, ""))).resolve()
    if not preflight_path.is_file() or not leaf_path.is_file():
        raise FileNotFoundError("P1 runtime preflight/leaf attestation is missing")
    preflight = read_json(preflight_path)
    leaf = read_json(leaf_path)
    validate_runtime_attestation(preflight)
    validate_runtime_attestation(leaf, reference=preflight)
    if (
        int(runtime.get("expected_visible_gpu_count", -1)) != FORMAL_WORLD_SIZE
        or leaf["runtime_class"].get("visible_gpu_count") != FORMAL_WORLD_SIZE
        or leaf["runtime_class"].get("container_digest")
        != "sha256:" + str(runtime.get("container_image_sha256", ""))
        or leaf["runtime_class"].get("dependency_lock_sha256")
        != runtime.get("dependency_lock_sha256")
        or runtime.get("before_numpy_model_cuda_data_checkpoint") is not True
    ):
        raise ValueError("P1 runtime attestation changed allocation or ordering")
    return {
        "preflight_path": str(preflight_path),
        "preflight_file_sha256": sha256_file(preflight_path),
        "leaf_path": str(leaf_path),
        "leaf_file_sha256": sha256_file(leaf_path),
        "runtime_class_fingerprint": leaf["runtime_class_fingerprint"],
        "runtime_class": dict(leaf["runtime_class"]),
    }


def validate_p1_deployment_shape(deployment: Mapping[str, Any]) -> dict[str, Any]:
    checked = dict(deployment)
    jobs = checked.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    cost_jobs = jobs.get("cost") if isinstance(jobs, Mapping) else None
    runtime = checked.get("runtime_attestation")
    cost = checked.get("cost_protocol")
    policy = checked.get("dependency_policy")
    if (
        checked.get("schema_version") != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or checked.get("study_id") != P1_STUDY_ID
        or checked.get("mode") != "p1"
        or checked.get("status") != "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX"
        or tuple(checked.get("arms", ())) != P1_FIRST_SCREEN_ARM_ORDER
        or tuple(checked.get("seeds", ())) != (P1_DEVELOPMENT_SEED,)
        or int(checked.get("seed", -1)) != P1_DEVELOPMENT_SEED
        or int(checked.get("accuracy_cells", -1)) != 5
        or int(checked.get("cost_leaves", -1)) != 8
        or not isinstance(stage_jobs, Mapping)
        or set(stage_jobs) != set(P1_FIRST_SCREEN_ARM_ORDER)
        or any(
            not isinstance(stage_jobs[arm], Mapping)
            or set(stage_jobs[arm]) != {str(P1_DEVELOPMENT_SEED)}
            or not str(stage_jobs[arm][str(P1_DEVELOPMENT_SEED)]).isdigit()
            for arm in P1_FIRST_SCREEN_ARM_ORDER
        )
        or not isinstance(cost_jobs, Mapping)
        or set(cost_jobs) != set(P1_COST_LEAF_SPECS)
        or any(not str(cost_jobs[leaf_id]).isdigit() for leaf_id in P1_COST_LEAF_SPECS)
        or not str(jobs.get("runtime_preflight", "")).isdigit()
        or not str(jobs.get("finalizer", "")).isdigit()
        or not isinstance(runtime, Mapping)
        or int(runtime.get("expected_visible_gpu_count", -1)) != FORMAL_WORLD_SIZE
        or runtime.get("before_numpy_model_cuda_data_checkpoint") is not True
        or not isinstance(cost, Mapping)
        or cost.get("leaf_specs") != P1_COST_LEAF_SPECS
        or int(cost.get("physical_windows", -1)) != PHYSICAL_WINDOWS
        or int(cost.get("video_clusters", -1)) != 40
        or int(cost.get("warmup_windows_before_each_pass", -1))
        != WARMUP_WINDOWS_PER_PASS
        or int(cost.get("power_interval_ms", -1)) != POWER_INTERVAL_MS
        or int(cost.get("bootstrap_replicates", -1)) != 10_000
        or float(cost.get("q_over_dn_upper_bound_limit", -1.0)) != 0.85
        or cost.get("dn_only_controlling_denominator") is not True
        or cost.get("do_mandatory_report_only") is not True
        or not isinstance(policy, Mapping)
        or any(
            policy.get(field) is not True
            for field in (
                "all_fifteen_jobs_held_until_receipts_immutable",
                "accuracy_afterany_runtime_preflight",
                "cost_afterany_runtime_preflight_and_source_stages",
                "finalizer_afterany_all_fourteen_predecessors",
                "release_all_fifteen_atomically",
            )
        )
        or policy.get("resume_allowed") is not False
        or policy.get("retry_allowed") is not False
        or policy.get("requeue_allowed") is not False
        or checked.get("official_test_opened") is not False
        or checked.get("paper_claim_allowed") is not False
        or not _self_hash_matches(checked, field="deployment_sha256")
    ):
        raise ValueError("P1 deployment shape changed")
    return checked


def _validate_deployment(
    *,
    run_root: Path,
    expected_commit: str,
    arm: str,
    seed: int,
    slurm_job_id: str,
    task: str = "accuracy",
    leaf_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    p1_cell = arm in P1_FIRST_SCREEN_ARM_ORDER or task == "cost"
    expected_arm_order = (
        P1_FIRST_SCREEN_ARM_ORDER if p1_cell else FORMAL_DEVELOPMENT_ARM_ORDER
    )
    expected_seeds = (
        (P1_DEVELOPMENT_SEED,) if p1_cell else FORMAL_DEVELOPMENT_SEEDS
    )
    deployment_path = run_root / "control" / "deployment.json"
    deployment = read_json(deployment_path)
    if p1_cell:
        validate_p1_deployment_shape(deployment)
    jobs = deployment.get("jobs")
    stage_jobs = jobs.get("stage") if isinstance(jobs, Mapping) else None
    arm_jobs = stage_jobs.get(arm) if isinstance(stage_jobs, Mapping) else None
    cost_jobs = jobs.get("cost") if isinstance(jobs, Mapping) else None
    expected_status = (
        "SUBMITTED_ZOOMTOKEN_P1_DNURQ_MATRIX"
        if p1_cell
        else "SUBMITTED_OFFICIAL_COMPARABLE_DEVELOPMENT_MATRIX"
    )
    expected_job_matches = (
        isinstance(cost_jobs, Mapping)
        and leaf_id in P1_COST_LEAF_SPECS
        and str(cost_jobs.get(str(leaf_id), "")) == slurm_job_id
        if task == "cost"
        else isinstance(arm_jobs, Mapping)
        and str(arm_jobs.get(str(seed), "")) == slurm_job_id
    )
    if (
        deployment.get("schema_version")
        != FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA
        or deployment.get("status")
        != expected_status
        or deployment.get("runtime_commit") != expected_commit
        or Path(str(deployment.get("run_root", ""))).resolve() != run_root
        or tuple(deployment.get("arms", ())) != expected_arm_order
        or tuple(deployment.get("seeds", ())) != expected_seeds
        or not _self_hash_matches(deployment, field="deployment_sha256")
        or not expected_job_matches
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
    arm_spec = _first_screen_arm_spec(arm) if arm == "DO" else development_arm_spec(arm)
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
    expected_spec = _first_screen_arm_spec(arm) if arm == "DO" else development_arm_spec(arm)
    seed_allowed = (
        int(seed) == P1_DEVELOPMENT_SEED
        if arm in P1_FIRST_SCREEN_ARM_ORDER
        else development_seed_allowed(arm=arm, seed=seed)
    )
    if (
        result.get("schema_version") != FORMAL_DEVELOPMENT_RESULT_SCHEMA
        or result.get("status")
        != "PASS_OFFICIAL_COMPARABLE_DEVELOPMENT_ONLY"
        or arm
        not in (*FORMAL_DEVELOPMENT_ARM_ORDER, *P1_FIRST_SCREEN_ARM_ORDER)
        or not seed_allowed
        or int(result.get("epochs", -1)) != FORMAL_EPOCHS
        or result.get("arm_spec") != expected_spec
        or not isinstance(binding, Mapping)
        or validate_formal_development_binding(binding, seed=seed)
        != dict(binding)
        or binding.get("arm") != _p1_execution_arm(arm)
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
    runtime_attestation = result.get("runtime_attestation")
    if arm in P1_FIRST_SCREEN_ARM_ORDER:
        if (
            not isinstance(runtime_attestation, Mapping)
            or not _is_sha256(runtime_attestation.get("preflight_file_sha256"))
            or not _is_sha256(runtime_attestation.get("leaf_file_sha256"))
            or not _is_sha256(runtime_attestation.get("runtime_class_fingerprint"))
            or not isinstance(runtime_attestation.get("runtime_class"), Mapping)
            or int(
                runtime_attestation["runtime_class"].get("visible_gpu_count", -1)
            )
            != FORMAL_WORLD_SIZE
        ):
            raise ValueError("P1 stage result lacks its exact runtime class receipt")
    elif runtime_attestation is not None:
        raise ValueError("legacy formal result contains a P1 runtime receipt")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("accuracy", "cost"), default="accuracy")
    parser.add_argument(
        "--arm",
        choices=(
            *FORMAL_DEVELOPMENT_ARM_ORDER,
            "DO",
            *P1_MATCHED_RUNNER_ARM_ORDER,
        ),
    )
    parser.add_argument("--leaf-id", choices=tuple(P1_COST_LEAF_SPECS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-config", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--development-annotation", type=Path)
    parser.add_argument("--class-map", type=Path)
    parser.add_argument("--development-video-root", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(
    args: argparse.Namespace,
    *,
    cell_root: Path,
) -> dict[str, Any]:
    if args.arm is None or any(
        getattr(args, name) is None
        for name in (
            "source_config",
            "manifest",
            "development_annotation",
            "class_map",
            "development_video_root",
            "pretrained",
        )
    ):
        raise ValueError("accuracy task requires its arm and all frozen inputs")
    expected_commit = str(args.expected_commit).lower()
    if _current_commit() != expected_commit:
        raise RuntimeError("formal development source commit changed")
    _assert_clean_source_snapshot()
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
        task="accuracy",
    )
    runtime_attestation = None
    if args.arm in P1_FIRST_SCREEN_ARM_ORDER:
        source_receipt = deployment.get("source_configs", {}).get(args.arm)
        source_path = args.source_config.resolve()
        if (
            not isinstance(source_receipt, Mapping)
            or source_path != Path(str(source_receipt.get("path", ""))).resolve()
            or not source_path.is_file()
            or sha256_file(source_path) != source_receipt.get("sha256")
        ):
            raise ValueError("P1 source-config identity differs from deployment")
        runtime_attestation = _read_p1_runtime_attestation(
            deployment,
            arm=args.arm,
        )
    storage_receipt_path = _write_accuracy_storage_preflight(
        run_root=run_root,
        cell_root=cell_root,
        arm=args.arm,
        seed=args.seed,
    )
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
        arm=_p1_execution_arm(args.arm),
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
    spec = _first_screen_arm_spec(args.arm)
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
    if runtime_attestation is not None:
        result["runtime_attestation"] = runtime_attestation
    result["stage_result_sha256"] = canonical_sha256(result)
    return validate_formal_stage_result(
        result,
        expected_arm=args.arm,
        expected_seed=args.seed,
        expected_commit=expected_commit,
    )


def _p1_cost_route_summary(arm: str, audit: Mapping[str, Any]) -> dict[str, Any]:
    spec = _first_screen_arm_spec(arm)
    route_mode = str(spec["route_mode"])
    expected_tokens = (
        P1_DENSE_PHYSICAL_TOKENS if arm in {"DO", "DN"} else P1_WINDOW_TOKEN_BUDGET
    )
    common = {
        "arm": arm,
        "route_mode": route_mode,
        "target_k": spec["tokens_per_tubelet"],
        "dynamic_k_t": bool(spec.get("dynamic_k_t", False)),
        "selected_physical_tokens": expected_tokens,
        "executed_physical_tokens": expected_tokens,
        "duplicate_selected_physical_tokens": 0,
        "padded_heavy_tokens": 0,
        "uses_gt_for_route": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_test_evidence": False,
    }
    if arm == "Q":
        packed = audit.get("packed")
        if (
            audit.get("routing_schema") != "georoute_dynamic_global_routing_v2"
            or audit.get("route_mode") != "dynamic_scnr"
            or audit.get("policy_estimator") != "straight_through"
            or audit.get("target_k") is not None
            or int(audit.get("window_token_budget", -1)) != P1_WINDOW_TOKEN_BUDGET
            or audit.get("window_budget_is_global") is not True
            or audit.get("fixed_per_tubelet_k") is not False
            or audit.get("k_t_allows_zero") is not True
            or audit.get("zero_carrier_mode") != "masked_zero"
            or audit.get("heavy_valid_mask_matches_k_t") is not True
            or int(audit.get("requested_physical_tokens_per_window", -1))
            != P1_WINDOW_TOKEN_BUDGET
            or int(audit.get("unique_physical_tokens_per_window", -1))
            != P1_WINDOW_TOKEN_BUDGET
            or int(audit.get("padded_heavy_tokens_per_window", -1)) != 0
            or int(audit.get("executed_patch_tokens_per_window", -1))
            != P1_WINDOW_TOKEN_BUDGET
            or int(audit.get("heavy_backbone_forward_count", -1)) != 1
            or audit.get("uses_gt_for_route") is not False
            or audit.get("uses_teacher") is not False
            or audit.get("uses_oracle") is not False
            or audit.get("uses_test_evidence") is not False
            or not isinstance(packed, Mapping)
            or packed.get("execution_mode") != "true_clip_ragged_no_padding"
            or packed.get("adapter_execution")
            != "coordinate_lineage_true_ragged"
            or int(packed.get("padded_heavy_tokens_per_window", -1)) != 0
        ):
            raise ValueError("P1 Q cost replay changed dynamic exact-B execution")
        if len(packed["clip_token_counts"]) != 1:
            raise ValueError("P1 Q cost replay requires batch-one ragged telemetry")
        clip_counts = [int(value) for value in packed["clip_token_counts"][0]]
        common.update(
            {
                "k_t_min": int(audit.get("k_t_min", -1)),
                "k_t_max": int(audit.get("k_t_max", -1)),
                "k_t_zero_count": int(audit.get("k_t_zero_count", -1)),
                "clip_token_counts": clip_counts,
                "attention_pairs": sum(value**2 for value in clip_counts),
                "physical_indices_sha256": audit["physical_indices_sha256"],
            }
        )
    else:
        expected_k = (
            FORMAL_NATIVE_TOKEN_COUNT
            if spec["tokens_per_tubelet"] is None
            else int(spec["tokens_per_tubelet"])
        )
        if (
            audit.get("route_mode") != route_mode
            or audit.get("policy_estimator") != spec["policy_estimator"]
            or int(audit.get("target_k", -1)) != expected_k
            or int(audit.get("selected_unique_count_min", -1)) != expected_k
            or int(audit.get("selected_unique_count_max", -1)) != expected_k
            or int(audit.get("selected_duplicate_count", -1)) != 0
            or int(audit.get("heavy_backbone_forward_count", -1)) != 1
            or audit.get("uses_gt_for_route") is not False
            or audit.get("uses_teacher") is not False
            or audit.get("uses_oracle") is not False
            or audit.get("uses_test_evidence") is not False
        ):
            raise ValueError("P1 cost control route audit changed")
    return common


def _profile_p1_cost_pass(
    *,
    torch: Any,
    device: Any,
    arm: str,
    pass_index: int,
    leaf_id: str,
    stage: Mapping[str, Any],
    expected_population_sha256: str | None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    str,
]:
    from mmengine.config import Config
    from torch.nn.parallel import DistributedDataParallel

    from opentad.cores.test_engine import gather_ddp_results
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed
    from tools.bata.profile_georoute_dynamic_floor_m2 import (
        _build_cost_cuda_events,
        _invalid_cost_cuda_stages,
        _population_descriptor,
        _read_cost_cuda_timings,
    )
    from tools.bata.profile_spatial_zoom_s1 import (
        _measure_wall_ms,
        _move_to_device,
        _sample_identity,
        _strip_ddp_prefix,
    )

    config_path = Path(str(stage.get("config_path", ""))).resolve()
    if not config_path.is_file() or sha256_file(config_path) != stage.get("config_sha256"):
        raise ValueError("P1 cost source config changed after accuracy execution")
    cfg = Config.fromfile(str(config_path))
    binding = dict(stage["binding"])
    checkpoint_receipt = stage["checkpoint_receipt"]
    checkpoint_path = Path(str(checkpoint_receipt["path"])).resolve()
    validate_formal_checkpoint_sidecar(checkpoint_path, binding=binding)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    descriptors, population_sha256, accuracy_population_sha256 = (
        _population_descriptor(dataset)
    )
    if (
        len(descriptors) != PHYSICAL_WINDOWS
        or len({str(row["video_id"]) for row in descriptors}) != 40
        or accuracy_population_sha256
        != stage["telemetry_summary"]["population_sha256"]
        or (
            expected_population_sha256 is not None
            and population_sha256 != expected_population_sha256
        )
    ):
        raise ValueError("P1 cost pass changed the frozen 136-window/40-video population")
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=0,
    )
    if len(loader) != PHYSICAL_WINDOWS:
        raise ValueError("P1 cost loader changed the complete population")

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model_cfg.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    model_cfg.backbone.custom.georoute_role_calibration_telemetry_enabled = False
    model = build_detector(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != FORMAL_EPOCHS - 1 or "state_dict_ema" not in checkpoint:
        raise ValueError("P1 cost checkpoint is not the frozen final EMA")
    model.load_state_dict(_strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True)
    del checkpoint
    model = model.to(device).eval()
    ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0)
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)
    if not bool(cfg.solver.amp):
        raise ValueError("P1 cost replay must preserve AMP inference")
    wrapper = getattr(model, "backbone", None)
    wrapped = getattr(wrapper, "model", None)
    heavy = getattr(wrapped, "backbone", None)
    events, method_events = _build_cost_cuda_events(
        torch, model=model, wrapper=wrapper, heavy=heavy
    )

    def forward_once(batch: Mapping[str, Any]) -> Any:
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            return ddp_model(
                **batch,
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=external_cls,
            )

    iterator = iter(loader)

    def next_batch() -> Any:
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    set_seed(P1_DEVELOPMENT_SEED, deterministic_warn_only=True)
    warmup_rows: list[dict[str, Any]] = []
    for ordinal in range(WARMUP_WINDOWS_PER_PASS):
        cpu_batch = next_batch()
        warmup_rows.append(
            {
                "schema_version": "zoomtoken_p1_cost_warmup_identity_v001",
                "leaf_id": leaf_id,
                "pass_index": pass_index,
                "arm": arm,
                "measurement_phase": "warmup",
                "warmup": True,
                "warmup_ordinal": ordinal,
                **_sample_identity(cpu_batch, ordinal),
            }
        )
        forward_once(_move_to_device(cpu_batch, device))
    synchronize()
    iterator = iter(loader)

    samples: list[dict[str, Any]] = []
    energy_windows: list[tuple[float, float]] = []
    video_rows: dict[str, list[dict[str, Any]]] = {}
    final_energy_window: tuple[float, float] | None = None
    try:
        for ordinal, descriptor in enumerate(descriptors):
            synchronize()
            continuous_started = time.perf_counter()
            energy_started = time.monotonic_ns() / 1_000_000_000.0
            cpu_batch, input_ms = _measure_wall_ms(next_batch, synchronize=synchronize)
            identity = _sample_identity(cpu_batch, ordinal)
            expected_physical = (
                f"{descriptor['video_id']}:{int(descriptor['window_center_first'])}"
            )
            if identity["physical_window_id"] != expected_physical:
                raise ValueError("P1 cost loader order changed")
            torch.cuda.reset_peak_memory_stats(device)
            gpu_batch, h2d_ms = _measure_wall_ms(
                lambda: _move_to_device(cpu_batch, device), synchronize=synchronize
            )
            events.reset()
            for event in method_events.values():
                event.reset()
            post_result, _ = _measure_wall_ms(
                lambda: forward_once(gpu_batch), synchronize=synchronize
            )
            if not isinstance(post_result, Mapping):
                raise ValueError("P1 cost detector returned no result mapping")
            for video_id, rows in post_result.items():
                video_rows.setdefault(str(video_id), []).extend(rows)
            raw_audit = getattr(wrapper, "latest_georoute_audit", None)
            if not isinstance(raw_audit, Mapping):
                raise ValueError("P1 cost replay lacks its selector-boundary route audit")
            route_audit = _p1_cost_route_summary(arm, raw_audit)
            synchronize()
            continuous_ended = time.perf_counter()
            energy_ended = time.monotonic_ns() / 1_000_000_000.0
            component_timings = _read_cost_cuda_timings(events, method_events)
            invalid = _invalid_cost_cuda_stages(component_timings)
            if invalid:
                raise RuntimeError(
                    "P1 cost instrumentation missed CUDA stages: " + ", ".join(invalid)
                )
            expected_tokens = (
                P1_DENSE_PHYSICAL_TOKENS
                if arm in {"DO", "DN"}
                else P1_WINDOW_TOKEN_BUDGET
            )
            samples.append(
                {
                    "schema_version": "zoomtoken_p1_cost_sample_v001",
                    "leaf_id": leaf_id,
                    "pass_index": pass_index,
                    "arm": arm,
                    "sample_ordinal": ordinal,
                    "measurement_phase": "measured",
                    "warmup": False,
                    "population_sha256": population_sha256,
                    "exact_window_budget": P1_WINDOW_TOKEN_BUDGET,
                    "selected_physical_tokens": expected_tokens,
                    "executed_physical_tokens": expected_tokens,
                    "duplicate_selected_physical_tokens": 0,
                    "padded_heavy_tokens": 0,
                    "input_pipeline_serial_ms": input_ms,
                    "h2d_ms": h2d_ms,
                    **component_timings,
                    "decode_to_window_output_wall_ms": (
                        continuous_ended - continuous_started
                    )
                    * 1000.0,
                    "final_video_nms_ms": 0.0,
                    "end_to_end_serial_ms": (
                        continuous_ended - continuous_started
                    )
                    * 1000.0,
                    "peak_gpu_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024**2)
                    ),
                    "peak_gpu_reserved_mb": (
                        torch.cuda.max_memory_reserved(device) / (1024**2)
                    ),
                    "gross_gpu_energy_j_per_sample": None,
                    "route_audit": route_audit,
                    **identity,
                }
            )
            energy_windows.append((energy_started, energy_ended))
            del cpu_batch, gpu_batch, post_result
        synchronize()
        final_started = time.monotonic_ns() / 1_000_000_000.0
        finalized = gather_ddp_results(1, video_rows, cfg.post_processing)
        synchronize()
        final_ended = time.monotonic_ns() / 1_000_000_000.0
        final_energy_window = (final_started, final_ended)
        if not isinstance(finalized, Mapping):
            raise ValueError("P1 cost final video NMS returned no mapping")
        amortized_nms_ms = (final_ended - final_started) * 1000.0 / len(samples)
        for sample in samples:
            sample["final_video_nms_ms"] = amortized_nms_ms
            sample["end_to_end_serial_ms"] += amortized_nms_ms
    finally:
        events.close()
        for event in reversed(tuple(method_events.values())):
            event.close()
    if final_energy_window is None:
        raise RuntimeError("P1 cost replay did not execute final video NMS")
    for sample, energy_window in zip(samples, energy_windows):
        sample["energy_window_monotonic_s"] = list(energy_window)
        sample["nms_energy_window_monotonic_s"] = list(final_energy_window)
    pass_receipt = {
        "pass_index": pass_index,
        "arm": arm,
        "sample_count": len(samples),
        "population_sha256": population_sha256,
        "checkpoint_sha256": checkpoint_receipt["sha256"],
        "config_sha256": stage["config_sha256"],
        "sample_manifest_sha256": canonical_sha256(
            [sample["window_id"] for sample in samples]
        ),
        "diagnostic_telemetry_inside_timed_forward": False,
        "training_or_resume_executed": False,
    }
    pass_receipt["pass_sha256"] = canonical_sha256(pass_receipt)
    del ddp_model, model, loader, dataset
    torch.cuda.empty_cache()
    return samples, warmup_rows, pass_receipt, population_sha256


def _execute_p1_cost(args: argparse.Namespace, *, leaf_root: Path) -> dict[str, Any]:
    if args.leaf_id is None or args.arm is not None or int(args.seed) != P1_DEVELOPMENT_SEED:
        raise ValueError("P1 cost task requires one frozen leaf and seed 3407")
    expected_commit = str(args.expected_commit).lower()
    if _current_commit() != expected_commit:
        raise RuntimeError("P1 cost source commit changed")
    _assert_clean_source_snapshot()
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "")
    visible = [
        value
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value
    ]
    if not slurm_job_id.isdigit() or len(visible) != FORMAL_WORLD_SIZE:
        raise RuntimeError("P1 cost requires the frozen two-GPU Slurm runtime class")
    run_root = args.run_root.resolve()
    if not _inside(run_root, Path("/data/run01/sczc063/yuzibo").resolve()):
        raise ValueError("P1 cost run root leaves write boundary")
    deployment, _protocol, _preflight = _validate_deployment(
        run_root=run_root,
        expected_commit=expected_commit,
        arm="",
        seed=args.seed,
        slurm_job_id=slurm_job_id,
        task="cost",
        leaf_id=args.leaf_id,
    )
    runtime_attestation = _read_p1_runtime_attestation(
        deployment,
        leaf_id=args.leaf_id,
    )
    if leaf_root.exists():
        raise FileExistsError("P1 cost leaf exists; refusing overwrite or resume")
    spec = p1_cost_leaf_spec(args.leaf_id)
    sequence = p1_cost_leaf_sequence(args.leaf_id)
    stage_results: dict[str, dict[str, Any]] = {}
    for arm in set(sequence):
        stage_path = run_root / _p1_cell_relative_path(
            arm=arm, seed=P1_DEVELOPMENT_SEED
        ) / "stage_result.json"
        if not stage_path.is_file():
            raise FileNotFoundError(f"P1 cost source stage is missing: {arm}")
        stage_results[arm] = validate_formal_stage_result(
            read_json(stage_path),
            expected_arm=arm,
            expected_seed=P1_DEVELOPMENT_SEED,
            expected_commit=expected_commit,
        )
    # Everything above is stdlib/runtime-attestation work.  Framework, CUDA,
    # checkpoint and data access begin only after exact preflight/leaf equality.
    import torch
    import torch.distributed as dist

    from tools.bata.profile_georoute_dynamic_floor_m2 import _write_jsonl
    from tools.bata.profile_spatial_zoom_s1 import integrate_energy
    from tools.bata.spatial_zoom_s1_power import NvmlSidecarPowerSampler

    if (
        int(os.environ.get("WORLD_SIZE", -1)) != 1
        or int(os.environ.get("RANK", -1)) != 0
        or int(os.environ.get("LOCAL_RANK", -1)) != 0
        or not torch.cuda.is_available()
        or dist.is_initialized()
    ):
        raise RuntimeError("P1 cost requires fresh torchrun world1 on cuda:0")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("P1 NVML sidecar requires Linux CPU affinity")
    available_cpus = tuple(sorted(os.sched_getaffinity(0)))
    if len(available_cpus) < 5:
        raise RuntimeError("P1 cost requires four detector CPUs plus one sidecar CPU")
    allocated_cpus = available_cpus[:5]
    detector_cpus = allocated_cpus[:4]
    sidecar_cpu = allocated_cpus[4]
    os.sched_setaffinity(0, set(detector_cpus))
    selector = visible[0]
    uuid_query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid",
            "--format=csv,noheader,nounits",
            "-i",
            selector,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    expected_uuid = uuid_query.stdout.strip()
    if uuid_query.returncode != 0 or not expected_uuid.startswith("GPU-"):
        raise RuntimeError("P1 cost could not bind NVML to logical cuda:0")
    leaf_root.mkdir(parents=True, exist_ok=False)
    sampler = NvmlSidecarPowerSampler(
        expected_uuid=expected_uuid,
        interval_ms=POWER_INTERVAL_MS,
        scratch_dir=Path("/tmp")
        / f"job{slurm_job_id}_zoomtoken_p1_{args.leaf_id.lower()}",
        attempt_prefix=leaf_root / "power_sidecar",
        sidecar_cpu_id=sidecar_cpu,
        detector_cpu_ids=detector_cpus,
        allocated_cpu_ids=allocated_cpus,
    )
    all_rows: list[dict[str, Any]] = []
    warmup_rows: list[dict[str, Any]] = []
    pass_receipts: list[dict[str, Any]] = []
    population_sha256: str | None = None
    sampler.start()
    time.sleep(sampler.interval_s * 1.5)
    try:
        for pass_index, arm in enumerate(sequence):
            rows, warmups, pass_receipt, population_sha256 = _profile_p1_cost_pass(
                torch=torch,
                device=device,
                arm=arm,
                pass_index=pass_index,
                leaf_id=args.leaf_id,
                stage=stage_results[arm],
                expected_population_sha256=population_sha256,
            )
            all_rows.extend(rows)
            warmup_rows.extend(warmups)
            pass_receipts.append(pass_receipt)
    finally:
        time.sleep(sampler.interval_s * 1.5)
        sampler.stop()
    pass_counts = {
        index: sum(int(row["pass_index"]) == index for row in all_rows)
        for index in range(4)
    }
    for row in all_rows:
        start, end = map(float, row["energy_window_monotonic_s"])
        nms_start, nms_end = map(float, row["nms_energy_window_monotonic_s"])
        sample_energy = integrate_energy(sampler.samples, start=start, end=end)
        nms_energy = integrate_energy(sampler.samples, start=nms_start, end=nms_end)
        if sample_energy is None or nms_energy is None:
            raise RuntimeError("P1 cost power trace has incomplete coverage")
        row["gross_gpu_energy_j_per_sample"] = sample_energy + nms_energy / pass_counts[
            int(row["pass_index"])
        ]
        row["sample_sha256"] = canonical_sha256(row)
    validate_p1_cost_rows(all_rows, leaf_id=args.leaf_id)
    validate_p1_cost_warmup_rows(warmup_rows, leaf_id=args.leaf_id)
    measured_path = leaf_root / "measured_samples.jsonl"
    warmup_path = leaf_root / "warmup_identities.jsonl"
    power_path = leaf_root / "power_trace.jsonl"
    _write_jsonl(measured_path, all_rows)
    _write_jsonl(warmup_path, warmup_rows)
    power_origin = sampler.samples[0][0]
    _write_jsonl(
        power_path,
        [
            {
                "sequence": index,
                "monotonic_s": timestamp,
                "timestamp_ms": (timestamp - power_origin) * 1000.0,
                "power_w": power,
            }
            for index, (timestamp, power) in enumerate(sampler.samples)
        ],
    )
    receipt = add_self_hash(
        {
            "schema_version": "zoomtoken_p1_cost_leaf_v001",
            "study_id": P1_STUDY_ID,
            "status": "COMPLETE_P1_COST_LEAF",
            "runtime_commit": expected_commit,
            "leaf_id": args.leaf_id,
            "comparator": spec["comparator"],
            "order": spec["order"],
            "sequence": list(sequence),
            "seed": P1_DEVELOPMENT_SEED,
            "slurm_job_id": slurm_job_id,
            "runtime_attestation": runtime_attestation,
            "warmup_windows_before_each_pass": WARMUP_WINDOWS_PER_PASS,
            "measured_windows_per_pass": PHYSICAL_WINDOWS,
            "measured_pass_count": 4,
            "measured_rows": len(all_rows),
            "population_sha256": population_sha256,
            "pass_receipts": pass_receipts,
            "artifacts": {
                "measured_samples": {
                    "path": str(measured_path.resolve()),
                    "sha256": sha256_file(measured_path),
                },
                "warmup_identities": {
                    "path": str(warmup_path.resolve()),
                    "sha256": sha256_file(warmup_path),
                },
                "power_trace": {
                    "path": str(power_path.resolve()),
                    "sha256": sha256_file(power_path),
                },
                "sidecar_report": {
                    "path": str(sampler.attempt_report_path.resolve()),
                    "sha256": sha256_file(sampler.attempt_report_path),
                },
            },
            "training_or_resume_executed": False,
            "metric_evaluation_executed": False,
            "held_out_test_opened": False,
            "authoritative_decision": False,
        },
        field="receipt_sha256",
    )
    if dist.is_initialized():
        dist.destroy_process_group()
    return receipt


def main() -> int:
    args = _parse_args()
    run_root = args.run_root.resolve()
    if args.task == "cost":
        if args.leaf_id is None or args.arm is not None:
            raise ValueError("cost task requires --leaf-id and forbids --arm")
        leaf_root = run_root / p1_cost_leaf_relative_path(args.leaf_id)
        try:
            receipt = _execute_p1_cost(args, leaf_root=leaf_root)
        except Exception as error:
            if leaf_root.is_dir():
                trace = traceback.format_exc()
                failure: dict[str, Any] = {
                    "schema_version": "zoomtoken_p1_cost_failure_v001",
                    "study_id": P1_STUDY_ID,
                    "status": "FAIL_P1_COST_LEAF",
                    "leaf_id": args.leaf_id,
                    "seed": int(args.seed),
                    "expected_runtime_commit": str(args.expected_commit).lower(),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "exception_type": type(error).__name__,
                    "exception_message": str(error)[:2000],
                    "traceback_sha256": hashlib.sha256(
                        trace.encode("utf-8", errors="replace")
                    ).hexdigest(),
                    "performance_inference_allowed": False,
                    "paper_claim_allowed": False,
                }
                failure["failure_sha256"] = canonical_sha256(failure)
                _atomic_write_json(leaf_root / "cost_failure.json", failure)
            raise
        _atomic_write_json(leaf_root / "receipt.json", receipt)
        print(json.dumps(receipt, sort_keys=True))
        return 0

    if args.arm is None or args.leaf_id is not None:
        raise ValueError("accuracy task requires --arm and forbids --leaf-id")
    _first_screen_arm_spec(args.arm)
    seed_allowed = (
        int(args.seed) == P1_DEVELOPMENT_SEED
        if args.arm in P1_FIRST_SCREEN_ARM_ORDER
        else development_seed_allowed(arm=args.arm, seed=args.seed)
    )
    if not seed_allowed:
        raise ValueError("formal seed is outside the frozen set")
    cell_root = run_root / (
        _p1_cell_relative_path(arm=args.arm, seed=args.seed)
        if args.arm in P1_FIRST_SCREEN_ARM_ORDER
        else formal_cell_relative_path(arm=args.arm, seed=args.seed)
    )
    if cell_root.exists():
        raise FileExistsError(
            "formal development cell exists; refusing overwrite or resume"
        )
    try:
        result = _execute(args, cell_root=cell_root)
    except Exception as error:
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
        failure_path = (
            cell_root / "stage_failure.json"
            if cell_root.is_dir()
            else run_root
            / "control"
            / "stage_failures"
            / f"{args.arm}_seed{args.seed}.json"
        )
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(failure_path, failure)
        raise
    _atomic_write_json(cell_root / "stage_result.json", result)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
