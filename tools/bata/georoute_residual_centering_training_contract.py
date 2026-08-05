"""Matched G1 training contract for the SCNR residual-centering intervention.

This single-seed development screen compares two fresh G1 trainings whose only
method variable is whether ``delta_residual`` is centered over every valid
candidate in the complete window.  It can authorize a later paired cost replay;
it cannot open official test, additional seeds, or a paper claim.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from tools.bata.georoute_dynamic_floor_m2_contract import (
    DYNAMIC_FLOOR_M2_EPOCHS,
    DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_TRAINING_ACCURACY_SCHEMA,
    DYNAMIC_FLOOR_M2_SEED,
    DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
    bind_dynamic_floor_m2_config,
    summarize_dynamic_floor_m2_telemetry,
    validate_dynamic_floor_m2_checkpoint_sidecar,
    validate_dynamic_floor_m2_config,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file
from tools.bata.run_georoute_role_instrumentation_pair import (
    compare_prediction_artifacts,
)
from tools.bata.georoute_stage_runner import parse_official_style_map


RESIDUAL_CENTERING_TRAINING_STUDY_ID = (
    "scnr_residual_centering_matched_training_v1"
)
RESIDUAL_CENTERING_TRAINING_CONTRACT_SCHEMA = (
    "scnr_residual_centering_matched_training_contract_v1"
)
RESIDUAL_CENTERING_TRAINING_STAGE_SCHEMA = (
    "scnr_residual_centering_matched_training_stage_v1"
)
RESIDUAL_CENTERING_TRAINING_FINALIZATION_SCHEMA = (
    "scnr_residual_centering_matched_training_finalization_v1"
)
RESIDUAL_CENTERING_TRAINING_DEPLOYMENT_SCHEMA = (
    "scnr_residual_centering_matched_training_deployment_v1"
)
RESIDUAL_CENTERING_TRAINING_PRECHECK_SCHEMA = (
    "scnr_residual_centering_matched_training_precheck_v1"
)
RESIDUAL_CENTERING_BASE_ARM = "native_1cell_main"
RESIDUAL_CENTERING_TRAIN_BATCHES_PER_EPOCH = 160
RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES = (
    DYNAMIC_FLOOR_M2_EPOCHS * RESIDUAL_CENTERING_TRAIN_BATCHES_PER_EPOCH
)
RESIDUAL_CENTERING_MEAN_ABS_TOLERANCE = 1e-4
RESIDUAL_CENTERING_ACCURACY_REPLAYS = ("accuracy_a", "accuracy_b")

RESIDUAL_CENTERING_TRAINING_VARIANTS: dict[str, dict[str, Any]] = {
    "none_control": {
        "slug": "none_control",
        "branch_calibration_mode": "none",
        "branch_calibration_scope": "disabled",
        "role": "matched_g1_control",
    },
    "residual_window_center": {
        "slug": "residual_window_center",
        "branch_calibration_mode": "residual_window_center",
        "branch_calibration_scope": "complete_window_all_valid_candidates",
        "role": "matched_g1_residual_centering_treatment",
    },
}
RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER = tuple(
    RESIDUAL_CENTERING_TRAINING_VARIANTS
)


def residual_centering_training_variant_spec(variant: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(RESIDUAL_CENTERING_TRAINING_VARIANTS[variant])
    except KeyError as error:
        raise ValueError(
            f"unsupported residual-centering training variant {variant!r}"
        ) from error


def residual_centering_training_cell_relative_path(
    *, variant: str, seed: int = DYNAMIC_FLOOR_M2_SEED
) -> Path:
    if int(seed) != DYNAMIC_FLOOR_M2_SEED:
        raise ValueError("residual-centering matched training permits only seed 3407")
    spec = residual_centering_training_variant_spec(variant)
    return Path("development") / str(spec["slug"]) / f"seed{seed}"


def _normalized_training_recipe(cfg: Any) -> dict[str, Any]:
    """Canonicalize only registered path, phase, and intervention differences."""

    payload = copy.deepcopy(cfg.to_dict())
    payload["work_dir"] = "__MATCHED_WORK_DIR__"
    custom = payload["model"]["backbone"]["custom"]
    custom["georoute_branch_calibration_mode"] = "__MATCHED_BRANCH_MODE__"
    custom["georoute_diagnostic_telemetry_enabled"] = False
    custom["georoute_role_calibration_telemetry_enabled"] = False
    payload["georoute_diagnostic_telemetry"] = {"enabled": False}
    payload["georoute_development_profile"] = {"enabled": False}
    payload.pop("georoute_phase_m_binding", None)
    payload.pop("georoute_residual_centering_training_binding", None)
    for name in (
        "georoute_dynamic_floor_m2_binding",
        "georoute_runtime_binding",
    ):
        inherited = payload.get(name)
        if isinstance(inherited, dict):
            inherited["work_dir"] = "__MATCHED_WORK_DIR__"
            inherited.pop("binding_sha256", None)
    telemetry_binding = payload.get("georoute_telemetry_binding")
    if isinstance(telemetry_binding, dict):
        telemetry_binding["arm"] = "__MATCHED_VARIANT__"
    return payload


def _shared_protocol(
    base_binding: Mapping[str, Any], *, normalized_recipe_sha256: str
) -> dict[str, Any]:
    """Extract the cross-variant fields that must remain byte-identical."""

    return {
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "runtime_commit": base_binding["runtime_commit"],
        "source_config_sha256": base_binding["source_config_sha256"],
        "manifest_file_sha256": base_binding["manifest_file_sha256"],
        "development_annotation_sha256": base_binding["development_annotation"][
            "sha256"
        ],
        "class_map_sha256": base_binding["class_map_sha256"],
        "development_video_root": base_binding["development_video_root"],
        "pretrained_checkpoint_sha256": base_binding[
            "pretrained_checkpoint_sha256"
        ],
        "base_arm": RESIDUAL_CENTERING_BASE_ARM,
        "base_arm_spec_sha256": base_binding["arm_spec_sha256"],
        "training_video_ids": list(base_binding["training_video_ids"]),
        "evaluation_video_ids": list(base_binding["evaluation_video_ids"]),
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "epochs": DYNAMIC_FLOOR_M2_EPOCHS,
        "train_batches_per_epoch": RESIDUAL_CENTERING_TRAIN_BATCHES_PER_EPOCH,
        "expected_successful_updates": (
            RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
        ),
        "world_size": 1,
        "local_batch": 1,
        "global_batch": 1,
        "window_token_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "route_mode": "dynamic_scnr",
        "policy_estimator": "straight_through",
        "roi_extent_floor_mode": "native_cells",
        "roi_extent_floor_cells": 1,
        "zero_carrier_mode": "masked_zero",
        "k_t_allows_zero": True,
        "fixed_context_quota": False,
        "fixed_per_tubelet_k": False,
        "q_ctx_used": False,
        "amp": True,
        "ema": True,
        "fp16_compress": False,
        "deterministic_training": "warn_only",
        "max_amp_retries_per_batch": 8,
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "fresh_training_per_variant": True,
        "old_g1_checkpoint_reused": False,
        "single_method_variable": "georoute_branch_calibration_mode",
        "normalized_complete_training_recipe_sha256": normalized_recipe_sha256,
        "accuracy_replays": list(RESIDUAL_CENTERING_ACCURACY_REPLAYS),
        "accuracy_execution": "same_job_same_gpu_serial_strict_math_sdpa",
        "cost_attached": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def bind_residual_centering_training_config(
    *,
    source_config_path: str | Path,
    variant: str,
    seed: int,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    runtime_commit: str,
):
    """Bind one fresh G1 training cell and its immutable shared protocol."""

    if int(seed) != DYNAMIC_FLOOR_M2_SEED:
        raise ValueError("residual-centering matched training seed changed")
    spec = residual_centering_training_variant_spec(variant)
    cfg = bind_dynamic_floor_m2_config(
        source_config_path=source_config_path,
        arm=RESIDUAL_CENTERING_BASE_ARM,
        seed=seed,
        work_dir=work_dir,
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        development_video_root=development_video_root,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        runtime_commit=runtime_commit,
    )
    custom = cfg.model.backbone.custom
    custom.georoute_branch_calibration_mode = spec["branch_calibration_mode"]
    custom.georoute_diagnostic_telemetry_enabled = False
    custom.georoute_role_calibration_telemetry_enabled = False
    cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    cfg.georoute_development_profile = dict(enabled=False)
    if "georoute_phase_m_binding" in cfg:
        del cfg["georoute_phase_m_binding"]
    cfg.georoute_telemetry_binding = dict(
        schema_version="georoute_telemetry_world_binding_v1",
        study_id=RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        arm=variant,
        seed=DYNAMIC_FLOOR_M2_SEED,
        world_size=1,
        local_batch=1,
        development_only=True,
        official_test_opened=False,
    )

    base_binding = dict(cfg.georoute_dynamic_floor_m2_binding)
    normalized_recipe_sha256 = canonical_sha256(_normalized_training_recipe(cfg))
    shared = _shared_protocol(
        base_binding,
        normalized_recipe_sha256=normalized_recipe_sha256,
    )
    binding: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_TRAINING_CONTRACT_SCHEMA,
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "variant": variant,
        "variant_spec": spec,
        "variant_spec_sha256": canonical_sha256(spec),
        "branch_calibration_mode": spec["branch_calibration_mode"],
        "branch_calibration_scope": spec["branch_calibration_scope"],
        "base_arm": RESIDUAL_CENTERING_BASE_ARM,
        "base_binding_sha256": base_binding["binding_sha256"],
        "shared_protocol": shared,
        "shared_protocol_sha256": canonical_sha256(shared),
        "training_work_dir": str(Path(work_dir).resolve()),
        "fresh_training": True,
        "old_g1_checkpoint_reused": False,
        "only_method_variable": "georoute_branch_calibration_mode",
        "strict_duplicate_accuracy_required": True,
        "cost_gate_requires_accuracy_screen_pass": True,
        "additional_seed_gate_requires_accuracy_and_cost": True,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_residual_centering_training_binding = binding
    validate_residual_centering_training_config(cfg, variant=variant, phase="train")
    return cfg


def configure_residual_centering_accuracy(
    cfg: Any,
    *,
    variant: str,
    replay: str,
    work_dir: str | Path,
    training_config_sha256: str,
) -> Any:
    """Build one strict, metric-enabled duplicate accuracy replay."""

    if replay not in RESIDUAL_CENTERING_ACCURACY_REPLAYS:
        raise ValueError(f"unsupported residual-centering accuracy replay {replay!r}")
    bound = copy.deepcopy(cfg)
    binding = dict(bound.georoute_residual_centering_training_binding)
    spec = residual_centering_training_variant_spec(variant)
    bound.work_dir = str(Path(work_dir).resolve())
    custom = bound.model.backbone.custom
    custom.georoute_branch_calibration_mode = spec["branch_calibration_mode"]
    custom.georoute_diagnostic_telemetry_enabled = True
    custom.georoute_role_calibration_telemetry_enabled = False
    bound.georoute_diagnostic_telemetry = dict(enabled=True)
    bound.georoute_development_profile = dict(enabled=False)
    phase: dict[str, Any] = {
        "schema_version": (
            DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_TRAINING_ACCURACY_SCHEMA
        ),
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "variant": variant,
        "base_arm": RESIDUAL_CENTERING_BASE_ARM,
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "replay": replay,
        "source_experiment_commit": binding["shared_protocol"]["runtime_commit"],
        "runtime_commit": binding["shared_protocol"]["runtime_commit"],
        "study_binding_sha256": binding["binding_sha256"],
        "shared_protocol_sha256": binding["shared_protocol_sha256"],
        "training_config_sha256": str(training_config_sha256),
        "branch_calibration_mode": spec["branch_calibration_mode"],
        "branch_calibration_scope": spec["branch_calibration_scope"],
        "role_calibration_telemetry_enabled": False,
        "matched_training_accuracy_only": True,
        "same_slurm_job": True,
        "same_visible_gpu": True,
        "serial_execution": True,
        "strict_deterministic_algorithms": True,
        "sdp_backend": "math",
        "tf32_enabled": False,
        "metric_evaluation_enabled": True,
        "diagnostic_telemetry_enabled": True,
        "timed_cost_enabled": False,
        "fixed_role_quota_used": False,
        "q_ctx_used": False,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "oracle_used": False,
        "raw_prediction_cache_used": False,
    }
    phase["binding_sha256"] = canonical_sha256(phase)
    bound.georoute_phase_m_binding = phase
    validate_residual_centering_training_config(
        bound, variant=variant, phase="accuracy"
    )
    return bound


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def validate_residual_centering_training_config(
    cfg: Any, *, variant: str, phase: str
) -> dict[str, Any]:
    """Validate both the inherited G1 contract and the one-variable overlay."""

    if phase not in {"train", "accuracy"}:
        raise ValueError(f"unsupported residual-centering phase {phase!r}")
    base = validate_dynamic_floor_m2_config(
        cfg, arm=RESIDUAL_CENTERING_BASE_ARM, phase=phase
    )
    spec = residual_centering_training_variant_spec(variant)
    binding = cfg.get("georoute_residual_centering_training_binding")
    if not isinstance(binding, Mapping):
        raise ValueError("residual-centering training binding is missing")
    binding = dict(binding)
    shared = binding.get("shared_protocol")
    custom = cfg.model.backbone.custom
    if (
        not _self_hash_matches(binding, field="binding_sha256")
        or binding.get("schema_version")
        != RESIDUAL_CENTERING_TRAINING_CONTRACT_SCHEMA
        or binding.get("study_id") != RESIDUAL_CENTERING_TRAINING_STUDY_ID
        or binding.get("variant") != variant
        or binding.get("variant_spec") != spec
        or binding.get("variant_spec_sha256") != canonical_sha256(spec)
        or binding.get("branch_calibration_mode")
        != spec["branch_calibration_mode"]
        or binding.get("branch_calibration_scope")
        != spec["branch_calibration_scope"]
        or binding.get("base_arm") != RESIDUAL_CENTERING_BASE_ARM
        or binding.get("base_binding_sha256") != base["binding_sha256"]
        or not isinstance(shared, Mapping)
        or binding.get("shared_protocol_sha256") != canonical_sha256(shared)
        or dict(shared)
        != _shared_protocol(
            base,
            normalized_recipe_sha256=canonical_sha256(
                _normalized_training_recipe(cfg)
            ),
        )
        or binding.get("fresh_training") is not True
        or binding.get("old_g1_checkpoint_reused") is not False
        or binding.get("only_method_variable")
        != "georoute_branch_calibration_mode"
        or binding.get("strict_duplicate_accuracy_required") is not True
        or custom.georoute_branch_calibration_mode
        != spec["branch_calibration_mode"]
        or custom.georoute_role_calibration_telemetry_enabled is not False
    ):
        raise ValueError("residual-centering matched-training binding is invalid")
    phase_binding = cfg.get("georoute_phase_m_binding")
    if phase == "train":
        if phase_binding:
            raise ValueError("residual-centering training must not carry an accuracy binding")
        if (
            custom.georoute_diagnostic_telemetry_enabled is not False
            or cfg.georoute_diagnostic_telemetry.get("enabled") is not False
        ):
            raise ValueError("residual-centering training must disable telemetry")
    else:
        if not isinstance(phase_binding, Mapping):
            raise ValueError("residual-centering accuracy binding is missing")
        phase_binding = dict(phase_binding)
        if (
            not _self_hash_matches(phase_binding, field="binding_sha256")
            or phase_binding.get("schema_version")
            != DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_TRAINING_ACCURACY_SCHEMA
            or phase_binding.get("study_id")
            != RESIDUAL_CENTERING_TRAINING_STUDY_ID
            or phase_binding.get("variant") != variant
            or phase_binding.get("replay")
            not in RESIDUAL_CENTERING_ACCURACY_REPLAYS
            or phase_binding.get("study_binding_sha256")
            != binding["binding_sha256"]
            or phase_binding.get("shared_protocol_sha256")
            != binding["shared_protocol_sha256"]
            or phase_binding.get("branch_calibration_mode")
            != spec["branch_calibration_mode"]
            or phase_binding.get("strict_deterministic_algorithms") is not True
            or phase_binding.get("sdp_backend") != "math"
            or phase_binding.get("tf32_enabled") is not False
            or phase_binding.get("metric_evaluation_enabled") is not True
            or custom.georoute_diagnostic_telemetry_enabled is not True
            or cfg.georoute_diagnostic_telemetry.get("enabled") is not True
        ):
            raise ValueError("residual-centering accuracy binding is invalid")
    return binding


def _load_telemetry(payload_or_path: Mapping[str, Any] | str | Path) -> dict[str, Any]:
    if isinstance(payload_or_path, Mapping):
        return dict(payload_or_path)
    payload = json.loads(Path(payload_or_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("residual-centering telemetry must be a JSON object")
    return payload


def residual_centering_route_payload_sha256(
    payload_or_path: Mapping[str, Any] | str | Path,
) -> str:
    telemetry = _load_telemetry(payload_or_path)
    records = telemetry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("residual-centering route telemetry is empty")
    routes = []
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(
            record.get("route"), Mapping
        ):
            raise ValueError("residual-centering route telemetry is invalid")
        routes.append(
            {
                "dataset_index": record.get("dataset_index"),
                "video_id": record.get("video_id"),
                "route": dict(record["route"]),
            }
        )
    return canonical_sha256(routes)


def summarize_residual_centering_training_branch(
    payload_or_path: Mapping[str, Any] | str | Path,
    *,
    expected_mode: str,
) -> dict[str, Any]:
    """Validate the identity or centered residual transform over every window."""

    if expected_mode not in {"none", "residual_window_center"}:
        raise ValueError("unsupported residual-centering branch mode")
    telemetry = _load_telemetry(payload_or_path)
    records = telemetry.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("residual-centering branch telemetry is empty")
    expected_scope = (
        "complete_window_all_valid_candidates"
        if expected_mode == "residual_window_center"
        else "disabled"
    )
    before_values: list[float] = []
    after_values: list[float] = []
    valid_counts: list[int] = []
    for record in records:
        route = record.get("route") if isinstance(record, Mapping) else None
        branch = route.get("branch_calibration") if isinstance(route, Mapping) else None
        if not isinstance(branch, Mapping):
            raise ValueError("residual-centering branch receipt is missing")
        tubelets = route.get("tubelet_count")
        items = route.get("item_count")
        valid_count = branch.get("valid_candidate_count")
        before = branch.get("residual_valid_mean_before")
        after = branch.get("residual_valid_mean_after")
        numeric = (
            not isinstance(before, bool)
            and isinstance(before, (int, float))
            and math.isfinite(float(before))
            and not isinstance(after, bool)
            and isinstance(after, (int, float))
            and math.isfinite(float(after))
        )
        mode_valid = (
            abs(float(after)) <= RESIDUAL_CENTERING_MEAN_ABS_TOLERANCE
            if numeric and expected_mode == "residual_window_center"
            else math.isclose(
                float(before), float(after), rel_tol=0.0, abs_tol=1e-12
            )
            if numeric
            else False
        )
        if (
            branch.get("schema_version")
            != "scnr_dynamic_branch_calibration_window_v1"
            or branch.get("mode") != expected_mode
            or branch.get("target") != "delta_residual"
            or branch.get("scope") != expected_scope
            or branch.get("changes_q_base") is not False
            or branch.get("changes_delta_roi") is not False
            or branch.get("changes_context_zero_modifier") is not False
            or branch.get("changes_budget_or_role_quota") is not False
            or branch.get("mean_detached") is not False
            or isinstance(valid_count, bool)
            or not isinstance(valid_count, int)
            or isinstance(tubelets, bool)
            or not isinstance(tubelets, int)
            or isinstance(items, bool)
            or not isinstance(items, int)
            or int(valid_count) != int(tubelets) * int(items)
            or not numeric
            or not mode_valid
        ):
            raise ValueError("residual-centering training branch receipt is invalid")
        valid_counts.append(int(valid_count))
        before_values.append(float(before))
        after_values.append(float(after))
    return {
        "schema_version": "scnr_residual_centering_training_branch_summary_v1",
        "mode": expected_mode,
        "scope": expected_scope,
        "record_count": len(records),
        "valid_candidate_count_min": min(valid_counts),
        "valid_candidate_count_max": max(valid_counts),
        "residual_valid_mean_before_min": min(before_values),
        "residual_valid_mean_before_max": max(before_values),
        "residual_valid_mean_after_max_abs": max(map(abs, after_values)),
        "centered_mean_abs_tolerance": RESIDUAL_CENTERING_MEAN_ABS_TOLERANCE,
        "identity_required": expected_mode == "none",
        "receipts_valid": True,
    }


def _metrics_from_log(path: Path) -> dict[str, float]:
    metrics = {
        key: float(value)
        for key, value in parse_official_style_map(
            path.read_text(encoding="utf-8", errors="replace")
        ).items()
    }
    metrics["high_iou_composite"] = 0.5 * (
        metrics["mAP@0.6"] + metrics["mAP@0.7"]
    )
    return metrics


def _validated_file_receipt(receipt: Any, *, label: str) -> Path:
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{label} receipt is missing")
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} path is invalid")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or str(path.resolve()) != raw_path
        or not path.is_file()
        or path.is_symlink()
        or receipt.get("sha256") != sha256_file(path)
    ):
        raise ValueError(f"{label} artifact receipt changed")
    return path


def residual_centering_reachability_gate(
    telemetry_summary: Mapping[str, Any],
) -> dict[str, Any]:
    roles = telemetry_summary.get("roles")
    counts = roles.get("counts") if isinstance(roles, Mapping) else None
    fractions = roles.get("fractions") if isinstance(roles, Mapping) else None
    role_order = ("context", "roi", "residual")
    if (
        not isinstance(counts, Mapping)
        or not isinstance(fractions, Mapping)
        or set(counts) != set(role_order)
        or set(fractions) != set(role_order)
        or any(
            isinstance(counts[role], bool)
            or not isinstance(counts[role], int)
            or counts[role] < 0
            for role in role_order
        )
        or any(
            isinstance(fractions[role], bool)
            or not isinstance(fractions[role], (int, float))
            or not math.isfinite(float(fractions[role]))
            or not 0.0 <= float(fractions[role]) <= 1.0
            for role in role_order
        )
    ):
        raise ValueError("residual-centering selected-role summary is missing")
    conditions = {
        "selected_context_positive": int(counts.get("context", 0)) > 0,
        "selected_roi_positive": int(counts.get("roi", 0)) > 0,
        "selected_residual_fraction_below_one": float(
            fractions.get("residual", 1.0)
        )
        < 1.0,
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "selected_role_counts": {key: int(counts[key]) for key in role_order},
        "selected_role_fractions": {
            key: float(fractions[key]) for key in role_order
        },
    }


def validate_residual_centering_training_stage_result(
    result: Mapping[str, Any],
    *,
    expected_variant: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    """Recompute the complete training/duplicate-accuracy evidence from artifacts."""

    result = dict(result)
    variant = str(result.get("variant", ""))
    spec = residual_centering_training_variant_spec(variant)
    if expected_variant is not None and variant != expected_variant:
        raise ValueError("residual-centering stage variant mismatch")
    if expected_commit is not None and result.get("runtime_commit") != expected_commit:
        raise ValueError("residual-centering stage commit mismatch")
    binding = result.get("binding")
    configs = result.get("config_receipts")
    checkpoint = result.get("checkpoint_receipt")
    replays = result.get("accuracy_replays")
    comparison = result.get("duplicate_integrity")
    prediction_comparison = (
        comparison.get("prediction") if isinstance(comparison, Mapping) else None
    )
    stage_shared = binding.get("shared_protocol") if isinstance(binding, Mapping) else None
    if (
        result.get("schema_version") != RESIDUAL_CENTERING_TRAINING_STAGE_SCHEMA
        or result.get("status")
        != "PASS_RESIDUAL_CENTERING_MATCHED_TRAINING_AND_DUPLICATE_ACCURACY"
        or result.get("study_id") != RESIDUAL_CENTERING_TRAINING_STUDY_ID
        or result.get("variant_spec") != spec
        or result.get("variant_spec_sha256") != canonical_sha256(spec)
        or int(result.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or int(result.get("epochs", -1)) != DYNAMIC_FLOOR_M2_EPOCHS
        or int(result.get("expected_successful_updates", -1))
        != RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
        or not isinstance(binding, Mapping)
        or not _self_hash_matches(binding, field="binding_sha256")
        or result.get("binding_sha256") != binding.get("binding_sha256")
        or not isinstance(stage_shared, Mapping)
        or result.get("runtime_commit")
        != stage_shared.get("runtime_commit")
        or not isinstance(configs, Mapping)
        or set(configs) != {"train", *RESIDUAL_CENTERING_ACCURACY_REPLAYS}
        or not isinstance(checkpoint, Mapping)
        or not isinstance(replays, Mapping)
        or set(replays) != set(RESIDUAL_CENTERING_ACCURACY_REPLAYS)
        or not isinstance(comparison, Mapping)
        or not isinstance(prediction_comparison, Mapping)
        or prediction_comparison.get("raw_sha256_parity") is not True
        or prediction_comparison.get("json_semantic_parity") is not True
        or comparison.get("route_payload_sha256_parity") is not True
        or comparison.get("metrics_parity") is not True
        or result.get("fresh_training") is not True
        or result.get("old_g1_checkpoint_reused") is not False
        or result.get("cost_attached") is not False
        or result.get("additional_seeds_opened") is not False
        or result.get("official_test_opened") is not False
        or result.get("paper_claim_allowed") is not False
        or not _self_hash_matches(result, field="stage_result_sha256")
    ):
        raise ValueError("residual-centering stage result contract failed")

    from mmengine.config import Config

    config_paths = {
        name: _validated_file_receipt(
            configs[name], label=f"residual-centering {name} config"
        )
        for name in ("train", *RESIDUAL_CENTERING_ACCURACY_REPLAYS)
    }
    bound_configs = {
        name: Config.fromfile(str(path)) for name, path in config_paths.items()
    }
    observed_binding = validate_residual_centering_training_config(
        bound_configs["train"], variant=variant, phase="train"
    )
    if dict(observed_binding) != dict(binding):
        raise ValueError("residual-centering train config changed its binding")
    training_config_sha = canonical_sha256(bound_configs["train"].to_dict())
    for replay in RESIDUAL_CENTERING_ACCURACY_REPLAYS:
        replay_binding = validate_residual_centering_training_config(
            bound_configs[replay], variant=variant, phase="accuracy"
        )
        phase_binding = bound_configs[replay].georoute_phase_m_binding
        if (
            dict(replay_binding) != dict(binding)
            or phase_binding.replay != replay
            or phase_binding.training_config_sha256 != training_config_sha
        ):
            raise ValueError("residual-centering accuracy config lineage changed")

    checkpoint_path = _validated_file_receipt(
        checkpoint, label="residual-centering checkpoint"
    )
    sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint_path,
        binding=bound_configs["train"].georoute_dynamic_floor_m2_binding,
        cfg=bound_configs["train"],
    )
    metadata = sidecar["experiment_metadata"]
    sidecar_path = Path(str(checkpoint_path) + ".metadata.json").resolve()
    if (
        checkpoint.get("size_bytes") != checkpoint_path.stat().st_size
        or checkpoint.get("sidecar_path") != str(sidecar_path)
        or checkpoint.get("sidecar_sha256") != sha256_file(sidecar_path)
        or checkpoint.get("state_key") != "state_dict_ema"
        or int(metadata.get("train_batches_per_epoch", -1))
        != RESIDUAL_CENTERING_TRAIN_BATCHES_PER_EPOCH
        or int(metadata.get("successful_updates", -1))
        != RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
        or int(checkpoint.get("successful_updates", -1))
        != RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES
    ):
        raise ValueError("residual-centering checkpoint proof is incomplete")

    recomputed: dict[str, Any] = {}
    for replay in RESIDUAL_CENTERING_ACCURACY_REPLAYS:
        receipt = replays[replay]
        if not isinstance(receipt, Mapping):
            raise ValueError("residual-centering accuracy replay receipt is missing")
        artifacts = receipt.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != {
            "prediction",
            "telemetry",
            "accuracy_log",
        }:
            raise ValueError("residual-centering accuracy artifacts are incomplete")
        paths = {
            name: _validated_file_receipt(
                artifacts[name], label=f"residual-centering {replay} {name}"
            )
            for name in artifacts
        }
        metrics = _metrics_from_log(paths["accuracy_log"])
        telemetry = summarize_dynamic_floor_m2_telemetry(paths["telemetry"])
        branch = summarize_residual_centering_training_branch(
            paths["telemetry"], expected_mode=spec["branch_calibration_mode"]
        )
        route_sha = residual_centering_route_payload_sha256(paths["telemetry"])
        if (
            dict(receipt.get("metrics", {})) != metrics
            or dict(receipt.get("telemetry_summary", {})) != telemetry
            or dict(receipt.get("branch_summary", {})) != branch
            or receipt.get("route_payload_sha256") != route_sha
            or receipt.get("population_sha256") != telemetry["population_sha256"]
        ):
            raise ValueError("residual-centering accuracy replay is not reproducible")
        recomputed[replay] = {
            "paths": paths,
            "metrics": metrics,
            "telemetry": telemetry,
            "branch": branch,
            "route_sha": route_sha,
        }

    left = recomputed["accuracy_a"]
    right = recomputed["accuracy_b"]
    prediction_comparison = compare_prediction_artifacts(
        left["paths"]["prediction"], right["paths"]["prediction"]
    )
    if (
        dict(comparison["prediction"]) != prediction_comparison
        or prediction_comparison["raw_sha256_parity"] is not True
        or left["route_sha"] != right["route_sha"]
        or left["metrics"] != right["metrics"]
        or left["telemetry"]["population_sha256"]
        != right["telemetry"]["population_sha256"]
        or left["branch"] != right["branch"]
    ):
        raise ValueError("residual-centering duplicate accuracy is not exact")
    if variant == "residual_window_center":
        reachability = residual_centering_reachability_gate(left["telemetry"])
        if not reachability["passed"] or result.get("reachability_gate") != reachability:
            raise ValueError("residual-centering treatment lost role reachability")
    elif result.get("reachability_gate") is not None:
        raise ValueError("residual-centering control must not claim treatment reachability")
    return result


def finalize_residual_centering_training(
    stage_results: Mapping[str, Mapping[str, Any]],
    *,
    expected_commit: str,
    expected_job_ids: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the pre-registered single-seed accuracy screen, fail closed."""

    validated: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER:
        try:
            stage = validate_residual_centering_training_stage_result(
                stage_results[variant],
                expected_variant=variant,
                expected_commit=expected_commit,
            )
            if expected_job_ids is not None and str(stage.get("slurm_job_id")) != str(
                expected_job_ids[variant]
            ):
                raise ValueError("stage result belongs to another Slurm job")
            validated[variant] = stage
        except (AttributeError, KeyError, OverflowError, TypeError, ValueError) as error:
            errors[variant] = str(error)
    complete = not errors and set(validated) == set(
        RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    )
    contrasts: dict[str, float] = {}
    conditions: dict[str, bool] = {}
    if complete:
        none = validated["none_control"]
        center = validated["residual_window_center"]
        if (
            none["binding"]["shared_protocol_sha256"]
            != center["binding"]["shared_protocol_sha256"]
        ):
            complete = False
            errors["matched_protocol"] = "cross-variant shared protocol hash mismatch"
        else:
            none_metrics = none["accuracy_replays"]["accuracy_a"]["metrics"]
            center_metrics = center["accuracy_replays"]["accuracy_a"]["metrics"]
            contrasts = {
                key: float(center_metrics[key]) - float(none_metrics[key])
                for key in none_metrics
            }
            conditions = {
                "mAP@0.6_strictly_improved": contrasts["mAP@0.6"] > 0.0,
                "mAP@0.7_strictly_improved": contrasts["mAP@0.7"] > 0.0,
                "average_mAP_non_degraded": contrasts["average_mAP"] >= 0.0,
            }
    passed = complete and all(conditions.values())
    result: dict[str, Any] = {
        "schema_version": RESIDUAL_CENTERING_TRAINING_FINALIZATION_SCHEMA,
        "study_id": RESIDUAL_CENTERING_TRAINING_STUDY_ID,
        "status": (
            "PASS_ACCURACY_SCREEN_PAIRED_COST_AUTHORIZED"
            if passed
            else "HOLD_COMPLETE_ACCURACY_SCREEN_NO_COST"
            if complete
            else "FAIL_INCOMPLETE_EMPTY_CONTRAST_NO_COST"
        ),
        "decision": (
            "RUN_SAME_GPU_ABBA_BAAB_FULL_STACK_COST"
            if passed
            else "HOLD_RESIDUAL_CENTERING_NO_COST"
            if complete
            else "INCOMPLETE_NO_INFERENCE"
        ),
        "runtime_commit": expected_commit,
        "expected_stage_job_ids": (
            {
                variant: str(expected_job_ids[variant])
                for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
            }
            if expected_job_ids is not None
            else None
        ),
        "validated_variants": sorted(validated),
        "errors": errors,
        "accuracy_screen_conditions": conditions if complete else {},
        "center_minus_none_metrics_pp": contrasts if complete else {},
        "paired_cost_authorized": passed,
        "paired_cost_protocol": (
            "same_gpu_counterbalanced_ABBA_plus_BAAB_full_stack"
            if passed
            else None
        ),
        "seeds_3408_3409_opened": False,
        "additional_seed_gate": "requires_accuracy_screen_and_paired_cost_pass",
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    result["finalization_sha256"] = canonical_sha256(result)
    return result


def validate_frozen_residual_centering_training_contract() -> None:
    if RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER != (
        "none_control",
        "residual_window_center",
    ):
        raise RuntimeError("residual-centering matched-training variant order changed")
    modes = tuple(
        RESIDUAL_CENTERING_TRAINING_VARIANTS[variant]["branch_calibration_mode"]
        for variant in RESIDUAL_CENTERING_TRAINING_VARIANT_ORDER
    )
    if modes != ("none", "residual_window_center"):
        raise RuntimeError("residual-centering matched-training intervention changed")
    if RESIDUAL_CENTERING_EXPECTED_SUCCESSFUL_UPDATES != 9600:
        raise RuntimeError("residual-centering successful-update target changed")


validate_frozen_residual_centering_training_contract()
