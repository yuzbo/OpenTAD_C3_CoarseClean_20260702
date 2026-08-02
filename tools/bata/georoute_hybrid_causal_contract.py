"""Frozen contract for the Hybrid-centered GeoRoute causal pilot.

The study is a development-only, one-seed screen.  It can admit a separately
frozen confirmatory study, but it cannot select an official-test model or create
a paper/efficiency claim.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Mapping

from tools.bata.georoute_experiment_contract import (
    assert_development_annotation,
    canonical_sha256,
    load_development_manifest,
    sha256_file,
)


HYBRID_CAUSAL_STUDY_ID = "georoute_hybrid_causal_pilot_v1"
HYBRID_CAUSAL_CONTRACT_SCHEMA = "georoute_hybrid_causal_contract_v1"
HYBRID_CAUSAL_STAGE_RESULT_SCHEMA = "georoute_hybrid_causal_stage_result_v1"
HYBRID_CAUSAL_FINALIZATION_SCHEMA = "georoute_hybrid_causal_finalization_v1"
HYBRID_CAUSAL_P0_SUITE_SCHEMA = "georoute_hybrid_causal_p0_suite_v1"
HYBRID_CAUSAL_DEPLOYMENT_SCHEMA = "georoute_hybrid_causal_deployment_v1"
HYBRID_CAUSAL_SEED_DERIVATION = "sha256_first8_mod10000_v1"
HYBRID_CAUSAL_SEED = 5227
HYBRID_CAUSAL_EPOCHS = 20
HYBRID_CAUSAL_K = 64
HYBRID_CAUSAL_ROUTE_RNG_SCHEMA = "georoute_route_private_rng_v1"
HYBRID_CAUSAL_DETECTOR_RISK_KEYS = ("cls_loss", "reg_loss")


def _derived_seed(study_id: str) -> int:
    prefix = hashlib.sha256(study_id.encode("utf-8")).hexdigest()[:8]
    return int(prefix, 16) % 10000


def _arm(
    *,
    arm_id: str,
    slug: str,
    route_mode: str,
    estimator: str,
    context: int,
    roi: int,
    residual: int,
    geometry_shift: int = 0,
    intervention: str,
) -> dict[str, Any]:
    return {
        "arm_id": arm_id,
        "slug": slug,
        "route_mode": route_mode,
        "policy_estimator": estimator,
        "context_tokens": int(context),
        "roi_tokens": int(roi),
        "residual_tokens": int(residual),
        "selected_tokens": (
            "all_valid" if route_mode == "dense" else HYBRID_CAUSAL_K
        ),
        "geometry_temporal_shift_tubelets": int(geometry_shift),
        "absolute_position_enabled": True,
        "absolute_coordinates_enabled": False,
        "roi_relative_coordinates_enabled": False,
        "geometry_projection_enabled": False,
        "geometry_side_channel": False,
        "pooling_mode": "uniform_selected",
        "geometry_smoothness_weight": 0.0,
        "area_prior_weight": 0.0,
        "policy_temperature": 0.7,
        "score_function_temporal_reduction": "mean",
        "route_study_seed": HYBRID_CAUSAL_SEED,
        "support_only": True,
        "intervention": intervention,
    }


HYBRID_CAUSAL_ARM_SPECS: dict[str, dict[str, Any]] = {
    "dense_native": _arm(
        arm_id="A0",
        slug="a0_dense",
        route_mode="dense",
        estimator="none",
        context=0,
        roi=0,
        residual=0,
        intervention="all_220_valid_native_tokens",
    ),
    "fixed_lattice_k64": _arm(
        arm_id="A1",
        slug="a1_fixed64",
        route_mode="uniform",
        estimator="none",
        context=0,
        roi=0,
        residual=0,
        intervention="deterministic_row_major_uniform_k64",
    ),
    "random_lattice_k64": _arm(
        arm_id="A2",
        slug="a2_random64",
        route_mode="random",
        estimator="none",
        context=0,
        roi=0,
        residual=0,
        intervention="stateless_data_independent_random_k64",
    ),
    "residual_pl_k64_support_only": _arm(
        arm_id="A3",
        slug="a3_respl64",
        route_mode="structured_context_residual",
        estimator="score_function",
        context=0,
        roi=0,
        residual=64,
        intervention="unstructured_residual_sequential_pl_k64",
    ),
    "context8_residual56_pl_support_only": _arm(
        arm_id="A4",
        slug="a4_ctx8res56",
        route_mode="structured_context_residual",
        estimator="score_function",
        context=8,
        roi=0,
        residual=56,
        intervention="replace_eight_residual_slots_with_deterministic_context",
    ),
    "context8_roi56_pl_support_only": _arm(
        arm_id="A5",
        slug="a5_ctx8roi56",
        route_mode="structured_context_roi",
        estimator="score_function",
        context=8,
        roi=56,
        residual=0,
        intervention="continuous_geometry_roi_family_without_residual",
    ),
    "hybrid_ctx8_roi28_res28_st_support_only": _arm(
        arm_id="A6",
        slug="a6_hybrid_st",
        route_mode="structured_hybrid",
        estimator="straight_through",
        context=8,
        roi=28,
        residual=28,
        intervention="branch_aligned_st_estimator_control",
    ),
    "hybrid_ctx8_roi28_res28_pl_support_only": _arm(
        arm_id="A7",
        slug="a7_hybrid_pl",
        route_mode="structured_hybrid",
        estimator="score_function",
        context=8,
        roi=28,
        residual=28,
        intervention="main_sequential_conditional_pl_candidate",
    ),
    "hybrid_ctx8_roi28_res28_pl_geometry_shift127": _arm(
        arm_id="A8",
        slug="a8_hybrid_shift127",
        route_mode="structured_hybrid_geometry_shift",
        estimator="score_function",
        context=8,
        roi=28,
        residual=28,
        geometry_shift=127,
        intervention="cyclic_geometry_trajectory_content_misalignment_control",
    ),
}
HYBRID_CAUSAL_ARM_ORDER = tuple(HYBRID_CAUSAL_ARM_SPECS)

HYBRID_CAUSAL_CONTRASTS = {
    "pl_residual_vs_fixed": (
        "residual_pl_k64_support_only",
        "fixed_lattice_k64",
    ),
    "pl_residual_vs_random": (
        "residual_pl_k64_support_only",
        "random_lattice_k64",
    ),
    "context_floor_increment": (
        "context8_residual56_pl_support_only",
        "residual_pl_k64_support_only",
    ),
    "roi_family_vs_context_residual": (
        "context8_roi56_pl_support_only",
        "context8_residual56_pl_support_only",
    ),
    "hybrid_vs_context_residual": (
        "hybrid_ctx8_roi28_res28_pl_support_only",
        "context8_residual56_pl_support_only",
    ),
    "hybrid_vs_context_roi": (
        "hybrid_ctx8_roi28_res28_pl_support_only",
        "context8_roi56_pl_support_only",
    ),
    "pl_vs_st_within_hybrid": (
        "hybrid_ctx8_roi28_res28_pl_support_only",
        "hybrid_ctx8_roi28_res28_st_support_only",
    ),
    "aligned_vs_geometry_shift": (
        "hybrid_ctx8_roi28_res28_pl_support_only",
        "hybrid_ctx8_roi28_res28_pl_geometry_shift127",
    ),
    "hybrid_vs_dense": (
        "hybrid_ctx8_roi28_res28_pl_support_only",
        "dense_native",
    ),
}


def hybrid_causal_arm_spec(name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(HYBRID_CAUSAL_ARM_SPECS[name])
    except KeyError as exc:
        raise ValueError(f"unsupported Hybrid causal arm {name!r}") from exc


def hybrid_causal_cell_relative_path(*, arm: str, seed: int) -> Path:
    if int(seed) != HYBRID_CAUSAL_SEED:
        raise ValueError("Hybrid causal pilot permits only its frozen seed")
    spec = hybrid_causal_arm_spec(arm)
    return Path("pilot") / f"{spec['slug']}_{arm}" / f"seed{seed}"


def validate_frozen_hybrid_causal_contract() -> None:
    if _derived_seed(HYBRID_CAUSAL_STUDY_ID) != HYBRID_CAUSAL_SEED:
        raise RuntimeError("Hybrid causal seed derivation changed")
    if tuple(spec["arm_id"] for spec in HYBRID_CAUSAL_ARM_SPECS.values()) != tuple(
        f"A{index}" for index in range(9)
    ):
        raise RuntimeError("Hybrid causal arm order changed")
    if len({spec["slug"] for spec in HYBRID_CAUSAL_ARM_SPECS.values()}) != 9:
        raise RuntimeError("Hybrid causal arm slugs must be unique")
    for name, spec in HYBRID_CAUSAL_ARM_SPECS.items():
        if (
            spec["absolute_position_enabled"] is not True
            or spec["absolute_coordinates_enabled"] is not False
            or spec["roi_relative_coordinates_enabled"] is not False
            or spec["geometry_projection_enabled"] is not False
            or spec["geometry_side_channel"] is not False
            or spec["pooling_mode"] != "uniform_selected"
            or spec["geometry_smoothness_weight"] != 0.0
            or spec["area_prior_weight"] != 0.0
            or spec["policy_temperature"] != 0.7
            or spec["score_function_temporal_reduction"] != "mean"
            or spec["support_only"] is not True
        ):
            raise RuntimeError(f"support-only invariant changed for {name}")
        if spec["route_mode"] not in {"dense", "uniform", "random"} and (
            spec["context_tokens"]
            + spec["roi_tokens"]
            + spec["residual_tokens"]
            != HYBRID_CAUSAL_K
        ):
            raise RuntimeError(f"structured quota changed for {name}")
    shift = HYBRID_CAUSAL_ARM_SPECS[
        "hybrid_ctx8_roi28_res28_pl_geometry_shift127"
    ]
    if shift["geometry_temporal_shift_tubelets"] != 127:
        raise RuntimeError("geometry-shift control changed")


def bind_hybrid_causal_config(
    *,
    source_config_path: str | Path,
    arm: str,
    seed: int,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
):
    """Bind one immutable, development-only arm config."""

    from mmengine.config import Config

    validate_frozen_hybrid_causal_contract()
    if int(seed) != HYBRID_CAUSAL_SEED:
        raise ValueError("Hybrid causal seed differs from its frozen derivation")
    spec = hybrid_causal_arm_spec(arm)
    source_config_path = Path(source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(development_annotation_path).resolve()
    class_map_path = Path(class_map_path).resolve()
    video_root = Path(development_video_root).resolve()
    pretrained_path = Path(pretrained_checkpoint_path).resolve()
    manifest = load_development_manifest(manifest_path)
    annotation = assert_development_annotation(annotation_path)
    required_ids = set(manifest["splits"]["fit"]) | set(manifest["splits"]["gate"])
    if not required_ids <= set(annotation["video_ids"]):
        raise ValueError("Hybrid causal manifest names absent development videos")
    for path in (source_config_path, class_map_path, pretrained_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not video_root.is_dir() or any(
        component.lower() in {"test", "testing", "test_videos", "official_test"}
        for component in video_root.parts
    ):
        raise ValueError("Hybrid causal video root must be a non-test directory")

    cfg = Config.fromfile(str(source_config_path))
    for split_name, block_list in (
        ("train", manifest["splits"]["gate"]),
        ("val", manifest["splits"]["fit"]),
        ("test", manifest["splits"]["fit"]),
    ):
        split = cfg.dataset[split_name]
        split.ann_file = annotation["path"]
        split.class_map = str(class_map_path)
        split.data_path = str(video_root)
        split.subset_name = "training"
        split.block_list = list(block_list)
    cfg.dataset.test.test_mode = True
    cfg.evaluation.ground_truth_filename = annotation["path"]
    cfg.evaluation.subset = "training"

    custom = cfg.model.backbone.custom
    custom.pretrain = str(pretrained_path)
    custom.georoute_route_mode = spec["route_mode"]
    custom.georoute_policy_estimator = spec["policy_estimator"]
    custom.georoute_tokens_per_tubelet = HYBRID_CAUSAL_K
    custom.georoute_context_tokens = spec["context_tokens"]
    custom.georoute_structured_context_tokens = spec["context_tokens"]
    custom.georoute_structured_roi_tokens = spec["roi_tokens"]
    custom.georoute_structured_residual_tokens = spec["residual_tokens"]
    custom.georoute_geometry_temporal_shift_tubelets = spec[
        "geometry_temporal_shift_tubelets"
    ]
    custom.georoute_absolute_position_enabled = True
    custom.georoute_absolute_coordinates_enabled = False
    custom.georoute_roi_relative_coordinates_enabled = False
    custom.georoute_geometry_projection_enabled = False
    custom.georoute_geometry_side_channel = False
    custom.georoute_pooling_mode = "uniform_selected"
    custom.georoute_geometry_smoothness_weight = 0.0
    custom.georoute_area_prior_weight = 0.0
    custom.georoute_policy_temperature = 0.7
    custom.georoute_score_function_weight = 1.0
    custom.georoute_score_function_baseline_momentum = 0.95
    custom.georoute_score_function_temporal_reduction = "mean"
    custom.georoute_route_study_seed = HYBRID_CAUSAL_SEED
    custom.georoute_random_seed = HYBRID_CAUSAL_SEED
    custom.georoute_diagnostic_telemetry_enabled = True
    custom.georoute_max_batch_size = 1

    cfg.solver.train.batch_size = 2
    cfg.solver.val.batch_size = 2
    cfg.solver.test.batch_size = 2
    cfg.solver.fp16_compress = False
    cfg.scheduler.max_epoch = HYBRID_CAUSAL_EPOCHS
    cfg.workflow.end_epoch = HYBRID_CAUSAL_EPOCHS
    cfg.workflow.val_start_epoch = HYBRID_CAUSAL_EPOCHS
    cfg.workflow.checkpoint_policy = "final_only"
    cfg.workflow.require_successful_update_hook = True
    cfg.workflow.schedule_and_ema_on_success_only = True
    cfg.workflow.max_amp_retries_per_batch = 8
    cfg.workflow.fail_on_skipped_update = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.post_processing.save_dict = True
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.work_dir = str(Path(work_dir).resolve())
    cfg.georoute_protocol.status = "hybrid_causal_exploratory_development_only"

    binding: dict[str, Any] = {
        "schema_version": HYBRID_CAUSAL_CONTRACT_SCHEMA,
        "study_id": HYBRID_CAUSAL_STUDY_ID,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": HYBRID_CAUSAL_SEED,
        "seed_derivation": HYBRID_CAUSAL_SEED_DERIVATION,
        "epochs": HYBRID_CAUSAL_EPOCHS,
        "source_config": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "fit_video_ids": manifest["splits"]["fit"],
        "gate_video_ids": manifest["splits"]["gate"],
        "development_annotation": annotation,
        "class_map_path": str(class_map_path),
        "class_map_sha256": sha256_file(class_map_path),
        "development_video_root": str(video_root),
        "pretrained_checkpoint_path": str(pretrained_path),
        "pretrained_checkpoint_sha256": sha256_file(pretrained_path),
        "work_dir": cfg.work_dir,
        "route_rng_schema": HYBRID_CAUSAL_ROUTE_RNG_SCHEMA,
        "detector_risk_keys": list(HYBRID_CAUSAL_DETECTOR_RISK_KEYS),
        "world_size": 2,
        "local_batch": 1,
        "global_batch": 2,
        "fp16_compress": False,
        "all_nine_complete_before_interpretation": True,
        "single_seed_screen_only": True,
        "old_free_first_selector_reused": False,
        "official_test_opened": False,
        "partial_survivor_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_hybrid_causal_binding = binding
    cfg.georoute_runtime_binding = binding
    cfg.georoute_telemetry_binding = dict(
        schema_version="georoute_telemetry_world_binding_v1",
        study_id=HYBRID_CAUSAL_STUDY_ID,
        arm=arm,
        seed=HYBRID_CAUSAL_SEED,
        world_size=2,
        local_batch=1,
        development_only=True,
        official_test_opened=False,
    )
    return cfg


def _finite_metrics(metrics: Any) -> bool:
    required = {
        "average_mAP",
        "mAP@0.3",
        "mAP@0.4",
        "mAP@0.5",
        "mAP@0.6",
        "mAP@0.7",
        "high_iou_composite",
    }
    return bool(
        isinstance(metrics, Mapping)
        and set(metrics) == required
        and all(
            isinstance(metrics[key], (int, float))
            and math.isfinite(float(metrics[key]))
            for key in required
        )
    )


def _expected_audit_role_counts(
    spec: Mapping[str, Any],
    *,
    expected_k: int,
) -> dict[str, int]:
    counts = {
        "context": 0,
        "roi": 0,
        "residual": 0,
        "free": 0,
        "dense": 0,
        "uniform": 0,
        "random": 0,
    }
    route_mode = str(spec["route_mode"])
    if route_mode.startswith("structured_"):
        counts.update(
            context=int(spec["context_tokens"]),
            roi=int(spec["roi_tokens"]),
            residual=int(spec["residual_tokens"]),
        )
    else:
        counts[route_mode] = int(expected_k)
    return counts


def validate_hybrid_causal_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
) -> None:
    unsigned = dict(result)
    observed_hash = unsigned.pop("stage_result_sha256", None)
    if observed_hash != canonical_sha256(unsigned):
        raise ValueError("Hybrid causal stage-result self-hash mismatch")
    arm = str(result.get("arm", ""))
    spec = hybrid_causal_arm_spec(arm)
    audit = result.get("routing_audit")
    profile = result.get("profile")
    telemetry = result.get("telemetry_summary")
    binding = result.get("binding")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("Hybrid causal result is bound to another arm")
    if (
        result.get("schema_version") != HYBRID_CAUSAL_STAGE_RESULT_SCHEMA
        or result.get("status") != "PASS_EXPLORATORY_DEVELOPMENT_ONLY"
        or result.get("study_id") != HYBRID_CAUSAL_STUDY_ID
        or int(result.get("seed", -1)) != HYBRID_CAUSAL_SEED
        or int(result.get("epochs", -1)) != HYBRID_CAUSAL_EPOCHS
        or result.get("arm_spec") != spec
        or result.get("arm_spec_sha256") != canonical_sha256(spec)
        or result.get("population_sha256") in {None, ""}
        or result.get("official_test_opened") is not False
        or result.get("partial_survivor_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
        or not _finite_metrics(result.get("metrics"))
        or not isinstance(profile, Mapping)
        or not isinstance(telemetry, Mapping)
        or not isinstance(binding, Mapping)
        or result.get("binding_sha256") != binding.get("binding_sha256")
        or result.get("binding_sha256") != canonical_sha256(
            {key: value for key, value in binding.items() if key != "binding_sha256"}
        )
        or int(binding.get("world_size", -1)) != 2
        or int(binding.get("local_batch", -1)) != 1
        or binding.get("fp16_compress") is not False
        or telemetry.get("population_sha256")
        != result.get("population_sha256")
        or telemetry.get("official_test_opened") is not False
        or telemetry.get("paper_claim_allowed") is not False
        or not isinstance(audit, Mapping)
        or int(audit.get("selected_duplicate_count", -1)) != 0
        or int(audit.get("heavy_backbone_forward_count", -1)) != 1
        or audit.get("uses_gt_for_route") is not False
        or audit.get("uses_teacher") is not False
        or audit.get("uses_oracle") is not False
        or audit.get("uses_test_evidence") is not False
    ):
        raise ValueError("Hybrid causal stage-result contract is invalid")
    if expected_commit is not None and result.get("runtime_commit") != expected_commit:
        raise ValueError("Hybrid causal stage-result commit mismatch")
    expected_k = int(audit.get("item_count", -1)) if spec["route_mode"] == "dense" else HYBRID_CAUSAL_K
    expected_roles = _expected_audit_role_counts(spec, expected_k=expected_k)
    if (
        expected_k <= 0
        or int(audit.get("target_k", -1)) != expected_k
        or int(audit.get("selected_unique_count_min", -1)) != expected_k
        or int(audit.get("selected_unique_count_max", -1)) != expected_k
        or audit.get("absolute_position_enabled") is not True
        or audit.get("absolute_coordinates_enabled") is not False
        or audit.get("roi_relative_coordinates_enabled") is not False
        or audit.get("geometry_projection_enabled") is not False
        or audit.get("pooling_mode") != "uniform_selected"
        or audit.get("route_mode") != spec["route_mode"]
        or audit.get("policy_estimator") != spec["policy_estimator"]
        or audit.get("role_counts") != expected_roles
        or telemetry.get("role_counts") != expected_roles
        or int(telemetry.get("target_k", -1)) != expected_k
        or audit.get("diagnostic_telemetry_enabled") is not False
        or int(audit.get("geometry_temporal_shift_tubelets", -1))
        != int(spec["geometry_temporal_shift_tubelets"])
    ):
        raise ValueError("Hybrid causal exact-K or representation audit changed")
    if str(spec["route_mode"]).startswith("structured_"):
        route_rng = audit.get("route_rng")
        if (
            audit.get("routing_schema")
            != "georoute_fixed_quota_structured_routing_v1"
            or not isinstance(route_rng, Mapping)
            or route_rng.get("schema_version")
            != HYBRID_CAUSAL_ROUTE_RNG_SCHEMA
            or route_rng.get("global_rng_consumed") is not False
        ):
            raise ValueError("Hybrid causal structured routing/RNG audit changed")
    p50 = profile.get("model_and_postprocess_p50_ms")
    if not isinstance(p50, (int, float)) or not math.isfinite(float(p50)) or float(p50) <= 0:
        raise ValueError("Hybrid causal profile lacks finite positive p50")
    scope = profile.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("diagnostic_route_telemetry_inside_timed_forward") is not False
        or scope.get("separate_from_accuracy_evaluation") is not True
    ):
        raise ValueError("Hybrid causal cost replay is confounded by route telemetry")


def _metric_delta(
    treatment: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, float]:
    return {
        key: float(treatment["metrics"][key]) - float(control["metrics"][key])
        for key in treatment["metrics"]
    }


def finalize_hybrid_causal_study(
    results: Mapping[str, Mapping[str, Any]],
    *,
    expected_commit: str,
) -> dict[str, Any]:
    """Seal all-nine results and emit only preregistered screen contrasts."""

    failures: dict[str, str] = {}
    valid: dict[str, Mapping[str, Any]] = {}
    for arm in HYBRID_CAUSAL_ARM_ORDER:
        result = results.get(arm)
        if result is None:
            failures[arm] = "MISSING_STAGE_RESULT"
            continue
        try:
            validate_hybrid_causal_stage_result(
                result,
                expected_arm=arm,
                expected_commit=expected_commit,
            )
            valid[arm] = result
        except Exception as error:
            failures[arm] = f"INVALID_STAGE_RESULT:{type(error).__name__}:{error}"
    populations = {str(result["population_sha256"]) for result in valid.values()}
    complete = len(valid) == 9 and not failures and len(populations) == 1
    contrasts: dict[str, Any] = {}
    if complete:
        for name, (treatment, control) in HYBRID_CAUSAL_CONTRASTS.items():
            contrasts[name] = {
                "treatment": treatment,
                "control": control,
                "metric_delta": _metric_delta(valid[treatment], valid[control]),
                "interpretation": "descriptive_single_seed_screen_only",
            }

    decision = "INCOMPLETE_NO_PERFORMANCE_INFERENCE"
    admission_checks: dict[str, bool] = {}
    if complete:
        h = {
            arm: float(result["metrics"]["high_iou_composite"])
            for arm, result in valid.items()
        }
        m7 = {
            arm: float(result["metrics"]["mAP@0.7"])
            for arm, result in valid.items()
        }
        main = "hybrid_ctx8_roi28_res28_pl_support_only"
        context_residual = "context8_residual56_pl_support_only"
        context_roi = "context8_roi56_pl_support_only"
        fixed = "fixed_lattice_k64"
        random = "random_lattice_k64"
        st = "hybrid_ctx8_roi28_res28_st_support_only"
        shifted = "hybrid_ctx8_roi28_res28_pl_geometry_shift127"
        dense = "dense_native"
        admission_checks = {
            "hybrid_gt_context_residual": h[main] > h[context_residual],
            "hybrid_gt_context_roi": h[main] > h[context_roi],
            "hybrid_gt_fixed": h[main] > h[fixed],
            "hybrid_gt_random": h[main] > h[random],
            "aligned_gt_geometry_shift": h[main] > h[shifted],
            "hybrid_mAP07_not_below_simple_controls": m7[main]
            >= max(m7[fixed], m7[random], m7[context_residual], m7[context_roi], m7[shifted]),
            "pl_not_below_st": h[main] >= h[st],
            "model_postprocess_p50_below_dense": float(
                valid[main]["profile"]["model_and_postprocess_p50_ms"]
            )
            < float(valid[dense]["profile"]["model_and_postprocess_p50_ms"]),
        }
        decision = (
            "ADMIT_SEPARATELY_FROZEN_CONFIRMATORY_STUDY"
            if all(admission_checks.values())
            else "HOLD_MECHANISM_AMBIGUOUS"
        )

    finalization: dict[str, Any] = {
        "schema_version": HYBRID_CAUSAL_FINALIZATION_SCHEMA,
        "status": (
            "COMPLETE_EXPLORATORY_SCREEN" if complete else "INCOMPLETE_EXPLORATORY_SCREEN"
        ),
        "decision": decision,
        "study_id": HYBRID_CAUSAL_STUDY_ID,
        "runtime_commit": expected_commit,
        "seed": HYBRID_CAUSAL_SEED,
        "arm_order": list(HYBRID_CAUSAL_ARM_ORDER),
        "valid_arm_count": len(valid),
        "failures": failures,
        "common_population": complete,
        "population_sha256": next(iter(populations)) if complete else None,
        "descriptive_contrasts": contrasts,
        "screen_admission_checks": admission_checks,
        "multiple_comparison_adjusted_claim_allowed": False,
        "single_seed_screen_only": True,
        "official_test_opened": False,
        "partial_survivor_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    finalization["finalization_sha256"] = canonical_sha256(finalization)
    return finalization


validate_frozen_hybrid_causal_contract()
