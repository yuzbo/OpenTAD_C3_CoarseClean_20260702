"""Frozen single-seed estimator/representation pilot contract for GeoRoute.

This study is deliberately independent of the historical seven-arm
NativeTokenSelect-first selector.  It estimates four causal contrasts on the
development split only and cannot authorize P2/P3, official test, or a paper
claim.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from tools.bata.georoute_experiment_contract import (
    assert_development_annotation,
    canonical_sha256,
    load_development_manifest,
    sha256_file,
)


PILOT_STUDY_ID = "georoute_estimator_representation_pilot_v1"
PILOT_CONTRACT_SCHEMA = "georoute_estimator_representation_pilot_contract_v1"
PILOT_DEPLOYMENT_SCHEMA = "georoute_estimator_representation_pilot_deployment_v2"
PILOT_STAGE_RESULT_SCHEMA = "georoute_estimator_representation_pilot_stage_result_v1"
PILOT_P0_SUITE_SCHEMA = "georoute_estimator_representation_pilot_p0_suite_v1"
PILOT_P0_FAILURE_SCHEMA = "georoute_estimator_representation_pilot_p0_failure_v1"
PILOT_FINALIZATION_SCHEMA = "georoute_estimator_representation_pilot_finalization_v2"
PILOT_SEED = 3407
PILOT_EPOCHS = 20
PILOT_K = 64
PILOT_POLICY_TEMPERATURE = 0.7
PILOT_SCORE_FUNCTION_WEIGHT = 1.0
PILOT_SCORE_FUNCTION_BASELINE_MOMENTUM = 0.95

REPRESENTATION_KEYS = (
    "absolute_coordinates_enabled",
    "roi_relative_coordinates_enabled",
    "geometry_projection_enabled",
)


def _arm(
    *,
    slug: str,
    route_mode: str,
    policy_estimator: str,
    representation_enabled: bool,
    geometry_side_channel: bool,
    roi_fraction: float,
    learned_geometry_enabled: bool,
    learned_residual_enabled: bool,
    causal_role: str,
) -> dict[str, Any]:
    return {
        "slug": slug,
        "route_mode": route_mode,
        "policy_estimator": policy_estimator,
        "tokens_per_tubelet": PILOT_K,
        "context_tokens": 0,
        "roi_fraction": float(roi_fraction),
        "geometry_side_channel": bool(geometry_side_channel),
        "absolute_position_enabled": True,
        "absolute_coordinates_enabled": bool(representation_enabled),
        "roi_relative_coordinates_enabled": bool(representation_enabled),
        "geometry_projection_enabled": bool(representation_enabled),
        "learned_geometry_enabled": bool(learned_geometry_enabled),
        "learned_residual_enabled": bool(learned_residual_enabled),
        "policy_temperature": PILOT_POLICY_TEMPERATURE,
        "score_function_weight": PILOT_SCORE_FUNCTION_WEIGHT,
        "score_function_baseline_momentum": (
            PILOT_SCORE_FUNCTION_BASELINE_MOMENTUM
        ),
        "geometry_smoothness_weight": 0.0,
        "area_prior_weight": 0.0,
        "pooling_mode": "uniform_selected",
        "adapter_mode": "coordinate_lineage_packed",
        "causal_role": causal_role,
    }


PILOT_ARMS: dict[str, dict[str, Any]] = {
    "residual_st_rep_off": _arm(
        slug="rst_off",
        route_mode="free",
        policy_estimator="straight_through",
        representation_enabled=False,
        geometry_side_channel=False,
        roi_fraction=0.0,
        learned_geometry_enabled=False,
        learned_residual_enabled=True,
        causal_role="estimator_baseline",
    ),
    "residual_pl_rep_off": _arm(
        slug="rpl_off",
        route_mode="free",
        policy_estimator="score_function",
        representation_enabled=False,
        geometry_side_channel=False,
        roi_fraction=0.0,
        learned_geometry_enabled=False,
        learned_residual_enabled=True,
        causal_role="estimator_intervention",
    ),
    "fixed_rep_off": _arm(
        slug="fix_off",
        route_mode="uniform",
        policy_estimator="none",
        representation_enabled=False,
        geometry_side_channel=False,
        roi_fraction=0.0,
        learned_geometry_enabled=False,
        learned_residual_enabled=False,
        causal_role="fixed_support_representation_baseline",
    ),
    "fixed_rep_on": _arm(
        slug="fix_on",
        route_mode="uniform",
        policy_estimator="none",
        representation_enabled=True,
        geometry_side_channel=True,
        roi_fraction=0.0,
        learned_geometry_enabled=True,
        learned_residual_enabled=False,
        causal_role="fixed_support_representation_intervention",
    ),
    "roi_pl_rep_off": _arm(
        slug="roi_off",
        route_mode="roi",
        policy_estimator="score_function",
        representation_enabled=False,
        geometry_side_channel=False,
        roi_fraction=1.0,
        learned_geometry_enabled=True,
        learned_residual_enabled=False,
        causal_role="learned_support_representation_baseline",
    ),
    "roi_pl_rep_on": _arm(
        slug="roi_on",
        route_mode="roi",
        policy_estimator="score_function",
        representation_enabled=True,
        geometry_side_channel=False,
        roi_fraction=1.0,
        learned_geometry_enabled=True,
        learned_residual_enabled=False,
        causal_role="learned_support_representation_intervention",
    ),
}
PILOT_ARM_ORDER = tuple(PILOT_ARMS)

PILOT_CONTRASTS = {
    "estimator_pl_minus_st_rep_off": (
        "residual_pl_rep_off",
        "residual_st_rep_off",
    ),
    "fixed_representation_on_minus_off": ("fixed_rep_on", "fixed_rep_off"),
    "roi_representation_on_minus_off": ("roi_pl_rep_on", "roi_pl_rep_off"),
    "roi_support_minus_residual_support_pl_rep_off": (
        "roi_pl_rep_off",
        "residual_pl_rep_off",
    ),
}


def validate_pilot_job_receipt(
    jobs: Any,
    *,
    expected_p0_finalizer: str | None = None,
) -> dict[str, Any]:
    """Normalize a JSON-sorted pilot job map without relying on key order."""

    if not isinstance(jobs, Mapping):
        raise ValueError("estimator pilot job receipt is not a mapping")
    p0 = jobs.get("p0")
    stage = jobs.get("stage")
    p0_finalizer = str(jobs.get("p0_finalizer", ""))
    if (
        not isinstance(p0, Mapping)
        or not isinstance(stage, Mapping)
        or set(p0) != set(PILOT_ARM_ORDER)
        or set(stage) != set(PILOT_ARM_ORDER)
        or not p0_finalizer.isdigit()
    ):
        raise ValueError("estimator pilot job receipt has the wrong arm set")
    normalized_p0 = {arm: str(p0[arm]) for arm in PILOT_ARM_ORDER}
    normalized_stage = {arm: str(stage[arm]) for arm in PILOT_ARM_ORDER}
    all_ids = [
        *normalized_p0.values(),
        p0_finalizer,
        *normalized_stage.values(),
    ]
    if any(not job_id.isdigit() for job_id in all_ids):
        raise ValueError("estimator pilot job receipt contains a nonnumeric job ID")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("estimator pilot job receipt reuses a Slurm job ID")
    if (
        expected_p0_finalizer is not None
        and p0_finalizer != str(expected_p0_finalizer)
    ):
        raise ValueError("estimator pilot P0 finalizer job ID is not self-bound")
    return {
        "p0": normalized_p0,
        "p0_finalizer": p0_finalizer,
        "stage": normalized_stage,
    }


def pilot_arm_spec(name: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(PILOT_ARMS[name])
    except KeyError as exc:
        raise ValueError(f"unsupported estimator pilot arm {name!r}") from exc


def pilot_cell_relative_path(*, arm: str, seed: int) -> Path:
    spec = pilot_arm_spec(arm)
    if int(seed) != PILOT_SEED:
        raise ValueError("estimator pilot permits only its frozen exploratory seed")
    return Path("pilot") / f"{spec['slug']}_{arm}" / f"seed{PILOT_SEED}"


def validate_frozen_pilot_contract() -> None:
    if tuple(PILOT_ARMS) != (
        "residual_st_rep_off",
        "residual_pl_rep_off",
        "fixed_rep_off",
        "fixed_rep_on",
        "roi_pl_rep_off",
        "roi_pl_rep_on",
    ):
        raise RuntimeError("estimator pilot arm order changed")
    if len({spec["slug"] for spec in PILOT_ARMS.values()}) != len(PILOT_ARMS):
        raise RuntimeError("estimator pilot arm slugs must be unique")
    for name, spec in PILOT_ARMS.items():
        if (
            spec["tokens_per_tubelet"] != PILOT_K
            or spec["context_tokens"] != 0
            or spec["absolute_position_enabled"] is not True
            or spec["pooling_mode"] != "uniform_selected"
            or spec["adapter_mode"] != "coordinate_lineage_packed"
            or spec["geometry_smoothness_weight"] != 0.0
            or spec["area_prior_weight"] != 0.0
        ):
            raise RuntimeError(f"shared estimator pilot invariant changed: {name}")
        representation_values = {bool(spec[key]) for key in REPRESENTATION_KEYS}
        if len(representation_values) != 1:
            raise RuntimeError(f"representation channels are not jointly isolated: {name}")
        if spec["route_mode"] == "free":
            if (
                spec["policy_estimator"]
                not in {"straight_through", "score_function"}
                or spec["roi_fraction"] != 0.0
                or spec["geometry_side_channel"]
                or any(spec[key] for key in REPRESENTATION_KEYS)
                or spec["learned_geometry_enabled"]
                or not spec["learned_residual_enabled"]
            ):
                raise RuntimeError(f"invalid residual-only pilot arm: {name}")
        elif spec["route_mode"] == "uniform":
            representation_on = all(spec[key] for key in REPRESENTATION_KEYS)
            if (
                spec["policy_estimator"] != "none"
                or spec["roi_fraction"] != 0.0
                or spec["geometry_side_channel"] is not representation_on
                or spec["learned_geometry_enabled"] is not representation_on
                or spec["learned_residual_enabled"]
            ):
                raise RuntimeError(f"invalid fixed-support pilot arm: {name}")
        elif spec["route_mode"] == "roi":
            if (
                spec["policy_estimator"] != "score_function"
                or spec["roi_fraction"] != 1.0
                or spec["geometry_side_channel"]
                or not spec["learned_geometry_enabled"]
                or spec["learned_residual_enabled"]
            ):
                raise RuntimeError(f"invalid ROI-support pilot arm: {name}")
        else:
            raise RuntimeError(f"unexpected estimator pilot route mode: {name}")


def bind_pilot_config(
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
    """Create one immutable development-only estimator pilot config."""

    from mmengine.config import Config

    validate_frozen_pilot_contract()
    if int(seed) != PILOT_SEED:
        raise ValueError("estimator pilot seed differs from the frozen seed")
    spec = pilot_arm_spec(arm)
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
        raise ValueError("estimator pilot manifest names absent development videos")
    for path in (source_config_path, class_map_path, pretrained_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    forbidden_video_root_components = {
        "test",
        "testing",
        "test_videos",
        "official_test",
    }
    if not video_root.is_dir() or any(
        component.lower() in forbidden_video_root_components
        for component in video_root.parts
    ):
        raise ValueError("estimator pilot video root must exist and must not be a test root")

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
    custom.georoute_tokens_per_tubelet = PILOT_K
    custom.georoute_context_tokens = spec["context_tokens"]
    custom.georoute_roi_fraction = spec["roi_fraction"]
    custom.georoute_geometry_side_channel = spec["geometry_side_channel"]
    custom.georoute_absolute_position_enabled = spec["absolute_position_enabled"]
    custom.georoute_absolute_coordinates_enabled = spec[
        "absolute_coordinates_enabled"
    ]
    custom.georoute_roi_relative_coordinates_enabled = spec[
        "roi_relative_coordinates_enabled"
    ]
    custom.georoute_geometry_projection_enabled = spec[
        "geometry_projection_enabled"
    ]
    custom.georoute_policy_temperature = spec["policy_temperature"]
    custom.georoute_score_function_weight = spec["score_function_weight"]
    custom.georoute_score_function_baseline_momentum = spec[
        "score_function_baseline_momentum"
    ]
    custom.georoute_geometry_smoothness_weight = spec[
        "geometry_smoothness_weight"
    ]
    custom.georoute_area_prior_weight = spec["area_prior_weight"]
    custom.georoute_random_seed = PILOT_SEED
    custom.georoute_pooling_mode = spec["pooling_mode"]
    custom.georoute_adapter_mode = spec["adapter_mode"]
    custom.georoute_diagnostic_telemetry_enabled = True

    cfg.scheduler.max_epoch = PILOT_EPOCHS
    cfg.workflow.end_epoch = PILOT_EPOCHS
    cfg.workflow.val_start_epoch = PILOT_EPOCHS
    cfg.workflow.checkpoint_policy = "final_only"
    cfg.work_dir = str(Path(work_dir).resolve())
    cfg.post_processing.save_dict = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_protocol.status = "exploratory_single_seed_pilot"

    binding: dict[str, Any] = {
        "schema_version": PILOT_CONTRACT_SCHEMA,
        "study_id": PILOT_STUDY_ID,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": PILOT_SEED,
        "epochs": PILOT_EPOCHS,
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
        "single_seed_exploratory": True,
        "old_selector_reused": False,
        "selector_emitted": False,
        "p2_p3_opened": False,
        "official_test_opened": False,
        "manual_roi_used": False,
        "gt_for_route_used": False,
        "teacher_for_route_used": False,
        "raw_prediction_cache_used": False,
        "checkpoint_policy": "final_only_atomic",
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_estimator_pilot_binding = binding
    cfg.georoute_runtime_binding = binding
    return cfg


validate_frozen_pilot_contract()
