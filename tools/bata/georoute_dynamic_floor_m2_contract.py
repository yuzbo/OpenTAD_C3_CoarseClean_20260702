"""Frozen development-only contract for dynamic SCNR ROI-floor M2.

M2 compares only the one-cell and two-cell native ROI floors.  Training and
accuracy replay are single-seed mechanism evidence.  Cost is measured later in
one counterbalanced, same-GPU full-stack replay.  No M2 artifact may select an
official-test model or support a paper claim.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.georoute_experiment_contract import (
    assert_development_annotation,
    canonical_sha256,
    load_development_manifest,
    sha256_file,
)


DYNAMIC_FLOOR_M2_STUDY_ID = "scnr_dynamic_floor_m2_v1"
DYNAMIC_FLOOR_M2_CONTRACT_SCHEMA = "scnr_dynamic_floor_m2_contract_v1"
DYNAMIC_FLOOR_M2_STAGE_RESULT_SCHEMA = "scnr_dynamic_floor_m2_stage_result_v1"
DYNAMIC_FLOOR_M2_COST_SCHEMA = "scnr_dynamic_floor_m2_full_stack_cost_v1"
DYNAMIC_FLOOR_M2_FINALIZATION_SCHEMA = "scnr_dynamic_floor_m2_finalization_v1"
DYNAMIC_FLOOR_M2_DEPLOYMENT_SCHEMA = "scnr_dynamic_floor_m2_deployment_v1"
DYNAMIC_FLOOR_M2_ROLE_REPLAY_SCHEMA = "georoute_phase_m_diagnostic_replay_v1"
DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA = (
    "georoute_role_instrumentation_neutrality_pair_v1"
)
DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA = (
    "georoute_role_instrumentation_strict_triplet_binding_v1"
)
DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA = (
    "scnr_residual_window_centering_probe_binding_v1"
)
DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA = (
    "scnr_dynamic_floor_m2_checkpoint_sidecar_v1"
)
DYNAMIC_FLOOR_M2_SEED = 3407
DYNAMIC_FLOOR_M2_EPOCHS = 60
DYNAMIC_FLOOR_M2_WORLD_SIZE = 1
DYNAMIC_FLOOR_M2_BATCH_SIZE = 1
DYNAMIC_FLOOR_M2_WINDOW_BUDGET = 384 * 64
DYNAMIC_FLOOR_M2_SOURCE_GRID_HW = (11, 20)
DYNAMIC_FLOOR_M2_COST_ORDER = (
    "native_1cell_main",
    "native_2cell_sensitivity",
    "native_2cell_sensitivity",
    "native_1cell_main",
)

DYNAMIC_FLOOR_M2_ARMS: dict[str, dict[str, Any]] = {
    "native_1cell_main": {
        "arm_id": "G1",
        "slug": "g1_native_1cell_main",
        "roi_extent_floor_mode": "native_cells",
        "roi_extent_floor_cells": 1,
        "role": "main_less_prior_floor",
    },
    "native_2cell_sensitivity": {
        "arm_id": "G2",
        "slug": "g2_native_2cell_sensitivity",
        "roi_extent_floor_mode": "native_cells",
        "roi_extent_floor_cells": 2,
        "role": "matched_floor_sensitivity",
    },
}
DYNAMIC_FLOOR_M2_ARM_ORDER = tuple(DYNAMIC_FLOOR_M2_ARMS)


def dynamic_floor_m2_arm_spec(arm: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(DYNAMIC_FLOOR_M2_ARMS[arm])
    except KeyError as error:
        raise ValueError(f"unsupported dynamic floor M2 arm {arm!r}") from error


def dynamic_floor_m2_cell_relative_path(*, arm: str, seed: int) -> Path:
    if int(seed) != DYNAMIC_FLOOR_M2_SEED:
        raise ValueError("dynamic floor M2 permits only seed 3407")
    spec = dynamic_floor_m2_arm_spec(arm)
    return Path("development") / str(spec["slug"]) / f"seed{seed}"


def require_clean_dynamic_floor_m2_checkout(
    *, expected_commit: str, root: str | Path
) -> None:
    """Fail closed unless HEAD, the frozen origin ref, and the worktree agree."""

    root = Path(root).resolve()
    expected_commit = str(expected_commit).lower()
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ValueError("dynamic floor M2 expected commit must be a full SHA")
    head = (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .lower()
    )
    origin = (
        subprocess.run(
            [
                "git",
                "rev-parse",
                "refs/remotes/origin/codex/spatial-zoom-s1-audit-fix-20260715",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
        .stdout.strip()
        .lower()
    )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if head != expected_commit or origin != expected_commit or status:
        raise RuntimeError(
            "dynamic floor M2 requires one exact clean origin-matched checkout"
        )


def require_dynamic_floor_m2_world1_slurm() -> None:
    visible = [
        item for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if item
    ]
    if (
        not str(os.environ.get("SLURM_JOB_ID", "")).isdigit()
        or len(visible) != 1
        or int(os.environ.get("WORLD_SIZE", "-1")) != 1
        or int(os.environ.get("RANK", "-1")) != 0
        or int(os.environ.get("LOCAL_RANK", "-1")) != 0
    ):
        raise RuntimeError(
            "dynamic floor M2 requires one Slurm GPU and one logical cuda:0 process"
        )


def validate_frozen_dynamic_floor_m2_contract() -> None:
    if DYNAMIC_FLOOR_M2_ARM_ORDER != (
        "native_1cell_main",
        "native_2cell_sensitivity",
    ):
        raise RuntimeError("dynamic floor M2 arm order changed")
    if tuple(
        DYNAMIC_FLOOR_M2_ARMS[arm]["roi_extent_floor_cells"]
        for arm in DYNAMIC_FLOOR_M2_ARM_ORDER
    ) != (1, 2):
        raise RuntimeError("dynamic floor M2 intervention changed")
    if DYNAMIC_FLOOR_M2_COST_ORDER != (
        DYNAMIC_FLOOR_M2_ARM_ORDER[0],
        DYNAMIC_FLOOR_M2_ARM_ORDER[1],
        DYNAMIC_FLOOR_M2_ARM_ORDER[1],
        DYNAMIC_FLOOR_M2_ARM_ORDER[0],
    ):
        raise RuntimeError("dynamic floor M2 cost order is not counterbalanced")
    for spec in DYNAMIC_FLOOR_M2_ARMS.values():
        if spec["roi_extent_floor_mode"] != "native_cells" or int(
            spec["roi_extent_floor_cells"]
        ) not in {1, 2}:
            raise RuntimeError("dynamic floor M2 contains an invalid ROI floor")


def bind_dynamic_floor_m2_config(
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
    runtime_commit: str,
):
    """Bind one immutable world-one development training/evaluation config."""

    from mmengine.config import Config

    validate_frozen_dynamic_floor_m2_contract()
    if int(seed) != DYNAMIC_FLOOR_M2_SEED:
        raise ValueError("dynamic floor M2 seed changed")
    runtime_commit = str(runtime_commit).lower()
    if len(runtime_commit) != 40 or any(
        c not in "0123456789abcdef" for c in runtime_commit
    ):
        raise ValueError("dynamic floor M2 runtime commit must be a full lowercase SHA")
    spec = dynamic_floor_m2_arm_spec(arm)
    source_config = Path(source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    annotation = assert_development_annotation(development_annotation_path)
    manifest = load_development_manifest(manifest_path)
    class_map = Path(class_map_path).resolve()
    video_root = Path(development_video_root).resolve()
    pretrained = Path(pretrained_checkpoint_path).resolve()
    for path in (source_config, class_map, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not video_root.is_dir() or any(
        component.lower() in {"test", "testing", "test_videos", "official_test"}
        for component in video_root.parts
    ):
        raise ValueError("dynamic floor M2 video root must be a non-test directory")
    fit_ids = list(manifest["splits"]["fit"])
    gate_ids = list(manifest["splits"]["gate"])
    if set(fit_ids) & set(gate_ids):
        raise ValueError("dynamic floor M2 Fit and Gate populations overlap")
    if not (set(fit_ids) | set(gate_ids)) <= set(annotation["video_ids"]):
        raise ValueError("dynamic floor M2 manifest names absent development videos")

    cfg = Config.fromfile(str(source_config))
    custom = cfg.model.backbone.custom
    if (
        str(custom.georoute_route_mode) != "dynamic_scnr"
        or str(custom.georoute_policy_estimator) != "straight_through"
        or int(custom.georoute_window_token_budget) != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or str(custom.georoute_zero_carrier_mode) != "masked_zero"
    ):
        raise ValueError(
            "dynamic floor M2 source config is not the approved main route"
        )
    for split_name, block_list in (
        ("train", gate_ids),
        ("val", fit_ids),
        ("test", fit_ids),
    ):
        split = cfg.dataset[split_name]
        split.ann_file = annotation["path"]
        split.class_map = str(class_map)
        split.data_path = str(video_root)
        split.subset_name = "training"
        split.block_list = list(block_list)
    cfg.dataset.test.test_mode = True
    cfg.evaluation.ground_truth_filename = annotation["path"]
    cfg.evaluation.subset = "training"

    custom.pretrain = str(pretrained)
    custom.georoute_roi_extent_floor_mode = spec["roi_extent_floor_mode"]
    custom.georoute_roi_extent_floor_cells = int(spec["roi_extent_floor_cells"])
    custom.georoute_random_seed = int(seed)
    custom.georoute_diagnostic_telemetry_enabled = False
    custom.georoute_amp_diagnostic_enabled = False
    custom.georoute_gradient_decomposition_enabled = False
    custom.georoute_p0_dense_reference_check = False
    custom.georoute_max_batch_size = 1
    for split_name in ("train", "val", "test"):
        cfg.solver[split_name].batch_size = DYNAMIC_FLOOR_M2_BATCH_SIZE
    cfg.solver.amp = True
    cfg.solver.ema = True
    cfg.solver.fp16_compress = False
    cfg.solver.static_graph = False
    cfg.scheduler.type = "LinearWarmupCosineAnnealingLR"
    cfg.scheduler.warmup_epoch = 2
    cfg.scheduler.max_epoch = DYNAMIC_FLOOR_M2_EPOCHS
    cfg.workflow.end_epoch = DYNAMIC_FLOOR_M2_EPOCHS
    cfg.workflow.val_start_epoch = DYNAMIC_FLOOR_M2_EPOCHS
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.checkpoint_policy = "final_only"
    cfg.workflow.max_amp_retries_per_batch = 8
    cfg.workflow.fail_on_skipped_update = True
    cfg.workflow.require_successful_update_hook = True
    cfg.workflow.schedule_and_ema_on_success_only = True
    cfg.workflow.capture_amp_rng_state = True
    cfg.workflow.fail_on_nonfinite_loss = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.post_processing.save_dict = True
    cfg.georoute_development_profile = dict(enabled=False)
    cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    cfg.work_dir = str(Path(work_dir).resolve())
    cfg.georoute_protocol.status = "dynamic_floor_m2_single_seed_development_only"

    binding: dict[str, Any] = {
        "schema_version": DYNAMIC_FLOOR_M2_CONTRACT_SCHEMA,
        "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": DYNAMIC_FLOOR_M2_SEED,
        "epochs": DYNAMIC_FLOOR_M2_EPOCHS,
        "runtime_commit": runtime_commit,
        "source_config": str(source_config),
        "source_config_sha256": sha256_file(source_config),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "fit_video_ids": fit_ids,
        "gate_video_ids": gate_ids,
        "training_video_ids": fit_ids,
        "training_block_list_video_ids": gate_ids,
        "evaluation_video_ids": gate_ids,
        "evaluation_block_list_video_ids": fit_ids,
        "development_annotation": annotation,
        "class_map_path": str(class_map),
        "class_map_sha256": sha256_file(class_map),
        "development_video_root": str(video_root),
        "pretrained_checkpoint_path": str(pretrained),
        "pretrained_checkpoint_sha256": sha256_file(pretrained),
        "work_dir": cfg.work_dir,
        "world_size": DYNAMIC_FLOOR_M2_WORLD_SIZE,
        "local_batch": DYNAMIC_FLOOR_M2_BATCH_SIZE,
        "global_batch": DYNAMIC_FLOOR_M2_BATCH_SIZE,
        "window_token_budget": DYNAMIC_FLOOR_M2_WINDOW_BUDGET,
        "successful_update_schedule": True,
        "amp_retry_replays_rng": True,
        "max_amp_retries_per_batch": 8,
        "fail_on_skipped_update": True,
        "deterministic_warn_only": True,
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "fp16_compress": False,
        "accuracy_and_cost_replays_separate": True,
        "cost_profile_order": list(DYNAMIC_FLOOR_M2_COST_ORDER),
        "single_seed_descriptive_only": True,
        "official_test_opened": False,
        "gt_for_route_used": False,
        "teacher_used": False,
        "raw_prediction_cache_used": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_dynamic_floor_m2_binding = binding
    cfg.georoute_runtime_binding = binding
    cfg.georoute_telemetry_binding = dict(
        schema_version="georoute_telemetry_world_binding_v1",
        study_id=DYNAMIC_FLOOR_M2_STUDY_ID,
        arm=arm,
        seed=DYNAMIC_FLOOR_M2_SEED,
        world_size=1,
        local_batch=1,
        development_only=True,
        official_test_opened=False,
    )
    return cfg


def validate_dynamic_floor_m2_config(
    cfg: Any, *, arm: str, phase: str | None = None
) -> dict[str, Any]:
    spec = dynamic_floor_m2_arm_spec(arm)
    binding = dict(cfg.georoute_dynamic_floor_m2_binding)
    unsigned = dict(binding)
    observed_hash = unsigned.pop("binding_sha256", None)
    if observed_hash != canonical_sha256(unsigned):
        raise ValueError("dynamic floor M2 binding self-hash mismatch")
    custom = cfg.model.backbone.custom
    if (
        binding.get("schema_version") != DYNAMIC_FLOOR_M2_CONTRACT_SCHEMA
        or binding.get("study_id") != DYNAMIC_FLOOR_M2_STUDY_ID
        or binding.get("arm") != arm
        or binding.get("arm_spec") != spec
        or int(binding.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or int(binding.get("epochs", -1)) != DYNAMIC_FLOOR_M2_EPOCHS
        or int(binding.get("world_size", -1)) != 1
        or int(binding.get("max_amp_retries_per_batch", -1)) != 8
        or binding.get("fail_on_skipped_update") is not True
        or binding.get("checkpoint_policy") != "final_epoch_ema_only_atomic"
        or binding.get("checkpoint_consumer_state_key") != "state_dict_ema"
        or custom.georoute_route_mode != "dynamic_scnr"
        or custom.georoute_policy_estimator != "straight_through"
        or int(custom.georoute_window_token_budget) != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        or custom.georoute_roi_extent_floor_mode != "native_cells"
        or int(custom.georoute_roi_extent_floor_cells)
        != int(spec["roi_extent_floor_cells"])
        or custom.georoute_zero_carrier_mode != "masked_zero"
        or custom.georoute_absolute_position_enabled is not True
        or custom.georoute_absolute_coordinates_enabled is not False
        or custom.georoute_roi_relative_coordinates_enabled is not False
        or custom.georoute_geometry_projection_enabled is not False
        or custom.georoute_geometry_side_channel is not False
        or float(custom.georoute_geometry_smoothness_weight) != 0.0
        or float(custom.georoute_area_prior_weight) != 0.0
        or cfg.solver.fp16_compress is not False
        or cfg.solver.static_graph is not False
        or cfg.solver.amp is not True
        or cfg.solver.ema is not True
        or int(cfg.scheduler.warmup_epoch) != 2
        or int(cfg.scheduler.max_epoch) != DYNAMIC_FLOOR_M2_EPOCHS
        or int(cfg.workflow.end_epoch) != DYNAMIC_FLOOR_M2_EPOCHS
        or int(cfg.workflow.val_start_epoch) != DYNAMIC_FLOOR_M2_EPOCHS
        or cfg.workflow.schedule_and_ema_on_success_only is not True
        or cfg.workflow.fail_on_skipped_update is not True
        or cfg.workflow.capture_amp_rng_state is not True
        or cfg.inference.load_from_raw_predictions is not False
        or cfg.inference.save_raw_prediction is not False
    ):
        raise ValueError("dynamic floor M2 config violates the frozen route/recipe")
    source_receipts = (
        (binding.get("source_config"), binding.get("source_config_sha256")),
        (binding.get("manifest_path"), binding.get("manifest_file_sha256")),
        (binding.get("class_map_path"), binding.get("class_map_sha256")),
        (
            binding.get("pretrained_checkpoint_path"),
            binding.get("pretrained_checkpoint_sha256"),
        ),
        (
            binding.get("development_annotation", {}).get("path"),
            binding.get("development_annotation", {}).get("sha256"),
        ),
    )
    for raw_path, digest in source_receipts:
        if (
            not isinstance(raw_path, str)
            or not Path(raw_path).is_file()
            or sha256_file(raw_path) != digest
        ):
            raise ValueError("dynamic floor M2 source artifact receipt changed")
    if (
        set(binding.get("training_video_ids", ()))
        & set(binding.get("evaluation_video_ids", ()))
        or not Path(str(binding.get("development_video_root", ""))).is_dir()
    ):
        raise ValueError("dynamic floor M2 development population binding changed")
    if phase not in {None, "train", "accuracy", "cost"}:
        raise ValueError(f"unsupported dynamic floor M2 phase {phase!r}")
    telemetry_enabled = bool(custom.georoute_diagnostic_telemetry_enabled)
    telemetry_block_enabled = bool(cfg.georoute_diagnostic_telemetry.get("enabled"))
    profile_enabled = bool(cfg.georoute_development_profile.get("enabled"))
    if phase == "accuracy" and (
        not telemetry_enabled or not telemetry_block_enabled or profile_enabled
    ):
        raise ValueError("dynamic floor M2 accuracy must enable out-of-band telemetry")
    if phase in {"train", "cost"} and (
        telemetry_enabled or telemetry_block_enabled or profile_enabled
    ):
        raise ValueError(f"dynamic floor M2 {phase} must disable diagnostic telemetry")
    return binding


def resolve_dynamic_floor_m2_accuracy_execution_commit(
    cfg: Any,
    *,
    binding: Mapping[str, Any],
) -> str:
    """Separate a frozen M2 model binding from diagnostic-only execution code."""

    binding = dict(binding)
    source_commit = str(binding.get("runtime_commit", "")).lower()
    phase_m = cfg.get("georoute_phase_m_binding")
    if not isinstance(phase_m, Mapping) or not phase_m:
        return source_commit
    phase_m = dict(phase_m)
    phase_schema = phase_m.get("schema_version")
    runtime_commit = str(phase_m.get("runtime_commit", "")).lower()
    sha_fields = (
        "source_bound_config_sha256",
        "source_checkpoint_sha256",
        "source_prediction_sha256",
        "source_population_sha256",
    )
    custom = cfg.model.backbone.custom
    if phase_schema == DYNAMIC_FLOOR_M2_RESIDUAL_CENTERING_PROBE_SCHEMA:
        probe_mode = phase_m.get("probe_mode")
        if (
            probe_mode not in {"centered_a", "centered_b"}
            or phase_m.get("variant") != binding.get("arm")
            or int(phase_m.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
            or phase_m.get("source_experiment_commit") != source_commit
            or len(runtime_commit) != 40
            or any(character not in "0123456789abcdef" for character in runtime_commit)
            or any(
                not isinstance(phase_m.get(field), str)
                or len(phase_m[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in phase_m[field].lower()
                )
                for field in sha_fields
            )
            or not isinstance(phase_m.get("source_dataset_count"), int)
            or isinstance(phase_m.get("source_dataset_count"), bool)
            or int(phase_m["source_dataset_count"]) <= 0
            or phase_m.get("branch_calibration_mode")
            != "residual_window_center"
            or phase_m.get("branch_calibration_scope")
            != "complete_window_all_valid_candidates"
            or phase_m.get("role_calibration_telemetry_enabled") is not True
            or phase_m.get("mechanism_probe_only") is not True
            or phase_m.get("training_performed") is not False
            or phase_m.get("same_slurm_job") is not True
            or phase_m.get("same_visible_gpu") is not True
            or phase_m.get("serial_execution") is not True
            or phase_m.get("strict_deterministic_algorithms") is not True
            or phase_m.get("sdp_backend") != "math"
            or phase_m.get("tf32_enabled") is not False
            or phase_m.get("fixed_role_quota_used") is not False
            or phase_m.get("q_ctx_used") is not False
            or phase_m.get("changes_route_or_execution") is not True
            or phase_m.get("metric_evaluation_enabled") is not False
            or phase_m.get("official_test_opened") is not False
            or phase_m.get("gt_for_route_used") is not False
            or phase_m.get("teacher_for_route_used") is not False
            or phase_m.get("oracle_used") is not False
            or phase_m.get("raw_prediction_cache_used") is not False
            or custom.georoute_branch_calibration_mode
            != "residual_window_center"
            or custom.georoute_diagnostic_telemetry_enabled is not True
            or custom.georoute_role_calibration_telemetry_enabled is not True
            or cfg.georoute_diagnostic_telemetry.get("enabled") is not True
            or cfg.georoute_development_profile.get("enabled") is not False
        ):
            raise ValueError(
                "dynamic floor M2 residual-centering probe execution binding is invalid"
            )
        return runtime_commit
    if phase_schema in {
        DYNAMIC_FLOOR_M2_ROLE_NEUTRALITY_PAIR_SCHEMA,
        DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA,
    }:
        strict_triplet = phase_schema == DYNAMIC_FLOOR_M2_ROLE_STRICT_TRIPLET_SCHEMA
        pair_mode = phase_m.get("pair_mode")
        expected_role_enabled = {
            "role_off": False,
            "role_on": True,
        }.get(pair_mode)
        if (
            expected_role_enabled is None
            or phase_m.get("variant") != binding.get("arm")
            or int(phase_m.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
            or phase_m.get("source_experiment_commit") != source_commit
            or len(runtime_commit) != 40
            or any(character not in "0123456789abcdef" for character in runtime_commit)
            or any(
                not isinstance(phase_m.get(field), str)
                or len(phase_m[field]) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in phase_m[field].lower()
                )
                for field in sha_fields
            )
            or not isinstance(phase_m.get("source_dataset_count"), int)
            or isinstance(phase_m.get("source_dataset_count"), bool)
            or int(phase_m["source_dataset_count"]) <= 0
            or phase_m.get("role_calibration_telemetry_enabled")
            is not expected_role_enabled
            or phase_m.get("instrumentation_only") is not (not strict_triplet)
            or phase_m.get("same_slurm_job") is not True
            or phase_m.get("same_visible_gpu") is not True
            or phase_m.get("serial_execution") is not True
            or phase_m.get("fixed_role_quota_used") is not False
            or phase_m.get("changes_route_or_execution") is not strict_triplet
            or phase_m.get("official_test_opened") is not False
            or phase_m.get("gt_for_route_used") is not False
            or phase_m.get("teacher_for_route_used") is not False
            or phase_m.get("oracle_used") is not False
            or phase_m.get("raw_prediction_cache_used") is not False
            or custom.georoute_diagnostic_telemetry_enabled is not True
            or custom.georoute_role_calibration_telemetry_enabled
            is not expected_role_enabled
            or cfg.georoute_diagnostic_telemetry.get("enabled") is not True
            or cfg.georoute_development_profile.get("enabled") is not False
            or (
                strict_triplet
                and (
                    phase_m.get("strict_deterministic_algorithms") is not True
                    or phase_m.get("sdp_backend") != "math"
                    or phase_m.get("tf32_enabled") is not False
                    or phase_m.get("deterministic_override_changes_heavy_execution")
                    is not True
                    or phase_m.get("role_calibration_instrumentation_only") is not True
                    or phase_m.get("role_calibration_changes_route_or_execution")
                    is not False
                    or phase_m.get("source_execution_reproduced") is not False
                    or phase_m.get("strict_determinism_diagnostic_only") is not True
                )
            )
        ):
            raise ValueError(
                "dynamic floor M2 role neutrality pair execution binding is invalid"
            )
        return runtime_commit
    if phase_m.get("role_calibration_telemetry_enabled") is not True:
        return source_commit
    if (
        phase_schema != DYNAMIC_FLOOR_M2_ROLE_REPLAY_SCHEMA
        or phase_m.get("variant") != binding.get("arm")
        or int(phase_m.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or phase_m.get("source_experiment_commit") != source_commit
        or len(runtime_commit) != 40
        or any(character not in "0123456789abcdef" for character in runtime_commit)
        or any(
            not isinstance(phase_m.get(field), str)
            or len(phase_m[field]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in phase_m[field].lower()
            )
            for field in sha_fields
        )
        or not isinstance(phase_m.get("source_dataset_count"), int)
        or isinstance(phase_m.get("source_dataset_count"), bool)
        or int(phase_m["source_dataset_count"]) <= 0
        or phase_m.get("instrumentation_only") is not True
        or phase_m.get("fixed_role_quota_used") is not False
        or phase_m.get("changes_route_or_execution") is not False
        or phase_m.get("official_test_opened") is not False
        or custom.georoute_diagnostic_telemetry_enabled is not True
        or custom.georoute_role_calibration_telemetry_enabled is not True
        or cfg.georoute_diagnostic_telemetry.get("enabled") is not True
        or cfg.georoute_development_profile.get("enabled") is not False
    ):
        raise ValueError("dynamic floor M2 role replay execution binding is invalid")
    return runtime_commit


def build_dynamic_floor_m2_cost_config(stage: Mapping[str, Any], *, arm: str) -> Any:
    """Rebuild the exact configuration hashed and executed by every cost pass."""

    from mmengine.config import Config

    accuracy_path = Path(stage["config_receipts"]["accuracy"]["path"])
    cfg = Config.fromfile(str(accuracy_path))
    cfg.model.backbone.custom.georoute_diagnostic_telemetry_enabled = False
    cfg.georoute_diagnostic_telemetry = dict(enabled=False)
    cfg.georoute_development_profile = dict(enabled=False)
    cfg.solver.test.batch_size = 1
    cfg.solver.test.num_workers = 0
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.post_processing.save_dict = False
    cfg.post_processing.sliding_window = True
    validate_dynamic_floor_m2_config(cfg, arm=arm, phase="cost")
    return cfg


def build_dynamic_floor_m2_checkpoint_metadata(
    cfg: Any,
    *,
    seed: int,
    epoch: int,
    successful_updates: int,
    train_batches_per_epoch: int,
    amp_skipped_attempts: int,
    max_amp_retries_observed: int,
    optimizer_attempts: int,
    consumed_batches: int,
    replay_attempts: int,
    scheduler_advances: int,
    ema_updates: int,
    world_size: int,
) -> dict[str, Any]:
    """Build the atomic proof that all 60 development epochs really completed."""

    arm = str(cfg.georoute_dynamic_floor_m2_binding.arm)
    binding = validate_dynamic_floor_m2_config(cfg, arm=arm, phase="train")
    train_batches_per_epoch = int(train_batches_per_epoch)
    expected_updates = DYNAMIC_FLOOR_M2_EPOCHS * train_batches_per_epoch
    if (
        int(seed) != DYNAMIC_FLOOR_M2_SEED
        or int(epoch) != DYNAMIC_FLOOR_M2_EPOCHS - 1
        or int(world_size) != DYNAMIC_FLOOR_M2_WORLD_SIZE
        or train_batches_per_epoch <= 0
        or int(successful_updates) != expected_updates
        or int(consumed_batches) != expected_updates
        or int(scheduler_advances) != expected_updates
        or int(ema_updates) != expected_updates
        or int(amp_skipped_attempts) < 0
        or int(replay_attempts) != int(amp_skipped_attempts)
        or int(optimizer_attempts) != expected_updates + int(amp_skipped_attempts)
        or not 0 <= int(max_amp_retries_observed) <= 8
    ):
        raise ValueError("dynamic floor M2 checkpoint update accounting is incomplete")
    metadata: dict[str, Any] = {
        "schema_version": DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA,
        "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
        "runtime_commit": binding["runtime_commit"],
        "binding_sha256": binding["binding_sha256"],
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "arm": arm,
        "seed": int(seed),
        "epoch": int(epoch),
        "epochs": DYNAMIC_FLOOR_M2_EPOCHS,
        "world_size": int(world_size),
        "global_batch_size": DYNAMIC_FLOOR_M2_BATCH_SIZE,
        "train_batches_per_epoch": train_batches_per_epoch,
        "successful_updates": int(successful_updates),
        "consumed_batches": int(consumed_batches),
        "optimizer_attempts": int(optimizer_attempts),
        "amp_skipped_attempts": int(amp_skipped_attempts),
        "replay_attempts": int(replay_attempts),
        "max_amp_retries_per_batch": 8,
        "max_amp_retries_observed": int(max_amp_retries_observed),
        "scheduler_advances": int(scheduler_advances),
        "ema_updates": int(ema_updates),
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "checkpoint_consumer_state_key": "state_dict_ema",
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def validate_dynamic_floor_m2_checkpoint_sidecar(
    checkpoint_path: str | Path,
    *,
    binding: Mapping[str, Any] | None = None,
    cfg: Any | None = None,
) -> dict[str, Any]:
    checkpoint = Path(checkpoint_path).resolve()
    sidecar_path = Path(str(checkpoint) + ".metadata.json")
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not sidecar_path.is_file()
        or sidecar_path.is_symlink()
    ):
        raise FileNotFoundError(
            "dynamic floor M2 checkpoint and atomic metadata sidecar are required"
        )
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if not isinstance(sidecar, Mapping):
        raise ValueError("dynamic floor M2 checkpoint sidecar must be an object")
    sidecar = dict(sidecar)
    unsigned_sidecar = dict(sidecar)
    observed_sidecar_hash = unsigned_sidecar.pop("sidecar_sha256", None)
    metadata = sidecar.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("dynamic floor M2 checkpoint sidecar lacks metadata")
    metadata = dict(metadata)
    unsigned_metadata = dict(metadata)
    observed_metadata_hash = unsigned_metadata.pop("metadata_sha256", None)
    train_batches = int(metadata.get("train_batches_per_epoch", -1))
    expected_updates = DYNAMIC_FLOOR_M2_EPOCHS * train_batches
    skipped = int(metadata.get("amp_skipped_attempts", -1))
    if (
        sidecar.get("schema_version") != DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA
        or str(Path(str(sidecar.get("checkpoint_path", ""))).resolve())
        != str(checkpoint)
        or sidecar.get("checkpoint_sha256") != sha256_file(checkpoint)
        or observed_sidecar_hash != canonical_sha256(unsigned_sidecar)
        or metadata.get("schema_version") != DYNAMIC_FLOOR_M2_CHECKPOINT_SIDECAR_SCHEMA
        or metadata.get("study_id") != DYNAMIC_FLOOR_M2_STUDY_ID
        or observed_metadata_hash != canonical_sha256(unsigned_metadata)
        or int(metadata.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or int(metadata.get("epoch", -1)) != DYNAMIC_FLOOR_M2_EPOCHS - 1
        or int(metadata.get("epochs", -1)) != DYNAMIC_FLOOR_M2_EPOCHS
        or int(metadata.get("world_size", -1)) != DYNAMIC_FLOOR_M2_WORLD_SIZE
        or int(metadata.get("global_batch_size", -1)) != DYNAMIC_FLOOR_M2_BATCH_SIZE
        or train_batches <= 0
        or int(metadata.get("successful_updates", -1)) != expected_updates
        or int(metadata.get("consumed_batches", -1)) != expected_updates
        or int(metadata.get("scheduler_advances", -1)) != expected_updates
        or int(metadata.get("ema_updates", -1)) != expected_updates
        or skipped < 0
        or int(metadata.get("replay_attempts", -1)) != skipped
        or int(metadata.get("optimizer_attempts", -1)) != expected_updates + skipped
        or int(metadata.get("max_amp_retries_per_batch", -1)) != 8
        or not 0 <= int(metadata.get("max_amp_retries_observed", -1)) <= 8
        or metadata.get("checkpoint_policy") != "final_epoch_ema_only_atomic"
        or metadata.get("checkpoint_consumer_state_key") != "state_dict_ema"
        or metadata.get("official_test_opened") is not False
        or metadata.get("paper_claim_allowed") is not False
    ):
        raise ValueError("dynamic floor M2 checkpoint sidecar is invalid")
    if binding is not None:
        binding = dict(binding)
        if (
            not _self_hash_matches(binding, field="binding_sha256")
            or metadata.get("runtime_commit") != binding.get("runtime_commit")
            or metadata.get("binding_sha256") != binding.get("binding_sha256")
            or metadata.get("arm") != binding.get("arm")
        ):
            raise ValueError("dynamic floor M2 checkpoint belongs to another cell")
    if cfg is not None and metadata.get("bound_config_sha256") != canonical_sha256(
        cfg.to_dict()
    ):
        raise ValueError("dynamic floor M2 checkpoint config hash mismatch")
    return sidecar


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty dynamic floor distribution")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    checked = [_finite(value, "distribution value") for value in values]
    return {
        "count": len(checked),
        "mean": sum(checked) / len(checked),
        "p05": _quantile(checked, 0.05),
        "p50": _quantile(checked, 0.50),
        "p95": _quantile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _integrate_power_samples(
    samples: Sequence[tuple[float, float]], *, start: float, end: float
) -> float | None:
    checked = sorted((float(timestamp), float(power)) for timestamp, power in samples)
    if (
        len(checked) < 2
        or checked[0][0] > start
        or checked[-1][0] < end
        or end <= start
    ):
        return None

    def interpolate(timestamp: float) -> float:
        for left, right in zip(checked[:-1], checked[1:]):
            if left[0] <= timestamp <= right[0]:
                width = right[0] - left[0]
                if width <= 0.0:
                    return right[1]
                weight = (timestamp - left[0]) / width
                return left[1] * (1.0 - weight) + right[1] * weight
        return checked[0][1] if timestamp <= checked[0][0] else checked[-1][1]

    clipped = [
        (start, interpolate(start)),
        *(
            (timestamp, power)
            for timestamp, power in checked
            if start < timestamp < end
        ),
        (end, interpolate(end)),
    ]
    return sum(
        0.5 * (left[1] + right[1]) * (right[0] - left[0])
        for left, right in zip(clipped[:-1], clipped[1:])
    )


def summarize_dynamic_floor_m2_telemetry(path: str | Path) -> dict[str, Any]:
    import json

    path = Path(path).resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    dataset_count = int(payload.get("dataset_count", -1))
    if (
        payload.get("schema_version") != "georoute_formal_development_telemetry_v1"
        or payload.get("development_only") is not True
        or payload.get("official_test_opened") is not False
        or payload.get("gt_for_route_used") is not False
        or payload.get("teacher_for_route_used") is not False
        or payload.get("oracle_used") is not False
        or payload.get("raw_prediction_cache_used") is not False
        or int(payload.get("world_size", -1)) != 1
        or int(payload.get("local_batch_size", -1)) != 1
        or dataset_count <= 0
        or not isinstance(records, list)
        or len(records) != dataset_count
        or int(payload.get("record_count", -1)) != dataset_count
        or int(payload.get("unique_dataset_count", -1)) != dataset_count
        or int(payload.get("sampler_padding_count", -1)) != 0
        or {int(record.get("dataset_index", -1)) for record in records}
        != set(range(dataset_count))
    ):
        raise ValueError("dynamic floor M2 telemetry population contract failed")

    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    k_values: list[int] = []
    attention_pairs: list[float] = []
    role_counts = Counter({name: 0 for name in ("context", "roi", "residual")})
    floor_rates = Counter()
    for record in records:
        route = record.get("route")
        if (
            not isinstance(route, Mapping)
            or route.get("schema_version")
            != "georoute_dynamic_diagnostic_window_telemetry_v1"
            or route.get("measurement_scope")
            != "accuracy_replay_only_excluded_from_timed_cost"
            or int(route.get("window_token_budget", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
            or route.get("source_grid_hw") != list(DYNAMIC_FLOOR_M2_SOURCE_GRID_HW)
            or int(route.get("batch_size", -1)) != 1
            or int(route.get("tubelet_count", -1)) != 384
            or int(route.get("item_count", -1)) != 220
            or not isinstance(route.get("selected_physical_index_sha256"), str)
            or len(route["selected_physical_index_sha256"]) != 64
            or route.get("gt_for_route_used") is not False
            or route.get("teacher_used") is not False
            or route.get("oracle_used") is not False
            or route.get("official_test_opened") is not False
            or route.get("paper_claim_allowed") is not False
        ):
            raise ValueError("dynamic floor M2 route telemetry schema changed")
        geometry_rows = route.get("geometry", {}).get("values")
        current_k = route.get("k_t", {}).get("values")
        per_tubelet_roles = route.get("roles", {}).get("per_tubelet_counts")
        if (
            not isinstance(geometry_rows, list)
            or len(geometry_rows) != 384
            or not isinstance(current_k, list)
            or len(current_k) != 384
            or not isinstance(per_tubelet_roles, list)
            or len(per_tubelet_roles) != 384
        ):
            raise ValueError("dynamic floor M2 telemetry lost tubelet attribution")
        for geometry, k_t, roles in zip(geometry_rows, current_k, per_tubelet_roles):
            if not isinstance(geometry, list) or len(geometry) != 4:
                raise ValueError("dynamic floor M2 geometry row is invalid")
            center_x = _finite(geometry[0], "ROI center x")
            center_y = _finite(geometry[1], "ROI center y")
            width = _finite(geometry[2], "ROI width")
            height = _finite(geometry[3], "ROI height")
            if isinstance(k_t, bool) or not isinstance(k_t, int):
                raise ValueError("dynamic floor M2 K_t must be an integer")
            if (
                not 0.0 <= center_x <= 1.0
                or not 0.0 <= center_y <= 1.0
                or not 0.0 < width <= 1.0
                or not 0.0 < height <= 1.0
                or k_t < 0
                or k_t > 220
                or not isinstance(roles, list)
                or len(roles) != 3
                or any(
                    isinstance(count, bool) or not isinstance(count, int) or count < 0
                    for count in roles
                )
                or sum(roles) != k_t
            ):
                raise ValueError("dynamic floor M2 K_t/role partition is invalid")
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            k_values.append(k_t)
            for name, count in zip(("context", "roi", "residual"), roles):
                role_counts[name] += int(count)
        if sum(map(int, current_k)) != DYNAMIC_FLOOR_M2_WINDOW_BUDGET:
            raise ValueError("dynamic floor M2 telemetry violates exact B")
        ragged = route.get("ragged_execution", {})
        if (
            int(ragged.get("requested_physical_tokens", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
            or int(ragged.get("unique_physical_tokens", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
            or int(ragged.get("padded_heavy_tokens", -1)) != 0
            or int(ragged.get("executed_patch_tokens", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
        ):
            raise ValueError("dynamic floor M2 telemetry lost true-ragged exact B")
        current_attention_pairs = _finite(
            ragged.get("attention_pairs"), "attention pairs"
        )
        if current_attention_pairs < 0.0:
            raise ValueError("dynamic floor M2 attention pairs must be non-negative")
        attention_pairs.append(current_attention_pairs)
        geometry_summary = route["geometry"]
        for key in (
            "width_floor_saturation_rate",
            "height_floor_saturation_rate",
            "width_ceiling_saturation_rate",
            "height_ceiling_saturation_rate",
        ):
            rate = _finite(geometry_summary.get(key), key)
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"dynamic floor M2 {key} must be in [0, 1]")
            floor_rates[key] += rate

    population_sha256 = str(payload.get("population_sha256", ""))
    if len(population_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in population_sha256.lower()
    ):
        raise ValueError("dynamic floor M2 population hash is invalid")
    total_roles = sum(role_counts.values())
    return {
        "schema_version": "scnr_dynamic_floor_m2_telemetry_summary_v1",
        "dataset_count": dataset_count,
        "record_count": len(records),
        "population_sha256": population_sha256,
        "telemetry_file_sha256": sha256_file(path),
        "geometry": {
            "width": _summary(widths),
            "height": _summary(heights),
            "area": _summary(areas),
            **{key: float(value) / dataset_count for key, value in floor_rates.items()},
        },
        "k_t": {
            "distribution": _summary(k_values),
            "zero_count": sum(value == 0 for value in k_values),
            "histogram": {
                str(key): count for key, count in sorted(Counter(k_values).items())
            },
        },
        "roles": {
            "counts": dict(role_counts),
            "fractions": {
                name: count / float(total_roles) for name, count in role_counts.items()
            },
        },
        "ragged_attention_pairs": _summary(attention_pairs),
        "development_only": True,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _validated_file_receipt(
    receipt: Any, *, label: str, require_size: bool = False
) -> Path:
    if not isinstance(receipt, Mapping):
        raise ValueError(f"{label} receipt is missing")
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError(f"{label} receipt path is invalid")
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or str(path.resolve()) != raw_path
        or not path.is_file()
        or path.is_symlink()
    ):
        raise ValueError(f"{label} artifact is not one canonical regular file")
    digest = str(receipt.get("sha256", receipt.get("file_sha256", "")))
    if digest != sha256_file(path):
        raise ValueError(f"{label} artifact hash mismatch")
    if require_size and int(receipt.get("size_bytes", -1)) != path.stat().st_size:
        raise ValueError(f"{label} artifact size mismatch")
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object")
    return payload


def _load_jsonl_objects(path: Path, *, label: str) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"{label} line {line_number} is not an object")
        rows.append(payload)
    if not rows:
        raise ValueError(f"{label} is empty")
    return rows


def validate_dynamic_floor_m2_stage_result(
    result: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
) -> dict[str, Any]:
    result = dict(result)
    arm = str(result.get("arm", ""))
    spec = dynamic_floor_m2_arm_spec(arm)
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("dynamic floor M2 stage arm mismatch")
    if expected_commit is not None and result.get("runtime_commit") != expected_commit:
        raise ValueError("dynamic floor M2 stage commit mismatch")
    binding = result.get("binding")
    metrics = result.get("metrics")
    telemetry = result.get("telemetry_summary")
    checkpoint = result.get("checkpoint_receipt")
    p0 = result.get("p0_receipt")
    configs = result.get("config_receipts")
    artifacts = result.get("artifact_receipts")
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
        result.get("schema_version") != DYNAMIC_FLOOR_M2_STAGE_RESULT_SCHEMA
        or result.get("status") != "PASS_DYNAMIC_FLOOR_M2_TRAINING_AND_ACCURACY"
        or result.get("study_id") != DYNAMIC_FLOOR_M2_STUDY_ID
        or result.get("arm_spec") != spec
        or result.get("arm_spec_sha256") != canonical_sha256(spec)
        or int(result.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or int(result.get("epochs", -1)) != DYNAMIC_FLOOR_M2_EPOCHS
        or not isinstance(binding, Mapping)
        or result.get("binding_sha256") != binding.get("binding_sha256")
        or not _self_hash_matches(binding, field="binding_sha256")
        or binding.get("arm") != arm
        or binding.get("arm_spec") != spec
        or not isinstance(metrics, Mapping)
        or set(metrics) != required_metrics
        or any(
            not math.isfinite(float(value)) or not 0.0 <= float(value) <= 100.0
            for value in metrics.values()
        )
        or not isinstance(telemetry, Mapping)
        or telemetry.get("schema_version")
        != "scnr_dynamic_floor_m2_telemetry_summary_v1"
        or telemetry.get("development_only") is not True
        or telemetry.get("official_test_opened") is not False
        or telemetry.get("paper_claim_allowed") is not False
        or result.get("population_sha256") != telemetry.get("population_sha256")
        or not isinstance(checkpoint, Mapping)
        or checkpoint.get("policy") != "final_epoch_ema_only_atomic"
        or int(checkpoint.get("epoch", -1)) != DYNAMIC_FLOOR_M2_EPOCHS - 1
        or not isinstance(checkpoint.get("sha256"), str)
        or len(checkpoint["sha256"]) != 64
        or not isinstance(p0, Mapping)
        or p0.get("status") != "PASS_NO_PERFORMANCE_P0"
        or p0.get("telemetry_status") != "PASS_NO_PERFORMANCE_TELEMETRY_P0"
        or not isinstance(configs, Mapping)
        or set(configs) != {"train", "accuracy"}
        or not isinstance(artifacts, Mapping)
        or set(artifacts) != {"prediction", "telemetry", "accuracy_log"}
        or result.get("cost_attached") is not False
        or result.get("single_seed_descriptive_only") is not True
        or result.get("official_test_opened") is not False
        or result.get("gt_for_route_used") is not False
        or result.get("teacher_used") is not False
        or result.get("raw_prediction_cache_used") is not False
        or result.get("paper_claim_allowed") is not False
        or not _self_hash_matches(result, field="stage_result_sha256")
    ):
        raise ValueError("dynamic floor M2 stage result contract failed")

    from mmengine.config import Config
    from tools.bata.georoute_stage_runner import parse_official_style_map

    config_paths = {
        phase: _validated_file_receipt(
            configs[phase], label=f"dynamic floor M2 {phase} config"
        )
        for phase in ("train", "accuracy")
    }
    bound_configs = {
        phase: Config.fromfile(str(config_paths[phase]))
        for phase in ("train", "accuracy")
    }
    for phase, cfg in bound_configs.items():
        observed_binding = validate_dynamic_floor_m2_config(cfg, arm=arm, phase=phase)
        if dict(observed_binding) != dict(binding):
            raise ValueError(f"dynamic floor M2 {phase} config changed binding")

    checkpoint_path = _validated_file_receipt(
        checkpoint, label="dynamic floor M2 checkpoint", require_size=True
    )
    sidecar = validate_dynamic_floor_m2_checkpoint_sidecar(
        checkpoint_path,
        binding=binding,
        cfg=bound_configs["train"],
    )
    sidecar_path = Path(str(checkpoint_path) + ".metadata.json").resolve()
    metadata = sidecar["experiment_metadata"]
    if (
        checkpoint.get("state_key") != "state_dict_ema"
        or checkpoint.get("sidecar_path") != str(sidecar_path)
        or checkpoint.get("sidecar_sha256") != sha256_file(sidecar_path)
        or checkpoint.get("metadata_sha256") != metadata.get("metadata_sha256")
        or int(checkpoint.get("successful_updates", -1))
        != int(metadata["successful_updates"])
    ):
        raise ValueError("dynamic floor M2 checkpoint receipt is not sidecar-bound")

    artifact_paths = {
        name: _validated_file_receipt(artifacts[name], label=f"dynamic floor M2 {name}")
        for name in ("prediction", "telemetry", "accuracy_log")
    }
    recomputed_telemetry = summarize_dynamic_floor_m2_telemetry(
        artifact_paths["telemetry"]
    )
    if dict(recomputed_telemetry) != dict(telemetry):
        raise ValueError("dynamic floor M2 telemetry summary is not reproducible")
    parsed_metrics = parse_official_style_map(
        artifact_paths["accuracy_log"].read_text(encoding="utf-8", errors="replace")
    )
    parsed_metrics["high_iou_composite"] = 0.5 * (
        parsed_metrics["mAP@0.6"] + parsed_metrics["mAP@0.7"]
    )
    if parsed_metrics != {key: float(value) for key, value in metrics.items()}:
        raise ValueError("dynamic floor M2 metrics differ from the accuracy log")
    prediction_payload = _load_json_object(
        artifact_paths["prediction"], label="dynamic floor M2 prediction"
    )
    prediction_results = prediction_payload.get("results")
    if not isinstance(prediction_results, Mapping) or not set(
        map(str, prediction_results)
    ) <= set(map(str, binding["evaluation_video_ids"])):
        raise ValueError("dynamic floor M2 prediction escaped development Gate")

    p0_path = _validated_file_receipt(p0, label="dynamic floor M2 P0")
    p0_report = _load_json_object(p0_path, label="dynamic floor M2 P0 report")
    from tools.bata.run_georoute_dynamic_stage1_p0 import (
        validate_dynamic_stage1_p0_report,
    )

    validate_dynamic_stage1_p0_report(p0_report)
    unsigned_p0 = dict(p0_report)
    observed_p0_hash = unsigned_p0.pop("report_sha256", None)
    p0_source = p0_report.get("source")
    p0_telemetry = p0_report.get("diagnostic_telemetry_p0")
    if (
        observed_p0_hash != canonical_sha256(unsigned_p0)
        or p0.get("report_sha256") != observed_p0_hash
        or p0_report.get("status") != "PASS_NO_PERFORMANCE_P0"
        or not isinstance(p0_source, Mapping)
        or p0_source.get("commit") != result.get("runtime_commit")
        or p0_source.get("head_matches_expected") is not True
        or p0_source.get("origin_ref_matches_expected") is not True
        or p0_source.get("tree_clean") is not True
        or not isinstance(p0_telemetry, Mapping)
        or p0_telemetry.get("status") != "PASS_NO_PERFORMANCE_TELEMETRY_P0"
    ):
        raise ValueError("dynamic floor M2 P0 receipt is not reproducible")
    return result


def validate_dynamic_floor_m2_cost_profile(
    profile: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    expected_execution_commit: str | None = None,
) -> dict[str, Any]:
    profile = dict(profile)
    if expected_commit is not None and profile.get("runtime_commit") != expected_commit:
        raise ValueError("dynamic floor M2 cost commit mismatch")
    if (
        expected_execution_commit is not None
        and profile.get("execution_commit") != expected_execution_commit
    ):
        raise ValueError("dynamic floor M2 cost execution commit mismatch")
    scope = profile.get("scope")
    arm_summaries = profile.get("arm_summaries")
    pass_receipts = profile.get("pass_receipts")
    artifacts = profile.get("artifact_receipts")
    stage_receipts = profile.get("stage_result_receipts")
    expected_scope = {
        "decode": True,
        "preprocess": True,
        "host_to_device": True,
        "scout": True,
        "route": True,
        "patch_embed": True,
        "backbone": True,
        "adapter": True,
        "detector": True,
        "nms": True,
        "diagnostic_telemetry_inside_timed_forward": False,
        "same_gpu_counterbalanced": True,
        "development_only": True,
    }
    if (
        profile.get("schema_version") != DYNAMIC_FLOOR_M2_COST_SCHEMA
        or profile.get("status") != "PASS_DYNAMIC_FLOOR_M2_FULL_STACK_COST"
        or profile.get("study_id") != DYNAMIC_FLOOR_M2_STUDY_ID
        or int(profile.get("seed", -1)) != DYNAMIC_FLOOR_M2_SEED
        or tuple(profile.get("profile_order", ())) != DYNAMIC_FLOOR_M2_COST_ORDER
        or not isinstance(scope, Mapping)
        or {key: scope.get(key) for key in expected_scope} != expected_scope
        or not isinstance(arm_summaries, Mapping)
        or set(arm_summaries) != set(DYNAMIC_FLOOR_M2_ARM_ORDER)
        or int(profile.get("warmup_samples_per_pass", -1)) != 50
        or int(profile.get("batch_size", -1)) != 1
        or int(profile.get("loader_workers", -1)) != 0
        or int(profile.get("world_size", -1)) != 1
        or int(profile.get("power_interval_ms", -1)) != 20
        or int(profile.get("raw_sample_count", -1)) <= 0
        or not isinstance(profile.get("population_sha256"), str)
        or len(profile["population_sha256"]) != 64
        or not isinstance(profile.get("accuracy_population_sha256"), str)
        or len(profile["accuracy_population_sha256"]) != 64
        or not isinstance(profile.get("run_root"), str)
        or not Path(profile["run_root"]).is_absolute()
        or not isinstance(pass_receipts, list)
        or len(pass_receipts) != len(DYNAMIC_FLOOR_M2_COST_ORDER)
        or not isinstance(artifacts, Mapping)
        or set(artifacts)
        != {
            "raw_samples",
            "power_trace",
            "sidecar_attempt_report",
            "sidecar_attempt_trace",
        }
        or not isinstance(stage_receipts, Mapping)
        or set(stage_receipts) != set(DYNAMIC_FLOOR_M2_ARM_ORDER)
        or not isinstance(profile.get("hardware_fingerprint"), str)
        or len(profile["hardware_fingerprint"]) != 64
        or not isinstance(profile.get("software_fingerprint"), str)
        or len(profile["software_fingerprint"]) != 64
        or profile.get("official_test_opened") is not False
        or profile.get("paper_claim_allowed") is not False
    ):
        raise ValueError("dynamic floor M2 cost profile header is invalid")
    run_root = Path(profile["run_root"]).resolve()
    if str(run_root) != profile["run_root"] or not run_root.is_dir():
        raise ValueError("dynamic floor M2 cost run root is not canonical")
    validated_stages: dict[str, dict[str, Any]] = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        stage_path = _validated_file_receipt(
            stage_receipts[arm], label=f"dynamic floor M2 cost stage {arm}"
        )
        try:
            stage_path.relative_to(run_root / "development")
        except ValueError as error:
            raise ValueError(
                "dynamic floor M2 cost stage path left the run root"
            ) from error
        stage_payload = _load_json_object(
            stage_path, label=f"dynamic floor M2 cost stage {arm}"
        )
        validated_stages[arm] = validate_dynamic_floor_m2_stage_result(
            stage_payload,
            expected_arm=arm,
            expected_commit=profile["runtime_commit"],
        )
        if (
            stage_receipts[arm].get("stage_result_sha256")
            != validated_stages[arm]["stage_result_sha256"]
            or validated_stages[arm]["population_sha256"]
            != profile["accuracy_population_sha256"]
        ):
            raise ValueError("dynamic floor M2 cost stage receipt changed")

    expected_cost_config_hashes = {}
    for arm, stage in validated_stages.items():
        cost_cfg = build_dynamic_floor_m2_cost_config(stage, arm=arm)
        expected_cost_config_hashes[arm] = canonical_sha256(cost_cfg.to_dict())
    pass_sample_counts = []
    for pass_index, (receipt, arm) in enumerate(
        zip(pass_receipts, DYNAMIC_FLOOR_M2_COST_ORDER)
    ):
        if not isinstance(receipt, Mapping):
            raise ValueError("dynamic floor M2 cost pass receipt is missing")
        unsigned_receipt = dict(receipt)
        observed_hash = unsigned_receipt.pop("pass_sha256", None)
        if (
            int(receipt.get("pass_index", -1)) != pass_index
            or receipt.get("arm") != arm
            or int(receipt.get("sample_count", -1)) <= 0
            or receipt.get("population_sha256") != profile.get("population_sha256")
            or receipt.get("accuracy_population_sha256")
            != profile.get("accuracy_population_sha256")
            or receipt.get("checkpoint_sha256")
            != validated_stages[arm]["checkpoint_receipt"]["sha256"]
            or receipt.get("bound_accuracy_config_sha256")
            != validated_stages[arm]["config_receipts"]["accuracy"]["sha256"]
            or receipt.get("cost_config_sha256") != expected_cost_config_hashes[arm]
            or receipt.get("diagnostic_telemetry_inside_timed_forward") is not False
            or observed_hash != canonical_sha256(unsigned_receipt)
        ):
            raise ValueError("dynamic floor M2 cost pass receipt is invalid")
        pass_sample_counts.append(int(receipt["sample_count"]))
    if len(set(pass_sample_counts)) != 1:
        raise ValueError("dynamic floor M2 cost passes changed population size")
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        summary = arm_summaries[arm]
        latency = summary.get("latency_ms") if isinstance(summary, Mapping) else None
        resources = summary.get("resources") if isinstance(summary, Mapping) else None
        if (
            not isinstance(summary, Mapping)
            or int(summary.get("pass_count", -1)) != 2
            or int(summary.get("sample_count", -1)) <= 0
            or not isinstance(summary.get("population_sha256"), str)
            or len(summary["population_sha256"]) != 64
            or not isinstance(latency, Mapping)
            or not isinstance(resources, Mapping)
        ):
            raise ValueError(f"dynamic floor M2 cost summary is invalid for {arm}")
        for stage in (
            "input_pipeline_serial_ms",
            "h2d_ms",
            "model_forward_ms",
            "postprocess_ms",
            "decode_to_window_output_wall_ms",
            "final_video_nms_ms",
            "end_to_end_serial_ms",
        ):
            stage_summary = latency.get(stage)
            if (
                not isinstance(stage_summary, Mapping)
                or _finite(stage_summary.get("p50"), f"{arm}.{stage}.p50") <= 0.0
                or _finite(stage_summary.get("p95"), f"{arm}.{stage}.p95") <= 0.0
            ):
                raise ValueError(
                    f"dynamic floor M2 cost stage is invalid: {arm}.{stage}"
                )
        for key in (
            "peak_gpu_allocated_mb",
            "peak_gpu_reserved_mb",
            "gross_gpu_energy_j",
        ):
            if _finite(resources.get(key), f"{arm}.{key}") <= 0.0:
                raise ValueError(f"dynamic floor M2 resource is invalid: {arm}.{key}")
    population_hashes = {
        arm_summaries[arm]["population_sha256"] for arm in DYNAMIC_FLOOR_M2_ARM_ORDER
    }
    if len(population_hashes) != 1:
        raise ValueError("dynamic floor M2 cost arms measured different populations")
    if population_hashes != {profile["population_sha256"]}:
        raise ValueError("dynamic floor M2 cost summary population hash changed")

    artifact_paths = {
        name: _validated_file_receipt(
            artifacts[name], label=f"dynamic floor M2 cost {name}"
        )
        for name in artifacts
    }
    for artifact_path in artifact_paths.values():
        try:
            artifact_path.relative_to(run_root / "cost")
        except ValueError as error:
            raise ValueError(
                "dynamic floor M2 cost artifact left the run root"
            ) from error
    raw_rows = _load_jsonl_objects(
        artifact_paths["raw_samples"], label="dynamic floor M2 raw cost samples"
    )
    if len(raw_rows) != int(profile["raw_sample_count"]) or len(raw_rows) != sum(
        pass_sample_counts
    ):
        raise ValueError("dynamic floor M2 raw cost sample count changed")
    rows_by_pass: dict[int, list[dict[str, Any]]] = {
        index: [] for index in range(len(DYNAMIC_FLOOR_M2_COST_ORDER))
    }
    for row in raw_rows:
        pass_index = int(row.get("pass_index", -1))
        unsigned_row = dict(row)
        observed_hash = unsigned_row.pop("sample_sha256", None)
        if (
            row.get("schema_version") != "scnr_dynamic_floor_m2_cost_sample_v1"
            or pass_index not in rows_by_pass
            or row.get("arm") != DYNAMIC_FLOOR_M2_COST_ORDER[pass_index]
            or row.get("population_sha256") != profile["population_sha256"]
            or observed_hash != canonical_sha256(unsigned_row)
            or _finite(row.get("gpu_energy_j"), "sample gpu energy") <= 0.0
            or _finite(row.get("peak_gpu_allocated_mb"), "sample peak allocated") <= 0.0
            or _finite(row.get("peak_gpu_reserved_mb"), "sample peak reserved") <= 0.0
        ):
            raise ValueError("dynamic floor M2 raw cost sample is invalid")
        for stage in (
            "input_pipeline_serial_ms",
            "h2d_ms",
            "model_forward_ms",
            "postprocess_ms",
            "decode_to_window_output_wall_ms",
            "final_video_nms_ms",
            "end_to_end_serial_ms",
        ):
            if _finite(row.get(stage), f"raw sample {stage}") <= 0.0:
                raise ValueError("dynamic floor M2 raw cost stage is invalid")
        for stage in (
            "backbone_wrapper_ms",
            "scout_ms",
            "patch_embed_ms",
            "heavy_backbone_ms",
            "sparse_adapter_ms",
            "projection_ms",
            "neck_ms",
            "head_ms",
        ):
            if _finite(row.get(stage), f"raw component {stage}") <= 0.0:
                raise ValueError("dynamic floor M2 CUDA component event is missing")
        energy_window = row.get("energy_window_monotonic_s")
        nms_window = row.get("nms_energy_window_monotonic_s")
        if (
            not isinstance(energy_window, list)
            or len(energy_window) != 2
            or not isinstance(nms_window, list)
            or len(nms_window) != 2
        ):
            raise ValueError("dynamic floor M2 raw energy window is missing")
        energy_start, energy_end = map(
            lambda value: _finite(value, "sample energy window"), energy_window
        )
        nms_start, nms_end = map(
            lambda value: _finite(value, "sample NMS energy window"), nms_window
        )
        expected_nms_ms = (
            (nms_end - nms_start) * 1000.0 / pass_sample_counts[pass_index]
        )
        if (
            energy_end <= energy_start
            or nms_end <= nms_start
            or not math.isclose(
                float(row["final_video_nms_ms"]),
                expected_nms_ms,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
            or not math.isclose(
                float(row["end_to_end_serial_ms"]),
                float(row["decode_to_window_output_wall_ms"])
                + float(row["final_video_nms_ms"]),
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("dynamic floor M2 NMS/full-stack timing is inconsistent")
        route_audit = row.get("route_audit")
        if (
            not isinstance(route_audit, Mapping)
            or int(route_audit.get("exact_window_budget", -1))
            != DYNAMIC_FLOOR_M2_WINDOW_BUDGET
            or int(route_audit.get("padded_heavy_tokens", -1)) != 0
        ):
            raise ValueError("dynamic floor M2 raw cost route audit is invalid")
        rows_by_pass[pass_index].append(row)
    for pass_index, rows in rows_by_pass.items():
        if (
            len(rows) != pass_sample_counts[pass_index]
            or sorted(int(row.get("sample_ordinal", -1)) for row in rows)
            != list(range(pass_sample_counts[pass_index]))
            or canonical_sha256([row["window_id"] for row in rows])
            != pass_receipts[pass_index]["sample_manifest_sha256"]
        ):
            raise ValueError("dynamic floor M2 raw cost pass lineage is invalid")
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        rows = [row for row in raw_rows if row["arm"] == arm]
        summary = arm_summaries[arm]
        if len(rows) != int(summary["sample_count"]):
            raise ValueError("dynamic floor M2 arm summary sample count changed")
        for stage in (
            "input_pipeline_serial_ms",
            "h2d_ms",
            "model_forward_ms",
            "postprocess_ms",
            "decode_to_window_output_wall_ms",
            "final_video_nms_ms",
            "end_to_end_serial_ms",
        ):
            values = [float(row[stage]) for row in rows]
            for quantile, probability in (("p50", 0.50), ("p95", 0.95)):
                if not math.isclose(
                    float(summary["latency_ms"][stage][quantile]),
                    _quantile(values, probability),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "dynamic floor M2 latency summary is not reproducible"
                    )
        expected_resources = {
            "peak_gpu_allocated_mb": max(
                float(row["peak_gpu_allocated_mb"]) for row in rows
            ),
            "peak_gpu_reserved_mb": max(
                float(row["peak_gpu_reserved_mb"]) for row in rows
            ),
            "gross_gpu_energy_j": sum(float(row["gpu_energy_j"]) for row in rows),
        }
        if any(
            not math.isclose(
                float(summary["resources"][key]),
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
            for key, value in expected_resources.items()
        ):
            raise ValueError("dynamic floor M2 resource summary is not reproducible")

    power_rows = _load_jsonl_objects(
        artifact_paths["power_trace"], label="dynamic floor M2 power trace"
    )
    power_samples = [
        (
            _finite(row.get("monotonic_s"), "power monotonic timestamp"),
            _finite(row.get("power_w"), "power value"),
        )
        for row in power_rows
    ]
    power_origin = power_samples[0][0]
    if any(
        int(row.get("sequence", -1)) != index
        or power <= 0.0
        or (index > 0 and timestamp <= power_samples[index - 1][0])
        or not math.isclose(
            _finite(row.get("timestamp_ms"), "normalized power timestamp"),
            (timestamp - power_origin) * 1000.0,
            rel_tol=1e-12,
            abs_tol=1e-9,
        )
        for index, (row, (timestamp, power)) in enumerate(
            zip(power_rows, power_samples)
        )
    ):
        raise ValueError("dynamic floor M2 normalized power trace is invalid")
    for row in raw_rows:
        start, end = map(float, row["energy_window_monotonic_s"])
        nms_start, nms_end = map(float, row["nms_energy_window_monotonic_s"])
        sample_energy = _integrate_power_samples(power_samples, start=start, end=end)
        nms_energy = _integrate_power_samples(
            power_samples, start=nms_start, end=nms_end
        )
        if sample_energy is None or nms_energy is None:
            raise ValueError(
                "dynamic floor M2 power trace does not cover timing windows"
            )
        expected_energy = (
            sample_energy + nms_energy / pass_sample_counts[int(row["pass_index"])]
        )
        if not math.isclose(
            float(row["gpu_energy_j"]),
            expected_energy,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("dynamic floor M2 energy is not reproducible")
    sidecar_report = _load_json_object(
        artifact_paths["sidecar_attempt_report"],
        label="dynamic floor M2 sidecar attempt report",
    )
    unsigned_sidecar = dict(sidecar_report)
    observed_sidecar_hash = unsigned_sidecar.pop("attempt_sha256", None)
    if (
        sidecar_report.get("status") != "PASS"
        or int(sidecar_report.get("interval_ms", -1)) != 20
        or sidecar_report.get("trace_io_inside_sampling_loop") is not False
        or sidecar_report.get("trace_file_sha256")
        != sha256_file(artifact_paths["sidecar_attempt_trace"])
        or observed_sidecar_hash != canonical_sha256(unsigned_sidecar)
    ):
        raise ValueError("dynamic floor M2 sidecar attempt receipt is invalid")
    if not _self_hash_matches(profile, field="profile_sha256"):
        raise ValueError("dynamic floor M2 cost profile self-hash mismatch")
    return profile


def finalize_dynamic_floor_m2(
    stage_results: Mapping[str, Mapping[str, Any]],
    cost_profile: Mapping[str, Any] | None,
    *,
    expected_commit: str,
    expected_cost_execution_commit: str | None = None,
) -> dict[str, Any]:
    expected_cost_execution_commit = str(
        expected_cost_execution_commit or expected_commit
    )
    validated: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    for arm in DYNAMIC_FLOOR_M2_ARM_ORDER:
        try:
            validated[arm] = validate_dynamic_floor_m2_stage_result(
                stage_results[arm],
                expected_arm=arm,
                expected_commit=expected_commit,
            )
        except (KeyError, TypeError, ValueError) as error:
            errors[arm] = str(error)
    validated_cost = None
    try:
        if cost_profile is None:
            raise ValueError("paired cost profile is missing")
        validated_cost = validate_dynamic_floor_m2_cost_profile(
            cost_profile,
            expected_commit=expected_commit,
            expected_execution_commit=expected_cost_execution_commit,
        )
    except (TypeError, ValueError) as error:
        errors["paired_cost"] = str(error)

    complete = not errors and set(validated) == set(DYNAMIC_FLOOR_M2_ARM_ORDER)
    contrasts: dict[str, Any] = {}
    if complete and validated_cost is not None:
        g1 = validated[DYNAMIC_FLOOR_M2_ARM_ORDER[0]]
        g2 = validated[DYNAMIC_FLOOR_M2_ARM_ORDER[1]]
        g1_cost = validated_cost["arm_summaries"][DYNAMIC_FLOOR_M2_ARM_ORDER[0]]
        g2_cost = validated_cost["arm_summaries"][DYNAMIC_FLOOR_M2_ARM_ORDER[1]]
        contrasts = {
            "g1_minus_g2_metrics_pp": {
                key: float(g1["metrics"][key]) - float(g2["metrics"][key])
                for key in g1["metrics"]
            },
            "g1_minus_g2_end_to_end_ms": {
                quantile: float(g1_cost["latency_ms"]["end_to_end_serial_ms"][quantile])
                - float(g2_cost["latency_ms"]["end_to_end_serial_ms"][quantile])
                for quantile in ("p50", "p95")
            },
            "g1_minus_g2_geometry": {
                "width_floor_saturation_rate": float(
                    g1["telemetry_summary"]["geometry"]["width_floor_saturation_rate"]
                )
                - float(
                    g2["telemetry_summary"]["geometry"]["width_floor_saturation_rate"]
                ),
                "height_floor_saturation_rate": float(
                    g1["telemetry_summary"]["geometry"]["height_floor_saturation_rate"]
                )
                - float(
                    g2["telemetry_summary"]["geometry"]["height_floor_saturation_rate"]
                ),
                "area_p50": float(g1["telemetry_summary"]["geometry"]["area"]["p50"])
                - float(g2["telemetry_summary"]["geometry"]["area"]["p50"]),
            },
        }
    result: dict[str, Any] = {
        "schema_version": DYNAMIC_FLOOR_M2_FINALIZATION_SCHEMA,
        "study_id": DYNAMIC_FLOOR_M2_STUDY_ID,
        "status": (
            "PASS_COMPLETE_DESCRIPTIVE_FLOOR_SENSITIVITY"
            if complete
            else "FAIL_INCOMPLETE_NO_FLOOR_INFERENCE"
        ),
        "decision": (
            "COMPLETE_DESCRIPTIVE_ONLY_M3_REQUIRED_FOR_FLOOR_SELECTION"
            if complete
            else "INCOMPLETE_NO_FLOOR_INFERENCE"
        ),
        "runtime_commit": expected_commit,
        "cost_execution_commit": expected_cost_execution_commit,
        "validated_arms": sorted(validated),
        "errors": errors,
        "descriptive_contrasts": contrasts if complete else {},
        "single_seed_floor_selection_allowed": False,
        "m3_confirmation_required": complete,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    result["finalization_sha256"] = canonical_sha256(result)
    return result


validate_frozen_dynamic_floor_m2_contract()
