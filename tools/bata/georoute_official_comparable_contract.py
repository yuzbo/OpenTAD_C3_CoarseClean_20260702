"""Frozen contracts for official-comparable GeoRoute evidence.

This module separates three things that must never be conflated:

1. the immutable upstream AdaTAD release anchor;
2. a no-performance resource/numerical preflight on the current source; and
3. a three-seed development matrix that is still forbidden from opening the
   THUMOS official validation/test population.

Only a later, separately sealed test-opening protocol may create paper-table
performance evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tools.bata.georoute_experiment_contract import (
    assert_development_annotation,
    canonical_sha256,
    load_development_manifest,
    sha256_file,
)


OFFICIAL_COMPARABLE_PROTOCOL_SCHEMA = (
    "georoute_official_comparable_protocol_manifest_v1"
)
OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA = (
    "georoute_official_comparable_preflight_finalization_v1"
)
OFFICIAL_DDP_WORLD2_KAT_SCHEMA = (
    "georoute_official_comparable_world2_fp32_ddp_kat_v1"
)
OFFICIAL_DDP_WORLD2_KAT_PASS = (
    "PASS_WORLD2_DEFAULT_FP32_DDP_REDUCTION_AND_UPDATE_ONLY"
)
OFFICIAL_DDP_WORLD2_KAT_FAIL = "FAIL_WORLD2_DEFAULT_FP32_DDP_REDUCTION"
FORMAL_DEVELOPMENT_BINDING_SCHEMA = (
    "georoute_official_comparable_development_binding_v1"
)
FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA = (
    "georoute_official_comparable_development_checkpoint_v1"
)
FORMAL_DEVELOPMENT_RESULT_SCHEMA = (
    "georoute_official_comparable_development_result_v1"
)
FORMAL_DEVELOPMENT_DEPLOYMENT_SCHEMA = (
    "georoute_official_comparable_development_deployment_v1"
)
FORMAL_DEVELOPMENT_FINALIZATION_SCHEMA = (
    "georoute_official_comparable_development_finalization_v1"
)

OFFICIAL_UPSTREAM_REPOSITORY = "https://github.com/sming256/OpenTAD.git"
OFFICIAL_UPSTREAM_RELEASE_COMMIT = (
    "01c58b9f2370e914150cf94d392208a4e211c053"
)
OFFICIAL_CONFIG_RELATIVE_PATH = (
    "configs/adatad/thumos/"
    "e2e_thumos_videomae_s_768x1_160_adapter.py"
)
OFFICIAL_CONFIG_SHA256 = (
    "5521b6ce28cc6770e662d3dfdd4621479bc228be6131e300a92285fb4961a49c"
)
OFFICIAL_RELEASED_MODEL_DRIVE_ID = "1HGUBroK90KBAkFqQreAVtHCIclJh7DmM"
OFFICIAL_RELEASED_LOG_DRIVE_ID = "1sqLsgkZsPReusv1lNUOg_nqE4nJX-YnD"
OFFICIAL_PUBLISHED_METRICS = {
    "mAP@0.3": 83.90,
    "mAP@0.4": 79.01,
    "mAP@0.5": 72.38,
    "mAP@0.6": 61.57,
    "mAP@0.7": 48.27,
    "average_mAP": 69.03,
}

# Exact released-checkpoint evaluation is deterministic up to the two-decimal
# reporting precision. A mismatch larger than 0.05 percentage points in any
# field means the evaluator/data/code anchor is not reproduced.
RELEASED_CHECKPOINT_EVAL_TOLERANCE_PP = 0.05

# A newly trained official-source run is stochastic. This wider criterion is
# only an admission check, never a replacement for the released-checkpoint
# evaluator anchor.
OFFICIAL_TRAIN_REPRO_AVERAGE_TOLERANCE_PP = 1.0
OFFICIAL_TRAIN_REPRO_PER_TIOU_TOLERANCE_PP = 1.5

FORMAL_DEVELOPMENT_SEEDS = (3407, 3408, 3409)
FORMAL_REFERENCE_SEED = 42
FORMAL_PREFLIGHT_SEED = 2311
FORMAL_EPOCHS = 60
FORMAL_SCHEDULER_WARMUP_EPOCHS = 5
FORMAL_SCHEDULER_MAX_EPOCHS = 100
FORMAL_WORLD_SIZE = 2
# OpenTAD's ``solver.*.batch_size`` is the job-global batch size: the
# dataloader divides it by ``world_size`` before constructing each rank's
# loader.  The pinned AdaTAD config freezes this value to two, so the matched
# two-rank recipe is one sample per rank and two samples per optimizer step.
FORMAL_CONFIG_BATCH_SIZE = 2
FORMAL_PER_RANK_BATCH_SIZE = FORMAL_CONFIG_BATCH_SIZE // FORMAL_WORLD_SIZE
FORMAL_GLOBAL_BATCH_SIZE = FORMAL_CONFIG_BATCH_SIZE
FORMAL_TOKEN_BUDGET = 64
FORMAL_SOURCE_HEIGHT = 180
FORMAL_SOURCE_WIDTH = 320
FORMAL_SOURCE_GRID_HW = (11, 20)
FORMAL_NATIVE_TOKEN_COUNT = 220
FORMAL_SCOUT_SIZE = 96

FORMAL_DEVELOPMENT_ARMS: dict[str, dict[str, Any]] = {
    "dense_native": {
        "route_mode": "dense",
        "policy_estimator": "none",
        "tokens_per_tubelet": None,
        "representation_enabled": False,
        "geometry_side_channel": False,
        "causal_role": "native_source_upper_compute_reference",
    },
    "fixed_lattice": {
        "route_mode": "uniform",
        "policy_estimator": "none",
        "tokens_per_tubelet": FORMAL_TOKEN_BUDGET,
        "representation_enabled": False,
        "geometry_side_channel": False,
        "causal_role": "deterministic_exact_k_control",
    },
    "random": {
        "route_mode": "random",
        "policy_estimator": "none",
        "tokens_per_tubelet": FORMAL_TOKEN_BUDGET,
        "representation_enabled": False,
        "geometry_side_channel": False,
        "causal_role": "seeded_data_independent_exact_k_control",
    },
    "residual_st_rep_off": {
        "route_mode": "free",
        "policy_estimator": "straight_through",
        "tokens_per_tubelet": FORMAL_TOKEN_BUDGET,
        "representation_enabled": False,
        "geometry_side_channel": False,
        "causal_role": "native_token_select_biased_estimator",
    },
    "residual_pl_rep_off": {
        "route_mode": "free",
        "policy_estimator": "score_function",
        "tokens_per_tubelet": FORMAL_TOKEN_BUDGET,
        "representation_enabled": False,
        "geometry_side_channel": False,
        "causal_role": "native_token_select_score_function_estimator",
    },
}
FORMAL_DEVELOPMENT_ARM_ORDER = tuple(FORMAL_DEVELOPMENT_ARMS)


def _full_hex(value: Any, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != length or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def formal_arm_spec(arm: str) -> dict[str, Any]:
    try:
        return copy.deepcopy(FORMAL_DEVELOPMENT_ARMS[arm])
    except KeyError as error:
        raise ValueError(f"unknown formal GeoRoute arm {arm!r}") from error


def formal_cell_relative_path(*, arm: str, seed: int) -> Path:
    formal_arm_spec(arm)
    if int(seed) not in FORMAL_DEVELOPMENT_SEEDS:
        raise ValueError("formal development seed is outside the frozen set")
    return Path("development") / arm / f"seed{int(seed)}"


def _validate_preflight_parent(
    path: str | Path,
    *,
    expected_file_sha256: str,
    expected_runtime_commit: str,
) -> dict[str, Any]:
    path = Path(path).resolve()
    expected_file_sha256 = _full_hex(
        expected_file_sha256,
        length=64,
        name="preflight finalization file sha256",
    )
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            "formal development requires the sealed preflight finalization"
        )
    if sha256_file(path) != expected_file_sha256:
        raise ValueError("formal preflight finalization file hash mismatch")
    parent = read_json(path)
    if (
        parent.get("schema_version") != OFFICIAL_COMPARABLE_PREFLIGHT_SCHEMA
        or parent.get("status") != "PASS_OFFICIAL_COMPARABLE_PREFLIGHT_ONLY"
        or parent.get("decision") != "FORMAL_DEVELOPMENT_MATRIX_AUTHORIZED"
        or parent.get("runtime_commit") != expected_runtime_commit
        or parent.get("formal_development_matrix_authorized") is not True
        or parent.get("official_protocol_freeze_authorized") is not False
        or parent.get("performance_metrics") != {}
        or parent.get("performance_inference_allowed") is not False
        or parent.get("official_test_opened") is not False
        or parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(
            parent, field="finalization_sha256"
        )
    ):
        raise ValueError(
            "formal preflight parent did not authorize the development matrix"
        )
    return parent


def bind_formal_development_config(
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
    preflight_finalization_path: str | Path,
    expected_preflight_file_sha256: str,
):
    """Create one official-semantics, development-only two-GPU config."""

    from mmengine.config import Config

    runtime_commit = _full_hex(
        runtime_commit, length=40, name="runtime_commit"
    )
    if int(seed) not in FORMAL_DEVELOPMENT_SEEDS:
        raise ValueError("formal development seed is outside the frozen set")
    spec = formal_arm_spec(arm)
    source_config_path = Path(source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(development_annotation_path).resolve()
    class_map_path = Path(class_map_path).resolve()
    video_root = Path(development_video_root).resolve()
    pretrained_path = Path(pretrained_checkpoint_path).resolve()
    work_dir = Path(work_dir).resolve()
    manifest = load_development_manifest(manifest_path)
    annotation = assert_development_annotation(annotation_path)
    required_ids = set(manifest["splits"]["fit"]) | set(
        manifest["splits"]["gate"]
    )
    if not required_ids <= set(annotation["video_ids"]):
        raise ValueError(
            "formal development manifest names absent development videos"
        )
    for path in (source_config_path, class_map_path, pretrained_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not video_root.is_dir() or any(
        component.lower()
        in {"test", "testing", "test_videos", "official_test"}
        for component in video_root.parts
    ):
        raise ValueError(
            "formal development video root must not be an official-test root"
        )
    preflight_parent = _validate_preflight_parent(
        preflight_finalization_path,
        expected_file_sha256=expected_preflight_file_sha256,
        expected_runtime_commit=runtime_commit,
    )

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
    custom.georoute_tokens_per_tubelet = FORMAL_TOKEN_BUDGET
    custom.georoute_context_tokens = 0
    custom.georoute_roi_fraction = 0.0
    custom.georoute_geometry_side_channel = bool(
        spec["geometry_side_channel"]
    )
    custom.georoute_absolute_position_enabled = True
    custom.georoute_absolute_coordinates_enabled = bool(
        spec["representation_enabled"]
    )
    custom.georoute_roi_relative_coordinates_enabled = bool(
        spec["representation_enabled"]
    )
    custom.georoute_geometry_projection_enabled = bool(
        spec["representation_enabled"]
    )
    custom.georoute_policy_temperature = 0.7
    custom.georoute_score_function_weight = 1.0
    custom.georoute_score_function_baseline_momentum = 0.95
    custom.georoute_score_function_temporal_reduction = "mean"
    custom.georoute_geometry_smoothness_weight = 0.0
    custom.georoute_area_prior_weight = 0.0
    custom.georoute_random_seed = int(seed)
    custom.georoute_max_batch_size = FORMAL_PER_RANK_BATCH_SIZE
    custom.georoute_pooling_mode = "uniform_selected"
    custom.georoute_adapter_mode = "coordinate_lineage_packed"
    custom.georoute_diagnostic_telemetry_enabled = True

    for split_name in ("train", "val", "test"):
        cfg.solver[split_name].batch_size = FORMAL_CONFIG_BATCH_SIZE
    cfg.solver.amp = True
    cfg.solver.fp16_compress = False
    cfg.solver.static_graph = True
    cfg.solver.ema = True
    cfg.scheduler.type = "LinearWarmupCosineAnnealingLR"
    cfg.scheduler.warmup_epoch = FORMAL_SCHEDULER_WARMUP_EPOCHS
    cfg.scheduler.max_epoch = FORMAL_SCHEDULER_MAX_EPOCHS
    cfg.workflow.end_epoch = FORMAL_EPOCHS
    cfg.workflow.val_start_epoch = FORMAL_EPOCHS
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.checkpoint_policy = "final_only"
    cfg.workflow.max_amp_retries_per_batch = 0
    cfg.workflow.fail_on_skipped_update = False
    cfg.workflow.require_successful_update_hook = True
    cfg.workflow.schedule_and_ema_on_success_only = False
    cfg.workflow.capture_amp_rng_state = False
    cfg.workflow.fail_on_nonfinite_loss = True
    cfg.post_processing.save_dict = True
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.georoute_development_profile = dict(enabled=True)
    cfg.georoute_diagnostic_telemetry = dict(enabled=True)
    cfg.georoute_protocol.status = (
        "official_comparable_three_seed_development_only"
    )
    cfg.georoute_protocol.official_test_open_allowed = False
    cfg.work_dir = str(work_dir)

    binding: dict[str, Any] = {
        "schema_version": FORMAL_DEVELOPMENT_BINDING_SCHEMA,
        "status": "BOUND_OFFICIAL_COMPARABLE_DEVELOPMENT_ONLY",
        "runtime_commit": runtime_commit,
        "arm": arm,
        "arm_spec": spec,
        "arm_spec_sha256": canonical_sha256(spec),
        "seed": int(seed),
        "epochs": FORMAL_EPOCHS,
        "source_config": str(source_config_path),
        "source_config_sha256": sha256_file(source_config_path),
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": manifest["manifest_file_sha256"],
        "fit_video_ids": list(manifest["splits"]["fit"]),
        "gate_video_ids": list(manifest["splits"]["gate"]),
        "training_video_ids": list(manifest["splits"]["fit"]),
        "evaluation_video_ids": list(manifest["splits"]["gate"]),
        "training_block_list_video_ids": list(
            manifest["splits"]["gate"]
        ),
        "evaluation_block_list_video_ids": list(
            manifest["splits"]["fit"]
        ),
        "development_annotation": annotation,
        "class_map_path": str(class_map_path),
        "class_map_sha256": sha256_file(class_map_path),
        "development_video_root": str(video_root),
        "pretrained_checkpoint_path": str(pretrained_path),
        "pretrained_checkpoint_sha256": sha256_file(pretrained_path),
        "work_dir": str(work_dir),
        "world_size": FORMAL_WORLD_SIZE,
        "config_batch_size": FORMAL_CONFIG_BATCH_SIZE,
        "per_rank_batch_size": FORMAL_PER_RANK_BATCH_SIZE,
        "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
        "optimizer": {
            "type": "AdamW",
            "lr": 1e-4,
            "weight_decay": 0.05,
            "clip_grad_norm": 1.0,
        },
        "scheduler": {
            "type": "LinearWarmupCosineAnnealingLR",
            "warmup_epoch": FORMAL_SCHEDULER_WARMUP_EPOCHS,
            "max_epoch": FORMAL_SCHEDULER_MAX_EPOCHS,
        },
        "amp": True,
        "fp16_compress": False,
        "ema": True,
        "static_graph": True,
        "deterministic_warn_only": True,
        "max_amp_retries_per_batch": 0,
        "fail_on_skipped_update": False,
        "schedule_and_ema_on_success_only": False,
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "preflight_parent": {
            "path": str(Path(preflight_finalization_path).resolve()),
            "file_sha256": expected_preflight_file_sha256,
            "finalization_sha256": preflight_parent[
                "finalization_sha256"
            ],
            "decision": preflight_parent["decision"],
        },
        "development_selection_allowed": True,
        "official_test_opened": False,
        "official_protocol_freeze_authorized": False,
        "paper_grade_result_record_emitted": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_official_development_binding = binding
    cfg.georoute_runtime_binding = binding
    return cfg


def validate_formal_development_binding(
    binding: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    binding = dict(binding)
    if (
        not _self_hash_matches(binding, field="binding_sha256")
        or binding.get("schema_version") != FORMAL_DEVELOPMENT_BINDING_SCHEMA
        or binding.get("status")
        != "BOUND_OFFICIAL_COMPARABLE_DEVELOPMENT_ONLY"
        or binding.get("arm") not in FORMAL_DEVELOPMENT_ARMS
        or binding.get("arm_spec")
        != FORMAL_DEVELOPMENT_ARMS[binding.get("arm")]
        or binding.get("arm_spec_sha256")
        != canonical_sha256(binding["arm_spec"])
        or int(binding.get("seed", -1)) not in FORMAL_DEVELOPMENT_SEEDS
        or int(binding.get("epochs", -1)) != FORMAL_EPOCHS
        or int(binding.get("world_size", -1)) != FORMAL_WORLD_SIZE
        or int(binding.get("config_batch_size", -1))
        != FORMAL_CONFIG_BATCH_SIZE
        or int(binding.get("per_rank_batch_size", -1))
        != FORMAL_PER_RANK_BATCH_SIZE
        or int(binding.get("global_batch_size", -1))
        != FORMAL_GLOBAL_BATCH_SIZE
        or binding.get("amp") is not True
        or binding.get("fp16_compress") is not False
        or binding.get("ema") is not True
        or binding.get("static_graph") is not True
        or binding.get("deterministic_warn_only") is not True
        or int(binding.get("max_amp_retries_per_batch", -1)) != 0
        or binding.get("fail_on_skipped_update") is not False
        or binding.get("schedule_and_ema_on_success_only") is not False
        or binding.get("checkpoint_policy")
        != "final_epoch_ema_only_atomic"
        or binding.get("development_selection_allowed") is not True
        or binding.get("official_test_opened") is not False
        or binding.get("official_protocol_freeze_authorized") is not False
        or binding.get("paper_grade_result_record_emitted") is not False
        or binding.get("paper_claim_allowed") is not False
    ):
        raise ValueError("formal development binding is invalid")
    if seed is not None and int(seed) != int(binding["seed"]):
        raise ValueError("formal development CLI seed differs from binding")
    _full_hex(
        binding.get("runtime_commit"),
        length=40,
        name="formal development runtime_commit",
    )
    for key in (
        "source_config_sha256",
        "manifest_file_sha256",
        "class_map_sha256",
        "pretrained_checkpoint_sha256",
    ):
        _full_hex(binding.get(key), length=64, name=key)
    if (
        list(binding.get("training_video_ids", []))
        != list(binding.get("fit_video_ids", []))
        or list(binding.get("evaluation_video_ids", []))
        != list(binding.get("gate_video_ids", []))
    ):
        raise ValueError("formal development population binding changed")
    return binding


def validate_formal_development_config(
    cfg: Any,
    *,
    seed: int,
) -> dict[str, Any]:
    if "georoute_official_development_binding" not in cfg:
        raise ValueError("config lacks formal GeoRoute development binding")
    binding = validate_formal_development_binding(
        cfg.georoute_official_development_binding,
        seed=seed,
    )
    arm_spec = binding["arm_spec"]
    custom = cfg.model.backbone.custom
    if (
        str(Path(cfg.work_dir).resolve()) != binding["work_dir"]
        or int(cfg.workflow.get("end_epoch", -1)) != FORMAL_EPOCHS
        or int(cfg.workflow.get("val_start_epoch", -1)) != FORMAL_EPOCHS
        or int(cfg.workflow.get("val_loss_interval", 0)) != -1
        or int(cfg.workflow.get("val_eval_interval", 0)) != -1
        or cfg.workflow.get("checkpoint_policy") != "final_only"
        or int(cfg.workflow.get("max_amp_retries_per_batch", -1)) != 0
        or cfg.workflow.get("fail_on_skipped_update") is not False
        or cfg.workflow.get("require_successful_update_hook") is not True
        or cfg.workflow.get("schedule_and_ema_on_success_only") is not False
        or cfg.workflow.get("fail_on_nonfinite_loss") is not True
        or cfg.solver.get("amp") is not True
        or cfg.solver.get("fp16_compress") is not False
        or cfg.solver.get("ema") is not True
        or cfg.solver.get("static_graph") is not True
        or any(
            int(cfg.solver[split].get("batch_size", -1))
            != FORMAL_CONFIG_BATCH_SIZE
            for split in ("train", "val", "test")
        )
        or cfg.scheduler.get("type")
        != "LinearWarmupCosineAnnealingLR"
        or int(cfg.scheduler.get("warmup_epoch", -1))
        != FORMAL_SCHEDULER_WARMUP_EPOCHS
        or int(cfg.scheduler.get("max_epoch", -1))
        != FORMAL_SCHEDULER_MAX_EPOCHS
        or custom.get("georoute_route_mode") != arm_spec["route_mode"]
        or custom.get("georoute_policy_estimator")
        != arm_spec["policy_estimator"]
        or int(custom.get("georoute_random_seed", -1)) != int(seed)
        or int(custom.get("georoute_max_batch_size", -1))
        != FORMAL_PER_RANK_BATCH_SIZE
        or custom.get("georoute_score_function_temporal_reduction")
        != "mean"
        or cfg.evaluation.get("subset") != "training"
        or cfg.georoute_protocol.get("official_test_open_allowed") is not False
    ):
        raise ValueError(
            "formal development config violates the frozen official-semantics "
            "matched protocol"
        )
    for split_name in ("train", "val", "test"):
        if cfg.dataset[split_name].get("subset_name") != "training":
            raise ValueError(
                "formal development config opened a non-training subset"
            )
    if (
        list(cfg.dataset.train.get("block_list", []))
        != list(binding["training_block_list_video_ids"])
        or list(cfg.dataset.val.get("block_list", []))
        != list(binding["evaluation_block_list_video_ids"])
        or list(cfg.dataset.test.get("block_list", []))
        != list(binding["evaluation_block_list_video_ids"])
    ):
        raise ValueError("formal development block-list binding changed")
    return binding


def require_formal_world2_slurm() -> str:
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = [
        item
        for item in str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if item
    ]
    world_size = int(os.environ.get("WORLD_SIZE", "-1"))
    if (
        not job_id.isdigit()
        or len(visible) != FORMAL_WORLD_SIZE
        or world_size != FORMAL_WORLD_SIZE
    ):
        raise RuntimeError(
            "formal GeoRoute development requires two Slurm GPUs and two ranks"
        )
    return job_id


def require_clean_formal_checkout(
    *,
    expected_commit: str,
    root: Path,
) -> None:
    expected_commit = _full_hex(
        expected_commit, length=40, name="expected_commit"
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip().lower()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    if head != expected_commit or status:
        raise RuntimeError(
            "formal GeoRoute development requires its exact clean commit"
        )


def build_formal_checkpoint_metadata(
    cfg: Any,
    *,
    seed: int,
    epoch: int,
    successful_updates: int,
    train_batches_per_epoch: int,
    amp_skipped_attempts: int,
    max_amp_retries_observed: int,
    world_size: int,
) -> dict[str, Any]:
    binding = validate_formal_development_config(cfg, seed=seed)
    if int(epoch) != FORMAL_EPOCHS - 1:
        raise ValueError("formal checkpoint must be the final epoch")
    if int(world_size) != FORMAL_WORLD_SIZE:
        raise ValueError("formal checkpoint world size changed")
    metadata: dict[str, Any] = {
        "schema_version": FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA,
        "runtime_commit": binding["runtime_commit"],
        "binding_sha256": binding["binding_sha256"],
        "arm": binding["arm"],
        "seed": int(seed),
        "epoch": int(epoch),
        "successful_updates": int(successful_updates),
        "train_batches_per_epoch": int(train_batches_per_epoch),
        "amp_skipped_attempts": int(amp_skipped_attempts),
        "max_amp_retries_observed": int(max_amp_retries_observed),
        "world_size": int(world_size),
        "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
        "checkpoint_policy": "final_epoch_ema_only_atomic",
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    metadata["metadata_sha256"] = canonical_sha256(metadata)
    return metadata


def validate_formal_checkpoint_sidecar(
    checkpoint_path: str | Path,
    *,
    binding: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify the atomic final-checkpoint commit marker and frozen binding."""

    checkpoint = Path(checkpoint_path).resolve()
    sidecar_path = Path(str(checkpoint) + ".metadata.json")
    if (
        not checkpoint.is_file()
        or checkpoint.is_symlink()
        or not sidecar_path.is_file()
        or sidecar_path.is_symlink()
    ):
        raise FileNotFoundError(
            "formal development checkpoint and metadata sidecar are required"
        )
    sidecar = read_json(sidecar_path)
    unsigned_sidecar = dict(sidecar)
    observed_sidecar_hash = unsigned_sidecar.pop("sidecar_sha256", None)
    metadata = sidecar.get("experiment_metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("formal checkpoint sidecar lacks experiment metadata")
    metadata = dict(metadata)
    unsigned_metadata = dict(metadata)
    observed_metadata_hash = unsigned_metadata.pop("metadata_sha256", None)
    if (
        sidecar.get("schema_version")
        != FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA
        or str(Path(str(sidecar.get("checkpoint_path", ""))).resolve())
        != str(checkpoint)
        or sidecar.get("checkpoint_sha256") != sha256_file(checkpoint)
        or observed_sidecar_hash != canonical_sha256(unsigned_sidecar)
        or metadata.get("schema_version")
        != FORMAL_DEVELOPMENT_CHECKPOINT_SIDECAR_SCHEMA
        or observed_metadata_hash != canonical_sha256(unsigned_metadata)
        or int(metadata.get("epoch", -1)) != FORMAL_EPOCHS - 1
        or int(metadata.get("world_size", -1)) != FORMAL_WORLD_SIZE
        or int(metadata.get("global_batch_size", -1))
        != FORMAL_GLOBAL_BATCH_SIZE
        or metadata.get("checkpoint_policy")
        != "final_epoch_ema_only_atomic"
        or int(metadata.get("max_amp_retries_observed", -1)) != 0
        or metadata.get("official_test_opened") is not False
        or metadata.get("paper_claim_allowed") is not False
    ):
        raise ValueError("formal checkpoint sidecar is invalid")
    train_batches = int(metadata.get("train_batches_per_epoch", -1))
    successful_updates = int(metadata.get("successful_updates", -1))
    skipped_attempts = int(metadata.get("amp_skipped_attempts", -1))
    expected_consumed_batches = FORMAL_EPOCHS * train_batches
    if (
        train_batches <= 0
        or successful_updates < 0
        or skipped_attempts < 0
        or successful_updates + skipped_attempts != expected_consumed_batches
    ):
        raise ValueError(
            "formal checkpoint update accounting differs from the frozen "
            "official transition semantics"
        )
    if binding is not None:
        validated_binding = validate_formal_development_binding(binding)
        if (
            metadata.get("runtime_commit")
            != validated_binding["runtime_commit"]
            or metadata.get("binding_sha256")
            != validated_binding["binding_sha256"]
            or metadata.get("arm") != validated_binding["arm"]
            or int(metadata.get("seed", -1))
            != int(validated_binding["seed"])
        ):
            raise ValueError(
                "formal checkpoint sidecar is bound to another cell"
            )
    return sidecar


def official_published_metrics_sha256() -> str:
    return canonical_sha256(OFFICIAL_PUBLISHED_METRICS)


def build_protocol_manifest(
    *,
    runtime_commit: str,
    runtime_origin_ref: str,
    current_official_config_path: str | Path,
    georoute_source_config_path: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    repair_parent: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the immutable protocol manifest before any formal job is submitted."""

    runtime_commit = _full_hex(
        runtime_commit, length=40, name="runtime_commit"
    )
    runtime_origin_ref = str(runtime_origin_ref)
    if (
        not runtime_origin_ref.startswith("refs/remotes/origin/")
        or any(
            character in runtime_origin_ref
            for character in (" ", "\t", "\n", "\r", "\x00")
        )
    ):
        raise ValueError("runtime_origin_ref must be a full origin ref")

    current_official_config_path = Path(current_official_config_path).resolve()
    georoute_source_config_path = Path(georoute_source_config_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    development_annotation_path = Path(
        development_annotation_path
    ).resolve()
    class_map_path = Path(class_map_path).resolve()
    development_video_root = Path(development_video_root).resolve()
    pretrained_checkpoint_path = Path(pretrained_checkpoint_path).resolve()
    for path in (
        current_official_config_path,
        georoute_source_config_path,
        manifest_path,
        development_annotation_path,
        class_map_path,
        pretrained_checkpoint_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not development_video_root.is_dir():
        raise FileNotFoundError(development_video_root)
    if any(
        part.lower() in {"test", "testing", "official_test", "test_videos"}
        for part in development_video_root.parts
    ):
        raise ValueError("formal development video root points at a test root")
    if sha256_file(current_official_config_path) != OFFICIAL_CONFIG_SHA256:
        raise ValueError(
            "current official-reference config differs from the pinned "
            "AdaTAD release config"
        )

    split_manifest = load_development_manifest(manifest_path)
    annotation = assert_development_annotation(development_annotation_path)
    required_ids = set(split_manifest["splits"]["fit"]) | set(
        split_manifest["splits"]["gate"]
    )
    if not required_ids <= set(annotation["video_ids"]):
        raise ValueError("formal development manifest references missing videos")

    repair_parent = dict(repair_parent)
    if (
        repair_parent.get("decision")
        != (
            "DDP_FP16_CAST_REPAIR_GATE_PASS_"
            "MATCHED_FORMAL_PROTOCOL_FREEZE_AUTHORIZED"
        )
        or repair_parent.get("all_arms_passed") is not True
        or repair_parent.get("repair_gate_passed") is not True
        or repair_parent.get("matched_formal_protocol_freeze_authorized")
        is not True
        or repair_parent.get("official_protocol_freeze_authorized") is not False
        or repair_parent.get("performance_metrics") != {}
        or repair_parent.get("performance_inference_allowed") is not False
        or repair_parent.get("official_test_opened") is not False
        or repair_parent.get("paper_claim_allowed") is not False
        or not _self_hash_matches(
            repair_parent, field="finalization_sha256"
        )
    ):
        raise ValueError(
            "formal protocol requires the sealed passing no-compression repair gate"
        )

    payload: dict[str, Any] = {
        "schema_version": OFFICIAL_COMPARABLE_PROTOCOL_SCHEMA,
        "status": "FROZEN_BEFORE_FORMAL_PREFLIGHT",
        "runtime_commit": runtime_commit,
        "runtime_origin_ref": runtime_origin_ref,
        "upstream_anchor": {
            "repository": OFFICIAL_UPSTREAM_REPOSITORY,
            "release_commit": OFFICIAL_UPSTREAM_RELEASE_COMMIT,
            "config_relative_path": OFFICIAL_CONFIG_RELATIVE_PATH,
            "config_sha256": OFFICIAL_CONFIG_SHA256,
            "released_model_drive_id": OFFICIAL_RELEASED_MODEL_DRIVE_ID,
            "released_log_drive_id": OFFICIAL_RELEASED_LOG_DRIVE_ID,
            "published_metrics": dict(OFFICIAL_PUBLISHED_METRICS),
            "published_metrics_sha256": official_published_metrics_sha256(),
            "released_checkpoint_eval_tolerance_pp": (
                RELEASED_CHECKPOINT_EVAL_TOLERANCE_PP
            ),
            "new_train_average_tolerance_pp": (
                OFFICIAL_TRAIN_REPRO_AVERAGE_TOLERANCE_PP
            ),
            "new_train_per_tiou_tolerance_pp": (
                OFFICIAL_TRAIN_REPRO_PER_TIOU_TOLERANCE_PP
            ),
        },
        "current_source_bridge": {
            "official_config_path": str(current_official_config_path),
            "official_config_sha256": sha256_file(
                current_official_config_path
            ),
            "georoute_source_config_path": str(
                georoute_source_config_path
            ),
            "georoute_source_config_sha256": sha256_file(
                georoute_source_config_path
            ),
            "fp16_compression_official_anchor": True,
            "fp16_compression_matched_native_arms": False,
            "bridge_sequence": [
                "upstream_release_official_recipe",
                "current_source_official_recipe",
                "current_source_official_recipe_no_fp16_compression",
                "current_source_native_dense_no_fp16_compression",
            ],
        },
        "development_inputs": {
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": split_manifest[
                "manifest_file_sha256"
            ],
            "fit_video_ids": list(split_manifest["splits"]["fit"]),
            "gate_video_ids": list(split_manifest["splits"]["gate"]),
            "development_annotation": annotation,
            "class_map_path": str(class_map_path),
            "class_map_sha256": sha256_file(class_map_path),
            "development_video_root": str(development_video_root),
            "pretrained_checkpoint_path": str(
                pretrained_checkpoint_path
            ),
            "pretrained_checkpoint_sha256": sha256_file(
                pretrained_checkpoint_path
            ),
        },
        "formal_development_matrix": {
            "arms": copy.deepcopy(FORMAL_DEVELOPMENT_ARMS),
            "arm_order": list(FORMAL_DEVELOPMENT_ARM_ORDER),
            "seeds": list(FORMAL_DEVELOPMENT_SEEDS),
            "epochs": FORMAL_EPOCHS,
            "scheduler": {
                "type": "LinearWarmupCosineAnnealingLR",
                "warmup_epoch": FORMAL_SCHEDULER_WARMUP_EPOCHS,
                "max_epoch": FORMAL_SCHEDULER_MAX_EPOCHS,
            },
            "config_batch_size": FORMAL_CONFIG_BATCH_SIZE,
            "per_rank_batch_size": FORMAL_PER_RANK_BATCH_SIZE,
            "world_size": FORMAL_WORLD_SIZE,
            "global_batch_size": FORMAL_GLOBAL_BATCH_SIZE,
            "token_budget": FORMAL_TOKEN_BUDGET,
            "native_source_hw": [
                FORMAL_SOURCE_HEIGHT,
                FORMAL_SOURCE_WIDTH,
            ],
            "native_source_grid_hw": list(FORMAL_SOURCE_GRID_HW),
            "native_source_token_count": FORMAL_NATIVE_TOKEN_COUNT,
            "fp16_compress": False,
            "amp": True,
            "ema": True,
            "static_graph": True,
            "checkpoint_selection": "final_epoch_ema_only",
            "official_test_opened": False,
        },
        "selection_rule": {
            "primary": "mean(mAP@0.6,mAP@0.7)",
            "native_selector_must_beat": ["fixed_lattice", "random"],
            "paired_seed_delta_required_positive_for_every_seed": True,
            "selector_cost_must_be_below_dense_for_every_seed": True,
            "development_cost_metric": (
                "model_and_postprocess_p50_ms_rank0_matched_only"
            ),
            "st_vs_pl_winner_requires_strict_accuracy_cost_pareto_dominance": True,
            "pareto_rule": (
                "paired_seed_noninferior_on_both_axes_and_strict_on_each_"
                "mean_axis"
            ),
            "ambiguous_or_failed_gate_action": "HOLD_NO_OFFICIAL_TEST",
            "geometry_zoom_allowed": False,
        },
        "sealed_test_policy": {
            "allowed_during_preflight": False,
            "allowed_during_development_matrix": False,
            "single_open_after_method_freeze": True,
            "released_checkpoint_anchor_evaluated_in_same_sealed_open": True,
            "paper_result_before_sealed_open": False,
        },
        "repair_parent": {
            "runtime_commit": repair_parent["runtime_commit"],
            "finalization_sha256": repair_parent["finalization_sha256"],
            "decision": repair_parent["decision"],
        },
        "performance_inference_allowed": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["protocol_sha256"] = canonical_sha256(payload)
    return validate_protocol_manifest(payload)


def validate_protocol_manifest(payload: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    if (
        not _self_hash_matches(payload, field="protocol_sha256")
        or payload.get("schema_version")
        != OFFICIAL_COMPARABLE_PROTOCOL_SCHEMA
        or payload.get("status") != "FROZEN_BEFORE_FORMAL_PREFLIGHT"
        or payload.get("performance_inference_allowed") is not False
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
    ):
        raise ValueError("official-comparable protocol manifest is invalid")
    _full_hex(
        payload.get("runtime_commit"),
        length=40,
        name="protocol runtime_commit",
    )
    upstream = _mapping(
        payload.get("upstream_anchor"), name="upstream anchor"
    )
    if (
        upstream.get("repository") != OFFICIAL_UPSTREAM_REPOSITORY
        or upstream.get("release_commit")
        != OFFICIAL_UPSTREAM_RELEASE_COMMIT
        or upstream.get("config_relative_path")
        != OFFICIAL_CONFIG_RELATIVE_PATH
        or upstream.get("config_sha256") != OFFICIAL_CONFIG_SHA256
        or upstream.get("published_metrics")
        != OFFICIAL_PUBLISHED_METRICS
        or upstream.get("published_metrics_sha256")
        != official_published_metrics_sha256()
    ):
        raise ValueError("official upstream anchor changed")
    matrix = _mapping(
        payload.get("formal_development_matrix"),
        name="formal development matrix",
    )
    if (
        matrix.get("arms") != FORMAL_DEVELOPMENT_ARMS
        or tuple(matrix.get("arm_order", ()))
        != FORMAL_DEVELOPMENT_ARM_ORDER
        or tuple(matrix.get("seeds", ())) != FORMAL_DEVELOPMENT_SEEDS
        or int(matrix.get("epochs", -1)) != FORMAL_EPOCHS
        or int(matrix.get("config_batch_size", -1))
        != FORMAL_CONFIG_BATCH_SIZE
        or int(matrix.get("per_rank_batch_size", -1))
        != FORMAL_PER_RANK_BATCH_SIZE
        or int(matrix.get("world_size", -1)) != FORMAL_WORLD_SIZE
        or int(matrix.get("global_batch_size", -1))
        != FORMAL_GLOBAL_BATCH_SIZE
        or matrix.get("native_source_hw")
        != [FORMAL_SOURCE_HEIGHT, FORMAL_SOURCE_WIDTH]
        or matrix.get("native_source_grid_hw")
        != list(FORMAL_SOURCE_GRID_HW)
        or int(matrix.get("native_source_token_count", -1))
        != FORMAL_NATIVE_TOKEN_COUNT
        or matrix.get("fp16_compress") is not False
        or matrix.get("official_test_opened") is not False
    ):
        raise ValueError("formal development matrix changed")
    return payload


def validate_world2_kat_receipt(
    payload: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    expected_slurm_job_id: str | None = None,
) -> dict[str, Any]:
    result = dict(payload)
    if not _self_hash_matches(result, field="kat_sha256"):
        raise ValueError("world2 DDP KAT self-hash mismatch")
    runtime_commit = _full_hex(
        result.get("runtime_commit"),
        length=40,
        name="world2 KAT runtime_commit",
    )
    slurm_job_id = str(result.get("slurm_job_id", ""))
    reduced = _mapping(
        result.get("reduced_scaled_gradient"),
        name="reduced scaled gradient",
    )
    unscaled = _mapping(
        result.get("unscaled_gradient"), name="unscaled gradient"
    )
    shadow = _mapping(
        result.get("detached_fp16_cast_shadow"),
        name="detached FP16 shadow",
    )
    expected_scaled = 80000.0
    if (
        result.get("schema_version") != OFFICIAL_DDP_WORLD2_KAT_SCHEMA
        or result.get("status") != OFFICIAL_DDP_WORLD2_KAT_PASS
        or not slurm_job_id.isdigit()
        or int(result.get("world_size", -1)) != FORMAL_WORLD_SIZE
        or result.get("backend") != "nccl"
        or result.get("comm_hook_registration_invoked") is not False
        or result.get("default_fp32_ddp_reduction_completed") is not True
        or result.get("optimizer_update_completed_on_all_ranks") is not True
        or result.get("rank_local_scaled_gradient_targets")
        != [70000.0, 90000.0]
        or reduced.get("dtype") != "torch.float32"
        or reduced.get("finite") is not True
        or not math.isclose(
            float(reduced.get("max_abs", math.nan)),
            expected_scaled,
            rel_tol=1e-6,
            abs_tol=1e-3,
        )
        or unscaled.get("dtype") != "torch.float32"
        or unscaled.get("finite") is not True
        or not math.isclose(
            float(unscaled.get("max_abs", math.nan)),
            expected_scaled / 65536.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or shadow.get("dtype") != "torch.float16"
        or shadow.get("finite") is not False
        or int(shadow.get("nonfinite_count", 0)) < 1
        or result.get("checkpoint_emitted") is not False
        or result.get("prediction_emitted") is not False
        or result.get("evaluator_invoked") is not False
        or result.get("official_test_opened") is not False
        or result.get("performance_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("world2 default FP32 DDP KAT receipt is invalid")
    if expected_commit is not None and runtime_commit != str(
        expected_commit
    ).lower():
        raise ValueError("world2 DDP KAT commit mismatch")
    if (
        expected_slurm_job_id is not None
        and slurm_job_id != str(expected_slurm_job_id)
    ):
        raise ValueError("world2 DDP KAT Slurm Job ID mismatch")
    return result


def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload
