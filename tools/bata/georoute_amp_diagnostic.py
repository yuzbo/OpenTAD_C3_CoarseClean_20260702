"""Fail-closed real-batch AMP diagnosis for the GeoRoute PL/ST pair.

The observer is opt-in and runs inside the production training engine.  It
publishes numerical provenance only: no checkpoint, prediction, evaluator,
official-test result, or paper claim is permitted.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tools.bata.georoute_estimator_pilot_contract import (
    PILOT_ARMS,
    PILOT_K,
    PILOT_SEED,
    bind_pilot_config,
    pilot_arm_spec,
)
from tools.bata.georoute_experiment_contract import canonical_sha256, sha256_file


AMP_DIAGNOSTIC_STUDY_ID = "georoute_real_batch_amp_diagnostic_v1"
AMP_DIAGNOSTIC_BINDING_SCHEMA = "georoute_real_batch_amp_binding_v3"
AMP_DIAGNOSTIC_RECEIPT_SCHEMA = "georoute_real_batch_amp_receipt_v1"
AMP_DIAGNOSTIC_STAGE_SCHEMA = "georoute_real_batch_amp_stage_v1"
AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA = "georoute_real_batch_amp_deployment_v1"
AMP_DIAGNOSTIC_FINALIZATION_SCHEMA = "georoute_real_batch_amp_finalization_v1"
AMP_DIAGNOSTIC_ARMS = (
    "residual_pl_rep_off",
    "residual_st_rep_off",
)
AMP_DIAGNOSTIC_MAX_BATCHES = 1
AMP_DIAGNOSTIC_RETRY_LIMIT = 12
AMP_DIAGNOSTIC_INITIAL_SCALE = 65536.0
AMP_DIAGNOSTIC_PL_MAX_LOCALIZED_SUCCESS_SCALE = 128.0
AMP_DIAGNOSTIC_PROFILE = "diagnostic"

AMP_STABILITY_PROFILE = "stability_gate"
AMP_STABILITY_STUDY_ID = "georoute_real_data_amp_stability_v1"
AMP_STABILITY_BINDING_SCHEMA = "georoute_real_data_amp_stability_binding_v1"
AMP_STABILITY_RECEIPT_SCHEMA = "georoute_real_data_amp_stability_receipt_v1"
AMP_STABILITY_STAGE_SCHEMA = "georoute_real_data_amp_stability_stage_v1"
AMP_STABILITY_DEPLOYMENT_SCHEMA = (
    "georoute_real_data_amp_stability_deployment_v1"
)
AMP_STABILITY_FINALIZATION_SCHEMA = (
    "georoute_real_data_amp_stability_finalization_v1"
)
AMP_STABILITY_MAX_BATCHES = 32
AMP_STABILITY_RETRY_LIMIT = 0

AMP_STABILITY_V2_PROFILE = "stability_official_semantics_v2"
AMP_STABILITY_V2_STUDY_ID = "georoute_real_data_amp_stability_v2"
AMP_STABILITY_V2_BINDING_SCHEMA = (
    "georoute_real_data_amp_stability_official_semantics_binding_v2"
)
AMP_STABILITY_V2_RECEIPT_SCHEMA = (
    "georoute_real_data_amp_stability_official_semantics_receipt_v2"
)
AMP_STABILITY_V2_STAGE_SCHEMA = (
    "georoute_real_data_amp_stability_official_semantics_stage_v2"
)
AMP_STABILITY_V2_DEPLOYMENT_SCHEMA = (
    "georoute_real_data_amp_stability_official_semantics_deployment_v2"
)
AMP_STABILITY_V2_FINALIZATION_SCHEMA = (
    "georoute_real_data_amp_stability_official_semantics_finalization_v2"
)
AMP_STABILITY_V2_SEED = 4417
AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS = (3407, 3408, 3409)
AMP_STABILITY_V2_MAX_BATCHES = 64
AMP_STABILITY_V2_RETRY_LIMIT = 0
AMP_STABILITY_V2_MAX_SKIPS = 2
AMP_STABILITY_V2_MAX_CONSECUTIVE_SKIPS = 1
AMP_STABILITY_V2_MIN_SCALE = 16384.0
AMP_STABILITY_V2_STABLE_TAIL_BATCHES = 16
AMP_STABILITY_V2_MAX_CROSS_ARM_SKIP_DELTA = 1
AMP_STABILITY_V2_MAX_FINAL_SCALE_RATIO = 2.0


def amp_protocol_spec(profile: str) -> dict[str, Any]:
    """Return the immutable numerical-only execution contract for one profile."""

    profile = str(profile)
    if profile == AMP_DIAGNOSTIC_PROFILE:
        return {
            "profile": AMP_DIAGNOSTIC_PROFILE,
            "study_id": AMP_DIAGNOSTIC_STUDY_ID,
            "binding_schema": AMP_DIAGNOSTIC_BINDING_SCHEMA,
            "receipt_schema": AMP_DIAGNOSTIC_RECEIPT_SCHEMA,
            "stage_schema": AMP_DIAGNOSTIC_STAGE_SCHEMA,
            "deployment_schema": AMP_DIAGNOSTIC_DEPLOYMENT_SCHEMA,
            "finalization_schema": AMP_DIAGNOSTIC_FINALIZATION_SCHEMA,
            "max_batches": AMP_DIAGNOSTIC_MAX_BATCHES,
            "retry_limit": AMP_DIAGNOSTIC_RETRY_LIMIT,
            "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
            "seed": PILOT_SEED,
            "temporal_reduction": "sum",
            "zero_failed_attempts_required": False,
            "cell_directory": "diagnostic",
            "receipt_filename": "amp_diagnostic.json",
            "rendezvous_stage": "ampdiag",
            "config_status": "real_batch_amp_diagnostic_only",
            "receipt_running_status": "RUNNING_DIAGNOSTIC_ONLY",
            "receipt_pass_status": "PASS_DIAGNOSTIC_EXECUTION_ONLY",
            "receipt_fail_status": "FAIL_DIAGNOSTIC_EXECUTION",
            "stage_pass_status": "PASS_STAGE_DIAGNOSTIC_ONLY",
            "stage_fail_status": "FAIL_STAGE_DIAGNOSTIC_EXECUTION",
            "stage_wrapper_fail_status": (
                "FAIL_STAGE_WRAPPER_PREVALIDATION_OR_SEALING"
            ),
            "deployment_status": (
                "SUBMITTED_REAL_BATCH_AMP_DIAGNOSTIC_ONLY"
            ),
            "finalizer_submission_status": (
                "SUBMITTED_DIAGNOSTIC_FINALIZER_AFTERANY"
            ),
            "stage_release_status": (
                "RELEASED_DIAGNOSTIC_STAGES_AFTER_IMMUTABLE_RECEIPTS"
            ),
            "complete_status": "COMPLETE_NUMERICAL_DIAGNOSTIC_ONLY",
            "incomplete_status": "INCOMPLETE_NUMERICAL_DIAGNOSTIC",
            "job_prefix": "gramp",
        }
    if profile == AMP_STABILITY_PROFILE:
        return {
            "profile": AMP_STABILITY_PROFILE,
            "study_id": AMP_STABILITY_STUDY_ID,
            "binding_schema": AMP_STABILITY_BINDING_SCHEMA,
            "receipt_schema": AMP_STABILITY_RECEIPT_SCHEMA,
            "stage_schema": AMP_STABILITY_STAGE_SCHEMA,
            "deployment_schema": AMP_STABILITY_DEPLOYMENT_SCHEMA,
            "finalization_schema": AMP_STABILITY_FINALIZATION_SCHEMA,
            "max_batches": AMP_STABILITY_MAX_BATCHES,
            "retry_limit": AMP_STABILITY_RETRY_LIMIT,
            "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
            "seed": PILOT_SEED,
            "temporal_reduction": "mean",
            "zero_failed_attempts_required": True,
            "cell_directory": "stability",
            "receipt_filename": "amp_stability.json",
            "rendezvous_stage": "ampstable",
            "config_status": "real_data_amp_stability_gate_only",
            "receipt_running_status": "RUNNING_STABILITY_GATE_ONLY",
            "receipt_pass_status": "PASS_STABILITY_GATE_EXECUTION_ONLY",
            "receipt_fail_status": "FAIL_STABILITY_GATE_EXECUTION",
            "stage_pass_status": "PASS_STAGE_STABILITY_GATE_ONLY",
            "stage_fail_status": "FAIL_STAGE_STABILITY_GATE_EXECUTION",
            "stage_wrapper_fail_status": (
                "FAIL_STABILITY_STAGE_WRAPPER_PREVALIDATION_OR_SEALING"
            ),
            "deployment_status": (
                "SUBMITTED_REAL_DATA_AMP_STABILITY_GATE_ONLY"
            ),
            "finalizer_submission_status": (
                "SUBMITTED_STABILITY_FINALIZER_AFTERANY"
            ),
            "stage_release_status": (
                "RELEASED_STABILITY_STAGES_AFTER_IMMUTABLE_RECEIPTS"
            ),
            "complete_status": (
                "COMPLETE_REAL_DATA_AMP_STABILITY_GATE_ONLY"
            ),
            "incomplete_status": "INCOMPLETE_REAL_DATA_AMP_STABILITY_GATE",
            "job_prefix": "grstab",
        }
    if profile == AMP_STABILITY_V2_PROFILE:
        return {
            "profile": AMP_STABILITY_V2_PROFILE,
            "study_id": AMP_STABILITY_V2_STUDY_ID,
            "binding_schema": AMP_STABILITY_V2_BINDING_SCHEMA,
            "receipt_schema": AMP_STABILITY_V2_RECEIPT_SCHEMA,
            "stage_schema": AMP_STABILITY_V2_STAGE_SCHEMA,
            "deployment_schema": AMP_STABILITY_V2_DEPLOYMENT_SCHEMA,
            "finalization_schema": AMP_STABILITY_V2_FINALIZATION_SCHEMA,
            "max_batches": AMP_STABILITY_V2_MAX_BATCHES,
            "retry_limit": AMP_STABILITY_V2_RETRY_LIMIT,
            "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
            "seed": AMP_STABILITY_V2_SEED,
            "temporal_reduction": "mean",
            "zero_failed_attempts_required": False,
            "max_skipped_attempts": AMP_STABILITY_V2_MAX_SKIPS,
            "max_consecutive_skips": AMP_STABILITY_V2_MAX_CONSECUTIVE_SKIPS,
            "minimum_scale": AMP_STABILITY_V2_MIN_SCALE,
            "stable_tail_batches": AMP_STABILITY_V2_STABLE_TAIL_BATCHES,
            "minimum_successful_updates": (
                AMP_STABILITY_V2_MAX_BATCHES - AMP_STABILITY_V2_MAX_SKIPS
            ),
            "max_cross_arm_skip_delta": (
                AMP_STABILITY_V2_MAX_CROSS_ARM_SKIP_DELTA
            ),
            "max_final_scale_ratio": AMP_STABILITY_V2_MAX_FINAL_SCALE_RATIO,
            "use_default_grad_scaler_constructor": True,
            "fail_on_skipped_update": False,
            "schedule_and_ema_on_success_only": False,
            "capture_amp_rng_state": True,
            "fail_on_nonfinite_loss": True,
            "cell_directory": "stability_v2",
            "receipt_filename": "amp_stability_v2.json",
            "rendezvous_stage": "ampstablev2",
            "config_status": "official_semantics_amp_stability_v2_only",
            "receipt_running_status": (
                "RUNNING_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_ONLY"
            ),
            "receipt_pass_status": (
                "PASS_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION_ONLY"
            ),
            "receipt_fail_status": (
                "FAIL_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION"
            ),
            "stage_pass_status": (
                "PASS_STAGE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_ONLY"
            ),
            "stage_fail_status": (
                "FAIL_STAGE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION"
            ),
            "stage_wrapper_fail_status": (
                "FAIL_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_STAGE_WRAPPER"
            ),
            "deployment_status": (
                "SUBMITTED_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_ONLY"
            ),
            "finalizer_submission_status": (
                "SUBMITTED_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_FINALIZER_AFTERANY"
            ),
            "stage_release_status": (
                "RELEASED_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_STAGES"
            ),
            "complete_status": (
                "COMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_ONLY"
            ),
            "incomplete_status": (
                "INCOMPLETE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2"
            ),
            "job_prefix": "grstabv2",
        }
    raise ValueError(f"unsupported AMP numerical protocol profile {profile!r}")


def amp_protocol_spec_for_binding(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    return amp_protocol_spec(
        str(binding.get("protocol_profile", AMP_DIAGNOSTIC_PROFILE))
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


def _full_hex(value: str, *, length: int, name: str) -> str:
    normalized = str(value).lower()
    if (
        len(normalized) != length
        or any(character not in "0123456789abcdef" for character in normalized)
    ):
        raise ValueError(f"{name} must be a full lowercase hexadecimal digest")
    return normalized


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def bind_amp_diagnostic_config(
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
    protocol_profile: str = AMP_DIAGNOSTIC_PROFILE,
    official_reference_config_path: str | Path | None = None,
):
    """Build one immutable, no-metric residual-PL/ST diagnostic config."""

    spec = amp_protocol_spec(protocol_profile)
    if arm not in AMP_DIAGNOSTIC_ARMS:
        raise ValueError("AMP diagnostic arm must be residual PL or matched ST")
    if int(seed) != int(spec["seed"]):
        raise ValueError("AMP diagnostic seed differs from its protocol profile")
    runtime_commit = _full_hex(
        runtime_commit,
        length=40,
        name="runtime_commit",
    )
    cfg = bind_pilot_config(
        source_config_path=source_config_path,
        arm=arm,
        # The pilot binder is reused only as a frozen configuration template.
        # The v2 execution/data-order seed is rebound below and is recorded
        # separately from this historical exploratory template seed.
        seed=PILOT_SEED,
        work_dir=work_dir,
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        development_video_root=development_video_root,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
    )
    parent_binding = dict(cfg.georoute_estimator_pilot_binding)
    work_dir = Path(work_dir).resolve()
    output_path = work_dir / str(spec["receipt_filename"])

    cfg.model.backbone.custom.georoute_amp_diagnostic_enabled = True
    cfg.model.backbone.custom.georoute_random_seed = int(spec["seed"])
    cfg.model.backbone.custom.georoute_score_function_temporal_reduction = (
        spec["temporal_reduction"]
    )
    cfg.workflow.end_epoch = 1
    cfg.workflow.val_start_epoch = 1
    cfg.workflow.val_loss_interval = -1
    cfg.workflow.val_eval_interval = -1
    cfg.workflow.disable_checkpoint = True
    cfg.workflow.max_train_iters = int(spec["max_batches"])
    cfg.workflow.max_amp_retries_per_batch = int(spec["retry_limit"])
    cfg.workflow.fail_on_skipped_update = bool(
        spec.get("fail_on_skipped_update", True)
    )
    cfg.workflow.require_successful_update_hook = True
    cfg.workflow.schedule_and_ema_on_success_only = bool(
        spec.get("schedule_and_ema_on_success_only", True)
    )
    cfg.workflow.capture_amp_rng_state = bool(
        spec.get("capture_amp_rng_state", False)
    )
    cfg.workflow.fail_on_nonfinite_loss = bool(
        spec.get("fail_on_nonfinite_loss", False)
    )
    cfg.post_processing.save_dict = False
    cfg.inference.load_from_raw_predictions = False
    cfg.inference.save_raw_prediction = False
    cfg.georoute_protocol.status = str(spec["config_status"])
    cfg.work_dir = str(work_dir)

    binding: dict[str, Any] = {
        "schema_version": spec["binding_schema"],
        "study_id": spec["study_id"],
        "protocol_profile": spec["profile"],
        "arm": arm,
        "arm_spec": pilot_arm_spec(arm),
        "seed": int(spec["seed"]),
        "token_budget": PILOT_K,
        "runtime_commit": runtime_commit,
        "work_dir": str(work_dir),
        "output_path": str(output_path),
        "max_batches": spec["max_batches"],
        "max_amp_retries_per_batch": spec["retry_limit"],
        "initial_scale": spec["initial_scale"],
        "score_function_temporal_reduction": spec[
            "temporal_reduction"
        ],
        "zero_failed_attempts_required": spec[
            "zero_failed_attempts_required"
        ],
        "source_config": parent_binding["source_config"],
        "source_config_sha256": parent_binding["source_config_sha256"],
        "manifest_path": parent_binding["manifest_path"],
        "manifest_file_sha256": parent_binding["manifest_file_sha256"],
        "fit_video_ids": list(parent_binding["fit_video_ids"]),
        "gate_video_ids": list(parent_binding["gate_video_ids"]),
        # SlidingWindowDataset.block_list excludes the named videos.  The failed
        # estimator pilot blocked Gate for train and Fit for val/test, so its
        # actual populations were Fit for training and Gate for development.
        # Bind both the included and excluded populations explicitly.
        "training_video_ids": list(parent_binding["fit_video_ids"]),
        "evaluation_video_ids": list(parent_binding["gate_video_ids"]),
        "training_block_list_video_ids": list(
            parent_binding["gate_video_ids"]
        ),
        "evaluation_block_list_video_ids": list(
            parent_binding["fit_video_ids"]
        ),
        "development_annotation": dict(
            parent_binding["development_annotation"]
        ),
        "class_map_path": parent_binding["class_map_path"],
        "class_map_sha256": parent_binding["class_map_sha256"],
        "development_video_root": parent_binding["development_video_root"],
        "pretrained_checkpoint_path": parent_binding[
            "pretrained_checkpoint_path"
        ],
        "pretrained_checkpoint_sha256": parent_binding[
            "pretrained_checkpoint_sha256"
        ],
        "parent_pilot_binding_sha256": parent_binding["binding_sha256"],
        "deterministic_same_config_reproduction": True,
        "exact_historical_batch_replay_claimed": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": True,
        "historical_pilot_seed_policy_matched": (
            spec["profile"] != AMP_STABILITY_V2_PROFILE
        ),
        "amp_diagnostic_telemetry_enabled": True,
        "checkpoint_disabled": True,
        "evaluator_invoked": False,
        "prediction_emitted": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "paper_claim_allowed": False,
    }
    if spec["profile"] == AMP_STABILITY_V2_PROFILE:
        if official_reference_config_path is None:
            raise ValueError(
                "official-semantics stability v2 requires an official reference config"
            )
        official_reference = Path(official_reference_config_path).resolve()
        if not official_reference.is_file() or official_reference.is_symlink():
            raise FileNotFoundError(
                "official-semantics stability v2 reference config is invalid"
            )
        from mmengine.config import Config

        official_cfg = Config.fromfile(str(official_reference))
        official_solver = _mapping(
            official_cfg.solver,
            name="official reference solver",
        )
        official_workflow = _mapping(
            official_cfg.workflow,
            name="official reference workflow",
        )
        official_transitions = {
            "amp_enabled": official_solver.get("amp") is True,
            "ema_enabled": official_solver.get("ema") is True,
            "clip_grad_l2norm": float(
                official_solver.get("clip_grad_norm", -1.0)
            ),
            "max_amp_retries_per_batch": int(
                official_workflow.get("max_amp_retries_per_batch", 0)
            ),
            "fail_on_skipped_update": bool(
                official_workflow.get("fail_on_skipped_update", False)
            ),
            "schedule_and_ema_on_success_only": bool(
                official_workflow.get(
                    "schedule_and_ema_on_success_only",
                    False,
                )
            ),
        }
        if official_transitions != {
            "amp_enabled": True,
            "ema_enabled": True,
            "clip_grad_l2norm": 1.0,
            "max_amp_retries_per_batch": 0,
            "fail_on_skipped_update": False,
            "schedule_and_ema_on_success_only": False,
        }:
            raise ValueError(
                "official reference no longer has the frozen AMP/scheduler/EMA "
                "transition semantics"
            )
        binding.update(
            {
                "template_seed": PILOT_SEED,
                "execution_seed": AMP_STABILITY_V2_SEED,
                "forbidden_future_paper_seeds": list(
                    AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS
                ),
                "paper_seed_disjoint": (
                    AMP_STABILITY_V2_SEED
                    not in AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS
                ),
                "use_default_grad_scaler_constructor": True,
                "observed_initial_scale_required": float(
                    AMP_DIAGNOSTIC_INITIAL_SCALE
                ),
                "fail_on_skipped_update": False,
                "max_amp_retries_per_batch": 0,
                "batch_replay_allowed": False,
                "schedule_and_ema_on_success_only": False,
                "scheduler_advances_per_consumed_batch": True,
                "ema_updates_per_consumed_batch": True,
                "capture_amp_rng_state": True,
                "fail_on_nonfinite_loss": True,
                "official_reference_config": str(official_reference),
                "official_reference_config_sha256": sha256_file(
                    official_reference
                ),
                "official_reference_transition_semantics": (
                    official_transitions
                ),
                "official_reference_transition_semantics_sha256": (
                    canonical_sha256(official_transitions)
                ),
                "official_prefix_transition_semantics_matched": True,
                "official_scheduler_advance_cadence_matched": True,
                "official_scheduler_hyperparameters_matched": False,
                "full_official_recipe_matched": False,
                "official_performance_comparable": False,
                "full_official_training_claimed": False,
                "development_prefix_only": True,
            }
        )
    binding["binding_sha256"] = canonical_sha256(binding)
    if "georoute_estimator_pilot_binding" in cfg:
        cfg.pop("georoute_estimator_pilot_binding")
    cfg.georoute_amp_diagnostic_binding = binding
    cfg.georoute_runtime_binding = binding
    return cfg


def validate_amp_diagnostic_binding(
    binding: Mapping[str, Any],
    *,
    seed: int | None = None,
) -> dict[str, Any]:
    binding = dict(_mapping(binding, name="AMP diagnostic binding"))
    if not _self_hash_matches(binding, field="binding_sha256"):
        raise ValueError("AMP diagnostic binding self-hash mismatch")
    spec = amp_protocol_spec_for_binding(binding)
    arm = str(binding.get("arm", ""))
    if (
        binding.get("schema_version") != spec["binding_schema"]
        or binding.get("study_id") != spec["study_id"]
        or arm not in AMP_DIAGNOSTIC_ARMS
        or binding.get("arm_spec") != PILOT_ARMS[arm]
        or int(binding.get("seed", -1)) != int(spec["seed"])
        or int(binding.get("token_budget", -1)) != PILOT_K
        or int(binding.get("max_batches", -1))
        != int(spec["max_batches"])
        or int(binding.get("max_amp_retries_per_batch", -1))
        != int(spec["retry_limit"])
        or float(binding.get("initial_scale", -1.0))
        != float(spec["initial_scale"])
        or str(
            binding.get(
                "score_function_temporal_reduction",
                "sum",
            )
        )
        != spec["temporal_reduction"]
        or bool(
            binding.get(
                "zero_failed_attempts_required",
                False,
            )
        )
        is not bool(spec["zero_failed_attempts_required"])
        or binding.get("deterministic_same_config_reproduction") is not True
        or binding.get("exact_historical_batch_replay_claimed") is not False
        or binding.get("deterministic_algorithms_enabled") is not True
        or binding.get("deterministic_warn_only") is not True
        or bool(binding.get("historical_pilot_seed_policy_matched"))
        is not bool(spec["profile"] != AMP_STABILITY_V2_PROFILE)
        or binding.get("amp_diagnostic_telemetry_enabled") is not True
        or binding.get("checkpoint_disabled") is not True
        or binding.get("evaluator_invoked") is not False
        or binding.get("prediction_emitted") is not False
        or binding.get("official_test_opened") is not False
        or binding.get("p2_p3_opened") is not False
        or binding.get("paper_claim_allowed") is not False
    ):
        raise ValueError("AMP diagnostic binding contract is invalid")
    if spec["profile"] == AMP_STABILITY_V2_PROFILE:
        if (
            int(binding.get("template_seed", -1)) != PILOT_SEED
            or int(binding.get("execution_seed", -1))
            != AMP_STABILITY_V2_SEED
            or tuple(binding.get("forbidden_future_paper_seeds", ()))
            != AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS
            or binding.get("paper_seed_disjoint") is not True
            or binding.get("use_default_grad_scaler_constructor") is not True
            or float(binding.get("observed_initial_scale_required", -1.0))
            != float(AMP_DIAGNOSTIC_INITIAL_SCALE)
            or binding.get("fail_on_skipped_update") is not False
            or binding.get("batch_replay_allowed") is not False
            or binding.get("schedule_and_ema_on_success_only") is not False
            or binding.get("scheduler_advances_per_consumed_batch") is not True
            or binding.get("ema_updates_per_consumed_batch") is not True
            or binding.get("capture_amp_rng_state") is not True
            or binding.get("fail_on_nonfinite_loss") is not True
            or binding.get("official_prefix_transition_semantics_matched")
            is not True
            or binding.get("official_scheduler_advance_cadence_matched")
            is not True
            or binding.get("official_scheduler_hyperparameters_matched")
            is not False
            or binding.get("full_official_recipe_matched") is not False
            or binding.get("official_performance_comparable") is not False
            or binding.get("full_official_training_claimed") is not False
            or binding.get("development_prefix_only") is not True
        ):
            raise ValueError(
                "official-semantics stability v2 binding contract is invalid"
            )
        _full_hex(
            str(binding.get("official_reference_config_sha256", "")),
            length=64,
            name="official_reference_config_sha256",
        )
        official_transitions = _mapping(
            binding.get("official_reference_transition_semantics"),
            name="official reference transition semantics",
        )
        if (
            dict(official_transitions)
            != {
                "amp_enabled": True,
                "ema_enabled": True,
                "clip_grad_l2norm": 1.0,
                "max_amp_retries_per_batch": 0,
                "fail_on_skipped_update": False,
                "schedule_and_ema_on_success_only": False,
            }
            or binding.get(
                "official_reference_transition_semantics_sha256"
            )
            != canonical_sha256(official_transitions)
        ):
            raise ValueError(
                "official-semantics stability v2 reference transition binding "
                "is invalid"
            )
    if seed is not None and int(seed) != int(binding["seed"]):
        raise ValueError("AMP diagnostic CLI seed differs from its binding")
    _full_hex(
        str(binding.get("runtime_commit", "")),
        length=40,
        name="AMP diagnostic runtime commit",
    )
    for key in (
        "source_config_sha256",
        "manifest_file_sha256",
        "class_map_sha256",
        "pretrained_checkpoint_sha256",
        "parent_pilot_binding_sha256",
    ):
        _full_hex(str(binding.get(key, "")), length=64, name=key)
    if (
        list(binding.get("training_video_ids", []))
        != list(binding.get("fit_video_ids", []))
        or list(binding.get("evaluation_video_ids", []))
        != list(binding.get("gate_video_ids", []))
        or list(binding.get("training_block_list_video_ids", []))
        != list(binding.get("gate_video_ids", []))
        or list(binding.get("evaluation_block_list_video_ids", []))
        != list(binding.get("fit_video_ids", []))
    ):
        raise ValueError("AMP diagnostic population binding changed")
    annotation = _mapping(
        binding.get("development_annotation"),
        name="development annotation",
    )
    _full_hex(str(annotation.get("sha256", "")), length=64, name="annotation")
    work_dir = Path(str(binding.get("work_dir", ""))).resolve()
    output_path = Path(str(binding.get("output_path", ""))).resolve()
    if output_path != work_dir / str(spec["receipt_filename"]):
        raise ValueError("AMP diagnostic output path is not work-dir bound")
    return binding


def validate_amp_diagnostic_config(cfg: Any, *, seed: int) -> dict[str, Any]:
    if "georoute_amp_diagnostic_binding" not in cfg:
        raise ValueError("config lacks GeoRoute AMP diagnostic binding")
    binding = validate_amp_diagnostic_binding(
        cfg.georoute_amp_diagnostic_binding,
        seed=seed,
    )
    workflow = _mapping(cfg.workflow, name="workflow")
    solver = _mapping(cfg.solver, name="solver")
    spec = amp_protocol_spec_for_binding(binding)
    if (
        str(Path(cfg.work_dir).resolve()) != binding["work_dir"]
        or int(workflow.get("end_epoch", -1)) != 1
        or int(workflow.get("max_train_iters", -1))
        != int(spec["max_batches"])
        or int(workflow.get("max_amp_retries_per_batch", -1))
        != int(spec["retry_limit"])
        or workflow.get("disable_checkpoint") is not True
        or workflow.get("fail_on_skipped_update")
        is not bool(spec.get("fail_on_skipped_update", True))
        or workflow.get("require_successful_update_hook") is not True
        or workflow.get("schedule_and_ema_on_success_only")
        is not bool(spec.get("schedule_and_ema_on_success_only", True))
        or int(workflow.get("val_start_epoch", -1)) < 1
        or int(workflow.get("val_loss_interval", 0)) != -1
        or int(workflow.get("val_eval_interval", 0)) != -1
        or solver.get("amp") is not True
        or float(solver.get("clip_grad_norm", -1.0)) <= 0.0
        or cfg.inference.get("load_from_raw_predictions") is not False
        or cfg.inference.get("save_raw_prediction") is not False
        or cfg.post_processing.get("save_dict") is not False
        or cfg.evaluation.get("subset") != "training"
        or cfg.model.backbone.custom.get(
            "georoute_amp_diagnostic_enabled"
        )
        is not True
        or cfg.model.backbone.custom.get(
            "georoute_score_function_temporal_reduction"
        )
        != spec["temporal_reduction"]
    ):
        raise ValueError("AMP diagnostic config violates its no-metric protocol")
    if spec["profile"] == AMP_STABILITY_V2_PROFILE and (
        workflow.get("capture_amp_rng_state") is not True
        or workflow.get("fail_on_nonfinite_loss") is not True
        or solver.get("ema") is not True
        or int(
            cfg.model.backbone.custom.get("georoute_random_seed", -1)
        )
        != AMP_STABILITY_V2_SEED
    ):
        raise ValueError(
            "official-semantics stability v2 config changed its frozen transitions"
        )
    for split_name in ("train", "val", "test"):
        if cfg.dataset[split_name].get("subset_name") != "training":
            raise ValueError("AMP diagnostic dataset left the development subset")
    if (
        list(cfg.dataset.train.get("block_list", []))
        != list(binding["training_block_list_video_ids"])
        or list(cfg.dataset.val.get("block_list", []))
        != list(binding["evaluation_block_list_video_ids"])
        or list(cfg.dataset.test.get("block_list", []))
        != list(binding["evaluation_block_list_video_ids"])
    ):
        raise ValueError("AMP diagnostic dataset block-list binding changed")
    return binding


def require_clean_git_checkout(*, expected_commit: str, root: Path) -> None:
    expected_commit = _full_hex(
        expected_commit,
        length=40,
        name="expected_commit",
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
        raise RuntimeError("AMP diagnostic requires its exact clean runtime commit")


def require_slurm_single_gpu() -> str:
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not job_id or not job_id.isdigit():
        raise RuntimeError("AMP diagnostic requires a numeric Slurm Job ID")
    if not visible or "," in visible:
        raise RuntimeError("AMP diagnostic requires one Slurm-visible GPU")
    return job_id


def diagnostic_cell_relative_path(
    *,
    arm: str,
    protocol_profile: str = AMP_DIAGNOSTIC_PROFILE,
) -> Path:
    if arm not in AMP_DIAGNOSTIC_ARMS:
        raise ValueError("unsupported AMP diagnostic arm")
    spec = amp_protocol_spec(protocol_profile)
    return Path(str(spec["cell_directory"])) / (
        f"{PILOT_ARMS[arm]['slug']}_{arm}"
    )


def validate_amp_diagnostic_job_receipt(
    jobs: Any,
    *,
    expected_finalizer: str | None = None,
) -> dict[str, Any]:
    if not isinstance(jobs, Mapping):
        raise ValueError("AMP diagnostic jobs must be a mapping")
    stages = jobs.get("stage")
    finalizer = str(jobs.get("finalizer", ""))
    if (
        not isinstance(stages, Mapping)
        or set(stages) != set(AMP_DIAGNOSTIC_ARMS)
        or not finalizer.isdigit()
    ):
        raise ValueError("AMP diagnostic job receipt has the wrong shape")
    normalized = {
        arm: str(stages[arm])
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    all_ids = [*normalized.values(), finalizer]
    if any(not job_id.isdigit() for job_id in all_ids):
        raise ValueError("AMP diagnostic job receipt contains a nonnumeric ID")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("AMP diagnostic job receipt reuses a Slurm ID")
    if expected_finalizer is not None and finalizer != str(expected_finalizer):
        raise ValueError("AMP diagnostic finalizer is not self-bound")
    return {"stage": normalized, "finalizer": finalizer}


def _tensor_bytes_and_descriptor(value: Any) -> tuple[bytes, dict[str, Any]]:
    import torch

    detached = value.detach().contiguous().to("cpu")
    raw = detached.view(torch.uint8).reshape(-1).numpy().tobytes()
    descriptor = {
        "kind": "tensor",
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return raw, descriptor


def _describe_data(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        _raw, descriptor = _tensor_bytes_and_descriptor(value)
        return descriptor
    if isinstance(value, Mapping):
        return {
            str(key): _describe_data(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_describe_data(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    representation = repr(value)
    return {
        "kind": "opaque",
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr_sha256": hashlib.sha256(
            representation.encode("utf-8", errors="replace")
        ).hexdigest(),
    }


def _tensor_numeric_summary(value: Any) -> dict[str, Any]:
    import torch

    detached = value.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = int(detached.numel())
    finite_values = detached.float().masked_select(finite_mask)
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "finite": finite_count == total_count,
        "finite_count": finite_count,
        "nonfinite_count": total_count - finite_count,
        "finite_min": (
            float(finite_values.min().item()) if finite_count else None
        ),
        "finite_max": (
            float(finite_values.max().item()) if finite_count else None
        ),
        "finite_mean": (
            float(finite_values.mean().item()) if finite_count else None
        ),
        "scalar_value": (
            float(finite_values.item())
            if total_count == 1 and finite_count == 1
            else None
        ),
    }


def _parameter_group(name: str) -> str:
    lowered = name.lower()
    if ".scout." in lowered or "score_function" in lowered:
        return "scout_score_function"
    if "sparse_adapter" in lowered or ".adapter." in lowered:
        return "adapter"
    if ".model.backbone." in lowered or ".backbone.backbone." in lowered:
        return "heavy_backbone"
    return "detector"


def _gradient_snapshot(model: Any) -> dict[str, Any]:
    import torch

    parameters: dict[str, Any] = {}
    grouped: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        if not bool(parameter.requires_grad):
            continue
        group = _parameter_group(str(name))
        group_summary = grouped.setdefault(
            group,
            {
                "parameter_count": 0,
                "parameters_with_gradient": 0,
                "missing_gradient_count": 0,
                "nonfinite_count": 0,
                "max_abs": None,
                "l2_norm": 0.0,
            },
        )
        group_summary["parameter_count"] += 1
        gradient = parameter.grad
        if gradient is None:
            parameters[str(name)] = {
                "group": group,
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "gradient_present": False,
            }
            group_summary["missing_gradient_count"] += 1
            continue
        detached = gradient.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = int(detached.numel())
        finite_values = detached.float().masked_select(finite_mask)
        max_abs = (
            float(finite_values.abs().max().item()) if finite_count else None
        )
        l2_tensor = (
            torch.linalg.vector_norm(finite_values)
            if finite_count
            else detached.new_zeros((), dtype=torch.float32)
        )
        l2_norm = (
            float(l2_tensor.item()) if bool(torch.isfinite(l2_tensor).item()) else None
        )
        parameters[str(name)] = {
            "group": group,
            "dtype": str(detached.dtype),
            "shape": list(detached.shape),
            "gradient_present": True,
            "finite_count": finite_count,
            "nonfinite_count": total_count - finite_count,
            "max_abs": max_abs,
            "l2_norm": l2_norm,
        }
        group_summary["parameters_with_gradient"] += 1
        group_summary["nonfinite_count"] += total_count - finite_count
        if max_abs is not None:
            group_summary["max_abs"] = max(
                float(group_summary["max_abs"] or 0.0),
                max_abs,
            )
        if l2_norm is not None:
            group_summary["l2_norm"] += l2_norm * l2_norm
    for summary in grouped.values():
        summary["l2_norm"] = math.sqrt(float(summary["l2_norm"]))
    return {
        "parameters": parameters,
        "groups": grouped,
        "has_nonfinite": any(
            int(summary["nonfinite_count"]) > 0 for summary in grouped.values()
        ),
        "nonfinite_groups": sorted(
            group
            for group, summary in grouped.items()
            if int(summary["nonfinite_count"]) > 0
        ),
    }


def _georoute_audit(model: Any) -> dict[str, Any] | None:
    unwrapped = getattr(model, "module", model)
    backbone = getattr(unwrapped, "backbone", None)
    audit = getattr(backbone, "latest_georoute_audit", None)
    if not isinstance(audit, Mapping):
        return None
    # Force a strict JSON round trip so accidental tensors/non-finite values
    # fail the diagnostic instead of producing an unauditable receipt.
    return json.loads(
        json.dumps(dict(audit), sort_keys=True, allow_nan=False)
    )


class RealBatchAmpDiagnosticObserver:
    """Record real-batch loss/gradient state without changing the graph."""

    def __init__(
        self,
        *,
        binding: Mapping[str, Any],
        output_path: str | Path,
        runtime_commit: str,
        slurm_job_id: str,
        rank: int,
    ) -> None:
        self.binding = validate_amp_diagnostic_binding(binding)
        self.spec = amp_protocol_spec_for_binding(self.binding)
        self.output_path = Path(output_path).resolve()
        if self.output_path != Path(self.binding["output_path"]).resolve():
            raise ValueError("AMP observer output differs from its binding")
        if int(rank) != 0:
            raise ValueError("AMP diagnostic observer is frozen to rank zero")
        if str(runtime_commit).lower() != self.binding["runtime_commit"]:
            raise ValueError("AMP observer runtime commit mismatch")
        if str(slurm_job_id) != str(os.environ.get("SLURM_JOB_ID", "")):
            raise ValueError("AMP observer Slurm Job ID is not process-bound")
        if self.output_path.exists():
            raise FileExistsError("AMP diagnostic receipt already exists")
        self.payload: dict[str, Any] = {
            "schema_version": self.spec["receipt_schema"],
            "status": self.spec["receipt_running_status"],
            "study_id": self.spec["study_id"],
            "protocol_profile": self.spec["profile"],
            "arm": self.binding["arm"],
            "runtime_commit": self.binding["runtime_commit"],
            "slurm_job_id": str(slurm_job_id),
            "binding": dict(self.binding),
            "events": [],
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        self._publish()

    def _publish(self) -> None:
        unsigned = dict(self.payload)
        unsigned.pop("receipt_sha256", None)
        self.payload["receipt_sha256"] = canonical_sha256(unsigned)
        _atomic_write_json(self.output_path, self.payload)

    def _append(self, event: dict[str, Any]) -> None:
        event["event_index"] = len(self.payload["events"])
        self.payload["events"].append(event)
        self._publish()

    def __call__(self, event: str, **payload: Any) -> None:
        common = {
            "event": str(event),
            "iter_idx": int(payload.get("iter_idx", -1)),
            "retry_count": int(payload.get("retry_count", 0)),
        }
        if event == "batch_start":
            data_descriptor = _describe_data(payload["data_dict"])
            cpu_rng = _describe_data(payload.get("cpu_rng_state"))
            cuda_rng = _describe_data(payload.get("cuda_rng_states"))
            self._append(
                {
                    **common,
                    "scale": float(payload["scale"]),
                    "successful_update_index": int(
                        payload["successful_update_index"]
                    ),
                    "data_descriptor": data_descriptor,
                    "data_fingerprint_sha256": canonical_sha256(
                        {"data": data_descriptor}
                    ),
                    "cpu_rng": cpu_rng,
                    "cpu_rng_sha256": canonical_sha256({"rng": cpu_rng}),
                    "cuda_rng": cuda_rng,
                    "cuda_rng_sha256": canonical_sha256({"rng": cuda_rng}),
                }
            )
            return
        if event == "forward_complete":
            losses = {
                str(name): _tensor_numeric_summary(value)
                for name, value in payload["losses"].items()
                if hasattr(value, "detach")
            }
            self._append(
                {
                    **common,
                    "scale": float(payload["scale"]),
                    "losses": losses,
                    "all_losses_finite": all(
                        bool(summary["finite"]) for summary in losses.values()
                    ),
                    "georoute_audit": _georoute_audit(payload["model"]),
                }
            )
            return
        if event in {
            "scaled_backward",
            "unscaled",
            "pre_clip",
            "post_clip",
        }:
            record = {
                **common,
                "scale": float(payload["scale"]),
                "gradient": _gradient_snapshot(payload["model"]),
            }
            if "clip_grad_l2norm" in payload:
                record["clip_grad_l2norm"] = float(
                    payload["clip_grad_l2norm"]
                )
            self._append(record)
            return
        if event == "scaler_result":
            self._append(
                {
                    **common,
                    "scale_before": float(payload["scale_before"]),
                    "scale_after": float(payload["scale_after"]),
                    "update_succeeded": bool(payload["update_succeeded"]),
                    "gradient": _gradient_snapshot(payload["model"]),
                }
            )
            return
        if event == "batch_complete":
            self._append(
                {
                    **common,
                    "scale": float(payload["scale"]),
                    "update_succeeded": bool(payload["update_succeeded"]),
                    "successful_updates": int(payload["successful_updates"]),
                }
            )
            return
        raise ValueError(f"unsupported AMP diagnostic event {event!r}")

    def _summary(
        self,
        update_audit: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        scaler_results = [
            event
            for event in self.payload["events"]
            if event.get("event") == "scaler_result"
        ]
        successful = [
            event for event in scaler_results if event.get("update_succeeded")
        ]
        failed = [
            event
            for event in scaler_results
            if event.get("update_succeeded") is False
        ]
        batches = [
            event
            for event in self.payload["events"]
            if event.get("event") == "batch_start"
        ]
        forwards = [
            event
            for event in self.payload["events"]
            if event.get("event") == "forward_complete"
        ]
        nonfinite_groups = sorted(
            {
                group
                for event in failed
                for group in event.get("gradient", {}).get(
                    "nonfinite_groups",
                    [],
                )
            }
        )
        success_flags = [
            bool(event.get("update_succeeded")) for event in scaler_results
        ]
        current_skip_streak = 0
        max_skip_streak = 0
        for succeeded in success_flags:
            if succeeded:
                current_skip_streak = 0
            else:
                current_skip_streak += 1
                max_skip_streak = max(max_skip_streak, current_skip_streak)
        scales_before = [
            float(event["scale_before"]) for event in scaler_results
        ]
        scales_after = [
            float(event["scale_after"]) for event in scaler_results
        ]
        all_scales = [*scales_before, *scales_after]
        stable_tail_batches = int(
            self.spec.get("stable_tail_batches", 0)
        )
        stable_tail = (
            success_flags[-stable_tail_batches:]
            if stable_tail_batches > 0
            else []
        )
        audit = dict(update_audit or {})
        return {
            "batch_count": len(batches),
            "data_fingerprint_sha256_by_batch": [
                event.get("data_fingerprint_sha256") for event in batches
            ],
            "cpu_rng_sha256_by_batch": [
                event.get("cpu_rng_sha256") for event in batches
            ],
            "cuda_rng_sha256_by_batch": [
                event.get("cuda_rng_sha256") for event in batches
            ],
            "data_fingerprint_sha256": (
                batches[0].get("data_fingerprint_sha256") if batches else None
            ),
            "cpu_rng_sha256": batches[0].get("cpu_rng_sha256") if batches else None,
            "cuda_rng_sha256": (
                batches[0].get("cuda_rng_sha256") if batches else None
            ),
            "optimizer_attempt_count": len(scaler_results),
            "failed_attempt_count": len(failed),
            "skipped_batch_indices": [
                int(event["iter_idx"]) for event in failed
            ],
            "failed_attempt_scales": [
                float(event["scale_before"]) for event in failed
            ],
            "first_successful_scale": (
                float(successful[0]["scale_before"]) if successful else None
            ),
            "successful_scales_by_batch": [
                float(event["scale_before"]) for event in successful
            ],
            "scale_before_by_attempt": scales_before,
            "scale_after_by_attempt": scales_after,
            "update_succeeded_by_attempt": success_flags,
            "max_consecutive_skipped_attempts": max_skip_streak,
            "minimum_observed_scale": min(all_scales) if all_scales else None,
            "final_scale": scales_after[-1] if scales_after else None,
            "observed_initial_scale": (
                float(batches[0]["scale"]) if batches else None
            ),
            "stable_tail_batches": stable_tail_batches,
            "stable_tail_success_count": sum(stable_tail),
            "stable_tail_all_success": (
                len(stable_tail) == stable_tail_batches
                and all(stable_tail)
                if stable_tail_batches > 0
                else None
            ),
            "retry_attempt_count": sum(
                int(event.get("retry_count", 0)) > 0
                for event in scaler_results
            ),
            "replay_attempt_count": int(audit.get("replay_attempts", 0)),
            "consumed_batch_count": int(
                audit.get("consumed_batches", len(batches))
            ),
            "scheduler_advance_count": int(
                audit.get("scheduler_advances", 0)
            ),
            "ema_update_count": int(audit.get("ema_updates", 0)),
            "failed_attempt_nonfinite_groups": nonfinite_groups,
            "forward_attempt_count": len(forwards),
            "all_forward_losses_finite": bool(forwards) and all(
                event.get("all_losses_finite") is True
                for event in forwards
            ),
        }

    def finalize_success(
        self,
        *,
        successful_updates: int,
        update_audit: Mapping[str, Any],
    ) -> None:
        if self.payload["status"] != self.spec["receipt_running_status"]:
            raise RuntimeError("AMP diagnostic observer was already finalized")
        summary = self._summary(update_audit)
        expected_batches = int(self.spec["max_batches"])
        common_complete = bool(
            summary["batch_count"] == expected_batches
            and summary["optimizer_attempt_count"] == expected_batches
            and summary["forward_attempt_count"] == expected_batches
            and summary["first_successful_scale"] is not None
            and summary["all_forward_losses_finite"] is True
        )
        if self.spec["profile"] == AMP_STABILITY_V2_PROFILE:
            v2_complete = bool(
                common_complete
                and int(successful_updates)
                >= int(self.spec["minimum_successful_updates"])
                and int(successful_updates) <= expected_batches
                and int(summary["failed_attempt_count"])
                <= int(self.spec["max_skipped_attempts"])
                and int(summary["max_consecutive_skipped_attempts"])
                <= int(self.spec["max_consecutive_skips"])
                and int(summary["retry_attempt_count"]) == 0
                and int(summary["replay_attempt_count"]) == 0
                and float(summary["observed_initial_scale"])
                == float(self.spec["initial_scale"])
                and float(summary["minimum_observed_scale"])
                >= float(self.spec["minimum_scale"])
                and float(summary["final_scale"])
                >= float(self.spec["minimum_scale"])
                and summary["stable_tail_all_success"] is True
                and int(summary["scheduler_advance_count"])
                == expected_batches
                and int(summary["ema_update_count"]) == expected_batches
                and int(summary["consumed_batch_count"])
                == expected_batches
                and int(update_audit.get("optimizer_attempts", -1))
                == expected_batches
                and int(update_audit.get("amp_skipped_attempts", -1))
                == int(summary["failed_attempt_count"])
                and int(update_audit.get("max_amp_retries_observed", -1))
                == 0
            )
            success_conditions_complete = v2_complete
        else:
            success_conditions_complete = bool(
                common_complete
                and int(successful_updates) == expected_batches
                and (
                    not bool(self.spec["zero_failed_attempts_required"])
                    or (
                        int(summary["failed_attempt_count"]) == 0
                        and all(
                            float(scale) == float(self.spec["initial_scale"])
                            for scale in summary["successful_scales_by_batch"]
                        )
                    )
                )
            )
        if not success_conditions_complete:
            raise RuntimeError("AMP diagnostic success conditions are incomplete")
        self.payload["status"] = self.spec["receipt_pass_status"]
        self.payload["summary"] = summary
        self.payload["successful_updates"] = int(successful_updates)
        self.payload["update_audit"] = dict(update_audit)
        self._publish()

    def finalize_failure(
        self,
        error: BaseException,
        *,
        successful_updates: int,
        update_audit: Mapping[str, Any],
    ) -> None:
        if self.payload["status"] != self.spec["receipt_running_status"]:
            return
        trace = traceback.format_exc()
        self.payload["status"] = self.spec["receipt_fail_status"]
        self.payload["summary"] = self._summary(update_audit)
        self.payload["successful_updates"] = int(successful_updates)
        self.payload["update_audit"] = dict(update_audit)
        self.payload["failure"] = {
            "exception_type": type(error).__name__,
            "exception_message": str(error)[:2000],
            "traceback_sha256": hashlib.sha256(
                trace.encode("utf-8", errors="replace")
            ).hexdigest(),
        }
        self._publish()


def _validate_amp_stability_v2_summary(
    *,
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    spec: Mapping[str, Any],
    status: str,
) -> None:
    expected_batches = int(spec["max_batches"])
    batch_count = int(summary.get("batch_count", -1))
    attempt_count = int(summary.get("optimizer_attempt_count", -1))
    if not (0 <= batch_count <= expected_batches):
        raise ValueError("AMP stability v2 batch count is invalid")
    if not (0 <= attempt_count <= batch_count):
        raise ValueError("AMP stability v2 optimizer-attempt count is invalid")
    for key, expected_length in (
        ("data_fingerprint_sha256_by_batch", batch_count),
        ("cpu_rng_sha256_by_batch", batch_count),
        ("cuda_rng_sha256_by_batch", batch_count),
        ("scale_before_by_attempt", attempt_count),
        ("scale_after_by_attempt", attempt_count),
        ("update_succeeded_by_attempt", attempt_count),
    ):
        values = summary.get(key)
        if not isinstance(values, list) or len(values) != expected_length:
            raise ValueError(
                f"AMP stability v2 summary has incomplete ordered telemetry: {key}"
            )
    if any(
        not isinstance(value, str) or len(value) != 64
        for key in (
            "data_fingerprint_sha256_by_batch",
            "cpu_rng_sha256_by_batch",
            "cuda_rng_sha256_by_batch",
        )
        for value in summary[key]
    ):
        raise ValueError("AMP stability v2 fingerprint sequence is invalid")
    success_flags = summary["update_succeeded_by_attempt"]
    if any(type(value) is not bool for value in success_flags):
        raise ValueError("AMP stability v2 update-success sequence is invalid")
    skipped = sum(not value for value in success_flags)
    successful = sum(success_flags)
    skipped_indices = summary.get("skipped_batch_indices")
    if (
        not isinstance(skipped_indices, list)
        or len(skipped_indices) != skipped
        or int(summary.get("failed_attempt_count", -1)) != skipped
        or int(payload.get("successful_updates", -1)) != successful
    ):
        raise ValueError("AMP stability v2 skip/update accounting is invalid")
    audit = _mapping(payload.get("update_audit"), name="update audit")
    if (
        int(audit.get("optimizer_attempts", -1)) != attempt_count
        or int(audit.get("amp_skipped_attempts", -1)) != skipped
        or int(audit.get("max_amp_retries_observed", -1)) != 0
        or int(audit.get("replay_attempts", -1))
        != int(summary.get("replay_attempt_count", -2))
        or int(audit.get("consumed_batches", -1))
        != int(summary.get("consumed_batch_count", -2))
        or int(audit.get("scheduler_advances", -1))
        != int(summary.get("scheduler_advance_count", -2))
        or int(audit.get("ema_updates", -1))
        != int(summary.get("ema_update_count", -2))
    ):
        raise ValueError("AMP stability v2 transition audit is inconsistent")
    observed_initial_scale = summary.get("observed_initial_scale")
    if batch_count > 0 and (
        not isinstance(summary.get("data_fingerprint_sha256"), str)
        or len(str(summary["data_fingerprint_sha256"])) != 64
        or not isinstance(observed_initial_scale, (int, float))
        or float(observed_initial_scale) != float(spec["initial_scale"])
    ):
        raise ValueError("AMP stability v2 initial provenance is incomplete")
    if status != spec["receipt_pass_status"]:
        if status != spec["receipt_fail_status"] or not isinstance(
            payload.get("failure"), Mapping
        ):
            raise ValueError("AMP stability v2 failure receipt is incomplete")
        return

    final_scale = summary.get("final_scale")
    minimum_scale = summary.get("minimum_observed_scale")
    if (
        batch_count != expected_batches
        or attempt_count != expected_batches
        or int(summary.get("forward_attempt_count", -1)) != expected_batches
        or summary.get("all_forward_losses_finite") is not True
        or successful < int(spec["minimum_successful_updates"])
        or skipped > int(spec["max_skipped_attempts"])
        or int(summary.get("max_consecutive_skipped_attempts", -1))
        > int(spec["max_consecutive_skips"])
        or int(summary.get("retry_attempt_count", -1)) != 0
        or int(summary.get("replay_attempt_count", -1)) != 0
        or not isinstance(minimum_scale, (int, float))
        or float(minimum_scale) < float(spec["minimum_scale"])
        or not isinstance(final_scale, (int, float))
        or float(final_scale) < float(spec["minimum_scale"])
        or int(summary.get("stable_tail_batches", -1))
        != int(spec["stable_tail_batches"])
        or int(summary.get("stable_tail_success_count", -1))
        != int(spec["stable_tail_batches"])
        or summary.get("stable_tail_all_success") is not True
        or int(summary.get("consumed_batch_count", -1)) != expected_batches
        or int(summary.get("scheduler_advance_count", -1))
        != expected_batches
        or int(summary.get("ema_update_count", -1)) != expected_batches
    ):
        raise ValueError(
            "passing AMP stability v2 receipt violates its frozen threshold"
        )


def validate_amp_diagnostic_receipt(
    payload: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
    expected_slurm_job_id: str | None = None,
    expected_profile: str | None = None,
) -> dict[str, Any]:
    payload = dict(_mapping(payload, name="AMP diagnostic receipt"))
    if not _self_hash_matches(payload, field="receipt_sha256"):
        raise ValueError("AMP diagnostic receipt self-hash mismatch")
    status = payload.get("status")
    arm = str(payload.get("arm", ""))
    validated_binding = validate_amp_diagnostic_binding(
        _mapping(payload.get("binding"), name="receipt binding")
    )
    spec = amp_protocol_spec_for_binding(validated_binding)
    if expected_profile is not None and spec["profile"] != expected_profile:
        raise ValueError("AMP diagnostic receipt protocol profile mismatch")
    if (
        payload.get("schema_version") != spec["receipt_schema"]
        or payload.get("study_id") != spec["study_id"]
        or str(
            payload.get(
                "protocol_profile",
                AMP_DIAGNOSTIC_PROFILE,
            )
        )
        != spec["profile"]
        or status
        not in {
            spec["receipt_pass_status"],
            spec["receipt_fail_status"],
        }
        or arm not in AMP_DIAGNOSTIC_ARMS
        or payload.get("checkpoint_emitted") is not False
        or payload.get("prediction_emitted") is not False
        or payload.get("evaluator_invoked") is not False
        or payload.get("official_test_opened") is not False
        or payload.get("paper_claim_allowed") is not False
    ):
        raise ValueError("AMP diagnostic receipt contract is invalid")
    if payload["binding"].get("runtime_commit") != payload.get("runtime_commit"):
        raise ValueError("AMP diagnostic receipt and binding commits differ")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("AMP diagnostic receipt arm mismatch")
    if (
        expected_commit is not None
        and payload.get("runtime_commit") != str(expected_commit).lower()
    ):
        raise ValueError("AMP diagnostic receipt commit mismatch")
    slurm_job_id = str(payload.get("slurm_job_id", ""))
    if not slurm_job_id.isdigit():
        raise ValueError("AMP diagnostic receipt lacks a numeric Slurm ID")
    if (
        expected_slurm_job_id is not None
        and slurm_job_id != str(expected_slurm_job_id)
    ):
        raise ValueError("AMP diagnostic receipt Slurm ID mismatch")
    summary = _mapping(payload.get("summary"), name="receipt summary")
    expected_batches = int(spec["max_batches"])
    if spec["profile"] == AMP_STABILITY_V2_PROFILE:
        _validate_amp_stability_v2_summary(
            payload=payload,
            summary=summary,
            spec=spec,
            status=str(status),
        )
        return payload
    if (
        int(summary.get("batch_count", -1)) != expected_batches
        or not isinstance(summary.get("data_fingerprint_sha256"), str)
        or len(str(summary["data_fingerprint_sha256"])) != 64
        or summary.get("all_forward_losses_finite") is not True
    ):
        raise ValueError("AMP diagnostic receipt summary is incomplete")
    if spec["profile"] == AMP_STABILITY_PROFILE:
        for key in (
            "data_fingerprint_sha256_by_batch",
            "cpu_rng_sha256_by_batch",
            "cuda_rng_sha256_by_batch",
            "successful_scales_by_batch",
        ):
            values = summary.get(key)
            if not isinstance(values, list) or len(values) != expected_batches:
                raise ValueError(
                    "AMP stability receipt lacks complete per-batch provenance"
                )
        if (
            any(
                not isinstance(value, str) or len(value) != 64
                for value in summary["data_fingerprint_sha256_by_batch"]
            )
            or int(summary.get("failed_attempt_count", -1)) != 0
            or int(summary.get("optimizer_attempt_count", -1))
            != expected_batches
            or any(
                float(scale) != float(spec["initial_scale"])
                for scale in summary["successful_scales_by_batch"]
            )
        ):
            raise ValueError("AMP stability zero-skip contract is invalid")
    if status == spec["receipt_pass_status"] and (
        summary.get("first_successful_scale") is None
        or int(payload.get("successful_updates", -1))
        != expected_batches
        or (
            bool(spec["zero_failed_attempts_required"])
            and (
                int(
                    _mapping(
                        payload.get("update_audit"),
                        name="update audit",
                    ).get("amp_skipped_attempts", -1)
                )
                != 0
            )
        )
    ):
        raise ValueError("passing AMP diagnostic lacks an optimizer update")
    return payload


def classify_amp_diagnostic_pair(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen, no-performance repair-authorization rule."""

    if set(receipts) != set(AMP_DIAGNOSTIC_ARMS):
        return {
            "decision": "DIAGNOSTIC_INCOMPLETE_NO_REPAIR",
            "root_cause_localized": False,
            "repair_authorized": False,
            "reason": "missing_or_extra_arm",
        }
    validated: dict[str, dict[str, Any]] = {}
    try:
        for arm in AMP_DIAGNOSTIC_ARMS:
            validated[arm] = validate_amp_diagnostic_receipt(
                receipts[arm],
                expected_arm=arm,
                expected_profile=AMP_DIAGNOSTIC_PROFILE,
            )
    except (TypeError, ValueError) as error:
        return {
            "decision": "DIAGNOSTIC_INCOMPLETE_NO_REPAIR",
            "root_cause_localized": False,
            "repair_authorized": False,
            "reason": f"invalid_receipt:{type(error).__name__}",
        }
    if any(
        receipt["status"] != "PASS_DIAGNOSTIC_EXECUTION_ONLY"
        for receipt in validated.values()
    ):
        return {
            "decision": "DIAGNOSTIC_INCOMPLETE_NO_REPAIR",
            "root_cause_localized": False,
            "repair_authorized": False,
            "reason": "one_or_more_arms_lacked_a_successful_optimizer_update",
        }

    summaries = {
        arm: validated[arm]["summary"]
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    fingerprints = {
        str(summary["data_fingerprint_sha256"])
        for summary in summaries.values()
    }
    cpu_rng_hashes = {
        str(summary.get("cpu_rng_sha256"))
        for summary in summaries.values()
    }
    cuda_rng_hashes = {
        str(summary.get("cuda_rng_sha256"))
        for summary in summaries.values()
    }
    matched_execution = (
        len(fingerprints) == 1
        and len(cpu_rng_hashes) == 1
        and len(cuda_rng_hashes) == 1
        and None not in {
            summary.get("cpu_rng_sha256")
            for summary in summaries.values()
        }
        and None not in {
            summary.get("cuda_rng_sha256")
            for summary in summaries.values()
        }
    )
    pl = summaries["residual_pl_rep_off"]
    st = summaries["residual_st_rep_off"]
    pl_groups = set(pl.get("failed_attempt_nonfinite_groups", []))
    st_groups = set(st.get("failed_attempt_nonfinite_groups", []))
    pl_scale = pl.get("first_successful_scale")
    st_scale = st.get("first_successful_scale")
    localized = bool(
        matched_execution
        and pl.get("all_forward_losses_finite") is True
        and st.get("all_forward_losses_finite") is True
        and int(pl.get("failed_attempt_count", 0)) > 0
        and pl_groups == {"scout_score_function"}
        and isinstance(pl_scale, (int, float))
        and float(pl_scale) <= AMP_DIAGNOSTIC_PL_MAX_LOCALIZED_SUCCESS_SCALE
        and int(st.get("failed_attempt_count", -1)) == 0
        and st_groups == set()
        and isinstance(st_scale, (int, float))
        and float(st_scale) == AMP_DIAGNOSTIC_INITIAL_SCALE
    )
    return {
        "decision": (
            "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
            if localized
            else "ROOT_CAUSE_NOT_LOCALIZED_HOLD"
        ),
        "root_cause_localized": localized,
        "repair_authorized": localized,
        "reason": (
            "PL_only_score_function_gradient_overflow_with_matched_ST_control"
            if localized
            else "frozen_localization_rule_not_satisfied"
        ),
        "matched_execution": matched_execution,
        "data_fingerprint_sha256": (
            next(iter(fingerprints)) if len(fingerprints) == 1 else None
        ),
        "cpu_rng_sha256": (
            next(iter(cpu_rng_hashes)) if len(cpu_rng_hashes) == 1 else None
        ),
        "cuda_rng_sha256": (
            next(iter(cuda_rng_hashes)) if len(cuda_rng_hashes) == 1 else None
        ),
        "pl_failed_attempt_count": int(pl.get("failed_attempt_count", 0)),
        "pl_first_successful_scale": pl_scale,
        "pl_failed_attempt_nonfinite_groups": sorted(pl_groups),
        "st_failed_attempt_count": int(st.get("failed_attempt_count", 0)),
        "st_first_successful_scale": st_scale,
        "st_failed_attempt_nonfinite_groups": sorted(st_groups),
    }


def classify_amp_stability_pair(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Require 32 matched real-data batches with no AMP skip in either arm."""

    if set(receipts) != set(AMP_DIAGNOSTIC_ARMS):
        return {
            "decision": "STABILITY_GATE_INCOMPLETE_HOLD",
            "stability_gate_passed": False,
            "official_protocol_freeze_authorized": False,
            "reason": "missing_or_extra_arm",
        }
    validated: dict[str, dict[str, Any]] = {}
    try:
        for arm in AMP_DIAGNOSTIC_ARMS:
            validated[arm] = validate_amp_diagnostic_receipt(
                receipts[arm],
                expected_arm=arm,
                expected_profile=AMP_STABILITY_PROFILE,
            )
    except (TypeError, ValueError) as error:
        return {
            "decision": "STABILITY_GATE_INCOMPLETE_HOLD",
            "stability_gate_passed": False,
            "official_protocol_freeze_authorized": False,
            "reason": f"invalid_receipt:{type(error).__name__}",
        }
    spec = amp_protocol_spec(AMP_STABILITY_PROFILE)
    if any(
        receipt["status"] != spec["receipt_pass_status"]
        for receipt in validated.values()
    ):
        return {
            "decision": "STABILITY_GATE_INCOMPLETE_HOLD",
            "stability_gate_passed": False,
            "official_protocol_freeze_authorized": False,
            "reason": "one_or_more_arms_failed_execution",
        }

    summaries = {
        arm: validated[arm]["summary"] for arm in AMP_DIAGNOSTIC_ARMS
    }
    data_sequences = {
        arm: list(summary["data_fingerprint_sha256_by_batch"])
        for arm, summary in summaries.items()
    }
    matched_data_sequence = (
        data_sequences[AMP_DIAGNOSTIC_ARMS[0]]
        == data_sequences[AMP_DIAGNOSTIC_ARMS[1]]
    )
    per_arm_zero_skip = {
        arm: bool(
            int(summary["batch_count"]) == int(spec["max_batches"])
            and int(summary["failed_attempt_count"]) == 0
            and int(summary["optimizer_attempt_count"])
            == int(spec["max_batches"])
            and float(summary["first_successful_scale"])
            == float(spec["initial_scale"])
            and all(
                float(scale) == float(spec["initial_scale"])
                for scale in summary["successful_scales_by_batch"]
            )
            and summary["all_forward_losses_finite"] is True
            and int(validated[arm]["successful_updates"])
            == int(spec["max_batches"])
            and validated[arm]["binding"][
                "score_function_temporal_reduction"
            ]
            == "mean"
        )
        for arm, summary in summaries.items()
    }
    passed = bool(
        matched_data_sequence and all(per_arm_zero_skip.values())
    )
    return {
        "decision": (
            "REAL_DATA_AMP_STABILITY_PASS_PROTOCOL_FREEZE_AUTHORIZED"
            if passed
            else "REAL_DATA_AMP_STABILITY_HOLD"
        ),
        "stability_gate_passed": passed,
        "official_protocol_freeze_authorized": passed,
        "reason": (
            "matched_32_batch_PL_ST_zero_skip_at_initial_scale"
            if passed
            else "frozen_stability_rule_not_satisfied"
        ),
        "matched_data_sequence": matched_data_sequence,
        "batch_count": int(spec["max_batches"]),
        "initial_scale": float(spec["initial_scale"]),
        "score_function_temporal_reduction": "mean",
        "per_arm_zero_skip": per_arm_zero_skip,
        "data_sequence_sha256": canonical_sha256(
            {"fingerprints": data_sequences[AMP_DIAGNOSTIC_ARMS[0]]}
        )
        if matched_data_sequence
        else None,
    }


def classify_amp_stability_v2_pair(
    receipts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen official-semantics 64-batch stability-v2 rule."""

    hold = {
        "decision": "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD",
        "stability_gate_passed": False,
        "official_protocol_freeze_authorized": False,
    }
    if set(receipts) != set(AMP_DIAGNOSTIC_ARMS):
        return {**hold, "reason": "missing_or_extra_arm"}
    validated: dict[str, dict[str, Any]] = {}
    try:
        for arm in AMP_DIAGNOSTIC_ARMS:
            validated[arm] = validate_amp_diagnostic_receipt(
                receipts[arm],
                expected_arm=arm,
                expected_profile=AMP_STABILITY_V2_PROFILE,
            )
    except (TypeError, ValueError) as error:
        return {
            **hold,
            "reason": f"invalid_receipt:{type(error).__name__}",
        }
    spec = amp_protocol_spec(AMP_STABILITY_V2_PROFILE)
    if any(
        receipt["status"] != spec["receipt_pass_status"]
        for receipt in validated.values()
    ):
        return {**hold, "reason": "one_or_more_arms_failed_execution"}

    summaries = {
        arm: validated[arm]["summary"] for arm in AMP_DIAGNOSTIC_ARMS
    }
    data_sequences = {
        arm: list(summary["data_fingerprint_sha256_by_batch"])
        for arm, summary in summaries.items()
    }
    matched_data_sequence = (
        data_sequences[AMP_DIAGNOSTIC_ARMS[0]]
        == data_sequences[AMP_DIAGNOSTIC_ARMS[1]]
    )
    matched_seed = all(
        int(validated[arm]["binding"]["execution_seed"])
        == AMP_STABILITY_V2_SEED
        and int(validated[arm]["binding"]["seed"])
        == AMP_STABILITY_V2_SEED
        for arm in AMP_DIAGNOSTIC_ARMS
    )
    matched_input_fields = (
        "source_config_sha256",
        "manifest_file_sha256",
        "development_annotation",
        "class_map_sha256",
        "development_video_root",
        "pretrained_checkpoint_sha256",
        "official_reference_config_sha256",
    )
    matched_inputs = all(
        validated[AMP_DIAGNOSTIC_ARMS[0]]["binding"].get(field)
        == validated[AMP_DIAGNOSTIC_ARMS[1]]["binding"].get(field)
        for field in matched_input_fields
    )
    skip_counts = {
        arm: int(summaries[arm]["failed_attempt_count"])
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    skip_delta = abs(
        skip_counts[AMP_DIAGNOSTIC_ARMS[0]]
        - skip_counts[AMP_DIAGNOSTIC_ARMS[1]]
    )
    final_scales = {
        arm: float(summaries[arm]["final_scale"])
        for arm in AMP_DIAGNOSTIC_ARMS
    }
    scale_floor = min(final_scales.values())
    final_scale_ratio = (
        max(final_scales.values()) / scale_floor
        if scale_floor > 0.0
        else math.inf
    )
    per_arm_pass = {
        arm: bool(
            int(summary["batch_count"]) == int(spec["max_batches"])
            and int(summary["optimizer_attempt_count"])
            == int(spec["max_batches"])
            and int(summary["failed_attempt_count"])
            <= int(spec["max_skipped_attempts"])
            and int(summary["max_consecutive_skipped_attempts"])
            <= int(spec["max_consecutive_skips"])
            and int(validated[arm]["successful_updates"])
            >= int(spec["minimum_successful_updates"])
            and float(summary["minimum_observed_scale"])
            >= float(spec["minimum_scale"])
            and float(summary["final_scale"]) >= float(spec["minimum_scale"])
            and summary["stable_tail_all_success"] is True
            and int(summary["retry_attempt_count"]) == 0
            and int(summary["replay_attempt_count"]) == 0
            and int(summary["scheduler_advance_count"])
            == int(spec["max_batches"])
            and int(summary["ema_update_count"]) == int(spec["max_batches"])
            and summary["all_forward_losses_finite"] is True
            and validated[arm]["binding"][
                "official_prefix_transition_semantics_matched"
            ]
            is True
            and validated[arm]["binding"]["full_official_training_claimed"]
            is False
        )
        for arm, summary in summaries.items()
    }
    passed = bool(
        matched_data_sequence
        and matched_seed
        and matched_inputs
        and skip_delta <= int(spec["max_cross_arm_skip_delta"])
        and final_scale_ratio <= float(spec["max_final_scale_ratio"])
        and all(per_arm_pass.values())
    )
    return {
        "decision": (
            "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_PASS_PROTOCOL_FREEZE_AUTHORIZED"
            if passed
            else "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD"
        ),
        "stability_gate_passed": passed,
        "official_protocol_freeze_authorized": passed,
        "reason": (
            "matched_bounded_dynamic_scaler_adaptation_with_stable_tail"
            if passed
            else "frozen_official_semantics_stability_v2_rule_not_satisfied"
        ),
        "matched_data_sequence": matched_data_sequence,
        "matched_execution_seed": matched_seed,
        "matched_input_bindings": matched_inputs,
        "batch_count": int(spec["max_batches"]),
        "execution_seed": AMP_STABILITY_V2_SEED,
        "per_arm_pass": per_arm_pass,
        "per_arm_skip_count": skip_counts,
        "cross_arm_skip_delta": skip_delta,
        "per_arm_final_scale": final_scales,
        "final_scale_ratio": final_scale_ratio,
        "data_sequence_sha256": (
            canonical_sha256(
                {
                    "fingerprints": data_sequences[
                        AMP_DIAGNOSTIC_ARMS[0]
                    ]
                }
            )
            if matched_data_sequence
            else None
        ),
    }
