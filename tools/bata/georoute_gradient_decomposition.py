"""Matched, no-performance gradient decomposition for GeoRoute PL versus ST.

This module is deliberately diagnostic-only.  It observes the production
GradScaler and DDP FP16 communication path without replacing the authoritative
PyTorch hook, changing an optimizer transition, or serializing raw tensors.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import math
import os
import subprocess
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as comm_hooks

from tools.bata.georoute_amp_diagnostic import (
    AMP_STABILITY_V2_PROFILE,
    AMP_STABILITY_V2_SEED,
    _describe_data,
    _tensor_numeric_summary,
    bind_amp_diagnostic_config,
)
from tools.bata.georoute_estimator_pilot_contract import PILOT_ARMS, PILOT_K
from tools.bata.georoute_experiment_contract import canonical_sha256


STUDY_ID = "georoute_pl_gradient_decomposition_diagnostic_v1"
PROFILE = "pl_gradient_decomposition_v1"
BINDING_SCHEMA = "georoute_pl_gradient_decomposition_binding_v1"
RECEIPT_SCHEMA = "georoute_pl_gradient_decomposition_receipt_v1"
STAGE_SCHEMA = "georoute_pl_gradient_decomposition_stage_v1"
DEPLOYMENT_SCHEMA = "georoute_pl_gradient_decomposition_deployment_v1"
FINALIZATION_SCHEMA = "georoute_pl_gradient_decomposition_finalization_v1"
KAT_SCHEMA = "georoute_pl_gradient_decomposition_cuda_kat_v1"

ARMS = ("residual_pl_rep_off", "residual_st_rep_off")
SEED = 7367
FORBIDDEN_SEEDS = (4417, 3407, 3408, 3409)
MAX_BATCHES = 64
EXPECTED_TEMPERATURE = 0.7
EXPECTED_WEIGHT = 1.0
EXPECTED_BASELINE_MOMENTUM = 0.95
EXPECTED_TEMPORAL_REDUCTION = "mean"
EXPECTED_TUBELETS = 384
EXPECTED_ITEMS = 220
EXPECTED_K = 64
FP64_REFERENCE_TUBELETS = (0, 127, 255, 383)
MAX_RECEIPT_BYTES = 256 * 1024 * 1024
MAX_NAMESPACE_BYTES = 2 * 1024 * 1024 * 1024
RENDEZVOUS_STAGE = "gradient-decomposition"

RUNNING_STATUS = "RUNNING_GRADIENT_DECOMPOSITION_ONLY"
PASS_STATUS = "PASS_GRADIENT_DECOMPOSITION_EXECUTION_ONLY"
FAIL_STATUS = "FAIL_GRADIENT_DECOMPOSITION_EXECUTION"
STAGE_PASS_STATUS = "PASS_STAGE_GRADIENT_DECOMPOSITION_ONLY"
STAGE_FAIL_STATUS = "FAIL_STAGE_GRADIENT_DECOMPOSITION_EXECUTION"
STAGE_WRAPPER_FAIL_STATUS = "FAIL_GRADIENT_DECOMPOSITION_STAGE_WRAPPER"
DEPLOYMENT_STATUS = "SUBMITTED_GRADIENT_DECOMPOSITION_DIAGNOSTIC_ONLY"
FINALIZER_SUBMISSION_STATUS = "SUBMITTED_GRADIENT_DECOMPOSITION_FINALIZER_AFTERANY"
STAGE_RELEASE_STATUS = "RELEASED_GRADIENT_DECOMPOSITION_STAGES"
COMPLETE_STATUS = "COMPLETE_GRADIENT_DECOMPOSITION_DIAGNOSTIC_ONLY"
INCOMPLETE_STATUS = "INCOMPLETE_GRADIENT_DECOMPOSITION_DIAGNOSTIC"

DECISION_REPAIR = "PL_NUMERICAL_MECHANISM_LOCALIZED_REPAIR_CLASS_IDENTIFIED"
DECISION_HOLD = "PL_NUMERICAL_MECHANISM_NOT_LOCALIZED_HOLD"
DECISION_INCOMPLETE = "GRADIENT_DECOMPOSITION_DIAGNOSTIC_INCOMPLETE"
KAT_PASS_STATUS = "PASS_GRADIENT_DECOMPOSITION_CUDA_KAT_ONLY"
KAT_FAIL_STATUS = "FAIL_GRADIENT_DECOMPOSITION_CUDA_KAT"

MECHANISM_CLASSES = {
    "DDP_FP16_CAST_OVERFLOW",
    "UPSTREAM_SCORE_NONFINITE",
    "SCOUT_VJP_NONFINITE",
    "SHARED_DETECTOR_BUCKET_OVERFLOW",
    "AMBIGUOUS_MIXED_FAILURE",
}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_RECEIPT_BYTES:
        raise RuntimeError("gradient-decomposition receipt exceeded its byte budget")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _self_hash_matches(payload: Mapping[str, Any], *, field: str) -> bool:
    unsigned = dict(payload)
    observed = unsigned.pop(field, None)
    return isinstance(observed, str) and observed == canonical_sha256(unsigned)


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


def _parameter_group(name: str) -> str:
    lowered = str(name).lower()
    if ".scout." in lowered or "score_function" in lowered:
        return "scout_score_function"
    if "sparse_adapter" in lowered or ".adapter." in lowered:
        return "adapter"
    if ".model.backbone." in lowered or ".backbone.backbone." in lowered:
        return "heavy_backbone"
    return "detector"


def require_clean_git_checkout(*, expected_commit: str, root: Path) -> None:
    expected_commit = _full_hex(
        expected_commit, length=40, name="gradient runtime commit"
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
            "gradient decomposition requires its exact clean runtime commit"
        )


def require_slurm_single_gpu() -> str:
    job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not job_id.isdigit() or not visible or "," in visible:
        raise RuntimeError(
            "gradient decomposition requires one Slurm-visible GPU"
        )
    return job_id


def cell_relative_path(arm: str) -> Path:
    if arm not in ARMS:
        raise ValueError("unsupported gradient-decomposition arm")
    slug = "pl" if arm == ARMS[0] else "st"
    return Path("diagnosis") / f"{slug}_{arm}"


def validate_job_receipt(
    jobs: Any, *, expected_finalizer: str | None = None
) -> dict[str, Any]:
    if not isinstance(jobs, Mapping):
        raise ValueError("gradient-decomposition jobs must be a mapping")
    stages = jobs.get("stage")
    finalizer = str(jobs.get("finalizer", ""))
    if (
        not isinstance(stages, Mapping)
        or set(stages) != set(ARMS)
        or not finalizer.isdigit()
    ):
        raise ValueError("gradient-decomposition job receipt has the wrong shape")
    normalized = {arm: str(stages[arm]) for arm in ARMS}
    all_ids = [*normalized.values(), finalizer]
    if any(not job_id.isdigit() for job_id in all_ids):
        raise ValueError("gradient-decomposition job receipt has a nonnumeric ID")
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("gradient-decomposition job receipt reuses a Slurm ID")
    if expected_finalizer is not None and finalizer != str(expected_finalizer):
        raise ValueError("gradient-decomposition finalizer is not self-bound")
    return {"stage": normalized, "finalizer": finalizer}


def bind_gradient_decomposition_config(
    *,
    source_config_path: str | Path,
    arm: str,
    work_dir: str | Path,
    manifest_path: str | Path,
    development_annotation_path: str | Path,
    class_map_path: str | Path,
    development_video_root: str | Path,
    pretrained_checkpoint_path: str | Path,
    official_reference_config_path: str | Path,
    runtime_commit: str,
    parent_evidence: Mapping[str, Any],
):
    """Bind the v2 execution semantics to a fresh diagnosis seed and schema."""

    if arm not in ARMS:
        raise ValueError("gradient decomposition requires the residual PL/ST pair")
    runtime_commit = _full_hex(
        runtime_commit, length=40, name="gradient runtime commit"
    )
    work_dir = Path(work_dir).resolve()
    cfg = bind_amp_diagnostic_config(
        source_config_path=source_config_path,
        arm=arm,
        seed=AMP_STABILITY_V2_SEED,
        work_dir=work_dir,
        manifest_path=manifest_path,
        development_annotation_path=development_annotation_path,
        class_map_path=class_map_path,
        development_video_root=development_video_root,
        pretrained_checkpoint_path=pretrained_checkpoint_path,
        runtime_commit=runtime_commit,
        protocol_profile=AMP_STABILITY_V2_PROFILE,
        official_reference_config_path=official_reference_config_path,
    )
    template = dict(cfg.georoute_amp_diagnostic_binding)
    cfg.pop("georoute_amp_diagnostic_binding")
    cfg.pop("georoute_runtime_binding")

    custom = cfg.model.backbone.custom
    custom.georoute_random_seed = SEED
    custom.georoute_gradient_decomposition_enabled = True
    custom.georoute_amp_diagnostic_enabled = True
    cfg.georoute_protocol.status = PROFILE
    output_path = work_dir / "gradient_decomposition.json"
    parent_evidence = dict(parent_evidence)
    parent_evidence_sha256 = canonical_sha256(parent_evidence)

    binding: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA,
        "study_id": STUDY_ID,
        "profile": PROFILE,
        "arm": arm,
        "arm_spec": dict(template["arm_spec"]),
        "seed": SEED,
        "mechanical_seed_derivation": (
            "1000+(int(sha256(study_id)[0:8],16)%9000)"
        ),
        "forbidden_seeds": list(FORBIDDEN_SEEDS),
        "runtime_commit": runtime_commit,
        "work_dir": str(work_dir),
        "output_path": str(output_path),
        "max_batches": MAX_BATCHES,
        "expected_tubelets": EXPECTED_TUBELETS,
        "expected_items": EXPECTED_ITEMS,
        "token_budget": EXPECTED_K,
        "temperature": EXPECTED_TEMPERATURE,
        "score_function_weight": EXPECTED_WEIGHT,
        "baseline_momentum": EXPECTED_BASELINE_MOMENTUM,
        "temporal_reduction": EXPECTED_TEMPORAL_REDUCTION,
        "pooling_mode": "uniform_selected",
        "geometry_side_channel": False,
        "batch_size": 1,
        "fp16_compress": True,
        "default_grad_scaler": True,
        "zero_retry": True,
        "zero_replay": True,
        "scheduler_advances_per_consumed_batch": True,
        "ema_updates_per_consumed_batch": True,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": True,
        "source_config": template["source_config"],
        "source_config_sha256": template["source_config_sha256"],
        "manifest_path": template["manifest_path"],
        "manifest_file_sha256": template["manifest_file_sha256"],
        "training_video_ids": list(template["training_video_ids"]),
        "evaluation_video_ids": list(template["evaluation_video_ids"]),
        "training_block_list_video_ids": list(
            template["training_block_list_video_ids"]
        ),
        "evaluation_block_list_video_ids": list(
            template["evaluation_block_list_video_ids"]
        ),
        "development_annotation": dict(template["development_annotation"]),
        "class_map_path": template["class_map_path"],
        "class_map_sha256": template["class_map_sha256"],
        "development_video_root": template["development_video_root"],
        "pretrained_checkpoint_path": template["pretrained_checkpoint_path"],
        "pretrained_checkpoint_sha256": template[
            "pretrained_checkpoint_sha256"
        ],
        "official_reference_config": template["official_reference_config"],
        "official_reference_config_sha256": template[
            "official_reference_config_sha256"
        ],
        "official_reference_transition_semantics": dict(
            template["official_reference_transition_semantics"]
        ),
        "official_reference_transition_semantics_sha256": template[
            "official_reference_transition_semantics_sha256"
        ],
        "official_prefix_transition_semantics_matched": True,
        "official_scheduler_advance_cadence_matched": True,
        "official_scheduler_hyperparameters_matched": False,
        "full_official_recipe_matched": False,
        "official_performance_comparable": False,
        "development_prefix_only": True,
        "template_amp_stability_v2_binding_sha256": template[
            "binding_sha256"
        ],
        "parent_evidence": parent_evidence,
        "parent_evidence_sha256": parent_evidence_sha256,
        # PL consumes CUDA RNG for Gumbel sampling while ST does not.  Requiring
        # all later CUDA states to match would be impossible without replay or
        # changing the estimator.  Match data/CPU state throughout and the
        # initial CUDA state once, then record the expected divergence.
        "rng_matching_policy": {
            "data_fingerprint_all_batches_equal": True,
            "cpu_rng_all_batches_equal": True,
            "cuda_rng_batch_zero_equal": True,
            "cuda_rng_after_batch_zero_equality_required": False,
            "reason": "PL_Gumbel_sampling_consumes_CUDA_RNG_ST_does_not",
        },
        "checkpoint_disabled": True,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    cfg.georoute_gradient_decomposition_binding = binding
    cfg.georoute_runtime_binding = binding
    cfg.work_dir = str(work_dir)
    return cfg


def validate_binding(
    binding: Mapping[str, Any], *, seed: int | None = None
) -> dict[str, Any]:
    result = dict(_mapping(binding, name="gradient-decomposition binding"))
    if not _self_hash_matches(result, field="binding_sha256"):
        raise ValueError("gradient-decomposition binding self-hash mismatch")
    arm = str(result.get("arm", ""))
    if (
        result.get("schema_version") != BINDING_SCHEMA
        or result.get("study_id") != STUDY_ID
        or result.get("profile") != PROFILE
        or arm not in ARMS
        or result.get("arm_spec") != PILOT_ARMS[arm]
        or int(result.get("seed", -1)) != SEED
        or int(result.get("seed", -1)) in FORBIDDEN_SEEDS
        or tuple(result.get("forbidden_seeds", ())) != FORBIDDEN_SEEDS
        or int(result.get("max_batches", -1)) != MAX_BATCHES
        or int(result.get("expected_tubelets", -1)) != EXPECTED_TUBELETS
        or int(result.get("expected_items", -1)) != EXPECTED_ITEMS
        or int(result.get("token_budget", -1)) != EXPECTED_K
        or float(result.get("temperature", -1.0)) != EXPECTED_TEMPERATURE
        or float(result.get("score_function_weight", -1.0)) != EXPECTED_WEIGHT
        or float(result.get("baseline_momentum", -1.0))
        != EXPECTED_BASELINE_MOMENTUM
        or result.get("temporal_reduction") != EXPECTED_TEMPORAL_REDUCTION
        or result.get("pooling_mode") != "uniform_selected"
        or result.get("geometry_side_channel") is not False
        or int(result.get("batch_size", -1)) != 1
        or result.get("fp16_compress") is not True
        or result.get("default_grad_scaler") is not True
        or result.get("zero_retry") is not True
        or result.get("zero_replay") is not True
        or result.get("scheduler_advances_per_consumed_batch") is not True
        or result.get("ema_updates_per_consumed_batch") is not True
        or result.get("deterministic_warn_only") is not True
        or result.get("official_prefix_transition_semantics_matched") is not True
        or result.get("official_scheduler_advance_cadence_matched") is not True
        or result.get("official_scheduler_hyperparameters_matched") is not False
        or result.get("full_official_recipe_matched") is not False
        or result.get("official_performance_comparable") is not False
        or result.get("checkpoint_disabled") is not True
        or result.get("prediction_emitted") is not False
        or result.get("evaluator_invoked") is not False
        or result.get("official_test_opened") is not False
        or result.get("p2_p3_opened") is not False
        or result.get("performance_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("gradient-decomposition binding contract is invalid")
    if seed is not None and int(seed) != SEED:
        raise ValueError("gradient-decomposition CLI seed differs from binding")
    _full_hex(result.get("runtime_commit"), length=40, name="runtime commit")
    for key in (
        "source_config_sha256",
        "manifest_file_sha256",
        "class_map_sha256",
        "pretrained_checkpoint_sha256",
        "official_reference_config_sha256",
        "template_amp_stability_v2_binding_sha256",
        "parent_evidence_sha256",
    ):
        _full_hex(result.get(key), length=64, name=key)
    if result.get("parent_evidence_sha256") != canonical_sha256(
        _mapping(result.get("parent_evidence"), name="parent evidence")
    ):
        raise ValueError("gradient-decomposition parent evidence changed")
    rng_policy = _mapping(
        result.get("rng_matching_policy"), name="RNG matching policy"
    )
    if rng_policy != {
        "data_fingerprint_all_batches_equal": True,
        "cpu_rng_all_batches_equal": True,
        "cuda_rng_batch_zero_equal": True,
        "cuda_rng_after_batch_zero_equality_required": False,
        "reason": "PL_Gumbel_sampling_consumes_CUDA_RNG_ST_does_not",
    }:
        raise ValueError("gradient-decomposition RNG matching policy changed")
    work_dir = Path(str(result.get("work_dir", ""))).resolve()
    if Path(str(result.get("output_path", ""))).resolve() != (
        work_dir / "gradient_decomposition.json"
    ):
        raise ValueError("gradient-decomposition output is not work-dir bound")
    return result


def validate_config(cfg: Any, *, seed: int) -> dict[str, Any]:
    if "georoute_gradient_decomposition_binding" not in cfg:
        raise ValueError("config lacks gradient-decomposition binding")
    if "georoute_amp_diagnostic_binding" in cfg:
        raise ValueError("gradient diagnosis cannot retain an AMP-v2 binding")
    binding = validate_binding(
        cfg.georoute_gradient_decomposition_binding, seed=seed
    )
    workflow = _mapping(cfg.workflow, name="workflow")
    solver = _mapping(cfg.solver, name="solver")
    custom = cfg.model.backbone.custom
    if (
        str(Path(cfg.work_dir).resolve()) != binding["work_dir"]
        or cfg.georoute_protocol.get("status") != PROFILE
        or int(workflow.get("end_epoch", -1)) != 1
        or int(workflow.get("max_train_iters", -1)) != MAX_BATCHES
        or int(workflow.get("max_amp_retries_per_batch", -1)) != 0
        or workflow.get("fail_on_skipped_update") is not False
        or workflow.get("schedule_and_ema_on_success_only") is not False
        or workflow.get("capture_amp_rng_state") is not True
        or workflow.get("fail_on_nonfinite_loss") is not True
        or workflow.get("disable_checkpoint") is not True
        or solver.get("amp") is not True
        or solver.get("ema") is not True
        or solver.get("fp16_compress") is not True
        or float(solver.get("clip_grad_norm", -1.0)) != 1.0
        or custom.get("georoute_amp_diagnostic_enabled") is not True
        or custom.get("georoute_gradient_decomposition_enabled") is not True
        or int(custom.get("georoute_random_seed", -1)) != SEED
        or float(custom.get("georoute_policy_temperature", -1.0))
        != EXPECTED_TEMPERATURE
        or float(custom.get("georoute_score_function_weight", -1.0))
        != EXPECTED_WEIGHT
        or float(
            custom.get("georoute_score_function_baseline_momentum", -1.0)
        )
        != EXPECTED_BASELINE_MOMENTUM
        or custom.get("georoute_score_function_temporal_reduction")
        != EXPECTED_TEMPORAL_REDUCTION
        or custom.get("georoute_pooling_mode") != "uniform_selected"
        or custom.get("georoute_geometry_side_channel") is not False
        or cfg.inference.get("load_from_raw_predictions") is not False
        or cfg.inference.get("save_raw_prediction") is not False
        or cfg.post_processing.get("save_dict") is not False
    ):
        raise ValueError("gradient-decomposition config violates its frozen protocol")
    return binding


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_count = int(finite.sum().item())
    total = int(detached.numel())
    finite_values = detached.masked_select(finite).to(torch.float64)
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "finite": finite_count == total,
        "finite_count": finite_count,
        "nonfinite_count": total - finite_count,
        "max_abs": (
            float(finite_values.abs().max().item()) if finite_count else None
        ),
        "l2_norm": (
            float(torch.linalg.vector_norm(finite_values).item())
            if finite_count
            else None
        ),
    }


def compact_gradient_snapshot(model: Any) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {}
    nonfinite_parameters: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = _parameter_group(name)
        aggregate = grouped.setdefault(
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
        aggregate["parameter_count"] += 1
        gradient = parameter.grad
        if gradient is None:
            aggregate["missing_gradient_count"] += 1
            continue
        aggregate["parameters_with_gradient"] += 1
        summary = tensor_stats(gradient)
        aggregate["nonfinite_count"] += int(summary["nonfinite_count"])
        if summary["max_abs"] is not None:
            aggregate["max_abs"] = max(
                float(aggregate["max_abs"] or 0.0), float(summary["max_abs"])
            )
        if summary["l2_norm"] is not None:
            aggregate["l2_norm"] += float(summary["l2_norm"]) ** 2
        if int(summary["nonfinite_count"]) > 0:
            nonfinite_parameters.append(
                {
                    "name": str(name),
                    "group": group,
                    **summary,
                }
            )
    for aggregate in grouped.values():
        aggregate["l2_norm"] = math.sqrt(float(aggregate["l2_norm"]))
    return {
        "groups": grouped,
        "nonfinite_parameters": nonfinite_parameters,
        "has_nonfinite": bool(nonfinite_parameters),
        "nonfinite_groups": sorted(
            group
            for group, summary in grouped.items()
            if int(summary["nonfinite_count"]) > 0
        ),
    }


@torch.no_grad()
def ordered_pl_log_prob_and_score(
    *,
    logits: torch.Tensor,
    ordered_indices: torch.Tensor,
    valid_mask: torch.Tensor,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    if logits.ndim != 3 or ordered_indices.ndim != 3:
        raise ValueError("PL logits/order must be [B,T,N] and [B,T,K]")
    if valid_mask.shape != logits.shape or valid_mask.dtype != torch.bool:
        raise ValueError("PL valid mask must be bool and match logits")
    if ordered_indices.shape[:2] != logits.shape[:2]:
        raise ValueError("PL order batch/time axes differ")
    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("PL temperature must be finite and positive")

    work_dtype = (
        torch.float64 if logits.dtype == torch.float64 else torch.float32
    )
    work = logits.detach().to(work_dtype)
    ordered = ordered_indices.detach().to(torch.long)
    available = valid_mask.detach().clone()
    score = torch.zeros_like(work)
    log_probability = torch.zeros(
        work.shape[:2], device=work.device, dtype=work_dtype
    )
    slot_records: list[dict[str, Any]] = []
    for slot in range(int(ordered.shape[-1])):
        choice = ordered[..., slot]
        if not bool(available.gather(-1, choice.unsqueeze(-1)).all().item()):
            raise ValueError("ordered PL path selects an unavailable token")
        masked = (work / float(temperature)).masked_fill(
            ~available, float("-inf")
        )
        log_probabilities = torch.log_softmax(masked, dim=-1)
        probabilities = torch.softmax(masked, dim=-1)
        chosen_log_probability = log_probabilities.gather(
            -1, choice.unsqueeze(-1)
        ).squeeze(-1)
        chosen_probability = probabilities.gather(
            -1, choice.unsqueeze(-1)
        ).squeeze(-1)
        log_probability.add_(chosen_log_probability)
        one_hot = torch.zeros_like(probabilities)
        one_hot.scatter_(-1, choice.unsqueeze(-1), 1.0)
        slot_score = (one_hot - probabilities) / float(temperature)
        score.add_(slot_score)
        entropy = -(
            probabilities
            * probabilities.clamp_min(torch.finfo(work_dtype).tiny).log()
        ).sum(dim=-1)
        slot_records.append(
            {
                "slot": slot,
                "remaining_min": int(available.sum(-1).min().item()),
                "remaining_max": int(available.sum(-1).max().item()),
                "chosen_probability_min": float(
                    chosen_probability.min().item()
                ),
                "chosen_probability_mean": float(
                    chosen_probability.mean().item()
                ),
                "chosen_probability_max": float(
                    chosen_probability.max().item()
                ),
                "remaining_entropy_mean": float(entropy.mean().item()),
                "slot_score": tensor_stats(slot_score),
                "cumulative_score": tensor_stats(score),
            }
        )
        available.scatter_(-1, choice.unsqueeze(-1), False)
    return log_probability, score, slot_records


def expected_scaled_logit_gradient(
    *,
    score: torch.Tensor,
    advantage: torch.Tensor,
    weight: float,
    temporal_reduction: str,
    loss_scale: float,
) -> torch.Tensor:
    if score.ndim != 3:
        raise ValueError("score must have shape [B,T,N]")
    if advantage.ndim != 0 or not bool(torch.isfinite(advantage).item()):
        raise ValueError("advantage must be one finite scalar")
    if temporal_reduction not in {"sum", "mean"}:
        raise ValueError("unsupported temporal reduction")
    if not math.isfinite(float(loss_scale)) or float(loss_scale) <= 0.0:
        raise ValueError("loss scale must be finite and positive")
    batch_size, tubelets, _ = score.shape
    denominator = float(batch_size)
    if temporal_reduction == "mean":
        denominator *= float(tubelets)
    coefficient = (
        float(loss_scale)
        * float(weight)
        * float(advantage.detach().to(torch.float32).item())
        / denominator
    )
    return score.to(torch.float32) * coefficient


def bucket_cast_telemetry(
    *,
    bucket_buffer: torch.Tensor,
    loss_scale: float,
    world_size: int,
) -> dict[str, Any]:
    if int(world_size) != 1:
        raise ValueError("gradient decomposition is frozen to one DDP process")
    original = bucket_buffer.detach()
    if original.dtype != torch.float32:
        raise ValueError("gradient decomposition expects FP32 pre-hook buckets")
    before = tensor_stats(original)
    hypothetical_unscaled = tensor_stats(
        original.to(torch.float32) / float(loss_scale)
    )
    # Match the authoritative PyTorch 2.0.1 hook order exactly: cast, then divide.
    # Both operations use a detached allocation and never mutate the real bucket.
    fp16_shadow = original.to(torch.float16).clone().div_(float(world_size))
    after_cast = tensor_stats(fp16_shadow)
    return {
        "fp32_pre_hook": before,
        "hypothetical_unscaled": hypothetical_unscaled,
        "fp16_shadow_cast_then_divide": after_cast,
        "cast_introduced_nonfinite": (
            before["finite"] is True and after_cast["finite"] is False
        ),
        "world_size": int(world_size),
    }


@dataclass
class ObservedFp16HookState:
    observer: "GradientDecompositionObserver"
    parameter_names: Mapping[int, str]
    process_group: Any = None


def observed_fp16_compress_hook(
    state: ObservedFp16HookState,
    bucket: dist.GradBucket,
):
    names = [
        state.parameter_names.get(id(parameter), "<unbound>")
        for parameter in bucket.parameters()
    ]
    state.observer.record_ddp_bucket(
        bucket_index=int(bucket.index()),
        parameter_names=names,
        bucket_buffer=bucket.buffer(),
    )
    # The production hook remains authoritative, including its returned Future.
    return comm_hooks.fp16_compress_hook(state.process_group, bucket)


class GradientDecompositionObserver:
    """Compact per-batch observer attached to the existing engine event stream."""

    def __init__(
        self,
        *,
        binding: Mapping[str, Any],
        output_path: str | Path,
        runtime_commit: str,
        slurm_job_id: str,
        rank: int,
    ) -> None:
        self.binding = validate_binding(binding)
        self.output_path = Path(output_path).resolve()
        if self.output_path != Path(self.binding["output_path"]).resolve():
            raise ValueError("gradient observer output differs from binding")
        if int(rank) != 0:
            raise ValueError("gradient observer is frozen to rank zero")
        if str(runtime_commit).lower() != self.binding["runtime_commit"]:
            raise ValueError("gradient observer runtime commit mismatch")
        if str(slurm_job_id) != str(os.environ.get("SLURM_JOB_ID", "")):
            raise ValueError("gradient observer Slurm Job ID is not process-bound")
        if self.output_path.exists():
            raise FileExistsError("gradient-decomposition receipt already exists")
        hook_source = inspect.getsource(comm_hooks.fp16_compress_hook)
        self.payload: dict[str, Any] = {
            "schema_version": RECEIPT_SCHEMA,
            "status": RUNNING_STATUS,
            "study_id": STUDY_ID,
            "profile": PROFILE,
            "arm": self.binding["arm"],
            "runtime_commit": self.binding["runtime_commit"],
            "slurm_job_id": str(slurm_job_id),
            "binding": dict(self.binding),
            "runtime": {
                "torch_version": str(torch.__version__),
                "cuda_version": str(torch.version.cuda),
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.is_available()
                    else None
                ),
                "nccl_version": (
                    list(torch.cuda.nccl.version())
                    if torch.cuda.is_available()
                    else None
                ),
                "fp16_compress_hook_source_sha256": hashlib.sha256(
                    hook_source.encode("utf-8")
                ).hexdigest(),
            },
            "bucket_layouts": {},
            "batches": [],
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        self._active: dict[str, Any] | None = None
        self._current_scale: float | None = None
        self._current_expected: torch.Tensor | None = None
        self._logit_gradient_record: dict[str, Any] | None = None
        self._bucket_records: list[dict[str, Any]] = []
        self._logit_hook: Any = None
        self._publish()

    @staticmethod
    def _backbone(model: Any) -> Any:
        unwrapped = getattr(model, "module", model)
        return unwrapped.backbone

    def _publish(self) -> None:
        unsigned = dict(self.payload)
        unsigned.pop("receipt_sha256", None)
        self.payload["receipt_sha256"] = canonical_sha256(unsigned)
        _atomic_write_json(self.output_path, self.payload)

    def record_ddp_bucket(
        self,
        *,
        bucket_index: int,
        parameter_names: Sequence[str],
        bucket_buffer: torch.Tensor,
    ) -> None:
        if self._active is None or self._current_scale is None:
            raise RuntimeError("DDP bucket observed outside a bound batch")
        names = [str(name) for name in parameter_names]
        if not names or any(name == "<unbound>" for name in names):
            raise RuntimeError("DDP bucket parameter layout is incomplete")
        groups = sorted({_parameter_group(name) for name in names})
        layout = {
            "parameter_names": names,
            "parameter_groups": groups,
        }
        layout_sha256 = canonical_sha256(layout)
        self.payload["bucket_layouts"].setdefault(layout_sha256, layout)
        self._bucket_records.append(
            {
                "bucket_index": int(bucket_index),
                "layout_sha256": layout_sha256,
                "parameter_groups": groups,
                "telemetry": bucket_cast_telemetry(
                    bucket_buffer=bucket_buffer,
                    loss_scale=self._current_scale,
                    world_size=dist.get_world_size(),
                ),
            }
        )

    def _capture_logit_gradient(self, gradient: torch.Tensor) -> torch.Tensor:
        actual = gradient.detach()
        record: dict[str, Any] = {"actual_scaled": tensor_stats(actual)}
        if self._current_expected is not None:
            expected = self._current_expected
            difference = actual.to(torch.float32) - expected
            actual64 = actual.to(torch.float64)
            expected64 = expected.to(torch.float64)
            dot = torch.sum(actual64 * expected64)
            denominator = torch.linalg.vector_norm(
                actual64
            ) * torch.linalg.vector_norm(expected64)
            cosine = (
                float((dot / denominator).item())
                if float(denominator.item()) > 0.0
                else None
            )
            record.update(
                {
                    "expected_scaled": tensor_stats(expected),
                    "difference": tensor_stats(difference),
                    "cosine": cosine,
                    "direction_positive": float(dot.item()) > 0.0,
                    "close": bool(
                        torch.allclose(
                            actual.to(torch.float32),
                            expected,
                            rtol=1.0e-4,
                            atol=1.0e-6,
                        )
                    ),
                }
            )
        self._logit_gradient_record = record
        return gradient

    def _classify_failed_attempt(self) -> dict[str, Any]:
        actual = self._logit_gradient_record or {}
        actual_finite = (
            actual.get("actual_scaled", {}).get("finite") is True
        )
        expected_summary = actual.get("expected_scaled")
        expected_finite = (
            expected_summary is None or expected_summary.get("finite") is True
        )
        pre_nonfinite = [
            record
            for record in self._bucket_records
            if record["telemetry"]["fp32_pre_hook"]["finite"] is False
        ]
        cast_overflow = [
            record
            for record in self._bucket_records
            if record["telemetry"]["cast_introduced_nonfinite"] is True
        ]
        affected_groups = sorted(
            {
                group
                for record in [*pre_nonfinite, *cast_overflow]
                for group in record["parameter_groups"]
            }
        )
        if not expected_finite or not actual_finite:
            mechanism = "UPSTREAM_SCORE_NONFINITE"
        elif pre_nonfinite and any(
            "scout_score_function" in record["parameter_groups"]
            for record in pre_nonfinite
        ):
            mechanism = "SCOUT_VJP_NONFINITE"
        elif cast_overflow and not pre_nonfinite:
            mechanism = "DDP_FP16_CAST_OVERFLOW"
        else:
            mechanism = "AMBIGUOUS_MIXED_FAILURE"
        return {
            "mechanism_class": mechanism,
            "affected_parameter_groups": affected_groups,
            "detector_only": bool(affected_groups)
            and set(affected_groups) == {"detector"},
            "fp32_pre_hook_nonfinite_bucket_count": len(pre_nonfinite),
            "fp16_cast_overflow_bucket_count": len(cast_overflow),
            "analytic_expected_finite": expected_finite,
            "actual_logit_gradient_finite": actual_finite,
        }

    def __call__(self, event: str, **payload: Any) -> None:
        if event == "batch_start":
            if self._active is not None:
                raise RuntimeError("gradient observer already has an active batch")
            self._current_scale = float(payload["scale"])
            self._current_expected = None
            self._logit_gradient_record = None
            self._bucket_records = []
            data_descriptor = _describe_data(payload["data_dict"])
            cpu_rng = _describe_data(payload.get("cpu_rng_state"))
            cuda_rng = _describe_data(payload.get("cuda_rng_states"))
            self._active = {
                "iter_idx": int(payload["iter_idx"]),
                "retry_count": int(payload.get("retry_count", 0)),
                "scale_at_batch_start": self._current_scale,
                "successful_update_index": int(
                    payload["successful_update_index"]
                ),
                "data_descriptor": data_descriptor,
                "data_fingerprint_sha256": canonical_sha256(
                    {"data": data_descriptor}
                ),
                "cpu_rng_sha256": canonical_sha256({"rng": cpu_rng}),
                "cuda_rng_sha256": canonical_sha256({"rng": cuda_rng}),
            }
            return

        if self._active is None:
            raise RuntimeError(f"gradient event {event!r} has no active batch")

        if event == "forward_complete":
            backbone = self._backbone(payload["model"])
            probe = backbone.peek_gradient_decomposition_payload()
            logits = probe["logits"]
            if logits.shape != (
                1,
                EXPECTED_TUBELETS,
                EXPECTED_ITEMS,
            ) or int(probe["target_k"]) != EXPECTED_K:
                raise RuntimeError(
                    "gradient diagnosis left the frozen B/T/N/K geometry"
                )
            analytic: dict[str, Any] | None = None
            if probe["policy_estimator"] == "score_function":
                log_probability, score, slots = ordered_pl_log_prob_and_score(
                    logits=logits,
                    ordered_indices=probe["ordered_indices"],
                    valid_mask=probe["valid_mask"],
                    temperature=float(probe["temperature"]),
                )
                registered = probe["ordered_log_prob"].to(torch.float32)
                if not torch.allclose(
                    log_probability,
                    registered,
                    rtol=1.0e-5,
                    atol=1.0e-6,
                ):
                    raise FloatingPointError(
                        "analytic PL log probability disagrees with production"
                    )
                self._current_expected = expected_scaled_logit_gradient(
                    score=score,
                    advantage=probe["advantage"],
                    weight=float(probe["weight"]),
                    temporal_reduction=str(probe["temporal_reduction"]),
                    loss_scale=float(self._current_scale),
                )
                subset = [
                    index
                    for index in FP64_REFERENCE_TUBELETS
                    if index < logits.shape[1]
                ]
                reference_logp, reference_score, _ = (
                    ordered_pl_log_prob_and_score(
                        logits=logits[:, subset].to(torch.float64),
                        ordered_indices=probe["ordered_indices"][:, subset],
                        valid_mask=probe["valid_mask"][:, subset],
                        temperature=float(probe["temperature"]),
                    )
                )
                fp32_subset = score[:, subset]
                analytic = {
                    "log_probability": tensor_stats(log_probability),
                    "score": tensor_stats(score),
                    "slots": slots,
                    "expected_scaled_logit_gradient": tensor_stats(
                        self._current_expected
                    ),
                    "fp64_reference": {
                        "tubelet_indices": subset,
                        "log_probability": tensor_stats(reference_logp),
                        "score": tensor_stats(reference_score),
                        "fp32_vs_fp64_score_max_abs_difference": float(
                            (
                                fp32_subset.to(torch.float64)
                                - reference_score.to(torch.float64)
                            )
                            .abs()
                            .max()
                            .item()
                        ),
                    },
                }
            self._logit_hook = logits.register_hook(
                self._capture_logit_gradient
            )
            losses = {
                str(name): _tensor_numeric_summary(value)
                for name, value in payload["losses"].items()
                if torch.is_tensor(value)
            }
            self._active["forward"] = {
                "losses": losses,
                "all_losses_finite": all(
                    summary["finite"] is True for summary in losses.values()
                ),
                "probe": {
                    key: (
                        _tensor_numeric_summary(value)
                        if torch.is_tensor(value)
                        else value
                    )
                    for key, value in probe.items()
                    if key
                    not in {
                        "logits",
                        "ordered_indices",
                        "valid_mask",
                        "ordered_log_prob",
                    }
                },
                "analytic_pl": analytic,
            }
            return

        if event == "scaled_backward":
            if self._logit_gradient_record is None:
                raise RuntimeError("residual-logit gradient hook did not fire")
            self._active["scaled_backward"] = {
                "logit_gradient": copy.deepcopy(self._logit_gradient_record),
                "ddp_buckets": copy.deepcopy(self._bucket_records),
                "parameter_gradient": compact_gradient_snapshot(payload["model"]),
            }
            return

        if event in {"unscaled", "pre_clip", "post_clip"}:
            self._active[event] = {
                "scale": float(payload["scale"]),
                "parameter_gradient": compact_gradient_snapshot(payload["model"]),
                **(
                    {"clip_grad_l2norm": float(payload["clip_grad_l2norm"])}
                    if "clip_grad_l2norm" in payload
                    else {}
                ),
            }
            return

        if event == "scaler_result":
            succeeded = bool(payload["update_succeeded"])
            record = {
                "scale_before": float(payload["scale_before"]),
                "scale_after": float(payload["scale_after"]),
                "update_succeeded": succeeded,
                "parameter_gradient": compact_gradient_snapshot(payload["model"]),
                "failure_classification": (
                    None if succeeded else self._classify_failed_attempt()
                ),
            }
            self._active["scaler_result"] = record
            return

        if event == "batch_complete":
            backbone = self._backbone(payload["model"])
            backbone.clear_gradient_decomposition_payload()
            if self._logit_hook is not None:
                self._logit_hook.remove()
            self._logit_hook = None
            self._active["batch_complete"] = {
                "update_succeeded": bool(payload["update_succeeded"]),
                "successful_updates": int(payload["successful_updates"]),
                "scale": float(payload["scale"]),
            }
            self.payload["batches"].append(self._active)
            self._active = None
            self._current_expected = None
            self._current_scale = None
            self._bucket_records = []
            self._logit_gradient_record = None
            # Publish once per consumed batch.  This bounds I/O while retaining
            # a crash-resilient, self-hashed prefix.
            self._publish()
            return

        raise ValueError(f"unsupported gradient-decomposition event {event!r}")

    def _summary(
        self, update_audit: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        batches = list(self.payload["batches"])
        results = [
            batch.get("scaler_result")
            for batch in batches
            if isinstance(batch.get("scaler_result"), Mapping)
        ]
        failures = [
            result for result in results if result["update_succeeded"] is False
        ]
        directions = [
            batch.get("scaled_backward", {})
            .get("logit_gradient", {})
            .get("direction_positive")
            for batch in batches
            if self.binding["arm"] == ARMS[0]
        ]
        audit = dict(update_audit or {})
        return {
            "batch_count": len(batches),
            "optimizer_attempt_count": len(results),
            "successful_update_count": sum(
                result["update_succeeded"] is True for result in results
            ),
            "failed_attempt_count": len(failures),
            "failed_batch_indices": [
                int(batch["iter_idx"])
                for batch in batches
                if batch.get("scaler_result", {}).get("update_succeeded") is False
            ],
            "failure_classes": [
                result["failure_classification"]["mechanism_class"]
                for result in failures
            ],
            "data_fingerprint_sha256_by_batch": [
                batch["data_fingerprint_sha256"] for batch in batches
            ],
            "cpu_rng_sha256_by_batch": [
                batch["cpu_rng_sha256"] for batch in batches
            ],
            "cuda_rng_sha256_by_batch": [
                batch["cuda_rng_sha256"] for batch in batches
            ],
            "scale_before_by_batch": [
                float(result["scale_before"]) for result in results
            ],
            "scale_after_by_batch": [
                float(result["scale_after"]) for result in results
            ],
            "update_succeeded_by_batch": [
                bool(result["update_succeeded"]) for result in results
            ],
            "all_forward_losses_finite": bool(batches)
            and all(
                batch.get("forward", {}).get("all_losses_finite") is True
                for batch in batches
            ),
            "complete_bucket_telemetry": bool(batches)
            and all(
                bool(
                    batch.get("scaled_backward", {}).get("ddp_buckets")
                )
                for batch in batches
            ),
            "all_pl_directions_positive": (
                bool(directions)
                and all(value is True for value in directions)
                if self.binding["arm"] == ARMS[0]
                else None
            ),
            "consumed_batch_count": int(
                audit.get("consumed_batches", len(batches))
            ),
            "replay_attempt_count": int(audit.get("replay_attempts", 0)),
            "retry_attempt_count": sum(
                int(batch.get("retry_count", 0)) > 0 for batch in batches
            ),
            "scheduler_advance_count": int(
                audit.get("scheduler_advances", 0)
            ),
            "ema_update_count": int(audit.get("ema_updates", 0)),
        }

    def finalize_success(
        self,
        *,
        successful_updates: int,
        update_audit: Mapping[str, Any],
    ) -> None:
        if self.payload["status"] != RUNNING_STATUS:
            raise RuntimeError("gradient observer was already finalized")
        summary = self._summary(update_audit)
        if (
            self._active is not None
            or int(summary["batch_count"]) != MAX_BATCHES
            or int(summary["optimizer_attempt_count"]) != MAX_BATCHES
            or int(summary["consumed_batch_count"]) != MAX_BATCHES
            or int(summary["retry_attempt_count"]) != 0
            or int(summary["replay_attempt_count"]) != 0
            or int(summary["scheduler_advance_count"]) != MAX_BATCHES
            or int(summary["ema_update_count"]) != MAX_BATCHES
            or summary["all_forward_losses_finite"] is not True
            or summary["complete_bucket_telemetry"] is not True
            or int(successful_updates)
            != int(summary["successful_update_count"])
        ):
            raise RuntimeError(
                "gradient-decomposition execution conditions are incomplete"
            )
        self.payload["status"] = PASS_STATUS
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
        if self.payload["status"] != RUNNING_STATUS:
            return
        if self._active is not None:
            self._active["incomplete_failure"] = {
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
            }
            self.payload["batches"].append(self._active)
            self._active = None
        trace = traceback.format_exc()
        self.payload["status"] = FAIL_STATUS
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


def validate_receipt(
    payload: Mapping[str, Any],
    *,
    expected_arm: str | None = None,
    expected_commit: str | None = None,
    expected_slurm_job_id: str | None = None,
) -> dict[str, Any]:
    result = dict(_mapping(payload, name="gradient-decomposition receipt"))
    if not _self_hash_matches(result, field="receipt_sha256"):
        raise ValueError("gradient-decomposition receipt self-hash mismatch")
    binding = validate_binding(
        _mapping(result.get("binding"), name="receipt binding")
    )
    arm = str(result.get("arm", ""))
    status = result.get("status")
    if (
        result.get("schema_version") != RECEIPT_SCHEMA
        or result.get("study_id") != STUDY_ID
        or result.get("profile") != PROFILE
        or arm not in ARMS
        or status not in {PASS_STATUS, FAIL_STATUS}
        or result.get("runtime_commit") != binding["runtime_commit"]
        or result.get("checkpoint_emitted") is not False
        or result.get("prediction_emitted") is not False
        or result.get("evaluator_invoked") is not False
        or result.get("official_test_opened") is not False
        or result.get("performance_inference_allowed") is not False
        or result.get("paper_claim_allowed") is not False
    ):
        raise ValueError("gradient-decomposition receipt contract is invalid")
    if expected_arm is not None and arm != expected_arm:
        raise ValueError("gradient-decomposition receipt arm mismatch")
    if (
        expected_commit is not None
        and result["runtime_commit"] != str(expected_commit).lower()
    ):
        raise ValueError("gradient-decomposition receipt commit mismatch")
    job_id = str(result.get("slurm_job_id", ""))
    if not job_id.isdigit() or (
        expected_slurm_job_id is not None
        and job_id != str(expected_slurm_job_id)
    ):
        raise ValueError("gradient-decomposition receipt Slurm ID mismatch")
    runtime = _mapping(result.get("runtime"), name="runtime")
    _full_hex(
        runtime.get("fp16_compress_hook_source_sha256"),
        length=64,
        name="fp16 hook source hash",
    )
    batches = result.get("batches")
    summary = _mapping(result.get("summary"), name="receipt summary")
    if not isinstance(batches, list):
        raise ValueError("gradient-decomposition receipt batches are missing")
    if status == PASS_STATUS and (
        len(batches) != MAX_BATCHES
        or int(summary.get("batch_count", -1)) != MAX_BATCHES
        or int(summary.get("optimizer_attempt_count", -1)) != MAX_BATCHES
        or int(summary.get("consumed_batch_count", -1)) != MAX_BATCHES
        or int(summary.get("retry_attempt_count", -1)) != 0
        or int(summary.get("replay_attempt_count", -1)) != 0
        or int(summary.get("scheduler_advance_count", -1)) != MAX_BATCHES
        or int(summary.get("ema_update_count", -1)) != MAX_BATCHES
        or summary.get("all_forward_losses_finite") is not True
        or summary.get("complete_bucket_telemetry") is not True
    ):
        raise ValueError("passing gradient-decomposition receipt is incomplete")
    return result


def classify_pair(receipts: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    hold = {
        "decision": DECISION_HOLD,
        "repair_class_identified": False,
        "repair_class": None,
        "repair_authorized": False,
    }
    if set(receipts) != set(ARMS):
        return {**hold, "reason": "missing_or_extra_arm"}
    try:
        validated = {
            arm: validate_receipt(receipts[arm], expected_arm=arm)
            for arm in ARMS
        }
    except (TypeError, ValueError) as error:
        return {
            **hold,
            "reason": f"invalid_receipt:{type(error).__name__}",
        }
    if any(receipt["status"] != PASS_STATUS for receipt in validated.values()):
        return {**hold, "reason": "one_or_more_arms_failed_execution"}
    summaries = {arm: validated[arm]["summary"] for arm in ARMS}
    data_matched = (
        summaries[ARMS[0]]["data_fingerprint_sha256_by_batch"]
        == summaries[ARMS[1]]["data_fingerprint_sha256_by_batch"]
    )
    cpu_rng_matched = (
        summaries[ARMS[0]]["cpu_rng_sha256_by_batch"]
        == summaries[ARMS[1]]["cpu_rng_sha256_by_batch"]
    )
    cuda_sequences = {
        arm: summaries[arm]["cuda_rng_sha256_by_batch"] for arm in ARMS
    }
    initial_cuda_rng_matched = bool(
        cuda_sequences[ARMS[0]]
        and cuda_sequences[ARMS[1]]
        and cuda_sequences[ARMS[0]][0] == cuda_sequences[ARMS[1]][0]
    )
    full_cuda_rng_matched = cuda_sequences[ARMS[0]] == cuda_sequences[ARMS[1]]
    input_keys = (
        "source_config_sha256",
        "manifest_file_sha256",
        "development_annotation",
        "class_map_sha256",
        "development_video_root",
        "pretrained_checkpoint_sha256",
        "official_reference_config_sha256",
        "parent_evidence_sha256",
    )
    inputs_matched = all(
        validated[ARMS[0]]["binding"].get(key)
        == validated[ARMS[1]]["binding"].get(key)
        for key in input_keys
    )
    pl_batches = validated[ARMS[0]]["batches"]
    st_batches = validated[ARMS[1]]["batches"]
    pl_failures = [
        batch
        for batch in pl_batches
        if batch["scaler_result"]["update_succeeded"] is False
    ]
    failure_classes = [
        batch["scaler_result"]["failure_classification"]["mechanism_class"]
        for batch in pl_failures
    ]
    unique_classes = set(failure_classes)
    direction_consistent = (
        summaries[ARMS[0]].get("all_pl_directions_positive") is True
    )
    if unique_classes == {"DDP_FP16_CAST_OVERFLOW"} and pl_failures:
        detector_only = all(
            batch["scaler_result"]["failure_classification"]["detector_only"]
            for batch in pl_failures
        )
        if detector_only:
            st_by_index = {int(batch["iter_idx"]): batch for batch in st_batches}
            shared = all(
                int(batch["iter_idx"]) in st_by_index
                and st_by_index[int(batch["iter_idx"])]["scaler_result"][
                    "update_succeeded"
                ]
                is False
                and st_by_index[int(batch["iter_idx"])]["scaler_result"][
                    "failure_classification"
                ]["mechanism_class"]
                == "DDP_FP16_CAST_OVERFLOW"
                for batch in pl_failures
            )
            if shared:
                unique_classes = {"SHARED_DETECTOR_BUCKET_OVERFLOW"}
                failure_classes = [
                    "SHARED_DETECTOR_BUCKET_OVERFLOW" for _ in pl_failures
                ]
    repair_class = (
        next(iter(unique_classes)) if len(unique_classes) == 1 else None
    )
    identified = bool(
        data_matched
        and cpu_rng_matched
        and initial_cuda_rng_matched
        and inputs_matched
        and all(
            summary["all_forward_losses_finite"] is True
            and summary["complete_bucket_telemetry"] is True
            for summary in summaries.values()
        )
        and pl_failures
        and repair_class in MECHANISM_CLASSES
        and repair_class != "AMBIGUOUS_MIXED_FAILURE"
        and direction_consistent
    )
    return {
        "decision": DECISION_REPAIR if identified else DECISION_HOLD,
        "repair_class_identified": identified,
        "repair_class": repair_class if identified else None,
        "repair_authorized": identified,
        "reason": (
            "one_PL_specific_mechanism_class_with_complete_matched_telemetry"
            if identified
            else "frozen_gradient_decomposition_rule_not_satisfied"
        ),
        "data_sequence_matched": data_matched,
        "cpu_rng_sequence_matched": cpu_rng_matched,
        "initial_cuda_rng_matched": initial_cuda_rng_matched,
        "full_cuda_rng_sequence_matched": full_cuda_rng_matched,
        "post_batch_zero_cuda_rng_divergence_expected": True,
        "input_bindings_matched": inputs_matched,
        "pl_failed_attempt_count": len(pl_failures),
        "pl_failure_classes": failure_classes,
        "pl_gradient_direction_consistent": direction_consistent,
    }
