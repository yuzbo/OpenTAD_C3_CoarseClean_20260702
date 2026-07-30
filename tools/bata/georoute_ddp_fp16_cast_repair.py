"""Contracts for the no-performance DDP FP16-cast repair gate."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from tools.bata.georoute_amp_diagnostic import (
    AMP_REPAIR_INTERVENTION,
    AMP_REPAIR_REGISTERED_CLASS,
    AMP_REPAIR_STUDY_ID,
)
from tools.bata.georoute_experiment_contract import canonical_sha256


KAT_SCHEMA = "georoute_ddp_fp16_cast_repair_cuda_kat_v1"
KAT_PASS_STATUS = "PASS_DDP_FP16_CAST_REPAIR_CUDA_KAT_ONLY"
KAT_FAIL_STATUS = "FAIL_DDP_FP16_CAST_REPAIR_CUDA_KAT"
KAT_LOSS_SCALE = 65536.0
KAT_SCALED_GRADIENT = 70000.0


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


def validate_kat_receipt(
    payload: Mapping[str, Any],
    *,
    expected_commit: str | None = None,
    expected_slurm_job_id: str | None = None,
) -> dict[str, Any]:
    """Validate a CUDA/DDP proof that FP32 reduction preserves the test gradient."""

    result = dict(payload)
    if not _self_hash_matches(result, field="kat_sha256"):
        raise ValueError("DDP FP16-cast repair KAT self-hash mismatch")
    runtime_commit = _full_hex(
        result.get("runtime_commit"),
        length=40,
        name="KAT runtime commit",
    )
    slurm_job_id = str(result.get("slurm_job_id", ""))
    scaled = result.get("scaled_fp32_gradient")
    unscaled = result.get("unscaled_fp32_gradient")
    shadow = result.get("detached_fp16_cast_shadow")
    if (
        result.get("schema_version") != KAT_SCHEMA
        or result.get("status") != KAT_PASS_STATUS
        or result.get("study_id") != AMP_REPAIR_STUDY_ID
        or result.get("registered_repair_class") != AMP_REPAIR_REGISTERED_CLASS
        or result.get("registered_single_variable_intervention")
        != AMP_REPAIR_INTERVENTION
        or float(result.get("loss_scale", -1.0)) != KAT_LOSS_SCALE
        or not slurm_job_id.isdigit()
        or int(result.get("world_size", -1)) != 1
        or result.get("comm_hook_registration_invoked") is not False
        or result.get("ddp_default_fp32_reduction_completed") is not True
        or result.get("optimizer_update_completed") is not True
        or not isinstance(scaled, Mapping)
        or scaled.get("dtype") != "torch.float32"
        or scaled.get("finite") is not True
        or not math.isclose(
            float(scaled.get("max_abs", math.nan)),
            KAT_SCALED_GRADIENT,
            rel_tol=1e-6,
            abs_tol=1e-3,
        )
        or not isinstance(unscaled, Mapping)
        or unscaled.get("dtype") != "torch.float32"
        or unscaled.get("finite") is not True
        or not math.isclose(
            float(unscaled.get("max_abs", math.nan)),
            KAT_SCALED_GRADIENT / KAT_LOSS_SCALE,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        or not isinstance(shadow, Mapping)
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
        raise ValueError("DDP FP16-cast repair CUDA KAT receipt is invalid")
    if expected_commit is not None and runtime_commit != str(expected_commit).lower():
        raise ValueError("DDP FP16-cast repair KAT commit mismatch")
    if (
        expected_slurm_job_id is not None
        and slurm_job_id != str(expected_slurm_job_id)
    ):
        raise ValueError("DDP FP16-cast repair KAT Slurm Job ID mismatch")
    return result
