import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.georoute_amp_diagnostic import (
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_INITIAL_SCALE,
    AMP_REPAIR_INTERVENTION,
    AMP_REPAIR_PROFILE,
    AMP_REPAIR_REGISTERED_CLASS,
    AMP_REPAIR_SEED,
    amp_protocol_spec,
    bind_amp_diagnostic_config,
    classify_amp_repair_pair,
    validate_amp_diagnostic_binding,
    validate_amp_diagnostic_config,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_ddp_fp16_cast_repair import (
    KAT_LOSS_SCALE,
    KAT_PASS_STATUS,
    KAT_SCALED_GRADIENT,
    KAT_SCHEMA,
    validate_kat_receipt,
)
from tools.bata.georoute_experiment_contract import canonical_sha256


def _rehash(payload: dict, field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = canonical_sha256(unsigned)


def _repair_binding_and_config(tmp_path: Path, arm: str):
    manifest = tmp_path / "manifest.json"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.json"
    pretrained = tmp_path / "pretrained.pth"
    videos = tmp_path / "videos"
    videos.mkdir(exist_ok=True)
    manifest.write_text(
        json.dumps({"splits": {"fit": ["fit"], "gate": ["gate"]}}),
        encoding="utf-8",
    )
    annotation.write_text(
        json.dumps(
            {
                "database": {
                    "fit": {"subset": "training"},
                    "gate": {"subset": "training"},
                }
            }
        ),
        encoding="utf-8",
    )
    class_map.write_text("{}", encoding="utf-8")
    pretrained.write_bytes(b"test")
    root = Path(__file__).resolve().parents[1]
    cfg = bind_amp_diagnostic_config(
        source_config_path=(
            root
            / "configs"
            / "adatad"
            / "thumos"
            / "georoute_adatad_development_base.py"
        ),
        arm=arm,
        seed=AMP_REPAIR_SEED,
        work_dir=tmp_path / f"cell_{arm}",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=videos,
        pretrained_checkpoint_path=pretrained,
        runtime_commit="a" * 40,
        protocol_profile=AMP_REPAIR_PROFILE,
        official_reference_config_path=(
            root
            / "configs"
            / "adatad"
            / "thumos"
            / "e2e_thumos_videomae_s_768x1_160_adapter.py"
        ),
    )
    return cfg, dict(cfg.georoute_amp_diagnostic_binding)


def _repair_receipt(
    tmp_path: Path,
    arm: str,
    *,
    job_id: str,
    skip_indices: tuple[int, ...] = (),
) -> dict:
    _cfg, binding = _repair_binding_and_config(tmp_path, arm)
    spec = amp_protocol_spec(AMP_REPAIR_PROFILE)
    batch_count = int(spec["max_batches"])
    skip_set = set(skip_indices)
    scale = AMP_DIAGNOSTIC_INITIAL_SCALE
    scale_before = []
    scale_after = []
    success_flags = []
    for index in range(batch_count):
        scale_before.append(scale)
        succeeded = index not in skip_set
        success_flags.append(succeeded)
        if not succeeded:
            scale /= 2.0
        scale_after.append(scale)
    fingerprints = [
        hashlib.sha256(f"batch-{index}".encode()).hexdigest()
        for index in range(batch_count)
    ]
    current_streak = 0
    max_streak = 0
    for succeeded in success_flags:
        current_streak = 0 if succeeded else current_streak + 1
        max_streak = max(max_streak, current_streak)
    summary = {
        "batch_count": batch_count,
        "data_fingerprint_sha256_by_batch": fingerprints,
        "cpu_rng_sha256_by_batch": ["3" * 64] * batch_count,
        "cuda_rng_sha256_by_batch": ["4" * 64] * batch_count,
        "data_fingerprint_sha256": fingerprints[0],
        "cpu_rng_sha256": "3" * 64,
        "cuda_rng_sha256": "4" * 64,
        "optimizer_attempt_count": batch_count,
        "failed_attempt_count": len(skip_indices),
        "skipped_batch_indices": list(skip_indices),
        "failed_attempt_scales": [scale_before[index] for index in skip_indices],
        "first_successful_scale": next(
            scale_before[index]
            for index, succeeded in enumerate(success_flags)
            if succeeded
        ),
        "successful_scales_by_batch": [
            scale_before[index]
            for index, succeeded in enumerate(success_flags)
            if succeeded
        ],
        "scale_before_by_attempt": scale_before,
        "scale_after_by_attempt": scale_after,
        "update_succeeded_by_attempt": success_flags,
        "max_consecutive_skipped_attempts": max_streak,
        "minimum_observed_scale": min([*scale_before, *scale_after]),
        "final_scale": scale_after[-1],
        "observed_initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "stable_tail_batches": int(spec["stable_tail_batches"]),
        "stable_tail_success_count": sum(
            success_flags[-int(spec["stable_tail_batches"]) :]
        ),
        "stable_tail_all_success": all(
            success_flags[-int(spec["stable_tail_batches"]) :]
        ),
        "retry_attempt_count": 0,
        "replay_attempt_count": 0,
        "consumed_batch_count": batch_count,
        "scheduler_advance_count": batch_count,
        "ema_update_count": batch_count,
        "failed_attempt_nonfinite_groups": (
            ["detector"] if skip_indices else []
        ),
        "forward_attempt_count": batch_count,
        "all_forward_losses_finite": True,
    }
    receipt = {
        "schema_version": spec["receipt_schema"],
        "status": spec["receipt_pass_status"],
        "study_id": spec["study_id"],
        "protocol_profile": spec["profile"],
        "arm": arm,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": binding,
        "events": [],
        "summary": summary,
        "successful_updates": sum(success_flags),
        "update_audit": {
            "optimizer_attempts": batch_count,
            "amp_skipped_attempts": len(skip_indices),
            "max_amp_retries_observed": 0,
            "consumed_batches": batch_count,
            "replay_attempts": 0,
            "scheduler_advances": batch_count,
            "ema_updates": batch_count,
        },
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    return receipt


def test_repair_profile_is_one_preregistered_communication_change(tmp_path):
    cfg, binding = _repair_binding_and_config(
        tmp_path, "residual_pl_rep_off"
    )
    spec = amp_protocol_spec(AMP_REPAIR_PROFILE)

    assert spec["seed"] == 2307
    assert spec["max_batches"] == 64
    assert spec["ddp_fp16_compress_enabled"] is False
    assert spec["registered_repair_class"] == "DDP_FP16_CAST_OVERFLOW"
    assert cfg.solver.fp16_compress is False
    assert binding["registered_intervention_count"] == 1
    assert binding["registered_single_variable_intervention"] == (
        "disable_ddp_fp16_compression"
    )
    assert binding["estimator_changed"] is False
    assert binding["objective_changed"] is False
    validate_amp_diagnostic_config(cfg, seed=AMP_REPAIR_SEED)

    cfg.solver.fp16_compress = True
    with pytest.raises(ValueError, match="no-metric protocol"):
        validate_amp_diagnostic_config(cfg, seed=AMP_REPAIR_SEED)


def test_repair_pair_passes_only_as_no_performance_protocol_gate(tmp_path):
    receipts = {
        AMP_DIAGNOSTIC_ARMS[0]: _repair_receipt(
            tmp_path,
            AMP_DIAGNOSTIC_ARMS[0],
            job_id="1211001",
            skip_indices=(2,),
        ),
        AMP_DIAGNOSTIC_ARMS[1]: _repair_receipt(
            tmp_path,
            AMP_DIAGNOSTIC_ARMS[1],
            job_id="1211002",
            skip_indices=(5,),
        ),
    }
    for arm, receipt in receipts.items():
        validate_amp_diagnostic_receipt(
            receipt,
            expected_arm=arm,
            expected_profile=AMP_REPAIR_PROFILE,
        )
    decision = classify_amp_repair_pair(receipts)

    assert decision["repair_gate_passed"] is True
    assert decision["matched_formal_protocol_freeze_authorized"] is True
    assert decision["official_protocol_freeze_authorized"] is False
    assert decision["registered_repair_matched"] is True
    assert decision["ddp_fp16_compress_enabled"] is False

    changed = copy.deepcopy(receipts[AMP_DIAGNOSTIC_ARMS[0]])
    changed["binding"]["ddp_fp16_compress_enabled"] = True
    _rehash(changed["binding"], "binding_sha256")
    _rehash(changed, "receipt_sha256")
    with pytest.raises(ValueError, match="binding contract"):
        validate_amp_diagnostic_receipt(
            changed,
            expected_profile=AMP_REPAIR_PROFILE,
        )


def test_repair_kat_contract_proves_fp32_survives_removed_cast():
    payload = {
        "schema_version": KAT_SCHEMA,
        "status": KAT_PASS_STATUS,
        "study_id": "georoute_ddp_fp16_cast_repair_gate_v1",
        "runtime_commit": "a" * 40,
        "slurm_job_id": "1212001",
        "world_size": 1,
        "registered_repair_class": AMP_REPAIR_REGISTERED_CLASS,
        "registered_single_variable_intervention": AMP_REPAIR_INTERVENTION,
        "loss_scale": KAT_LOSS_SCALE,
        "comm_hook_registration_invoked": False,
        "ddp_default_fp32_reduction_completed": True,
        "scaled_fp32_gradient": {
            "dtype": "torch.float32",
            "finite": True,
            "nonfinite_count": 0,
            "max_abs": KAT_SCALED_GRADIENT,
        },
        "detached_fp16_cast_shadow": {
            "dtype": "torch.float16",
            "finite": False,
            "nonfinite_count": 2,
            "max_abs": None,
        },
        "unscaled_fp32_gradient": {
            "dtype": "torch.float32",
            "finite": True,
            "nonfinite_count": 0,
            "max_abs": KAT_SCALED_GRADIENT / KAT_LOSS_SCALE,
        },
        "optimizer_update_completed": True,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    payload["kat_sha256"] = canonical_sha256(payload)

    validate_kat_receipt(
        payload,
        expected_commit="a" * 40,
        expected_slurm_job_id="1212001",
    )
