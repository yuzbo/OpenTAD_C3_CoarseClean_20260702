import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.deploy_georoute_amp_diagnostic import (
    _validate_stability_v1_parent,
)
from tools.bata.georoute_amp_diagnostic import (
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_INITIAL_SCALE,
    AMP_STABILITY_FINALIZATION_SCHEMA,
    AMP_STABILITY_STUDY_ID,
    AMP_STABILITY_V2_BINDING_SCHEMA,
    AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS,
    AMP_STABILITY_V2_MAX_BATCHES,
    AMP_STABILITY_V2_PROFILE,
    AMP_STABILITY_V2_RECEIPT_SCHEMA,
    AMP_STABILITY_V2_SEED,
    AMP_STABILITY_V2_STUDY_ID,
    amp_protocol_spec,
    bind_amp_diagnostic_config,
    classify_amp_stability_v2_pair,
    diagnostic_cell_relative_path,
    validate_amp_diagnostic_binding,
    validate_amp_diagnostic_config,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_estimator_pilot_contract import PILOT_ARMS, PILOT_K
from tools.bata.georoute_experiment_contract import (
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_amp_diagnostic_stage_runner import (
    validate_amp_diagnostic_stage_result,
)
from tools.bata.georoute_stage_runner import build_torchrun_prefix


def _rehash(payload: dict, field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = canonical_sha256(unsigned)


def _v2_binding(tmp_path: Path, arm: str) -> dict:
    work_dir = tmp_path / diagnostic_cell_relative_path(
        arm=arm,
        protocol_profile=AMP_STABILITY_V2_PROFILE,
    )
    binding = {
        "schema_version": AMP_STABILITY_V2_BINDING_SCHEMA,
        "study_id": AMP_STABILITY_V2_STUDY_ID,
        "protocol_profile": AMP_STABILITY_V2_PROFILE,
        "arm": arm,
        "arm_spec": PILOT_ARMS[arm],
        "seed": AMP_STABILITY_V2_SEED,
        "token_budget": PILOT_K,
        "runtime_commit": "a" * 40,
        "work_dir": str(work_dir.resolve()),
        "output_path": str((work_dir / "amp_stability_v2.json").resolve()),
        "max_batches": AMP_STABILITY_V2_MAX_BATCHES,
        "max_amp_retries_per_batch": 0,
        "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "score_function_temporal_reduction": "mean",
        "zero_failed_attempts_required": False,
        "source_config": str((tmp_path / "source.py").resolve()),
        "source_config_sha256": "b" * 64,
        "manifest_path": str((tmp_path / "manifest.json").resolve()),
        "manifest_file_sha256": "c" * 64,
        "fit_video_ids": ["fit_a"],
        "gate_video_ids": ["gate_a"],
        "training_video_ids": ["fit_a"],
        "evaluation_video_ids": ["gate_a"],
        "training_block_list_video_ids": ["gate_a"],
        "evaluation_block_list_video_ids": ["fit_a"],
        "development_annotation": {
            "path": str((tmp_path / "annotation.json").resolve()),
            "sha256": "d" * 64,
        },
        "class_map_path": str((tmp_path / "classes.txt").resolve()),
        "class_map_sha256": "e" * 64,
        "development_video_root": str((tmp_path / "videos").resolve()),
        "pretrained_checkpoint_path": str(
            (tmp_path / "pretrained.pth").resolve()
        ),
        "pretrained_checkpoint_sha256": "f" * 64,
        "parent_pilot_binding_sha256": "1" * 64,
        "deterministic_same_config_reproduction": True,
        "exact_historical_batch_replay_claimed": False,
        "deterministic_algorithms_enabled": True,
        "deterministic_warn_only": True,
        "historical_pilot_seed_policy_matched": False,
        "amp_diagnostic_telemetry_enabled": True,
        "checkpoint_disabled": True,
        "evaluator_invoked": False,
        "prediction_emitted": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "paper_claim_allowed": False,
        "template_seed": 3407,
        "execution_seed": AMP_STABILITY_V2_SEED,
        "forbidden_future_paper_seeds": list(
            AMP_STABILITY_V2_FORBIDDEN_PAPER_SEEDS
        ),
        "paper_seed_disjoint": True,
        "use_default_grad_scaler_constructor": True,
        "observed_initial_scale_required": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "fail_on_skipped_update": False,
        "batch_replay_allowed": False,
        "schedule_and_ema_on_success_only": False,
        "scheduler_advances_per_consumed_batch": True,
        "ema_updates_per_consumed_batch": True,
        "capture_amp_rng_state": True,
        "fail_on_nonfinite_loss": True,
        "official_reference_config": str(
            (tmp_path / "official_reference.py").resolve()
        ),
        "official_reference_config_sha256": "2" * 64,
        "official_reference_transition_semantics": {
            "amp_enabled": True,
            "ema_enabled": True,
            "clip_grad_l2norm": 1.0,
            "max_amp_retries_per_batch": 0,
            "fail_on_skipped_update": False,
            "schedule_and_ema_on_success_only": False,
        },
        "official_reference_transition_semantics_sha256": canonical_sha256(
            {
                "amp_enabled": True,
                "ema_enabled": True,
                "clip_grad_l2norm": 1.0,
                "max_amp_retries_per_batch": 0,
                "fail_on_skipped_update": False,
                "schedule_and_ema_on_success_only": False,
            }
        ),
        "official_prefix_transition_semantics_matched": True,
        "official_scheduler_advance_cadence_matched": True,
        "official_scheduler_hyperparameters_matched": False,
        "full_official_recipe_matched": False,
        "official_performance_comparable": False,
        "full_official_training_claimed": False,
        "development_prefix_only": True,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _fingerprints() -> list[str]:
    return [
        hashlib.sha256(f"batch-{index}".encode()).hexdigest()
        for index in range(AMP_STABILITY_V2_MAX_BATCHES)
    ]


def _v2_receipt(
    tmp_path: Path,
    arm: str,
    *,
    job_id: str,
    skip_indices: tuple[int, ...] = (),
) -> dict:
    skip_set = set(skip_indices)
    scale = AMP_DIAGNOSTIC_INITIAL_SCALE
    scale_before = []
    scale_after = []
    success_flags = []
    for index in range(AMP_STABILITY_V2_MAX_BATCHES):
        scale_before.append(scale)
        succeeded = index not in skip_set
        success_flags.append(succeeded)
        if not succeeded:
            scale /= 2.0
        scale_after.append(scale)
    current_streak = 0
    max_streak = 0
    for succeeded in success_flags:
        current_streak = 0 if succeeded else current_streak + 1
        max_streak = max(max_streak, current_streak)
    fingerprints = _fingerprints()
    successful_scales = [
        scale_before[index]
        for index, succeeded in enumerate(success_flags)
        if succeeded
    ]
    summary = {
        "batch_count": AMP_STABILITY_V2_MAX_BATCHES,
        "data_fingerprint_sha256_by_batch": fingerprints,
        "cpu_rng_sha256_by_batch": ["3" * 64]
        * AMP_STABILITY_V2_MAX_BATCHES,
        "cuda_rng_sha256_by_batch": ["4" * 64]
        * AMP_STABILITY_V2_MAX_BATCHES,
        "data_fingerprint_sha256": fingerprints[0],
        "cpu_rng_sha256": "3" * 64,
        "cuda_rng_sha256": "4" * 64,
        "optimizer_attempt_count": AMP_STABILITY_V2_MAX_BATCHES,
        "failed_attempt_count": len(skip_indices),
        "skipped_batch_indices": list(skip_indices),
        "failed_attempt_scales": [
            scale_before[index] for index in skip_indices
        ],
        "first_successful_scale": successful_scales[0],
        "successful_scales_by_batch": successful_scales,
        "scale_before_by_attempt": scale_before,
        "scale_after_by_attempt": scale_after,
        "update_succeeded_by_attempt": success_flags,
        "max_consecutive_skipped_attempts": max_streak,
        "minimum_observed_scale": min([*scale_before, *scale_after]),
        "final_scale": scale_after[-1],
        "observed_initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "stable_tail_batches": 16,
        "stable_tail_success_count": sum(success_flags[-16:]),
        "stable_tail_all_success": all(success_flags[-16:]),
        "retry_attempt_count": 0,
        "replay_attempt_count": 0,
        "consumed_batch_count": AMP_STABILITY_V2_MAX_BATCHES,
        "scheduler_advance_count": AMP_STABILITY_V2_MAX_BATCHES,
        "ema_update_count": AMP_STABILITY_V2_MAX_BATCHES,
        "failed_attempt_nonfinite_groups": (
            ["detector"] if skip_indices else []
        ),
        "forward_attempt_count": AMP_STABILITY_V2_MAX_BATCHES,
        "all_forward_losses_finite": True,
    }
    payload = {
        "schema_version": AMP_STABILITY_V2_RECEIPT_SCHEMA,
        "status": (
            "PASS_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION_ONLY"
        ),
        "study_id": AMP_STABILITY_V2_STUDY_ID,
        "protocol_profile": AMP_STABILITY_V2_PROFILE,
        "arm": arm,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": _v2_binding(tmp_path, arm),
        "events": [],
        "summary": summary,
        "successful_updates": sum(success_flags),
        "update_audit": {
            "optimizer_attempts": AMP_STABILITY_V2_MAX_BATCHES,
            "amp_skipped_attempts": len(skip_indices),
            "max_amp_retries_observed": 0,
            "consumed_batches": AMP_STABILITY_V2_MAX_BATCHES,
            "replay_attempts": 0,
            "scheduler_advances": AMP_STABILITY_V2_MAX_BATCHES,
            "ema_updates": AMP_STABILITY_V2_MAX_BATCHES,
        },
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _passing_pair(tmp_path: Path) -> dict[str, dict]:
    return {
        "residual_pl_rep_off": _v2_receipt(
            tmp_path,
            "residual_pl_rep_off",
            job_id="1210001",
            skip_indices=(2, 10),
        ),
        "residual_st_rep_off": _v2_receipt(
            tmp_path,
            "residual_st_rep_off",
            job_id="1210002",
            skip_indices=(5,),
        ),
    }


def test_v2_profile_freezes_official_transition_semantics_and_disjoint_seed(
    tmp_path,
):
    spec = amp_protocol_spec(AMP_STABILITY_V2_PROFILE)
    binding = validate_amp_diagnostic_binding(
        _v2_binding(tmp_path, "residual_pl_rep_off")
    )

    assert spec["seed"] == 4417
    assert spec["max_batches"] == 64
    assert spec["retry_limit"] == 0
    assert spec["max_skipped_attempts"] == 2
    assert spec["max_consecutive_skips"] == 1
    assert spec["minimum_scale"] == 16384.0
    assert spec["stable_tail_batches"] == 16
    assert binding["use_default_grad_scaler_constructor"] is True
    assert binding["fail_on_skipped_update"] is False
    assert binding["schedule_and_ema_on_success_only"] is False
    assert binding["paper_seed_disjoint"] is True
    assert binding["official_scheduler_hyperparameters_matched"] is False
    assert binding["official_performance_comparable"] is False
    assert binding["full_official_training_claimed"] is False

    with pytest.raises(ValueError, match="seed differs"):
        bind_amp_diagnostic_config(
            source_config_path=tmp_path / "missing.py",
            arm="residual_pl_rep_off",
            seed=3407,
            work_dir=tmp_path / "cell",
            manifest_path=tmp_path / "manifest.json",
            development_annotation_path=tmp_path / "annotation.json",
            class_map_path=tmp_path / "classes.txt",
            development_video_root=tmp_path / "videos",
            pretrained_checkpoint_path=tmp_path / "pretrained.pth",
            runtime_commit="a" * 40,
            protocol_profile=AMP_STABILITY_V2_PROFILE,
            official_reference_config_path=tmp_path / "official.py",
        )


def test_v2_real_binding_verifies_official_reference_transition_semantics(
    tmp_path,
):
    manifest = tmp_path / "manifest.json"
    annotation = tmp_path / "annotation.json"
    class_map = tmp_path / "class_map.json"
    pretrained = tmp_path / "pretrained.pth"
    videos = tmp_path / "videos"
    videos.mkdir()
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
    source_config = (
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_adatad_development_base.py"
    )
    official_reference = (
        root
        / "configs"
        / "adatad"
        / "thumos"
        / "e2e_thumos_videomae_s_768x1_160_adapter.py"
    )

    cfg = bind_amp_diagnostic_config(
        source_config_path=source_config,
        arm="residual_pl_rep_off",
        seed=AMP_STABILITY_V2_SEED,
        work_dir=tmp_path / "cell",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=videos,
        pretrained_checkpoint_path=pretrained,
        runtime_commit="a" * 40,
        protocol_profile=AMP_STABILITY_V2_PROFILE,
        official_reference_config_path=official_reference,
    )
    binding = validate_amp_diagnostic_config(
        cfg,
        seed=AMP_STABILITY_V2_SEED,
    )

    assert cfg.workflow.fail_on_skipped_update is False
    assert cfg.workflow.max_amp_retries_per_batch == 0
    assert cfg.workflow.schedule_and_ema_on_success_only is False
    assert cfg.workflow.capture_amp_rng_state is True
    assert cfg.workflow.fail_on_nonfinite_loss is True
    assert cfg.model.backbone.custom.georoute_random_seed == 4417
    assert binding["official_reference_config_sha256"] == sha256_file(
        official_reference
    )
    assert binding["official_prefix_transition_semantics_matched"] is True
    assert binding["official_performance_comparable"] is False


def test_v2_pair_passes_bounded_nonconsecutive_backoff_with_stable_tail(
    tmp_path,
):
    receipts = _passing_pair(tmp_path)
    for arm, receipt in receipts.items():
        validate_amp_diagnostic_receipt(
            receipt,
            expected_arm=arm,
            expected_profile=AMP_STABILITY_V2_PROFILE,
        )

    decision = classify_amp_stability_v2_pair(receipts)
    assert decision["decision"] == (
        "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_PASS_"
        "PROTOCOL_FREEZE_AUTHORIZED"
    )
    assert decision["stability_gate_passed"] is True
    assert decision["official_protocol_freeze_authorized"] is True
    assert decision["matched_data_sequence"] is True
    assert decision["cross_arm_skip_delta"] == 1
    assert decision["final_scale_ratio"] == 2.0


@pytest.mark.parametrize(
    "skip_indices, reason",
    [
        ((2, 3), "consecutive"),
        ((2, 10, 20), "excessive"),
        ((2, 55), "unstable_tail"),
    ],
)
def test_v2_rejects_consecutive_excessive_or_unstable_tail_skips(
    tmp_path,
    skip_indices,
    reason,
):
    receipt = _v2_receipt(
        tmp_path,
        "residual_pl_rep_off",
        job_id="1210001",
        skip_indices=skip_indices,
    )
    with pytest.raises(ValueError, match="frozen threshold"):
        validate_amp_diagnostic_receipt(
            receipt,
            expected_profile=AMP_STABILITY_V2_PROFILE,
        )
    assert reason


def test_v2_pair_holds_on_ordered_data_mismatch(tmp_path):
    receipts = _passing_pair(tmp_path)
    mismatched = copy.deepcopy(receipts)
    mismatched["residual_st_rep_off"]["summary"][
        "data_fingerprint_sha256_by_batch"
    ][7] = "9" * 64
    _rehash(mismatched["residual_st_rep_off"], "receipt_sha256")

    decision = classify_amp_stability_v2_pair(mismatched)
    assert decision["decision"] == (
        "OFFICIAL_SEMANTICS_AMP_STABILITY_V2_HOLD"
    )
    assert decision["matched_data_sequence"] is False
    assert decision["official_protocol_freeze_authorized"] is False


def test_v2_stage_result_and_rendezvous_are_bound_to_seed_4417(tmp_path):
    arm = "residual_pl_rep_off"
    job_id = "1210001"
    receipt = _v2_receipt(tmp_path, arm, job_id=job_id)
    _command, rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=job_id,
        stage="ampstablev2",
        variant=arm,
        seed=AMP_STABILITY_V2_SEED,
    )
    result = {
        "schema_version": (
            "georoute_real_data_amp_stability_official_semantics_stage_v2"
        ),
        "status": (
            "PASS_STAGE_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_ONLY"
        ),
        "study_id": AMP_STABILITY_V2_STUDY_ID,
        "protocol_profile": AMP_STABILITY_V2_PROFILE,
        "arm": arm,
        "seed": AMP_STABILITY_V2_SEED,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": receipt["binding"],
        "binding_sha256": receipt["binding"]["binding_sha256"],
        "diagnostic_receipt": receipt,
        "artifact_audit": {
            "checkpoint_payload_count": 0,
            "temporary_payload_count": 0,
            "prediction_payload_count": 0,
            "evaluator_output_count": 0,
        },
        "rendezvous": rendezvous,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
        "execution_error": None,
    }
    result["stage_result_sha256"] = canonical_sha256(result)

    validate_amp_diagnostic_stage_result(
        result,
        expected_arm=arm,
        expected_commit="a" * 40,
        expected_job_id=job_id,
        expected_profile=AMP_STABILITY_V2_PROFILE,
    )
    wrong_seed = copy.deepcopy(result)
    wrong_seed["seed"] = 3407
    _rehash(wrong_seed, "stage_result_sha256")
    with pytest.raises(ValueError, match="contract"):
        validate_amp_diagnostic_stage_result(
            wrong_seed,
            expected_profile=AMP_STABILITY_V2_PROFILE,
        )


def test_v2_failure_receipt_preserves_partial_provenance_without_passing(
    tmp_path,
):
    receipt = _v2_receipt(
        tmp_path,
        "residual_pl_rep_off",
        job_id="1210001",
    )
    receipt["status"] = (
        "FAIL_OFFICIAL_SEMANTICS_AMP_STABILITY_V2_EXECUTION"
    )
    receipt["summary"]["batch_count"] = 3
    receipt["summary"]["data_fingerprint_sha256_by_batch"] = receipt[
        "summary"
    ]["data_fingerprint_sha256_by_batch"][:3]
    receipt["summary"]["cpu_rng_sha256_by_batch"] = receipt["summary"][
        "cpu_rng_sha256_by_batch"
    ][:3]
    receipt["summary"]["cuda_rng_sha256_by_batch"] = receipt["summary"][
        "cuda_rng_sha256_by_batch"
    ][:3]
    for key in (
        "scale_before_by_attempt",
        "scale_after_by_attempt",
        "update_succeeded_by_attempt",
    ):
        receipt["summary"][key] = receipt["summary"][key][:2]
    receipt["summary"]["optimizer_attempt_count"] = 2
    receipt["summary"]["forward_attempt_count"] = 3
    receipt["summary"]["successful_updates"] = 2
    receipt["summary"]["failed_attempt_count"] = 0
    receipt["summary"]["skipped_batch_indices"] = []
    receipt["summary"]["consumed_batch_count"] = 3
    receipt["summary"]["scheduler_advance_count"] = 2
    receipt["summary"]["ema_update_count"] = 2
    receipt["successful_updates"] = 2
    receipt["update_audit"].update(
        optimizer_attempts=2,
        amp_skipped_attempts=0,
        consumed_batches=3,
        scheduler_advances=2,
        ema_updates=2,
    )
    receipt["failure"] = {
        "exception_type": "FloatingPointError",
        "exception_message": "partial numerical failure",
        "traceback_sha256": "5" * 64,
    }
    _rehash(receipt, "receipt_sha256")

    validated = validate_amp_diagnostic_receipt(
        receipt,
        expected_profile=AMP_STABILITY_V2_PROFILE,
    )
    assert validated["status"].startswith("FAIL_")


def test_v2_requires_sealed_stability_v1_hold_parent(tmp_path):
    path = tmp_path / "finalization.json"
    payload = {
        "schema_version": AMP_STABILITY_FINALIZATION_SCHEMA,
        "study_id": AMP_STABILITY_STUDY_ID,
        "status": "INCOMPLETE_REAL_DATA_AMP_STABILITY_GATE",
        "decision": "STABILITY_GATE_INCOMPLETE_HOLD",
        "runtime_commit": "8" * 40,
        "stability_gate_passed": False,
        "official_protocol_freeze_authorized": False,
        "performance_metrics": {},
        "performance_inference_allowed": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["finalization_sha256"] = canonical_sha256(payload)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    validated = _validate_stability_v1_parent(
        path,
        expected_file_sha256=sha256_file(path),
        expected_runtime_commit="8" * 40,
    )
    assert validated["decision"] == "STABILITY_GATE_INCOMPLETE_HOLD"
    payload["decision"] = "REAL_DATA_AMP_STABILITY_PASS_PROTOCOL_FREEZE_AUTHORIZED"
    _rehash(payload, "finalization_sha256")
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fail-closed HOLD"):
        _validate_stability_v1_parent(
            path,
            expected_file_sha256=sha256_file(path),
            expected_runtime_commit="8" * 40,
        )


def test_v2_surface_forbids_performance_artifacts_and_uses_default_scaler():
    root = Path(__file__).resolve().parents[1]
    train_source = (root / "tools" / "train.py").read_text(encoding="utf-8")
    finalizer_source = (
        root / "tools" / "bata" / "finalize_georoute_amp_diagnostic.py"
    ).read_text(encoding="utf-8")

    assert '"use_default_grad_scaler_constructor"' in train_source
    assert "scaler = GradScaler()" in train_source
    assert '"performance_metrics": {}' in finalizer_source
    assert '"performance_inference_allowed": False' in finalizer_source
    assert '"official_test_opened": False' in finalizer_source
    assert '"paper_claim_allowed": False' in finalizer_source
