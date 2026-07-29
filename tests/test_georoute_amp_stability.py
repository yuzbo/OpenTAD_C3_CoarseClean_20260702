import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.bata.georoute_amp_diagnostic import (
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_INITIAL_SCALE,
    AMP_STABILITY_BINDING_SCHEMA,
    AMP_STABILITY_MAX_BATCHES,
    AMP_STABILITY_PROFILE,
    AMP_STABILITY_RECEIPT_SCHEMA,
    AMP_STABILITY_STUDY_ID,
    amp_protocol_spec,
    bind_amp_diagnostic_config,
    classify_amp_stability_pair,
    diagnostic_cell_relative_path,
    validate_amp_diagnostic_binding,
    validate_amp_diagnostic_config,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_estimator_pilot_contract import (
    PILOT_ARMS,
    PILOT_K,
    PILOT_SEED,
)
from tools.bata.georoute_experiment_contract import (
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import build_torchrun_prefix
from tools.bata.georoute_amp_diagnostic_stage_runner import (
    validate_amp_diagnostic_stage_result,
)
from tools.bata.deploy_georoute_amp_diagnostic import (
    _validate_diagnostic_parent_deployment,
)


def _stability_binding(tmp_path: Path, arm: str) -> dict:
    work_dir = tmp_path / diagnostic_cell_relative_path(
        arm=arm,
        protocol_profile=AMP_STABILITY_PROFILE,
    )
    binding = {
        "schema_version": AMP_STABILITY_BINDING_SCHEMA,
        "study_id": AMP_STABILITY_STUDY_ID,
        "protocol_profile": AMP_STABILITY_PROFILE,
        "arm": arm,
        "arm_spec": PILOT_ARMS[arm],
        "seed": PILOT_SEED,
        "token_budget": PILOT_K,
        "runtime_commit": "a" * 40,
        "work_dir": str(work_dir.resolve()),
        "output_path": str((work_dir / "amp_stability.json").resolve()),
        "max_batches": AMP_STABILITY_MAX_BATCHES,
        "max_amp_retries_per_batch": 0,
        "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "score_function_temporal_reduction": "mean",
        "zero_failed_attempts_required": True,
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
        "historical_pilot_seed_policy_matched": True,
        "amp_diagnostic_telemetry_enabled": True,
        "checkpoint_disabled": True,
        "evaluator_invoked": False,
        "prediction_emitted": False,
        "official_test_opened": False,
        "p2_p3_opened": False,
        "paper_claim_allowed": False,
    }
    binding["binding_sha256"] = canonical_sha256(binding)
    return binding


def _fingerprints() -> list[str]:
    return [
        hashlib.sha256(f"batch-{index}".encode("utf-8")).hexdigest()
        for index in range(AMP_STABILITY_MAX_BATCHES)
    ]


def _stability_receipt(
    tmp_path: Path,
    arm: str,
    *,
    job_id: str,
) -> dict:
    fingerprints = _fingerprints()
    payload = {
        "schema_version": AMP_STABILITY_RECEIPT_SCHEMA,
        "status": "PASS_STABILITY_GATE_EXECUTION_ONLY",
        "study_id": AMP_STABILITY_STUDY_ID,
        "protocol_profile": AMP_STABILITY_PROFILE,
        "arm": arm,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": _stability_binding(tmp_path, arm),
        "events": [],
        "summary": {
            "batch_count": AMP_STABILITY_MAX_BATCHES,
            "data_fingerprint_sha256_by_batch": fingerprints,
            "cpu_rng_sha256_by_batch": ["3" * 64]
            * AMP_STABILITY_MAX_BATCHES,
            "cuda_rng_sha256_by_batch": ["4" * 64]
            * AMP_STABILITY_MAX_BATCHES,
            "data_fingerprint_sha256": fingerprints[0],
            "cpu_rng_sha256": "3" * 64,
            "cuda_rng_sha256": "4" * 64,
            "optimizer_attempt_count": AMP_STABILITY_MAX_BATCHES,
            "failed_attempt_count": 0,
            "failed_attempt_scales": [],
            "first_successful_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
            "successful_scales_by_batch": [
                AMP_DIAGNOSTIC_INITIAL_SCALE
            ]
            * AMP_STABILITY_MAX_BATCHES,
            "failed_attempt_nonfinite_groups": [],
            "forward_attempt_count": AMP_STABILITY_MAX_BATCHES,
            "all_forward_losses_finite": True,
        },
        "successful_updates": AMP_STABILITY_MAX_BATCHES,
        "update_audit": {
            "optimizer_attempts": AMP_STABILITY_MAX_BATCHES,
            "amp_skipped_attempts": 0,
            "max_amp_retries_observed": 0,
        },
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def _rehash(payload: dict, field: str) -> None:
    unsigned = dict(payload)
    unsigned.pop(field, None)
    payload[field] = canonical_sha256(unsigned)


def test_stability_pair_requires_matched_32_batch_zero_skip_execution(
    tmp_path,
):
    receipts = {
        "residual_pl_rep_off": _stability_receipt(
            tmp_path,
            "residual_pl_rep_off",
            job_id="1208001",
        ),
        "residual_st_rep_off": _stability_receipt(
            tmp_path,
            "residual_st_rep_off",
            job_id="1208002",
        ),
    }
    for arm, receipt in receipts.items():
        validate_amp_diagnostic_receipt(
            receipt,
            expected_arm=arm,
            expected_profile=AMP_STABILITY_PROFILE,
        )

    decision = classify_amp_stability_pair(receipts)
    assert decision["decision"] == (
        "REAL_DATA_AMP_STABILITY_PASS_PROTOCOL_FREEZE_AUTHORIZED"
    )
    assert decision["stability_gate_passed"] is True
    assert decision["official_protocol_freeze_authorized"] is True
    assert decision["matched_data_sequence"] is True
    assert decision["batch_count"] == 32


def test_stability_pair_holds_on_data_mismatch_or_any_amp_skip(tmp_path):
    receipts = {
        "residual_pl_rep_off": _stability_receipt(
            tmp_path,
            "residual_pl_rep_off",
            job_id="1208001",
        ),
        "residual_st_rep_off": _stability_receipt(
            tmp_path,
            "residual_st_rep_off",
            job_id="1208002",
        ),
    }
    mismatched = copy.deepcopy(receipts)
    mismatched["residual_st_rep_off"]["summary"][
        "data_fingerprint_sha256_by_batch"
    ][7] = "9" * 64
    _rehash(mismatched["residual_st_rep_off"], "receipt_sha256")
    decision = classify_amp_stability_pair(mismatched)
    assert decision["decision"] == "REAL_DATA_AMP_STABILITY_HOLD"
    assert decision["official_protocol_freeze_authorized"] is False

    skipped = copy.deepcopy(receipts["residual_pl_rep_off"])
    skipped["summary"]["failed_attempt_count"] = 1
    skipped["summary"]["optimizer_attempt_count"] = 33
    skipped["summary"]["failed_attempt_scales"] = [
        AMP_DIAGNOSTIC_INITIAL_SCALE
    ]
    skipped["update_audit"]["optimizer_attempts"] = 33
    skipped["update_audit"]["amp_skipped_attempts"] = 1
    _rehash(skipped, "receipt_sha256")
    with pytest.raises(ValueError, match="zero-skip"):
        validate_amp_diagnostic_receipt(
            skipped,
            expected_profile=AMP_STABILITY_PROFILE,
        )


def test_stability_binding_uses_mean_reduction_and_no_retry(tmp_path):
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
    source_config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "adatad"
        / "thumos"
        / "georoute_adatad_development_base.py"
    )

    cfg = bind_amp_diagnostic_config(
        source_config_path=source_config,
        arm="residual_pl_rep_off",
        seed=PILOT_SEED,
        work_dir=tmp_path / "cell",
        manifest_path=manifest,
        development_annotation_path=annotation,
        class_map_path=class_map,
        development_video_root=videos,
        pretrained_checkpoint_path=pretrained,
        runtime_commit="a" * 40,
        protocol_profile=AMP_STABILITY_PROFILE,
    )
    binding = validate_amp_diagnostic_config(cfg, seed=PILOT_SEED)
    spec = amp_protocol_spec(AMP_STABILITY_PROFILE)

    assert binding["max_batches"] == 32
    assert binding["max_amp_retries_per_batch"] == 0
    assert binding["zero_failed_attempts_required"] is True
    assert binding["score_function_temporal_reduction"] == "mean"
    assert cfg.workflow.max_train_iters == 32
    assert cfg.workflow.max_amp_retries_per_batch == 0
    assert (
        cfg.model.backbone.custom.georoute_score_function_temporal_reduction
        == "mean"
    )
    assert Path(binding["output_path"]).name == "amp_stability.json"
    assert spec["initial_scale"] == 65536.0
    assert binding["checkpoint_disabled"] is True
    assert binding["evaluator_invoked"] is False
    assert binding["official_test_opened"] is False
    assert binding["paper_claim_allowed"] is False


def test_stability_stage_result_is_profile_bound_and_no_metric(tmp_path):
    arm = "residual_pl_rep_off"
    job_id = "1208001"
    receipt = _stability_receipt(tmp_path, arm, job_id=job_id)
    _command, rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=job_id,
        stage="ampstable",
        variant=arm,
        seed=PILOT_SEED,
    )
    result = {
        "schema_version": "georoute_real_data_amp_stability_stage_v1",
        "status": "PASS_STAGE_STABILITY_GATE_ONLY",
        "study_id": AMP_STABILITY_STUDY_ID,
        "protocol_profile": AMP_STABILITY_PROFILE,
        "arm": arm,
        "seed": PILOT_SEED,
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
        expected_profile=AMP_STABILITY_PROFILE,
    )
    assert "metrics" not in result
    wrong_profile = copy.deepcopy(result)
    wrong_profile["protocol_profile"] = "diagnostic"
    _rehash(wrong_profile, "stage_result_sha256")
    with pytest.raises(ValueError):
        validate_amp_diagnostic_stage_result(
            wrong_profile,
            expected_profile=AMP_STABILITY_PROFILE,
        )


def test_stability_parent_reloads_immutable_diagnostic_inputs(tmp_path):
    deployment_path = tmp_path / "deployment.json"
    deployment = {
        "schema_version": "georoute_real_batch_amp_deployment_v1",
        "study_id": "georoute_real_batch_amp_diagnostic_v1",
        "runtime_commit": "c" * 40,
        "input_receipts": {
            "GEOROUTE_MANIFEST": {
                "path": "/frozen/manifest.json",
                "sha256": "1" * 64,
            }
        },
    }
    deployment["deployment_sha256"] = canonical_sha256(deployment)
    deployment_path.write_text(
        json.dumps(deployment, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    parent = {
        "deployment_path": str(deployment_path.resolve()),
        "deployment_file_sha256": sha256_file(deployment_path),
        "runtime_commit": "c" * 40,
    }

    assert (
        _validate_diagnostic_parent_deployment(parent)["input_receipts"]
        == deployment["input_receipts"]
    )
    deployment_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact changed"):
        _validate_diagnostic_parent_deployment(parent)


def test_stability_profile_is_not_a_performance_or_paper_result_surface():
    root = Path(__file__).resolve().parents[1]
    deployer = (
        root / "tools" / "bata" / "deploy_georoute_amp_diagnostic.py"
    ).read_text(encoding="utf-8")
    finalizer = (
        root / "tools" / "bata" / "finalize_georoute_amp_diagnostic.py"
    ).read_text(encoding="utf-8")

    assert "--parent-diagnostic-finalization" in deployer
    assert "repair-authorizing diagnostic" in deployer
    assert '"performance_metrics": {}' in finalizer
    assert '"performance_inference_allowed": False' in finalizer
    assert '"official_test_opened": False' in finalizer
    assert '"paper_claim_allowed": False' in finalizer
