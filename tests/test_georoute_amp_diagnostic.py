import copy
import json
from pathlib import Path

import pytest

from tools.bata.georoute_amp_diagnostic import (
    AMP_DIAGNOSTIC_ARMS,
    AMP_DIAGNOSTIC_BINDING_SCHEMA,
    AMP_DIAGNOSTIC_INITIAL_SCALE,
    AMP_DIAGNOSTIC_RECEIPT_SCHEMA,
    AMP_DIAGNOSTIC_STAGE_SCHEMA,
    AMP_DIAGNOSTIC_STUDY_ID,
    classify_amp_diagnostic_pair,
    diagnostic_cell_relative_path,
    validate_amp_diagnostic_binding,
    validate_amp_diagnostic_job_receipt,
    validate_amp_diagnostic_receipt,
)
from tools.bata.georoute_amp_diagnostic_stage_runner import (
    audit_no_performance_artifacts,
    validate_amp_diagnostic_stage_result,
)
from tools.bata.finalize_georoute_amp_diagnostic import (
    _validate_wrapper_failure,
    finalize_amp_diagnostic,
)
from tools.bata.deploy_georoute_amp_diagnostic import _validate_parent
from tools.bata.georoute_estimator_pilot_contract import (
    PILOT_ARMS,
    PILOT_FINALIZATION_SCHEMA,
    PILOT_K,
    PILOT_SEED,
    PILOT_STUDY_ID,
)
from tools.bata.georoute_experiment_contract import (
    canonical_sha256,
    sha256_file,
)
from tools.bata.georoute_stage_runner import build_torchrun_prefix


def _binding(tmp_path: Path, arm: str) -> dict:
    work_dir = tmp_path / diagnostic_cell_relative_path(arm=arm)
    binding = {
        "schema_version": AMP_DIAGNOSTIC_BINDING_SCHEMA,
        "study_id": AMP_DIAGNOSTIC_STUDY_ID,
        "arm": arm,
        "arm_spec": PILOT_ARMS[arm],
        "seed": PILOT_SEED,
        "token_budget": PILOT_K,
        "runtime_commit": "a" * 40,
        "work_dir": str(work_dir.resolve()),
        "output_path": str((work_dir / "amp_diagnostic.json").resolve()),
        "max_batches": 1,
        "max_amp_retries_per_batch": 12,
        "initial_scale": AMP_DIAGNOSTIC_INITIAL_SCALE,
        "source_config": str((tmp_path / "source.py").resolve()),
        "source_config_sha256": "b" * 64,
        "manifest_path": str((tmp_path / "manifest.json").resolve()),
        "manifest_file_sha256": "c" * 64,
        "fit_video_ids": ["fit_a"],
        "gate_video_ids": ["gate_a"],
        "training_video_ids": ["gate_a"],
        "evaluation_video_ids": ["fit_a"],
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


def _receipt(
    tmp_path: Path,
    arm: str,
    *,
    failed_attempts: int,
    first_successful_scale: float | None,
    nonfinite_groups=(),
    status: str = "PASS_DIAGNOSTIC_EXECUTION_ONLY",
    fingerprint: str = "2" * 64,
    job_id: str = "1207001",
) -> dict:
    payload = {
        "schema_version": AMP_DIAGNOSTIC_RECEIPT_SCHEMA,
        "status": status,
        "study_id": AMP_DIAGNOSTIC_STUDY_ID,
        "arm": arm,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": _binding(tmp_path, arm),
        "events": [],
        "summary": {
            "batch_count": 1,
            "data_fingerprint_sha256": fingerprint,
            "cpu_rng_sha256": "3" * 64,
            "cuda_rng_sha256": "4" * 64,
            "optimizer_attempt_count": failed_attempts + (
                1 if first_successful_scale is not None else 0
            ),
            "failed_attempt_count": failed_attempts,
            "failed_attempt_scales": [
                AMP_DIAGNOSTIC_INITIAL_SCALE / (2**index)
                for index in range(failed_attempts)
            ],
            "first_successful_scale": first_successful_scale,
            "failed_attempt_nonfinite_groups": list(nonfinite_groups),
            "forward_attempt_count": failed_attempts + 1,
            "all_forward_losses_finite": True,
        },
        "successful_updates": 1 if first_successful_scale is not None else 0,
        "update_audit": {
            "optimizer_attempts": failed_attempts + 1,
            "amp_skipped_attempts": failed_attempts,
            "max_amp_retries_observed": failed_attempts,
        },
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    if status == "FAIL_DIAGNOSTIC_EXECUTION":
        payload["failure"] = {
            "exception_type": "FloatingPointError",
            "exception_message": "synthetic",
            "traceback_sha256": "5" * 64,
        }
    payload["receipt_sha256"] = canonical_sha256(payload)
    return payload


def test_binding_preserves_historical_train_and_development_populations(
    tmp_path,
):
    binding = validate_amp_diagnostic_binding(
        _binding(tmp_path, "residual_pl_rep_off")
    )
    assert binding["training_video_ids"] == binding["gate_video_ids"]
    assert binding["evaluation_video_ids"] == binding["fit_video_ids"]

    tampered = copy.deepcopy(binding)
    tampered["training_video_ids"] = tampered["fit_video_ids"]
    tampered["binding_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "binding_sha256"}
    )
    with pytest.raises(ValueError, match="population binding changed"):
        validate_amp_diagnostic_binding(tampered)


def test_receipt_is_self_bound_to_arm_commit_and_slurm_job(tmp_path):
    receipt = _receipt(
        tmp_path,
        "residual_st_rep_off",
        failed_attempts=0,
        first_successful_scale=AMP_DIAGNOSTIC_INITIAL_SCALE,
    )
    validate_amp_diagnostic_receipt(
        receipt,
        expected_arm="residual_st_rep_off",
        expected_commit="a" * 40,
        expected_slurm_job_id="1207001",
    )
    tampered = copy.deepcopy(receipt)
    tampered["slurm_job_id"] = "1207002"
    with pytest.raises(ValueError, match="self-hash mismatch"):
        validate_amp_diagnostic_receipt(tampered)


def test_pair_localizes_only_matched_pl_score_function_overflow(tmp_path):
    receipts = {
        "residual_pl_rep_off": _receipt(
            tmp_path,
            "residual_pl_rep_off",
            failed_attempts=10,
            first_successful_scale=64.0,
            nonfinite_groups=("scout_score_function",),
            job_id="1207001",
        ),
        "residual_st_rep_off": _receipt(
            tmp_path,
            "residual_st_rep_off",
            failed_attempts=0,
            first_successful_scale=AMP_DIAGNOSTIC_INITIAL_SCALE,
            job_id="1207002",
        ),
    }
    decision = classify_amp_diagnostic_pair(receipts)
    assert decision["decision"] == "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
    assert decision["repair_authorized"] is True
    assert decision["matched_execution"] is True

    mismatched = copy.deepcopy(receipts)
    mismatched["residual_st_rep_off"]["summary"][
        "data_fingerprint_sha256"
    ] = "6" * 64
    unsigned = dict(mismatched["residual_st_rep_off"])
    unsigned.pop("receipt_sha256")
    mismatched["residual_st_rep_off"]["receipt_sha256"] = canonical_sha256(
        unsigned
    )
    decision = classify_amp_diagnostic_pair(mismatched)
    assert decision["decision"] == "ROOT_CAUSE_NOT_LOCALIZED_HOLD"
    assert decision["repair_authorized"] is False


def test_failed_arm_cannot_authorize_a_repair(tmp_path):
    receipts = {
        "residual_pl_rep_off": _receipt(
            tmp_path,
            "residual_pl_rep_off",
            failed_attempts=13,
            first_successful_scale=None,
            nonfinite_groups=("scout_score_function",),
            status="FAIL_DIAGNOSTIC_EXECUTION",
            job_id="1207001",
        ),
        "residual_st_rep_off": _receipt(
            tmp_path,
            "residual_st_rep_off",
            failed_attempts=0,
            first_successful_scale=AMP_DIAGNOSTIC_INITIAL_SCALE,
            job_id="1207002",
        ),
    }
    decision = classify_amp_diagnostic_pair(receipts)
    assert decision["decision"] == "DIAGNOSTIC_INCOMPLETE_NO_REPAIR"
    assert decision["repair_authorized"] is False


def test_stage_result_contains_no_metric_or_checkpoint_surface(tmp_path):
    arm = "residual_st_rep_off"
    job_id = "1207001"
    diagnostic = _receipt(
        tmp_path,
        arm,
        failed_attempts=0,
        first_successful_scale=AMP_DIAGNOSTIC_INITIAL_SCALE,
        job_id=job_id,
    )
    _command, rendezvous = build_torchrun_prefix(
        phase="train",
        slurm_job_id=job_id,
        stage="ampdiag",
        variant=arm,
        seed=PILOT_SEED,
    )
    result = {
        "schema_version": AMP_DIAGNOSTIC_STAGE_SCHEMA,
        "status": "PASS_STAGE_DIAGNOSTIC_ONLY",
        "study_id": AMP_DIAGNOSTIC_STUDY_ID,
        "arm": arm,
        "seed": PILOT_SEED,
        "runtime_commit": "a" * 40,
        "slurm_job_id": job_id,
        "binding": diagnostic["binding"],
        "binding_sha256": diagnostic["binding"]["binding_sha256"],
        "diagnostic_receipt": diagnostic,
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
    )
    assert "metrics" not in result


def test_artifact_audit_rejects_checkpoint_prediction_and_tmp(tmp_path):
    cell = tmp_path / "cell"
    cell.mkdir()
    assert audit_no_performance_artifacts(cell)["checkpoint_payload_count"] == 0
    (cell / "forbidden.pth").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="forbidden performance artifact"):
        audit_no_performance_artifacts(cell)


def test_job_receipt_requires_distinct_two_arm_leaves_and_finalizer():
    jobs = {
        "stage": {
            AMP_DIAGNOSTIC_ARMS[0]: "1207001",
            AMP_DIAGNOSTIC_ARMS[1]: "1207002",
        },
        "finalizer": "1207003",
    }
    assert validate_amp_diagnostic_job_receipt(jobs) == jobs
    duplicated = copy.deepcopy(jobs)
    duplicated["finalizer"] = "1207001"
    with pytest.raises(ValueError, match="reuses a Slurm ID"):
        validate_amp_diagnostic_job_receipt(duplicated)


def test_parent_finalization_is_bound_to_explicit_runtime_commit(tmp_path):
    path = tmp_path / "pilot_finalization.json"
    parent = {
        "schema_version": PILOT_FINALIZATION_SCHEMA,
        "study_id": PILOT_STUDY_ID,
        "status": "INCOMPLETE_EXPLORATORY_PILOT",
        "decision": "PILOT_INCOMPLETE_NO_PERFORMANCE_INFERENCE",
        "runtime_commit": "c" * 40,
        "failures": {"residual_pl_rep_off": {"status": "FAILED"}},
        "official_test_opened": False,
        "paper_claim_allowed": False,
    }
    parent["finalization_sha256"] = canonical_sha256(parent)
    path.write_text(json.dumps(parent, sort_keys=True) + "\n", encoding="utf-8")

    assert _validate_parent(
        path,
        expected_file_sha256=sha256_file(path),
        expected_runtime_commit="c" * 40,
    ) == parent
    with pytest.raises(ValueError, match="sealed failed pilot"):
        _validate_parent(
            path,
            expected_file_sha256=sha256_file(path),
            expected_runtime_commit="d" * 40,
        )


def test_wrapper_failure_is_bound_to_arm_commit_and_slurm_job():
    failure = {
        "schema_version": AMP_DIAGNOSTIC_STAGE_SCHEMA,
        "status": "FAIL_STAGE_WRAPPER_PREVALIDATION_OR_SEALING",
        "study_id": AMP_DIAGNOSTIC_STUDY_ID,
        "arm": "residual_pl_rep_off",
        "seed": PILOT_SEED,
        "expected_runtime_commit": "a" * 40,
        "observed_runtime_commit": "a" * 40,
        "slurm_job_id": "1207001",
        "exception_type": "RuntimeError",
        "exception_message": "synthetic",
        "traceback_sha256": "b" * 64,
        "checkpoint_emitted": False,
        "prediction_emitted": False,
        "evaluator_invoked": False,
        "official_test_opened": False,
        "performance_inference_allowed": False,
        "paper_claim_allowed": False,
    }
    failure["failure_sha256"] = canonical_sha256(failure)
    assert _validate_wrapper_failure(
        failure,
        expected_arm="residual_pl_rep_off",
        expected_commit="a" * 40,
        expected_job_id="1207001",
    ) == failure

    rebound = copy.deepcopy(failure)
    rebound["slurm_job_id"] = "1207002"
    rebound["failure_sha256"] = canonical_sha256(
        {
            key: value
            for key, value in rebound.items()
            if key != "failure_sha256"
        }
    )
    with pytest.raises(ValueError, match="contract is invalid"):
        _validate_wrapper_failure(
            rebound,
            expected_arm="residual_pl_rep_off",
            expected_commit="a" * 40,
            expected_job_id="1207001",
        )


def test_finalizer_authorizes_only_a_no_metric_repair_step(tmp_path):
    run_root = tmp_path / "run"
    stage_jobs = {
        "residual_pl_rep_off": "1207001",
        "residual_st_rep_off": "1207002",
    }
    for arm in AMP_DIAGNOSTIC_ARMS:
        job_id = stage_jobs[arm]
        cell = run_root / diagnostic_cell_relative_path(arm=arm)
        cell.mkdir(parents=True)
        diagnostic = _receipt(
            run_root,
            arm,
            failed_attempts=10 if arm == "residual_pl_rep_off" else 0,
            first_successful_scale=(
                64.0
                if arm == "residual_pl_rep_off"
                else AMP_DIAGNOSTIC_INITIAL_SCALE
            ),
            nonfinite_groups=(
                ("scout_score_function",)
                if arm == "residual_pl_rep_off"
                else ()
            ),
            job_id=job_id,
        )
        diagnostic_path = cell / "amp_diagnostic.json"
        diagnostic_path.write_text(
            json.dumps(diagnostic, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        config_path = run_root / "control" / "bound_configs" / f"{arm}.py"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("cfg = {}\n", encoding="utf-8")
        train_log = run_root / "control" / "train_logs" / f"{arm}.out"
        train_log.parent.mkdir(parents=True, exist_ok=True)
        train_log.write_text("diagnostic only\n", encoding="utf-8")
        _command, rendezvous = build_torchrun_prefix(
            phase="train",
            slurm_job_id=job_id,
            stage="ampdiag",
            variant=arm,
            seed=PILOT_SEED,
        )
        result = {
            "schema_version": AMP_DIAGNOSTIC_STAGE_SCHEMA,
            "status": "PASS_STAGE_DIAGNOSTIC_ONLY",
            "study_id": AMP_DIAGNOSTIC_STUDY_ID,
            "arm": arm,
            "seed": PILOT_SEED,
            "runtime_commit": "a" * 40,
            "slurm_job_id": job_id,
            "binding": diagnostic["binding"],
            "binding_sha256": diagnostic["binding"]["binding_sha256"],
            "diagnostic_receipt": diagnostic,
            "diagnostic_receipt_path": str(diagnostic_path.resolve()),
            "diagnostic_receipt_file_sha256": sha256_file(diagnostic_path),
            "bound_config_path": str(config_path.resolve()),
            "bound_config_sha256": sha256_file(config_path),
            "train_log_path": str(train_log.resolve()),
            "train_log_sha256": sha256_file(train_log),
            "rendezvous": rendezvous,
            "artifact_audit": audit_no_performance_artifacts(cell),
            "execution_error": None,
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        result["stage_result_sha256"] = canonical_sha256(result)
        (cell / "stage_result.json").write_text(
            json.dumps(result, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    finalization = finalize_amp_diagnostic(
        run_root=run_root,
        expected_commit="a" * 40,
        expected_stage_jobs=stage_jobs,
    )
    assert finalization["decision"] == (
        "ROOT_CAUSE_LOCALIZED_REPAIR_AUTHORIZED"
    )
    assert finalization["repair_authorized"] is True
    assert finalization["performance_metrics"] == {}
    assert finalization["performance_inference_allowed"] is False
    assert finalization["official_test_opened"] is False
    assert finalization["paper_claim_allowed"] is False


def test_launchers_and_deployer_freeze_diagnostic_only_scope():
    root = Path(__file__).resolve().parents[1]
    deployer = (
        root / "tools" / "bata" / "deploy_georoute_amp_diagnostic.py"
    ).read_text(encoding="utf-8")
    stage = (
        root / "scripts" / "run_georoute_amp_diagnostic_stage_slurm.sh"
    ).read_text(encoding="utf-8")
    finalizer = (
        root / "tools" / "bata" / "finalize_georoute_amp_diagnostic.py"
    ).read_text(encoding="utf-8")
    assert "_require_submit_capacity(additional_jobs=3)" in deployer
    assert deployer.index("_require_submit_capacity(additional_jobs=3)") < (
        deployer.index("run_root.mkdir")
    )
    # One call is inside the frozen two-arm loop; the second is the finalizer.
    assert deployer.count("test_only=True") == 2
    assert 'dependency_type="afterany"' in deployer
    assert "--hold" in deployer
    assert "tools/test.py" not in stage
    assert "--not_eval" not in stage
    assert '"performance_metrics": {}' in finalizer
    assert '"paper_claim_allowed": False' in finalizer
