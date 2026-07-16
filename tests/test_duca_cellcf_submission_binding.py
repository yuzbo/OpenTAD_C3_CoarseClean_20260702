from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPARE = (ROOT / "scripts/prepare_duca_cellcf_suite.sh").read_text(encoding="utf-8")
SUBMIT = (ROOT / "scripts/submit_duca_cellcf_suite.sh").read_text(encoding="utf-8")


def test_preparation_hash_binds_the_complete_formal_dag() -> None:
    assert "duca_cellcf_prepared_submission_v1" in PREPARE
    assert 'PREPARED_SUBMISSION="${RUN_ROOT}/prepared_submission.json"' in PREPARE
    assert 'MANIFEST_SHA256="$(sha256_file "${MANIFEST}")"' in PREPARE
    assert '"suite_manifest_sha256": manifest_sha256' in PREPARE
    assert '"job_file_sha256": row["sbatch_sha256"]' in PREPARE
    assert 'job_keys=(uniform transition_beta0 cellcf aggregate cost completion)' in PREPARE
    assert 'checkpoint_interval": 5' in PREPARE
    assert "checkpoint-every-5" in PREPARE

    for role in (
        "none",
        "afterok_three_arms",
        "afterok_aggregate",
        "afterok_aggregate_and_cost",
    ):
        assert f"DUCA_CELLCF_DEPENDENCY_ROLE={role}" in PREPARE
    assert "#SBATCH --clusters=${TARGET_CLUSTER}" in PREPARE
    assert r'[[ "\${SLURM_JOB_NAME:-}" ==' in PREPARE
    assert r'[[ "\${SLURM_CLUSTER_NAME:-}" ==' in PREPARE


def test_submitter_rejects_stale_receipts_and_closes_the_crash_window() -> None:
    assert "flock -n 9" in SUBMIT
    assert ".intent.json" in SUBMIT
    assert ".receipt.json" in SUBMIT
    assert "intent exists without receipt; reconcile Slurm manually" in SUBMIT
    assert "receipt exists without its bound intent" in SUBMIT
    assert "receipt does not bind the exact submission intent" in SUBMIT
    assert "receipt job_ref does not preserve jobid;cluster identity" in SUBMIT
    assert "validate_duca_cellcf_slurm_receipt" in SUBMIT
    assert 'command -v sacct' in SUBMIT
    assert 'command -v squeue' in SUBMIT
    assert "duca_cellcf_slurm_submission_v2" in SUBMIT

    for field in (
        "job_name",
        "job_file_sha256",
        "dependency_role",
        "dependency",
        "cluster",
        "suite_manifest_sha256",
        "prepared_submission_sha256",
        "git_commit",
        "seed",
    ):
        assert f'"{field}"' in SUBMIT

    assert '"--clusters=${target_cluster}"' in SUBMIT
    assert '"--job-name=${job_name}"' in SUBMIT
    assert 'sbatch_args+=("--dependency=${dependency}")' in SUBMIT


def test_cost_and_completion_are_mandatory_after_successful_aggregate() -> None:
    assert "SUBMIT_COST" not in SUBMIT
    assert 'cost_dependency="afterok:${aggregate_id}"' in SUBMIT
    assert 'completion_dependency="afterok:${aggregate_id}:${cost_id}"' in SUBMIT
    assert SUBMIT.index('read_prepared_job "aggregate"') < SUBMIT.index(
        'read_prepared_job "cost"'
    )
    assert SUBMIT.index('read_prepared_job "cost"') < SUBMIT.index(
        'read_prepared_job "completion"'
    )

    assert "DUCA_CELLCF_POST_RUN_EVIDENCE_JSON" in PREPARE
    assert "DUCA_CELLCF_POST_RUN_EVIDENCE_SHA256" in PREPARE
    assert "aggregate evidence does not cover exactly three CellCF arms" in PREPARE
    assert "--cost-evidence '${COST_EVIDENCE}'" in PREPARE
    assert "--require-cost-evidence" in PREPARE
    assert "aggregate_suite_evidence.json" in PREPARE
    assert "final_suite_evidence.json" in PREPARE
    assert 'payload.get("status") != "runs_complete_cost_pending"' in PREPARE


def test_slurm_jobs_use_generic_allocations_without_physical_gpu_override() -> None:
    assert "#SBATCH --gres=gpu:1" in PREPARE
    assert "CUDA_VISIBLE_DEVICES=" not in PREPARE
    assert '"--clusters=${target_cluster}"' in SUBMIT
    assert "--gres=gpu:1" not in SUBMIT
