from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_duca_cellcf_postrun_evidence.sh"


def test_postrun_submitter_binds_dual_commits_and_terminal_suite() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "DUCA_EVIDENCE_EXPECTED_COMMIT" in source
    assert "DUCA_CELLCF_TRAINED_REPO_ROOT" in source
    assert "aggregate_suite_evidence.json" in source
    assert "final_suite_evidence.json" in source
    assert "original cost job is not FAILED/1:0" in source
    assert "original completion job is not cancelled" in source
    assert "DUCA_CELLCF_COST_RECOVERY_MANIFEST" in source
    assert "--cost-recovery-manifest" in source
    assert "LEGACY_EXPOSURE132_COMMITS" in source
    assert "PRECHECK_ONLY" in source
    assert "export PYTHONNOUSERSITE=1" in source
    assert "unset PYTHONHOME PYTHONPATH" in source


def test_postrun_submitter_has_exact_fail_closed_dag() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    for key in (
        "convergence_uniform",
        "convergence_transition_beta0",
        "convergence_cellcf",
        "convergence_summary",
        "training_cost",
        "completion",
    ):
        assert key in source
    assert "afterok:${uniform_id}:${transition_id}:${cellcf_id}" in source
    assert "afterok:${summary_id}:${training_cost_id}" in source
    assert source.count("#SBATCH --gres=gpu:1") == 1
    assert "refusing" not in source.lower() or "already exists" in source
    assert "jobs.submitted.tsv" in source
    assert "postrun_evidence_candidate.json" in source
    assert "--candidate" in source
    assert "submission_intent.json" in source
    assert "duca_cellcf_postrun_submission_intent_v1" in source
    assert "validate_duca_cellcf_slurm_receipt" in source
    assert "--require-scheduler-script" in source
    assert "--require-submitted-with-hold" in source
    assert "--require-current-user-hold" in source
    assert "sbatch --parsable --hold" in source
    assert "for validation_attempt in $(seq 1 10)" in source
    assert "scancel" in source
    assert 'scontrol --clusters="${TARGET_CLUSTER}" release' in source
    assert "--comment=" in source


def test_postrun_submitter_uses_versioned_nonoverwriting_outputs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "postrun_submission_${EVIDENCE_COMMIT:0:7}_v1" in source
    assert 'POSTRUN_OUTPUT_ROOT="${CONTROL_ROOT}/artifacts"' in source
    assert '[[ ! -e "${RUN_ROOT}/convergence" ]]' in source
    assert '[[ ! -e "${RUN_ROOT}/training_cost" ]]' in source
    assert '[[ ! -e "${CONTROL_ROOT}" ]]' in source
    assert "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256" in source
    assert "evidence_git_commit" in source


def test_postrun_precheck_cannot_create_files_or_submit_jobs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    precheck = source.index('if [[ "${PRECHECK_ONLY:-0}" == "1" ]]')
    create_root = source.index('mkdir -p "${CONTROL_ROOT}/jobs"')
    first_submit = source.index('submit_job convergence_uniform ""')
    assert precheck < create_root < first_submit


def test_postrun_records_intent_before_first_scheduler_mutation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    intent = source.index('INTENT="${CONTROL_ROOT}/submission_intent.json"')
    first_submit = source.index('submit_job convergence_uniform ""')
    assert intent < first_submit
    assert "submission_intent_sha256" in source
    assert '"submission_token": (' in source
    assert "job file changed after submission intent" in source
    assert "aggregate evidence changed before" in source
    assert "final suite evidence changed before" in source
    assert "sbatch failed for ${key}; rollback reconciliation required" in source
    submitted = source.index(
        'write_job_receipt "${submitted_receipt}" "SUBMITTED_UNVERIFIED"'
    )
    scheduler_check = source.index(
        '"${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt',
        submitted + 1,
    )
    verified = source.index(
        'write_job_receipt "${verified_receipt}" "VERIFIED"',
        submitted + 1,
    )
    assert submitted < scheduler_check < verified
    assert "finalize_duca_cellcf_postrun_evidence" in source


def test_postrun_jobs_are_released_only_after_durable_manifest() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    first_submit = source.index('submit_job convergence_uniform ""')
    manifest_write = source.index(
        '"${PYTHON}" - "${CONTROL_ROOT}/submission_manifest.json"'
    )
    manifest_lock = source.index(
        'chmod 0400 "${CONTROL_ROOT}/submission_manifest.json"'
    )
    first_release = source.index(
        'scontrol --clusters="${TARGET_CLUSTER}" release'
    )
    hold_barrier = source.index("revalidate_all_current_holds")
    hold_barrier_call = source.index(
        "revalidate_all_current_holds",
        hold_barrier + 1,
    )
    committed = source.index("SUBMISSION_COMMITTED=1")
    assert (
        first_submit
        < manifest_write
        < manifest_lock
        < hold_barrier_call
        < first_release
        < committed
    )


def test_postrun_rollback_recovers_ambiguous_submission_and_proves_cancel() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    token_record = source.index('UNRESOLVED_SUBMISSION_TOKEN="${token}"')
    first_sbatch = source.index("sbatch --parsable --hold", token_record)
    assert token_record < first_sbatch
    assert "reconcile_duca_cellcf_slurm_submission" in source
    assert "recover-held-job" in source
    assert "verify-cancelled" in source
    assert "ROLLBACK_REQUESTED" in source
    assert "ROLLBACK_INCOMPLETE" in source
    assert 'rollback_status="ROLLED_BACK"' in source
    assert "cancellation_verified=1" in source
    assert "scancel_exit_code" in source
    assert "scancel --clusters" in source
    assert (
        "scancel --clusters=\"${TARGET_CLUSTER}\" \\\n"
        "        \"${SUBMITTED_JOB_IDS[@]}\" >/dev/null 2>&1 || true"
        not in source
    )
    assert "append_unique_job_id" in source
    append_normal = source.index('append_unique_job_id "${job_id}"')
    clear_token = source.index(
        'UNRESOLVED_SUBMISSION_TOKEN=""',
        append_normal,
    )
    assert append_normal < clear_token
    assert "trap 'terminate_on_signal 129' HUP" in source
    assert "trap 'terminate_on_signal 130' INT" in source
    assert "trap 'terminate_on_signal 143' TERM" in source


def test_postrun_release_barrier_revalidates_all_six_current_holds() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    barrier_start = source.index("revalidate_all_current_holds()")
    barrier_end = source.index("\n}\n\nsubmit_job()", barrier_start)
    barrier = source[barrier_start:barrier_end]

    assert '"${#SUBMITTED_JOB_IDS[@]}" -eq 6' in barrier
    assert "--require-scheduler-script" in barrier
    assert "--require-submitted-with-hold" in barrier
    assert "--require-current-user-hold" in barrier
    assert "release barrier hold validation failed" in barrier
