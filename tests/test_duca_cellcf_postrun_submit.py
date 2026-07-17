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
    assert "formal completion job is not uniquely COMPLETED/0:0" in source
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
    assert "postrun_evidence_complete.json" in source
    assert "submission_intent.json" in source
    assert "duca_cellcf_postrun_submission_intent_v1" in source
    assert "validate_duca_cellcf_slurm_receipt" in source
    assert "--require-scheduler-script" in source
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
    first_submit = source.index('uniform_id="$(submit_job')
    assert precheck < create_root < first_submit


def test_postrun_records_intent_before_first_scheduler_mutation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    intent = source.index('INTENT="${CONTROL_ROOT}/submission_intent.json"')
    first_submit = source.index('uniform_id="$(submit_job')
    assert intent < first_submit
    assert "submission_intent_sha256" in source
    assert '"submission_token": (' in source
    assert "job file changed after submission intent" in source
    assert "aggregate evidence changed before" in source
    assert "final suite evidence changed before" in source
    assert "sbatch failed for ${key}; reconcile" in source
    submitted = source.index(
        'write_job_receipt "${submitted_receipt}" "SUBMITTED_UNVERIFIED"'
    )
    scheduler_check = source.index(
        '"${PYTHON}" -m tools.bata.validate_duca_cellcf_slurm_receipt'
    )
    verified = source.index(
        'write_job_receipt "${verified_receipt}" "VERIFIED"',
        submitted + 1,
    )
    assert submitted < scheduler_check < verified
    assert "finalize_duca_cellcf_postrun_evidence" in source
