from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "submit_duca_cellcf_cost_recovery.sh"


def test_recovery_is_bounded_and_preserves_original_failure() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "original_failure_receipt.json" in source
    assert 'cost[3] != "FAILED"' in source
    assert 're.fullmatch(r"CANCELLED' in source
    assert 'completion[4] != "0:0"' in source
    assert "int(completion[5]) != 0" in source
    assert "do not rerun or" in source
    assert "three completed 132-epoch training arms" in source
    assert "refusing to reuse an existing recovery root" in source
    assert "refusing to overwrite" not in source


def test_recovery_uses_exact_cross_commit_cost_validation() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "--evidence-repo-root" in source
    assert "--expected-evidence-commit" in source
    assert "DUCA_EVIDENCE_EXPECTED_COMMIT" in source
    assert "DUCA_CELLCF_TRAINED_REPO_ROOT" in source
    assert "DUCA_CELLCF_COST_SAMPLES" in source
    assert "DUCA_CELLCF_COST_REPEATS" in source
    assert "run_duca_cellcf_cost_pair.sh" in source
    assert source.count("#SBATCH --gres=gpu:1") == 2
    assert "epoch_131.pth" in source
    assert "state_dict_ema" in source


def test_recovery_commits_ledger_before_releasing_held_jobs() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    first_submit = source.index("submit-held-pair")
    ledger_write = source.index("printf 'job_key")
    manifest_write = source.index('"duca_cellcf_cost_recovery_submission_v1"')
    first_release = source.index('scontrol -M "${TARGET_CLUSTER}" release')

    assert first_submit < ledger_write < manifest_write < first_release
    assert "scancel" in source
    assert "submission.lock" in source
    assert "scheduler binding mismatch" in source
    assert "write batch_script" in source
    assert "scheduler-owned script differs" in source
    assert "--require-current-user-hold" in source
    assert "Reason=JobHeldUser" in source
    assert "for validation_attempt in {1..10}" in source
    assert "scheduler identity validation failed for ${key}" in source
    assert "fsync-artifacts" in source
    assert '--directory "${FORMAL_ROOT}"' in source
    assert '--directory "${RECOVERY_ROOT}/logs"' in source
