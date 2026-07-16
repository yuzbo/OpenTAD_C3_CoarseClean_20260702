from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


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
    assert 'if ! raw_response="$(sbatch' in SUBMIT
    assert 'if ! binding="$(normalize_job_binding' in SUBMIT
    assert 'if ! intent_sha256="$(sha256_file' in SUBMIT
    assert "sbatch returned no valid job binding" in SUBMIT
    assert "parsed an invalid job id" in SUBMIT
    assert "prepared suite binding must contain exactly seven fields" in SUBMIT
    assert '--dependency "${dependency}"' in SUBMIT
    assert "os.fsync(directory_fd)" in SUBMIT


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


def _submit_once_harness(
    response: str,
    *,
    fail_intent_hash: bool = False,
    scheduler_ok: bool = True,
) -> str:
    normalize_start = SUBMIT.index("normalize_job_binding() {")
    normalize_end = SUBMIT.index("\n}\n\nwrite_submission_json()", normalize_start) + 2
    submit_start = SUBMIT.index("submit_once() {")
    submit_end = SUBMIT.index("\n}\n\nall_keys=()", submit_start) + 2
    functions = SUBMIT[normalize_start:normalize_end] + "\n\n" + SUBMIT[submit_start:submit_end]
    return "\n".join(
        (
            "set -u",
            "fail() { printf 'FAIL:%s\\n' \"$*\" >&2; exit 91; }",
            functions,
            "TMP_ROOT=$(mktemp -d)",
            "trap 'rm -rf \"$TMP_ROOT\"' EXIT",
            "RECEIPT_DIR=$TMP_ROOT/receipts",
            "mkdir -p \"$RECEIPT_DIR\"",
            "MANIFEST=$TMP_ROOT/manifest.json",
            "PREPARED_SUBMISSION=$TMP_ROOT/prepared.json",
            "JOB_FILE=$TMP_ROOT/job.sbatch",
            ": > \"$MANIFEST\"",
            ": > \"$PREPARED_SUBMISSION\"",
            ": > \"$JOB_FILE\"",
            "HASH=$(printf 'f%.0s' {1..64})",
            "MANIFEST_SHA256=$HASH",
            "PREPARED_SUBMISSION_SHA256=$HASH",
            "EXPECTED_COMMIT=$(printf 'a%.0s' {1..40})",
            "SEED=0",
            "PYTHON=python_stub",
            f"SBATCH_RESPONSE={shlex.quote(response)}",
            f"FAIL_INTENT_HASH={int(fail_intent_hash)}",
            f"SCHEDULER_OK={int(scheduler_ok)}",
            "CALLS=$TMP_ROOT/calls",
            (
                "sha256_file() { if [[ \"$FAIL_INTENT_HASH\" == 1 && \"$1\" == "
                "\"$RECEIPT_DIR/test.intent.json\" ]]; then return 7; fi; "
                "printf '%s\\n' \"$HASH\"; }"
            ),
            "write_submission_json() { printf '%s\\n' \"$2\" >> \"$CALLS\"; : > \"$1\"; }",
            (
                "python_stub() { printf 'SCHEDULER_VALIDATED\\n' >> \"$CALLS\"; "
                "[[ \"$SCHEDULER_OK\" == 1 ]]; }"
            ),
            "read_receipt_binding() { printf '123\\t123;n16r4\\tn16r4\\n'; }",
            "sbatch() { printf '%s' \"${SBATCH_RESPONSE:-}\"; return 0; }",
            "set +e",
            "BINDING=$(submit_once test cellcf-test \"$JOB_FILE\" \"$HASH\" none '' n16r4)",
            "RC=$?",
            "set -e",
            "printf 'RC=%s\\n' \"$RC\"",
            "printf 'CALLS='",
            "test ! -f \"$CALLS\" || tr '\\n' ',' < \"$CALLS\"",
            "printf '\\nBINDING=%s\\n' \"$BINDING\"",
            "test ! -e \"$RECEIPT_DIR/test.receipt.json\" || printf 'RECEIPT_PRESENT=1\\n'",
            "exit \"$RC\"",
        )
    )


def _run_submit_once_harness(
    response: str,
    *,
    fail_intent_hash: bool = False,
    scheduler_ok: bool = True,
) -> tuple[int, str, str]:
    result = subprocess.run(
        ["bash"],
        input=_submit_once_harness(
            response,
            fail_intent_hash=fail_intent_hash,
            scheduler_ok=scheduler_ok,
        ).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    return (
        result.returncode,
        result.stdout.decode("utf-8", errors="replace"),
        result.stderr.decode("utf-8", errors="replace"),
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
@pytest.mark.parametrize(
    "response",
    ["", "not-a-job", "0;n16r4", "000123;n16r4", "123", "123;other-cluster"],
)
def test_submit_once_never_writes_a_receipt_for_invalid_sbatch_binding(response: str) -> None:
    _, stdout, stderr = _run_submit_once_harness(response)
    assert "RC=91" in stdout
    assert "CALLS=INTENT_RECORDED," in stdout
    assert "RECEIPT_PRESENT=1" not in stdout
    assert "no valid job binding" in stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_submit_once_never_writes_receipt_when_intent_hash_fails() -> None:
    _, stdout, stderr = _run_submit_once_harness(
        "123;n16r4",
        fail_intent_hash=True,
    )
    assert "RC=91" in stdout
    assert "CALLS=INTENT_RECORDED," in stdout
    assert "SCHEDULER_VALIDATED" not in stdout
    assert "RECEIPT_PRESENT=1" not in stdout
    assert "failed to hash persisted" in stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_submit_once_validates_scheduler_before_writing_receipt() -> None:
    _, stdout, stderr = _run_submit_once_harness(
        "123;n16r4",
        scheduler_ok=False,
    )
    assert "RC=91" in stdout
    assert "CALLS=INTENT_RECORDED,SCHEDULER_VALIDATED," in stdout
    assert "SUBMITTED" not in stdout
    assert "RECEIPT_PRESENT=1" not in stdout
    assert "Slurm binding could not be verified" in stderr


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash is required")
def test_submit_once_writes_receipt_only_after_valid_jobid_cluster_binding() -> None:
    _, stdout, stderr = _run_submit_once_harness("123;n16r4")
    assert not stderr
    assert "RC=0" in stdout
    assert (
        "CALLS=INTENT_RECORDED,SCHEDULER_VALIDATED,SUBMITTED,"
        "SCHEDULER_VALIDATED," in stdout
    )
    assert "BINDING=123\t123;n16r4\tn16r4" in stdout
    assert "RECEIPT_PRESENT=1" in stdout
