from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.bata.validate_duca_cellcf_slurm_receipt import validate_slurm_receipt


def _runner(
    *,
    squeue: str = "",
    sacct: str = "",
    squeue_returncode: int = 0,
    squeue_stderr: str = "",
    calls: list[str] | None = None,
):
    def run(command, **_kwargs):
        if calls is not None:
            calls.append(command[0])
        output = squeue if command[0] == "squeue" else sacct
        return SimpleNamespace(
            returncode=squeue_returncode if command[0] == "squeue" else 0,
            stdout=output,
            stderr=squeue_stderr if command[0] == "squeue" else "",
        )

    return run


def test_live_receipt_reopens_exact_job_identity() -> None:
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-uniform-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        runner=_runner(
            squeue="123|cellcf-uniform-s0-abcdef0|RUNNING|cellcf-token\n"
        ),
    )
    assert result["source"] == "squeue"
    assert result["state"] == "RUNNING"


def test_completed_receipt_reopens_exact_accounting_identity() -> None:
    calls: list[str] = []
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-uniform-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        runner=_runner(
            sacct="123|cellcf-uniform-s0-abcdef0|COMPLETED|cellcf-token|n16r4\n",
            squeue_returncode=1,
            squeue_stderr="squeue: error: Invalid job id specified\n",
            calls=calls,
        ),
    )
    assert calls == ["squeue", "sacct"]
    assert result["source"] == "sacct"
    assert result["state"] == "COMPLETED"


def test_unrelated_squeue_failure_remains_fail_closed() -> None:
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="Slurm query failed"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            runner=_runner(
                squeue_returncode=1,
                squeue_stderr="squeue: error: Access denied\n",
                calls=calls,
            ),
        )
    assert calls == ["squeue"]


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED by 42", "TIMEOUT", "OUT_OF_MEMORY"])
def test_terminal_failure_receipt_is_never_reused(state: str) -> None:
    with pytest.raises(ValueError, match="non-reusable Slurm state"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            runner=_runner(
                sacct=f"123|cellcf-uniform-s0-abcdef0|{state}|cellcf-token|n16r4\n"
            ),
        )


def test_unknown_receipt_is_not_treated_as_idempotent() -> None:
    with pytest.raises(ValueError, match="absent or ambiguous"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            runner=_runner(),
        )


def test_receipt_comment_must_match_submission_token() -> None:
    with pytest.raises(ValueError, match="comment"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            runner=_runner(
                squeue="123|cellcf-uniform-s0-abcdef0|PENDING|another-token\n"
            ),
        )
