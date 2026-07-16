from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.bata.validate_duca_cellcf_slurm_receipt import validate_slurm_receipt


def _runner(
    *,
    squeue: str = "",
    sacct: str = "",
    predecessors: str = "",
    squeue_returncode: int = 0,
    squeue_stderr: str = "",
    calls: list[str] | None = None,
    time_formats: list[str | None] | None = None,
):
    def run(command, **_kwargs):
        if calls is not None:
            calls.append(command[0])
        if command[0] == "sacct" and time_formats is not None:
            time_formats.append((_kwargs.get("env") or {}).get("SLURM_TIME_FORMAT"))
        if command[0] == "squeue":
            output = squeue
        elif "SubmitLine%1024" in command[-1]:
            output = sacct
        else:
            output = predecessors
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
            squeue="123|cellcf-uniform-s0-abcdef0|RUNNING|cellcf-token|(null)\n"
        ),
    )
    assert result["source"] == "squeue"
    assert result["state"] == "RUNNING"


def test_completed_receipt_reopens_exact_accounting_identity() -> None:
    calls: list[str] = []
    time_formats: list[str | None] = []
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-uniform-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        runner=_runner(
            sacct=(
                "123|cellcf-uniform-s0-abcdef0|COMPLETED|cellcf-token|n16r4|"
                "sbatch --parsable --clusters=n16r4 job.sbatch|"
                "2026-07-16T10:00:00|2026-07-16T11:00:00\n"
            ),
            squeue_returncode=1,
            squeue_stderr="squeue: error: Invalid job id specified\n",
            calls=calls,
            time_formats=time_formats,
        ),
    )
    assert calls == ["squeue", "sacct"]
    assert time_formats == ["standard"]
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
                sacct=(
                    f"123|cellcf-uniform-s0-abcdef0|{state}|cellcf-token|n16r4|"
                    "sbatch --parsable --clusters=n16r4 job.sbatch|Unknown|Unknown\n"
                )
            ),
        )


def test_unknown_receipt_is_not_treated_as_idempotent() -> None:
    with pytest.raises(ValueError, match="absent"):
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
                squeue=(
                    "123|cellcf-uniform-s0-abcdef0|PENDING|another-token|(null)\n"
                )
            ),
        )


def test_live_pending_receipt_requires_exact_dependency() -> None:
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-aggregate-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        dependency="afterok:1:2:3",
        runner=_runner(
            squeue=(
                "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|"
                "afterok:1:2:3\n"
            )
        ),
    )
    assert result["source"] == "squeue"
    assert result["dependency"] == "afterok:1:2:3"


def test_running_receipt_reopens_dependency_from_accounting_submit_line() -> None:
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-aggregate-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        dependency="afterok:1:2:3",
        runner=_runner(
            squeue=(
                "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|(null)\n"
            ),
            sacct=(
                "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|n16r4|"
                "sbatch --parsable --dependency=afterok:1:2:3 job.sbatch|"
                "2026-07-16T12:00:00|Unknown\n"
            ),
            predecessors=(
                "1|COMPLETED|0:0|2026-07-16T11:00:00|2026-07-16T10:00:00|n16r4\n"
                "2|COMPLETED|0:0|2026-07-16T11:01:00|2026-07-16T10:00:00|n16r4\n"
                "3|COMPLETED|0:0|2026-07-16T11:02:00|2026-07-16T10:00:00|n16r4\n"
            ),
        ),
    )
    assert result["source"] == "sacct"
    assert result["dependency"] == "afterok:1:2:3"


def test_live_remaining_dependency_may_be_a_subset_of_original_afterok() -> None:
    result = validate_slurm_receipt(
        job_id=123,
        job_name="cellcf-aggregate-s0-abcdef0",
        token="cellcf-token",
        cluster="n16r4",
        dependency="afterok:1:2:3",
        runner=_runner(
            squeue=(
                "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|"
                "afterok:2:3\n"
            ),
            sacct=(
                "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|n16r4|"
                "sbatch --dependency=afterok:1:2:3 job.sbatch|Unknown|Unknown\n"
            ),
            predecessors=(
                "1|COMPLETED|0:0|2026-07-16T11:00:00|"
                "2026-07-16T10:00:00|n16r4\n"
            ),
        ),
    )
    assert result["source"] == "sacct"


def test_removed_afterok_dependency_must_be_successfully_completed() -> None:
    with pytest.raises(ValueError, match="not successfully completed"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1:2:3",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|"
                    "afterok:2:3\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|n16r4|"
                    "sbatch --dependency=afterok:1:2:3 job.sbatch|Unknown|Unknown\n"
                ),
                predecessors="1|RUNNING|0:0|Unknown|2026-07-16T10:00:00|n16r4\n",
            ),
        )


def test_started_target_must_not_precede_satisfied_afterok_dependencies() -> None:
    with pytest.raises(ValueError, match="target started before"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1:2:3",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|(null)\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|n16r4|"
                    "sbatch --dependency=afterok:1:2:3 job.sbatch|"
                    "2026-07-16T12:00:00|Unknown\n"
                ),
                predecessors=(
                    "1|COMPLETED|0:0|2026-07-16T11:00:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                    "2|COMPLETED|0:0|2026-07-16T12:30:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                    "3|COMPLETED|0:0|2026-07-16T11:30:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                ),
            ),
        )


def test_started_target_cannot_retain_unmet_dependencies() -> None:
    with pytest.raises(ValueError, match="started while dependencies remain"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1:2:3",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|"
                    "afterok:1:2:3\n"
                )
            ),
        )


def test_live_started_state_overrides_stale_pending_accounting_state() -> None:
    with pytest.raises(ValueError, match="target start timestamp is unavailable"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1:2:3",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|(null)\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|n16r4|"
                    "sbatch --dependency=afterok:1:2:3 job.sbatch|Unknown|Unknown\n"
                ),
                predecessors=(
                    "1|COMPLETED|0:0|2026-07-16T11:00:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                    "2|COMPLETED|0:0|2026-07-16T11:01:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                    "3|COMPLETED|0:0|2026-07-16T11:02:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                ),
            ),
        )


def test_accounting_started_state_cannot_retain_live_dependencies() -> None:
    with pytest.raises(ValueError, match="started while afterok dependencies remain"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1:2:3",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|"
                    "afterok:1:2:3\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|n16r4|"
                    "sbatch --dependency=afterok:1:2:3 job.sbatch|"
                    "2026-07-16T12:00:00|Unknown\n"
                ),
            ),
        )


def test_date_only_accounting_time_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid Slurm target start timestamp"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency="afterok:1",
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|(null)\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|RUNNING|cellcf-token|n16r4|"
                    "sbatch --dependency=afterok:1 job.sbatch|2026-07-16|Unknown\n"
                ),
                predecessors=(
                    "1|COMPLETED|0:0|2026-07-16T11:00:00|"
                    "2026-07-16T10:00:00|n16r4\n"
                ),
            ),
        )


@pytest.mark.parametrize(
    ("expected", "live"),
    [
        ("", "afterok:999"),
        ("afterok:1:2:3", "afterok:2:999"),
    ],
)
def test_live_dependency_conflict_is_not_hidden_by_matching_submit_line(
    expected: str,
    live: str,
) -> None:
    submit_option = f" --dependency={expected}" if expected else ""
    with pytest.raises(ValueError, match="live dependency mismatch"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency=expected,
            runner=_runner(
                squeue=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|"
                    f"{live}\n"
                ),
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|n16r4|"
                    f"sbatch{submit_option} job.sbatch|Unknown|Unknown\n"
                ),
            ),
        )


@pytest.mark.parametrize(
    ("expected", "submit_line"),
    [
        ("afterok:1:2:3", "sbatch --dependency=afterok:1:2:4 job.sbatch"),
        ("", "sbatch --dependency=afterok:1:2:3 job.sbatch"),
    ],
)
def test_accounting_receipt_rejects_wrong_dependency(
    expected: str,
    submit_line: str,
) -> None:
    with pytest.raises(ValueError, match="dependency mismatch"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-aggregate-s0-abcdef0",
            token="cellcf-token",
            cluster="n16r4",
            dependency=expected,
            runner=_runner(
                sacct=(
                    "123|cellcf-aggregate-s0-abcdef0|PENDING|cellcf-token|n16r4|"
                    f"{submit_line}|Unknown|Unknown\n"
                )
            ),
        )
