from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from tools.bata.validate_duca_cellcf_slurm_receipt import (
    validate_slurm_receipt,
)


JOB_FILE = Path(__file__).resolve()
JOB_FILE_SHA256 = hashlib.sha256(JOB_FILE.read_bytes()).hexdigest()
TOKEN = "cellcf-token"
CLUSTER = "n16r4"


def _submit_line(
    *,
    job_name: str,
    token: str = TOKEN,
    dependency: str = "",
    job_file: Path = JOB_FILE,
    cluster: str = CLUSTER,
) -> str:
    tokens = [
        "sbatch",
        "--parsable",
        f"--clusters={cluster}",
        f"--job-name={job_name}",
        f"--comment={token}",
    ]
    if dependency:
        tokens.append(f"--dependency={dependency}")
    tokens.append(str(job_file))
    return " ".join(tokens)


def _live(
    *,
    job_id: int = 123,
    job_name: str,
    state: str | list[str] = "PENDING",
    token: str = TOKEN,
    dependency: str | None = None,
    cluster: str = CLUSTER,
    job_file: Path = JOB_FILE,
    prefix: str | None = None,
) -> str:
    payload = {
        "jobs": [
            {
                "job_id": job_id,
                "name": job_name,
                "job_state": [state] if isinstance(state, str) else state,
                "comment": token,
                "cluster": cluster,
                "dependency": dependency,
                "command": str(job_file),
            }
        ],
        "errors": [],
        "warnings": [],
    }
    cluster_prefix = f"CLUSTER: {cluster}\n" if prefix is None else prefix
    return cluster_prefix + json.dumps(payload)


def _accounting(
    *,
    job_name: str,
    state: str = "PENDING",
    comment: str = TOKEN,
    dependency: str = "",
    submit_line: str | None = None,
    start: str = "Unknown",
    end: str = "Unknown",
    cluster: str = CLUSTER,
) -> str:
    line = submit_line or _submit_line(
        job_name=job_name,
        dependency=dependency,
        cluster=cluster,
    )
    return (
        f"123|{job_name}|{state}|{comment}|{cluster}|{line}|{start}|{end}\n"
    )


def _runner(
    *,
    squeue: str = "",
    sacct: str = "",
    predecessors: str = "",
    squeue_returncode: int = 0,
    squeue_stderr: str = "",
    batch_script: bytes | None = None,
    calls: list[str] | None = None,
    commands: list[list[str]] | None = None,
    text_modes: list[bool | None] | None = None,
    time_formats: list[str | None] | None = None,
):
    def run(command, **kwargs):
        if calls is not None:
            calls.append(command[0])
        if commands is not None:
            commands.append(list(command))
        if text_modes is not None:
            text_modes.append(kwargs.get("text"))
        if command[0] == "sacct" and time_formats is not None:
            time_formats.append((kwargs.get("env") or {}).get("SLURM_TIME_FORMAT"))
        if command[0] == "squeue":
            output = squeue
        elif command[0] == "scontrol":
            output = JOB_FILE.read_bytes() if batch_script is None else batch_script
        elif any("SubmitLine" in item for item in command):
            output = sacct
        else:
            output = predecessors
        return SimpleNamespace(
            returncode=squeue_returncode if command[0] == "squeue" else 0,
            stdout=output,
            stderr=squeue_stderr if command[0] == "squeue" else "",
        )

    return run


def _validate(**kwargs):
    kwargs.setdefault("job_file", JOB_FILE)
    kwargs.setdefault("job_file_sha256", JOB_FILE_SHA256)
    return validate_slurm_receipt(**kwargs)


def test_live_receipt_reopens_exact_job_identity() -> None:
    name = "cellcf-uniform-s0-abcdef0"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        runner=_runner(squeue=_live(job_name=name, state="RUNNING")),
    )
    assert result["source"] == "squeue"
    assert result["state"] == "RUNNING"
    assert result["job_file"] == str(JOB_FILE)


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (["RUNNING", "COMPLETING"], "RUNNING"),
        (["PENDING", "REQUEUE_HOLD"], "PENDING"),
        (["CONFIGURING", "SIGNALING"], "CONFIGURING"),
    ],
)
def test_live_multivalue_slurm_states_are_normalized(
    states: list[str],
    expected: str,
) -> None:
    name = "cellcf-uniform-s0-abcdef0"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        runner=_runner(squeue=_live(job_name=name, state=states)),
    )
    assert result["state"] == expected


def test_new_submission_verifies_the_script_bytes_stored_by_slurm() -> None:
    name = "cellcf-uniform-s0-abcdef0"
    commands: list[list[str]] = []
    text_modes: list[bool | None] = []
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        require_scheduler_script=True,
        runner=_runner(
            squeue=_live(job_name=name),
            commands=commands,
            text_modes=text_modes,
        ),
    )
    assert result["scheduler_script_verified"] is True
    assert result["job_file_sha256"] == JOB_FILE_SHA256
    assert commands[-1] == [
        "scontrol",
        f"--clusters={CLUSTER}",
        "write",
        "batch_script",
        "123",
        "-",
    ]
    assert text_modes[-1] is False


def test_new_submission_rejects_a_scheduler_script_hash_mismatch() -> None:
    name = "cellcf-uniform-s0-abcdef0"
    with pytest.raises(ValueError, match="stored batch script"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            require_scheduler_script=True,
            runner=_runner(
                squeue=_live(job_name=name),
                batch_script=b"#!/bin/bash\necho changed\n",
            ),
        )


def test_local_job_file_hash_must_match_before_scheduler_queries() -> None:
    with pytest.raises(ValueError, match="file hash mismatch"):
        _validate(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            job_file_sha256="0" * 64,
            runner=_runner(),
        )


def test_completed_receipt_reopens_exact_accounting_identity() -> None:
    name = "cellcf-uniform-s0-abcdef0"
    calls: list[str] = []
    time_formats: list[str | None] = []
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        runner=_runner(
            sacct=_accounting(
                job_name=name,
                state="COMPLETED",
                start="2026-07-16T10:00:00",
                end="2026-07-16T11:00:00",
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


@pytest.mark.parametrize("state", ["PENDING", "COMPLETED"])
def test_blank_accounting_comment_uses_exact_canonical_submit_line(state: str) -> None:
    name = "cellcf-transition_beta0-s0-abcdef0"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        runner=_runner(
            squeue=_live(job_name=name) if state == "PENDING" else "",
            sacct=_accounting(
                job_name=name,
                state=state,
                comment="",
                start=("Unknown" if state == "PENDING" else "2026-07-16T10:00:00"),
                end=("Unknown" if state == "PENDING" else "2026-07-16T11:00:00"),
            ),
        ),
    )
    assert result["source"] == "sacct"
    assert result["state"] == state


@pytest.mark.parametrize(
    "submit_line",
    [
        f"sbatch --parsable {shlex.quote(str(JOB_FILE))}",
        (
            "sbatch --parsable --clusters=n16r4 "
            "--job-name=cellcf-transition_beta0-s0-abcdef0 "
            f"--comment=wrong-token {shlex.quote(str(JOB_FILE))}"
        ),
        (
            "sbatch --parsable --clusters=n16r4 "
            "--job-name=cellcf-transition_beta0-s0-abcdef0 "
            "--comment=cellcf-token --comment=cellcf-token "
            f"{shlex.quote(str(JOB_FILE))}"
        ),
        (
            "sbatch --parsable --clusters=n16r4 "
            "--job-name=cellcf-transition_beta0-s0-abcdef0 "
            f"-- {shlex.quote(str(JOB_FILE))} --comment=cellcf-token"
        ),
        (
            "sbatch --parsable --clusters=n16r4 "
            "--job-name=cellcf-transition_beta0-s0-abcdef0 --comment= "
            f"{shlex.quote(str(JOB_FILE))}"
        ),
        (
            "sbatch --parsable --clusters=n16r4 "
            "--job-name=cellcf-transition_beta0-s0-abcdef0 "
            "--comment=' cellcf-token ' "
            f"{shlex.quote(str(JOB_FILE))}"
        ),
    ],
)
def test_accounting_requires_the_exact_canonical_submit_command(
    submit_line: str,
) -> None:
    name = "cellcf-transition_beta0-s0-abcdef0"
    with pytest.raises(ValueError, match="canonical submission"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(
                sacct=_accounting(
                    job_name=name,
                    comment="",
                    submit_line=submit_line,
                )
            ),
        )


def test_nonblank_accounting_comment_conflict_is_rejected_exactly() -> None:
    name = "cellcf-transition_beta0-s0-abcdef0"
    with pytest.raises(ValueError, match="comment"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(
                sacct=_accounting(job_name=name, comment=f" {TOKEN} ")
            ),
        )


@pytest.mark.parametrize("token", ["", " cellcf-token", "cellcf token", "x" * 257])
def test_submission_token_is_nonempty_and_not_normalized(token: str) -> None:
    with pytest.raises(ValueError, match="submission token"):
        _validate(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=token,
            cluster=CLUSTER,
            runner=_runner(),
        )


def test_job_file_must_be_absolute_and_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            job_file="relative.sbatch",
            job_file_sha256=JOB_FILE_SHA256,
            runner=_runner(),
        )
    with pytest.raises(ValueError, match="does not exist"):
        validate_slurm_receipt(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            job_file=tmp_path / "missing.sbatch",
            job_file_sha256=JOB_FILE_SHA256,
            runner=_runner(),
        )


def test_live_comment_and_command_are_exact() -> None:
    name = "cellcf-uniform-s0-abcdef0"
    with pytest.raises(ValueError, match="comment"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(squeue=_live(job_name=name, token=f" {TOKEN} ")),
        )
    with pytest.raises(ValueError, match="bound job file"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(
                squeue=_live(job_name=name, job_file=JOB_FILE.parent)
            ),
        )


@pytest.mark.parametrize(
    "squeue",
    [
        "not-json",
        "WRONG PREFIX\n" + json.dumps({"jobs": [], "errors": []}),
        json.dumps({"jobs": [], "errors": [{"error": "denied"}]}),
        json.dumps({"jobs": [{}, {}], "errors": []}),
    ],
)
def test_malformed_or_ambiguous_live_json_fails_closed(squeue: str) -> None:
    with pytest.raises(ValueError):
        _validate(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(squeue=squeue),
        )


def test_unrelated_squeue_failure_remains_fail_closed() -> None:
    calls: list[str] = []
    with pytest.raises(RuntimeError, match="Slurm query failed"):
        _validate(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(
                squeue_returncode=1,
                squeue_stderr="squeue: error: Access denied\n",
                calls=calls,
            ),
        )
    assert calls == ["squeue"]


@pytest.mark.parametrize("state", ["FAILED", "CANCELLED by 42", "TIMEOUT", "OUT_OF_MEMORY"])
def test_terminal_failure_receipt_is_never_reused(state: str) -> None:
    name = "cellcf-uniform-s0-abcdef0"
    with pytest.raises(ValueError, match="non-reusable Slurm state"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(sacct=_accounting(job_name=name, state=state)),
        )


def test_unknown_receipt_is_not_treated_as_idempotent() -> None:
    with pytest.raises(ValueError, match="absent"):
        _validate(
            job_id=123,
            job_name="cellcf-uniform-s0-abcdef0",
            token=TOKEN,
            cluster=CLUSTER,
            runner=_runner(),
        )


def test_live_pending_receipt_requires_exact_dependency() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        dependency=dependency,
        runner=_runner(
            squeue=_live(job_name=name, dependency=dependency),
        ),
    )
    assert result["source"] == "squeue"
    assert result["dependency"] == dependency


def test_running_receipt_reopens_dependency_from_canonical_submit_line() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        dependency=dependency,
        runner=_runner(
            squeue=_live(job_name=name, state="RUNNING"),
            sacct=_accounting(
                job_name=name,
                state="RUNNING",
                dependency=dependency,
                start="2026-07-16T12:00:00",
            ),
            predecessors=(
                "1|COMPLETED|0:0|2026-07-16T11:00:00|2026-07-16T10:00:00|n16r4\n"
                "2|COMPLETED|0:0|2026-07-16T11:01:00|2026-07-16T10:00:00|n16r4\n"
                "3|COMPLETED|0:0|2026-07-16T11:02:00|2026-07-16T10:00:00|n16r4\n"
            ),
        ),
    )
    assert result["source"] == "sacct"
    assert result["dependency"] == dependency


def test_live_remaining_dependency_may_be_a_subset_of_original_afterok() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    result = _validate(
        job_id=123,
        job_name=name,
        token=TOKEN,
        cluster=CLUSTER,
        dependency=dependency,
        runner=_runner(
            squeue=_live(job_name=name, dependency="afterok:2:3"),
            sacct=_accounting(job_name=name, dependency=dependency),
            predecessors=(
                "1|COMPLETED|0:0|2026-07-16T11:00:00|"
                "2026-07-16T10:00:00|n16r4\n"
            ),
        ),
    )
    assert result["source"] == "sacct"


def test_removed_afterok_dependency_must_be_successfully_completed() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    with pytest.raises(ValueError, match="not successfully completed"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(job_name=name, dependency="afterok:2:3"),
                sacct=_accounting(job_name=name, dependency=dependency),
                predecessors="1|RUNNING|0:0|Unknown|2026-07-16T10:00:00|n16r4\n",
            ),
        )


def test_started_target_must_not_precede_satisfied_afterok_dependencies() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    with pytest.raises(ValueError, match="target started before"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(job_name=name, state="RUNNING"),
                sacct=_accounting(
                    job_name=name,
                    state="RUNNING",
                    dependency=dependency,
                    start="2026-07-16T12:00:00",
                ),
                predecessors=(
                    "1|COMPLETED|0:0|2026-07-16T11:00:00|2026-07-16T10:00:00|n16r4\n"
                    "2|COMPLETED|0:0|2026-07-16T12:30:00|2026-07-16T10:00:00|n16r4\n"
                    "3|COMPLETED|0:0|2026-07-16T11:30:00|2026-07-16T10:00:00|n16r4\n"
                ),
            ),
        )


def test_started_target_cannot_retain_unmet_dependencies() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    with pytest.raises(ValueError, match="started while dependencies remain"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(
                    job_name=name,
                    state="RUNNING",
                    dependency=dependency,
                )
            ),
        )


def test_live_started_state_overrides_stale_pending_accounting_state() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    with pytest.raises(ValueError, match="target start timestamp is unavailable"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(job_name=name, state="RUNNING"),
                sacct=_accounting(job_name=name, dependency=dependency),
                predecessors=(
                    "1|COMPLETED|0:0|2026-07-16T11:00:00|2026-07-16T10:00:00|n16r4\n"
                    "2|COMPLETED|0:0|2026-07-16T11:01:00|2026-07-16T10:00:00|n16r4\n"
                    "3|COMPLETED|0:0|2026-07-16T11:02:00|2026-07-16T10:00:00|n16r4\n"
                ),
            ),
        )


def test_accounting_started_state_cannot_retain_live_dependencies() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1:2:3"
    with pytest.raises(ValueError, match="started while afterok dependencies remain"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(job_name=name, dependency=dependency),
                sacct=_accounting(
                    job_name=name,
                    state="RUNNING",
                    dependency=dependency,
                    start="2026-07-16T12:00:00",
                ),
            ),
        )


def test_date_only_accounting_time_is_rejected() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    dependency = "afterok:1"
    with pytest.raises(ValueError, match="invalid Slurm target start timestamp"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=dependency,
            runner=_runner(
                squeue=_live(job_name=name, state="RUNNING"),
                sacct=_accounting(
                    job_name=name,
                    state="RUNNING",
                    dependency=dependency,
                    start="2026-07-16",
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
def test_live_dependency_conflict_is_rejected(expected: str, live: str) -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    with pytest.raises(ValueError, match="live dependency mismatch"):
        _validate(
            job_id=123,
            job_name=name,
            token=TOKEN,
            cluster=CLUSTER,
            dependency=expected,
            runner=_runner(
                squeue=_live(job_name=name, dependency=live),
                sacct=_accounting(job_name=name, dependency=expected),
            ),
        )


def test_accounting_rejects_wrong_dependency_or_job_file() -> None:
    name = "cellcf-aggregate-s0-abcdef0"
    expected = "afterok:1:2:3"
    wrong_dependency = _submit_line(job_name=name, dependency="afterok:1:2:4")
    wrong_file = _submit_line(job_name=name, dependency=expected, job_file=JOB_FILE.parent)
    for line in (wrong_dependency, wrong_file):
        with pytest.raises(ValueError, match="canonical submission"):
            _validate(
                job_id=123,
                job_name=name,
                token=TOKEN,
                cluster=CLUSTER,
                dependency=expected,
                runner=_runner(
                    sacct=_accounting(
                        job_name=name,
                        dependency=expected,
                        submit_line=line,
                    )
                ),
            )
