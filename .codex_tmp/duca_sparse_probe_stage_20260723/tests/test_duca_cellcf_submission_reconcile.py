from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.reconcile_duca_cellcf_slurm_submission import (
    fsync_artifacts,
    recover_unique_held_job_id,
    submit_held_job,
    submit_held_job_pair,
    verify_cancelled_job_ids,
)


def _runner(stdout: str):
    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


class _FakeSlurm:
    def __init__(
        self,
        *,
        responses: list[str],
        holds: list[bool] | None = None,
        hidden_queries: int = 0,
    ) -> None:
        self.responses = list(responses)
        self.holds = list(holds or [True] * len(responses))
        self.jobs: dict[int, dict] = {}
        self.next_id = 101
        self.hidden_queries = hidden_queries

    @staticmethod
    def _argument(command: list[str], prefix: str) -> str:
        return next(item[len(prefix) :] for item in command if item.startswith(prefix))

    def __call__(self, command, **_kwargs):
        command = list(command)
        if command[0] == "sbatch":
            response = self.responses.pop(0)
            if response == "reject":
                return SimpleNamespace(
                    returncode=1,
                    stdout="",
                    stderr="policy rejected the job",
                )
            job_id = self.next_id
            self.next_id += 1
            cluster = self._argument(command, "--clusters=")
            held = self.holds.pop(0)
            self.jobs[job_id] = {
                "job_id": job_id,
                "name": self._argument(command, "--job-name="),
                "comment": self._argument(command, "--comment="),
                "cluster": cluster,
                "command": str(Path(command[-1]).resolve()),
                "job_state": ["PENDING"],
                "hold": held,
                "state_reason": "JobHeldUser" if held else "Priority",
            }
            stdout = (
                f"{job_id};{cluster}\n"
                if response == "canonical"
                else f"{response}\n"
            )
            return SimpleNamespace(returncode=0, stdout=stdout, stderr="")
        if command[0] == "squeue":
            cluster = command[command.index("--clusters") + 1]
            if self.hidden_queries:
                self.hidden_queries -= 1
                jobs = []
            else:
                jobs = [
                    job
                    for job in self.jobs.values()
                    if job["cluster"] == cluster
                    and job["job_state"] != ["CANCELLED"]
                ]
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    f"CLUSTER: {cluster}\n"
                    + json.dumps({"jobs": jobs, "errors": []})
                ),
                stderr="",
            )
        if command[0] == "scancel":
            for raw_id in command[3:]:
                self.jobs[int(raw_id)]["job_state"] = ["CANCELLED"]
                self.jobs[int(raw_id)]["hold"] = False
                self.jobs[int(raw_id)]["state_reason"] = "None"
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if command[0] == "sacct":
            raw_ids = command[command.index("-j") + 1]
            rows = []
            for raw_id in raw_ids.split(","):
                job = self.jobs[int(raw_id)]
                state = (
                    "CANCELLED"
                    if job["job_state"] == ["CANCELLED"]
                    else job["job_state"][0]
                )
                rows.append(f"{raw_id}|{job['cluster']}|{state}")
            return SimpleNamespace(
                returncode=0,
                stdout="\n".join(rows) + "\n",
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")


def test_recover_unique_job_requires_current_user_hold(tmp_path: Path) -> None:
    job_file = tmp_path / "job.sbatch"
    job_file.write_text("#!/bin/bash\n", encoding="utf-8")
    base = {
        "job_id": 123,
        "name": "postrun-uniform",
        "comment": "token-123",
        "cluster": "n16r4",
        "command": str(job_file.resolve()),
        "job_state": ["PENDING"],
        "hold": True,
        "state_reason": "JobHeldUser",
    }
    output = "CLUSTER: n16r4\n" + json.dumps(
        {"jobs": [base], "errors": []}
    )

    assert recover_unique_held_job_id(
        token="token-123",
        job_name="postrun-uniform",
        cluster="n16r4",
        job_file=job_file,
        user="sczc063",
        runner=_runner(output),
    ) == 123

    running = dict(base, job_state=["RUNNING"], hold=False, state_reason="None")
    with pytest.raises(ValueError, match="currently held"):
        recover_unique_held_job_id(
            token="token-123",
            job_name="postrun-uniform",
            cluster="n16r4",
            job_file=job_file,
            user="sczc063",
            runner=_runner(
                "CLUSTER: n16r4\n"
                + json.dumps({"jobs": [running], "errors": []})
            ),
        )


def test_cancelled_verification_rejects_completed_or_missing_jobs() -> None:
    verify_cancelled_job_ids(
        job_ids=[11, 12],
        cluster="n16r4",
        runner=_runner(
            "11|n16r4|CANCELLED by 7\n12|n16r4|CANCELLED\n"
        ),
    )

    with pytest.raises(ValueError, match="not CANCELLED"):
        verify_cancelled_job_ids(
            job_ids=[11, 12],
            cluster="n16r4",
            runner=_runner("11|n16r4|CANCELLED\n12|n16r4|COMPLETED\n"),
        )
    with pytest.raises(ValueError, match="incomplete"):
        verify_cancelled_job_ids(
            job_ids=[11, 12],
            cluster="n16r4",
            runner=_runner("11|n16r4|CANCELLED\n"),
        )


def test_malformed_sbatch_binding_is_cancelled(tmp_path: Path) -> None:
    job_file = tmp_path / "cost.sbatch"
    job_file.write_text("#!/bin/bash\n", encoding="utf-8")
    slurm = _FakeSlurm(responses=["created-but-malformed"])

    with pytest.raises(ValueError, match="non-canonical"):
        submit_held_job(
            token="token-cost",
            job_name="cost-job",
            cluster="n16r4",
            job_file=job_file,
            user="sczc063",
            recovery_attempts=1,
            runner=slurm,
            sleeper=lambda _delay: None,
        )

    assert slurm.jobs[101]["job_state"] == ["CANCELLED"]


def test_non_user_hold_is_rejected_and_cancelled(tmp_path: Path) -> None:
    job_file = tmp_path / "cost.sbatch"
    job_file.write_text("#!/bin/bash\n", encoding="utf-8")
    slurm = _FakeSlurm(responses=["canonical"], holds=[False])

    with pytest.raises(ValueError, match="currently held"):
        submit_held_job(
            token="token-cost",
            job_name="cost-job",
            cluster="n16r4",
            job_file=job_file,
            user="sczc063",
            recovery_attempts=1,
            runner=slurm,
            sleeper=lambda _delay: None,
        )

    assert slurm.jobs[101]["job_state"] == ["CANCELLED"]


def test_rejected_sbatch_surfaces_scheduler_stderr(
    tmp_path: Path,
) -> None:
    job_file = tmp_path / "cost.sbatch"
    job_file.write_text("#!/bin/bash\n", encoding="utf-8")
    slurm = _FakeSlurm(responses=["reject"])

    with pytest.raises(
        RuntimeError,
        match="policy rejected the job",
    ):
        submit_held_job(
            token="token-cost",
            job_name="cost-job",
            cluster="n16r4",
            job_file=job_file,
            user="sczc063",
            recovery_attempts=1,
            runner=slurm,
            sleeper=lambda _delay: None,
        )

    assert slurm.jobs == {}


def test_successful_sbatch_waits_through_delayed_visibility(
    tmp_path: Path,
) -> None:
    job_file = tmp_path / "cost.sbatch"
    job_file.write_text("#!/bin/bash\n", encoding="utf-8")
    slurm = _FakeSlurm(
        responses=["canonical"],
        hidden_queries=3,
    )

    submission = submit_held_job(
        token="token-cost",
        job_name="cost-job",
        cluster="n16r4",
        job_file=job_file,
        user="sczc063",
        recovery_attempts=1,
        recovery_delay_seconds=0,
        runner=slurm,
        sleeper=lambda _delay: None,
    )

    assert submission.job_id == 101
    assert slurm.hidden_queries == 0
    assert slurm.jobs[101]["job_state"] == ["PENDING"]


def test_second_submission_failure_rolls_back_both_jobs(
    tmp_path: Path,
) -> None:
    cost_file = tmp_path / "cost.sbatch"
    completion_file = tmp_path / "completion.sbatch"
    cost_file.write_text("#!/bin/bash\n", encoding="utf-8")
    completion_file.write_text("#!/bin/bash\n", encoding="utf-8")
    slurm = _FakeSlurm(
        responses=["canonical", "malformed-completion-binding"]
    )

    with pytest.raises(ValueError, match="non-canonical"):
        submit_held_job_pair(
            cost_token="token-cost",
            cost_job_name="cost-job",
            cost_job_file=cost_file,
            completion_token="token-completion",
            completion_job_name="completion-job",
            completion_job_file=completion_file,
            cluster="n16r4",
            user="sczc063",
            recovery_attempts=1,
            runner=slurm,
            sleeper=lambda _delay: None,
        )

    assert {
        job_id: job["job_state"] for job_id, job in slurm.jobs.items()
    } == {101: ["CANCELLED"], 102: ["CANCELLED"]}


def test_fsync_failure_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(
        "tools.bata.reconcile_duca_cellcf_slurm_submission.os.fsync",
        fail_fsync,
    )
    with pytest.raises(OSError, match="injected fsync failure"):
        fsync_artifacts(files=[artifact], directories=[tmp_path])


@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX-only")
def test_fsync_artifacts_persists_files_and_directories(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "receipts" / "artifact.json"
    artifact.parent.mkdir()
    artifact.write_text("{}\n", encoding="utf-8")

    fsync_artifacts(
        files=[artifact],
        directories=[tmp_path, artifact.parent],
    )
