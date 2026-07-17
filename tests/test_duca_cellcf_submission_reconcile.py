from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.bata.reconcile_duca_cellcf_slurm_submission import (
    recover_unique_held_job_id,
    verify_cancelled_job_ids,
)


def _runner(stdout: str):
    def run(_command, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=stdout, stderr="")

    return run


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
