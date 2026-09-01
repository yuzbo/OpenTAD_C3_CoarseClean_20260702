from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.bata.finalize_duca_cellcf_postrun_evidence import (
    SnapshotRecords,
    _validate_cost_recovery_submission,
    canonical_sha256,
    sha256_file,
)


TRAINED_COMMIT = "1642f265e48391418a7c8a4a087e33e2b7bf6899"
EVIDENCE_COMMIT = "e" * 40


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _hashed(payload: dict) -> dict:
    result = dict(payload)
    result["artifact_sha256"] = canonical_sha256(result)
    return result


def _write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _case(
    tmp_path: Path,
    *,
    frozen_completion_state: str = "CANCELLED by 1258",
    live_completion_state: str | None = None,
    live_completion_elapsed: int = 0,
    reuse_original_cost_id: bool = False,
    reuse_original_completion_id: bool = False,
) -> dict:
    run_root = tmp_path / "formal"
    trained_root = tmp_path / "trained"
    evidence_root = tmp_path / "evidence"
    recovery_root = run_root / "cost_recovery_evidence_v1"
    for path in (run_root, trained_root, evidence_root, recovery_root):
        path.mkdir(parents=True, exist_ok=True)
    aggregate = run_root / "aggregate_suite_evidence.json"
    final_suite = run_root / "final_suite_evidence.json"
    cost_evidence = recovery_root / "cost" / "cellcf_vs_bare_uniform.json"
    _write_json(aggregate, {"ok": True})
    _write_json(final_suite, {"ok": True})
    _write_json(cost_evidence, {"ok": True})

    formal_ids = {
        "uniform": "9001",
        "transition_beta0": "9002",
        "cellcf": "9003",
        "aggregate": "9004",
        "cost": "9005",
        "completion": "9006",
    }
    formal_rows = []
    for key in formal_ids:
        formal_rows.append(
            {
                "job_key": key,
                "seed": "0",
                "commit": TRAINED_COMMIT,
                "training_profile": "exposure132",
                "job_id": formal_ids[key],
                "job_name": f"formal-{key}",
                "cluster": "n16r4",
            }
        )
    formal_ledger = run_root / "jobs.submitted.tsv"
    _write_tsv(formal_ledger, formal_rows)
    formal_ledger_sha = sha256_file(formal_ledger)

    scheduler_query = (
        recovery_root / "receipts" / "original_terminal_jobs.sacct"
    )
    scheduler_query.parent.mkdir(parents=True, exist_ok=True)
    scheduler_query.write_text(
        "9005|formal-cost|n16r4|FAILED|1:0|7853\n"
        f"9006|formal-completion|n16r4|{frozen_completion_state}|0:0|0\n",
        encoding="utf-8",
    )
    failure = _hashed(
        {
            "schema": "duca_cellcf_cost_recovery_original_failure_v1",
            "ok": True,
            "task": "offline_temporal_action_detection",
            "original_formal_ledger_path": str(formal_ledger.resolve()),
            "original_formal_ledger_sha256": formal_ledger_sha,
            "scheduler_query_path": str(scheduler_query.resolve()),
            "scheduler_query_sha256": sha256_file(scheduler_query),
            "cost": {
                "job_id": 9005,
                "job_name": "formal-cost",
                "cluster": "n16r4",
                "state": "FAILED",
                "exit_code": "1:0",
                "elapsed_raw_seconds": 7853,
            },
            "completion": {
                "job_id": 9006,
                "job_name": "formal-completion",
                "cluster": "n16r4",
                "state": frozen_completion_state,
                "exit_code": "0:0",
                "elapsed_raw_seconds": 0,
            },
        }
    )
    failure_path = recovery_root / "original_failure_receipt.json"
    _write_json(failure_path, failure)

    intent = _hashed(
        {
            "schema": "duca_cellcf_cost_recovery_intent_v1",
            "status": "INTENT_RECORDED",
            "task": "offline_temporal_action_detection",
            "formal_run_root": str(run_root.resolve()),
            "recovery_root": str(recovery_root.resolve()),
            "trained_repository": str(trained_root.resolve()),
            "trained_git_commit": TRAINED_COMMIT,
            "cost_producer_repository": str(evidence_root.resolve()),
            "cost_producer_evidence_commit": EVIDENCE_COMMIT,
            "target_cluster": "n16r4",
            "aggregate_evidence_path": str(aggregate.resolve()),
            "aggregate_evidence_sha256": sha256_file(aggregate),
            "original_formal_ledger_path": str(formal_ledger.resolve()),
            "original_formal_ledger_sha256": formal_ledger_sha,
            "original_failure_receipt_path": str(failure_path.resolve()),
            "original_failure_receipt_sha256": sha256_file(failure_path),
            "cost_root": str((recovery_root / "cost").resolve()),
            "cost_evidence_path": str(cost_evidence.resolve()),
            "final_suite_evidence_path": str(final_suite.resolve()),
            "recovery_scope": (
                "rerun cost profiling only; do not rerun the completed arms"
            ),
        }
    )
    intent_path = recovery_root / "submission_intent.json"
    _write_json(intent_path, intent)
    intent_sha = sha256_file(intent_path)

    recovery_ids = {
        "cost": formal_ids["cost"] if reuse_original_cost_id else "9201",
        "completion": (
            formal_ids["completion"]
            if reuse_original_completion_id
            else "9202"
        ),
    }
    recovery_rows = []
    manifest_jobs = []
    for key in ("cost", "completion"):
        job_file = recovery_root / "jobs" / f"{key}.sbatch"
        job_file.parent.mkdir(parents=True, exist_ok=True)
        job_file.write_text(
            f"#!/bin/bash\n#SBATCH --job-name=recovery-{key}\n",
            encoding="utf-8",
        )
        receipt = recovery_root / "receipts" / f"{key}.scheduler.txt"
        receipt.write_text(
            f"JobId={recovery_ids[key]} JobName=recovery-{key}\n",
            encoding="utf-8",
        )
        scheduler_script = (
            recovery_root
            / "receipts"
            / f"{key}.scheduler.sbatch"
        )
        scheduler_script.write_bytes(job_file.read_bytes())
        dependency = (
            "none"
            if key == "cost"
            else f"afterok:{recovery_ids['cost']}"
        )
        raw = f"{recovery_ids[key]};n16r4"
        recovery_rows.append(
            {
                "job_key": key,
                "job_id": recovery_ids[key],
                "job_ref": raw,
                "job_name": f"recovery-{key}",
                "cluster": "n16r4",
                "dependency": dependency,
                "sbatch_file": str(job_file.resolve()),
                "sbatch_sha256": sha256_file(job_file),
                "raw_sbatch_response": raw,
                "scheduler_receipt": str(receipt.resolve()),
                "scheduler_receipt_sha256": sha256_file(receipt),
                "scheduler_script": str(scheduler_script.resolve()),
                "scheduler_script_sha256": sha256_file(
                    scheduler_script
                ),
                "trained_commit": TRAINED_COMMIT,
                "cost_producer_evidence_commit": EVIDENCE_COMMIT,
                "submission_intent_sha256": intent_sha,
                "original_formal_ledger_sha256": formal_ledger_sha,
            }
        )
        manifest_jobs.append(
            {
                "job_key": key,
                "job_id": int(recovery_ids[key]),
                "job_name": f"recovery-{key}",
                "cluster": "n16r4",
                "dependency": dependency,
                "sbatch_file": str(job_file.resolve()),
                "sbatch_sha256": sha256_file(job_file),
                "raw_sbatch_response": raw,
                "scheduler_script": str(scheduler_script.resolve()),
                "scheduler_script_sha256": sha256_file(
                    scheduler_script
                ),
            }
        )
    recovery_ledger = recovery_root / "jobs.submitted.tsv"
    _write_tsv(recovery_ledger, recovery_rows)
    manifest = _hashed(
        {
            "schema": "duca_cellcf_cost_recovery_submission_v1",
            "ok": True,
            "status": "SUBMITTED_HELD_VERIFIED",
            "task": "offline_temporal_action_detection",
            "submission_intent_path": str(intent_path.resolve()),
            "submission_intent_sha256": intent_sha,
            "jobs_ledger_path": str(recovery_ledger.resolve()),
            "jobs_ledger_sha256": sha256_file(recovery_ledger),
            "original_failure_receipt_path": str(failure_path.resolve()),
            "original_failure_receipt_sha256": sha256_file(failure_path),
            "trained_git_commit": TRAINED_COMMIT,
            "cost_producer_evidence_commit": EVIDENCE_COMMIT,
            "aggregate_evidence_path": str(aggregate.resolve()),
            "aggregate_evidence_sha256": sha256_file(aggregate),
            "cost_evidence_path": str(cost_evidence.resolve()),
            "final_suite_evidence_path": str(final_suite.resolve()),
            "target_cluster": "n16r4",
            "jobs": manifest_jobs,
        }
    )
    manifest_path = recovery_root / "submission_manifest.json"
    _write_json(manifest_path, manifest)
    states = {
        (9005, "formal-cost"): ("FAILED", "1:0", 7853),
        (9006, "formal-completion"): (
            live_completion_state or frozen_completion_state,
            "0:0",
            live_completion_elapsed,
        ),
        (
            int(recovery_ids["cost"]),
            "recovery-cost",
        ): ("COMPLETED", "0:0", 8123),
        (
            int(recovery_ids["completion"]),
            "recovery-completion",
        ): ("COMPLETED", "0:0", 42),
    }

    def read_terminal(
        *, job_id: int, job_name: str, cluster: str
    ) -> dict:
        state, exit_code, elapsed = states[(job_id, job_name)]
        return {
            "ok": True,
            "job_id": job_id,
            "job_name": job_name,
            "cluster": cluster,
            "state": state,
            "exit_code": exit_code,
            "elapsed_raw_seconds": elapsed,
        }

    return {
        "kwargs": {
            "manifest_path": manifest_path,
            "manifest_sha256": sha256_file(manifest_path),
            "run_root": run_root.resolve(),
            "trained_root": trained_root.resolve(),
            "trained_commit": TRAINED_COMMIT,
            "evidence_root": evidence_root.resolve(),
            "evidence_commit": EVIDENCE_COMMIT,
            "aggregate_file": aggregate.resolve(),
            "aggregate_sha256": sha256_file(aggregate),
            "final_file": final_suite.resolve(),
            "cost_evidence_path": cost_evidence.resolve(),
            "formal_ledger_path": formal_ledger.resolve(),
            "formal_ledger_sha256": formal_ledger_sha,
            "terminal_state_reader": read_terminal,
        }
    }


def _validate(case: dict) -> dict:
    records = SnapshotRecords()
    try:
        return _validate_cost_recovery_submission(
            **case["kwargs"],
            records=records,
        )
    finally:
        records.close()


def test_cost_recovery_binds_original_failure_and_new_success(
    tmp_path: Path,
) -> None:
    payload = _validate(_case(tmp_path))

    assert payload["status"] == "complete_via_cost_recovery"
    assert payload["original_formal_dag_complete"] is False
    assert (
        payload["original_terminal_jobs"]["cost"]["live"]["state"]
        == "FAILED"
    )
    assert (
        payload["recovery_terminal_jobs"]["completion"]["scheduler"][
            "state"
        ]
        == "COMPLETED"
    )


def test_cost_recovery_rejects_forged_original_completion_success(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError, match="original completion frozen terminal state"
    ):
        _validate(
            _case(
                tmp_path,
                frozen_completion_state="COMPLETED",
            )
        )


@pytest.mark.parametrize(
    ("reuse_cost", "reuse_completion"),
    [(True, False), (False, True)],
)
def test_cost_recovery_jobs_cannot_impersonate_original_jobs(
    tmp_path: Path,
    reuse_cost: bool,
    reuse_completion: bool,
) -> None:
    with pytest.raises(
        ValueError, match="illegally reuses an original terminal job id"
    ):
        _validate(
            _case(
                tmp_path,
                reuse_original_cost_id=reuse_cost,
                reuse_original_completion_id=reuse_completion,
            )
        )


@pytest.mark.parametrize(
    ("state", "elapsed"),
    [
        ("CANCELLED by 999", 0),
        ("CANCELLED by 1258", 1),
    ],
)
def test_cost_recovery_rejects_original_completion_scheduler_drift(
    tmp_path: Path,
    state: str,
    elapsed: int,
) -> None:
    with pytest.raises(
        ValueError,
        match="scheduler state no longer matches recovery evidence",
    ):
        _validate(
            _case(
                tmp_path,
                live_completion_state=state,
                live_completion_elapsed=elapsed,
            )
        )
