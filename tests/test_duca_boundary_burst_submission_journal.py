from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.bata.duca_boundary_burst_submission_journal import (
    JournalError,
    ROLE_ORDER,
    initialize,
    inspect,
    record,
    reserve,
    seal,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
CLUSTER = "n16r4"


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    return tmp_path / "jobs.tsv", tmp_path / "jobs.complete.json"


def _dependency(role: str, job_ids: dict[str, str]) -> str:
    if role == "r0_holdout_map":
        return "none"
    if role == "p0":
        return f"afterok:{job_ids['r0_holdout_map']}"
    if role == "gate":
        return f"afterok:{job_ids['p0']}"
    if role in ROLE_ORDER[3:7]:
        return f"afterok:{job_ids['gate']}"
    return "afterok:" + ":".join(job_ids[item] for item in ROLE_ORDER[3:7])


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_each_submission_receipt_is_persisted_before_the_next_role(
    tmp_path: Path,
) -> None:
    journal, complete = _paths(tmp_path)
    initialize(journal, complete)
    job_ids: dict[str, str] = {}

    for index, role in enumerate(ROLE_ORDER, start=101):
        dependency = _dependency(role, job_ids)
        reserve(
            journal,
            complete,
            role=role,
            dependency=dependency,
            target_cluster=CLUSTER,
        )
        assert _rows(journal)[-1]["job_id"] == "PENDING"
        record(
            journal,
            complete,
            role=role,
            job_id=str(index),
            dependency=dependency,
            target_cluster=CLUSTER,
        )
        job_ids[role] = str(index)
        persisted = _rows(journal)
        assert persisted[-1] == {
            "role": role,
            "job_id": str(index),
            "dependency": dependency,
            "cluster": CLUSTER,
        }
        assert len(persisted) == len(job_ids)

    seal(
        journal,
        complete,
        expected_commit=COMMIT,
        target_cluster=CLUSTER,
    )
    assert (
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )
        == "COMPLETE"
    )
    payload = json.loads(complete.read_text(encoding="utf-8"))
    assert payload["job_count"] == len(ROLE_ORDER)


@pytest.mark.parametrize(
    "state", ("header_only", "pending", "recorded", "unsealed_full")
)
def test_any_partial_journal_fails_closed_and_cannot_be_reinitialized(
    tmp_path: Path,
    state: str,
) -> None:
    journal, complete = _paths(tmp_path)
    initialize(journal, complete)
    if state != "header_only":
        reserve(
            journal,
            complete,
            role="r0_holdout_map",
            dependency="none",
            target_cluster=CLUSTER,
        )
    if state in {"recorded", "unsealed_full"}:
        record(
            journal,
            complete,
            role="r0_holdout_map",
            job_id="101",
            dependency="none",
            target_cluster=CLUSTER,
        )
    if state == "unsealed_full":
        job_ids = {"r0_holdout_map": "101"}
        for index, role in enumerate(ROLE_ORDER[1:], start=102):
            dependency = _dependency(role, job_ids)
            reserve(
                journal,
                complete,
                role=role,
                dependency=dependency,
                target_cluster=CLUSTER,
            )
            record(
                journal,
                complete,
                role=role,
                job_id=str(index),
                dependency=dependency,
                target_cluster=CLUSTER,
            )
            job_ids[role] = str(index)

    with pytest.raises(JournalError, match="partial"):
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )
    with pytest.raises(JournalError, match="already exists"):
        initialize(journal, complete)


def test_complete_journal_rejects_hash_commit_and_dependency_drift(
    tmp_path: Path,
) -> None:
    journal, complete = _paths(tmp_path)
    initialize(journal, complete)
    job_ids: dict[str, str] = {}
    for index, role in enumerate(ROLE_ORDER, start=201):
        dependency = _dependency(role, job_ids)
        reserve(
            journal,
            complete,
            role=role,
            dependency=dependency,
            target_cluster=CLUSTER,
        )
        record(
            journal,
            complete,
            role=role,
            job_id=str(index),
            dependency=dependency,
            target_cluster=CLUSTER,
        )
        job_ids[role] = str(index)
    seal(
        journal,
        complete,
        expected_commit=COMMIT,
        target_cluster=CLUSTER,
    )

    with pytest.raises(JournalError, match="completion seal"):
        inspect(
            journal,
            complete,
            expected_commit="b" * 40,
            target_cluster=CLUSTER,
        )

    rows = _rows(journal)
    rows[-1]["job_id"] = "999"
    with journal.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "job_id", "dependency", "cluster"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(JournalError, match="completion seal"):
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )

    rows[-1]["job_id"] = job_ids["aggregate"]
    with journal.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("role", "job_id", "dependency", "cluster"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    text = journal.read_text(encoding="utf-8")
    journal.write_text(text.replace("afterok:201", "afterok:999", 1), encoding="utf-8")
    with pytest.raises(JournalError, match="dependency drift"):
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )


def test_launcher_orders_reservation_sbatch_receipt_and_completion_seal() -> None:
    source = (
        ROOT / "scripts" / "submit_duca_boundary_burst_official60_suite.sh"
    ).read_text(encoding="utf-8")

    reserve_index = source.index(
        'journal reserve --role "${role}" --dependency "${dependency}"'
    )
    submit_index = source.index('raw="$(sbatch "${sbatch_args[@]}" "${sbatch_file}")"')
    record_index = source.index('journal record --role "${role}" --job-id "${job_id}"')
    assert reserve_index < submit_index < record_index
    assert "journal initialize" in source
    assert "journal seal" in source
    assert "journal inspect >/dev/null" in source
    assert '> "${RUN_ROOT}/jobs.tsv"' not in source


def test_cli_argument_order_used_by_launcher_is_executable(tmp_path: Path) -> None:
    journal, complete = _paths(tmp_path)
    prefix = [
        sys.executable,
        "-m",
        "tools.bata.duca_boundary_burst_submission_journal",
        "--journal",
        str(journal),
        "--seal",
        str(complete),
        "--expected-commit",
        COMMIT,
        "--target-cluster",
        CLUSTER,
    ]

    absent = subprocess.run(
        [*prefix, "inspect"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert absent.stdout.strip() == "ABSENT"
    subprocess.run([*prefix, "initialize"], cwd=ROOT, check=True)
    subprocess.run(
        [
            *prefix,
            "reserve",
            "--role",
            "r0_holdout_map",
            "--dependency",
            "none",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            *prefix,
            "record",
            "--role",
            "r0_holdout_map",
            "--job-id",
            "101",
            "--dependency",
            "none",
        ],
        cwd=ROOT,
        check=True,
    )
    partial = subprocess.run(
        [*prefix, "inspect"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert partial.returncode != 0
    assert "partial submission journal" in partial.stderr

    job_ids = {"r0_holdout_map": "101"}
    for index, role in enumerate(ROLE_ORDER[1:], start=102):
        dependency = _dependency(role, job_ids)
        subprocess.run(
            [
                *prefix,
                "reserve",
                "--role",
                role,
                "--dependency",
                dependency,
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                *prefix,
                "record",
                "--role",
                role,
                "--job-id",
                str(index),
                "--dependency",
                dependency,
            ],
            cwd=ROOT,
            check=True,
        )
        job_ids[role] = str(index)
    subprocess.run([*prefix, "seal"], cwd=ROOT, check=True)
    complete_result = subprocess.run(
        [*prefix, "inspect"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert complete_result.stdout.strip() == "COMPLETE"
