from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from tools.bata.duca_boundary_burst_submission_journal import (
    AGGREGATE_ROLE,
    GENERATED_SBATCH_FILENAMES,
    JournalError,
    ROLE_ORDER,
    SELECTED_G0_ROLE,
    UNIFORM_ROLE,
    initialize,
    inspect,
    record,
    reserve,
    seal,
)


ROOT = Path(__file__).resolve().parents[1]
COMMIT = "a" * 40
CLUSTER = "n16r4"
ROLES = ROLE_ORDER


def _paths(tmp_path: Path) -> tuple[Path, Path]:
    submission = tmp_path / "submission"
    submission.mkdir()
    artifacts = {}
    for role, filename in GENERATED_SBATCH_FILENAMES.items():
        artifact = submission / filename
        artifact.write_text(f"#!/usr/bin/env bash\n# {role}\n", encoding="utf-8")
        artifacts[role] = {
            "path": str(artifact.resolve()),
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    manifest = tmp_path / "submission_manifest.json"
    payload = {
        "schema": "duca_boundary_burst_submission_v2",
        "ok": True,
        "fail_closed": True,
        "run_root": str(tmp_path.resolve()),
        "generated_sbatch_artifacts": artifacts,
    }
    payload["manifest_sha256"] = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (tmp_path / "submission_manifest.sha256").write_text(
        hashlib.sha256(manifest.read_bytes()).hexdigest() + "\n",
        encoding="utf-8",
    )
    return tmp_path / "jobs.tsv", tmp_path / "jobs.complete.json"


def _dependency(role: str, job_ids: dict[str, str]) -> str:
    if role == "r0_holdout_map":
        return "none"
    if role == "p0":
        return f"afterok:{job_ids['r0_holdout_map']}"
    if role == "gate":
        return f"afterok:{job_ids['p0']}"
    if role in (UNIFORM_ROLE, SELECTED_G0_ROLE):
        return f"afterok:{job_ids['gate']}"
    if role == AGGREGATE_ROLE:
        return f"afterok:{job_ids[UNIFORM_ROLE]}:{job_ids[SELECTED_G0_ROLE]}"
    raise AssertionError(role)


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_each_submission_receipt_is_persisted_before_the_next_role(
    tmp_path: Path,
) -> None:
    journal, complete = _paths(tmp_path)
    initialize(journal, complete)
    job_ids: dict[str, str] = {}
    roles = ROLE_ORDER

    for index, role in enumerate(roles, start=101):
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
    assert payload["job_count"] == len(roles)
    assert payload["main_official60_roles"] == [UNIFORM_ROLE, SELECTED_G0_ROLE]
    assert payload["selected_g0_runtime_routing"] == "sealed_frontend_decision"
    assert payload["run_root"] == str(tmp_path.resolve())
    assert payload["submission_manifest_path"] == str(
        (tmp_path / "submission_manifest.json").resolve()
    )


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
        for index, role in enumerate(ROLES[1:], start=102):
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
    for index, role in enumerate(ROLES, start=201):
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

    manifest = tmp_path / "submission_manifest.json"
    original_manifest = manifest.read_bytes()
    manifest.write_bytes(original_manifest + b" ")
    with pytest.raises(JournalError, match="submission manifest path/hash drift"):
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )
    manifest.write_bytes(original_manifest)
    uniform_sbatch = tmp_path / "submission" / GENERATED_SBATCH_FILENAMES[UNIFORM_ROLE]
    original_sbatch = uniform_sbatch.read_bytes()
    uniform_sbatch.write_bytes(original_sbatch + b"# drift\n")
    with pytest.raises(JournalError, match="submission artifact path/hash drift"):
        inspect(
            journal,
            complete,
            expected_commit=COMMIT,
            target_cluster=CLUSTER,
        )
    uniform_sbatch.write_bytes(original_sbatch)

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


def test_diagnostic_roles_cannot_enter_or_block_the_main_journal(
    tmp_path: Path,
) -> None:
    journal, complete = _paths(tmp_path)
    initialize(journal, complete)
    job_ids: dict[str, str] = {}
    for index, role in enumerate(ROLES[:4], start=301):
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

    for forbidden in (
        "gaussian_matched_g0",
        "boundary_burst_r2q3_g0",
        "boundary_burst_r4q5_g0",
    ):
        with pytest.raises(JournalError, match="unexpected next submission role"):
            reserve(
                journal,
                complete,
                role=forbidden,
                dependency=f"afterok:{job_ids['gate']}",
                target_cluster=CLUSTER,
            )

    selected_dependency = f"afterok:{job_ids['gate']}"
    reserve(
        journal,
        complete,
        role=SELECTED_G0_ROLE,
        dependency=selected_dependency,
        target_cluster=CLUSTER,
    )
    record(
        journal,
        complete,
        role=SELECTED_G0_ROLE,
        job_id="305",
        dependency=selected_dependency,
        target_cluster=CLUSTER,
    )
    job_ids[SELECTED_G0_ROLE] = "305"
    aggregate_dependency = _dependency(AGGREGATE_ROLE, job_ids)
    assert aggregate_dependency == "afterok:304:305"
    assert "gaussian" not in aggregate_dependency
    assert not {
        "boundary_burst_r2q3_g0",
        "boundary_burst_r4q5_g0",
    } & {row["role"] for row in _rows(journal)}


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
    assert 'submit_and_record "two_stage_exact_uniform" "${main_dependency}"' in source
    assert 'submit_and_record "r0_selected_boundary_burst_g0" "${main_dependency}"' in source
    assert 'aggregate_dependency="afterok:${uniform}:${selected_g0}"' in source
    assert "gaussian_matched_g0.sbatch" not in source
    assert "boundary_burst_r2q3_g0.sbatch" not in source
    assert "boundary_burst_r4q5_g0.sbatch" not in source
    assert 'for variant in "${variants[@]}"' not in source


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
    for index, role in enumerate(ROLES[1:], start=102):
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
