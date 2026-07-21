from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Iterable


SCHEMA = "duca_boundary_burst_submission_journal_v1"
HEADER = ("role", "job_id", "dependency", "cluster")
PENDING_JOB_ID = "PENDING"
ROLE_ORDER = (
    "r0_holdout_map",
    "p0",
    "gate",
    "two_stage_exact_uniform",
    "gaussian_matched_g0",
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
    "aggregate",
)


class JournalError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _exclusive_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _fsync_directory(path.parent)


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _serialize_rows(rows: Iterable[dict[str, str]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer, fieldnames=HEADER, delimiter="\t", lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def _expected_dependency(role: str, completed: dict[str, str]) -> str:
    if role == "r0_holdout_map":
        return "none"
    if role == "p0":
        return f"afterok:{completed['r0_holdout_map']}"
    if role == "gate":
        return f"afterok:{completed['p0']}"
    if role in ROLE_ORDER[3:7]:
        return f"afterok:{completed['gate']}"
    if role == "aggregate":
        arm_ids = [completed[item] for item in ROLE_ORDER[3:7]]
        return f"afterok:{':'.join(arm_ids)}"
    raise JournalError(f"unknown submission role: {role}")


def _read_rows(path: Path, *, target_cluster: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise JournalError(f"submission journal is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise JournalError("submission journal header is invalid")
        rows = list(reader)
    if len(rows) > len(ROLE_ORDER):
        raise JournalError("submission journal contains too many rows")

    completed: dict[str, str] = {}
    observed_ids: set[str] = set()
    for index, row in enumerate(rows):
        if None in row or any(row.get(field, "") == "" for field in HEADER):
            raise JournalError("submission journal contains a malformed row")
        expected_role = ROLE_ORDER[index]
        if row["role"] != expected_role:
            raise JournalError(
                f"submission journal role order drift: expected {expected_role}, "
                f"got {row['role']}"
            )
        if row["cluster"] != target_cluster:
            raise JournalError(f"submission journal cluster drift for {expected_role}")
        expected_dependency = _expected_dependency(expected_role, completed)
        if row["dependency"] != expected_dependency:
            raise JournalError(
                f"submission journal dependency drift for {expected_role}"
            )
        job_id = row["job_id"]
        if job_id == PENDING_JOB_ID:
            if index != len(rows) - 1:
                raise JournalError("only the final journal row may be pending")
            continue
        if not job_id.isdecimal() or int(job_id) <= 0:
            raise JournalError(f"invalid job id for {expected_role}: {job_id}")
        if job_id in observed_ids:
            raise JournalError(f"duplicate job id in submission journal: {job_id}")
        observed_ids.add(job_id)
        completed[expected_role] = job_id
    return rows


def initialize(journal_path: Path, seal_path: Path) -> None:
    if journal_path.exists() or seal_path.exists():
        raise JournalError("submission journal already exists; refusing to resubmit")
    _exclusive_write(journal_path, _serialize_rows([]))


def reserve(
    journal_path: Path,
    seal_path: Path,
    *,
    role: str,
    dependency: str,
    target_cluster: str,
) -> None:
    if seal_path.exists():
        raise JournalError("completed submission journal is immutable")
    rows = _read_rows(journal_path, target_cluster=target_cluster)
    if rows and rows[-1]["job_id"] == PENDING_JOB_ID:
        raise JournalError("prior submission intent is unresolved")
    if len(rows) >= len(ROLE_ORDER) or role != ROLE_ORDER[len(rows)]:
        raise JournalError(f"unexpected next submission role: {role}")
    completed = {row["role"]: row["job_id"] for row in rows}
    if dependency != _expected_dependency(role, completed):
        raise JournalError(f"dependency does not match the journal prefix for {role}")
    rows.append(
        {
            "role": role,
            "job_id": PENDING_JOB_ID,
            "dependency": dependency,
            "cluster": target_cluster,
        }
    )
    _atomic_write(journal_path, _serialize_rows(rows))


def record(
    journal_path: Path,
    seal_path: Path,
    *,
    role: str,
    job_id: str,
    dependency: str,
    target_cluster: str,
) -> None:
    if seal_path.exists():
        raise JournalError("completed submission journal is immutable")
    rows = _read_rows(journal_path, target_cluster=target_cluster)
    if not rows or rows[-1]["job_id"] != PENDING_JOB_ID:
        raise JournalError("submission receipt has no matching pending intent")
    pending = rows[-1]
    if pending["role"] != role or pending["dependency"] != dependency:
        raise JournalError("submission receipt does not match its pending intent")
    if not job_id.isdecimal() or int(job_id) <= 0:
        raise JournalError(f"invalid Slurm job id: {job_id}")
    if job_id in {row["job_id"] for row in rows[:-1]}:
        raise JournalError(f"duplicate Slurm job id: {job_id}")
    pending["job_id"] = job_id
    _atomic_write(journal_path, _serialize_rows(rows))


def _validate_complete(
    journal_path: Path,
    seal_path: Path,
    *,
    expected_commit: str,
    target_cluster: str,
) -> list[dict[str, str]]:
    rows = _read_rows(journal_path, target_cluster=target_cluster)
    if len(rows) != len(ROLE_ORDER) or any(
        row["job_id"] == PENDING_JOB_ID for row in rows
    ):
        raise JournalError("submission journal is partial")
    if not seal_path.is_file():
        raise JournalError("submission journal has no completion seal")
    try:
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JournalError(
            "submission journal completion seal is unreadable"
        ) from error
    expected = {
        "schema": SCHEMA,
        "complete": True,
        "git_commit": expected_commit,
        "target_cluster": target_cluster,
        "journal_path": str(journal_path.resolve()),
        "journal_sha256": _sha256(journal_path),
        "roles": list(ROLE_ORDER),
        "job_count": len(ROLE_ORDER),
    }
    if seal != expected:
        raise JournalError("submission journal completion seal does not match jobs.tsv")
    return rows


def seal(
    journal_path: Path,
    seal_path: Path,
    *,
    expected_commit: str,
    target_cluster: str,
) -> None:
    if seal_path.exists():
        raise JournalError("submission journal is already sealed")
    rows = _read_rows(journal_path, target_cluster=target_cluster)
    if len(rows) != len(ROLE_ORDER) or any(
        row["job_id"] == PENDING_JOB_ID for row in rows
    ):
        raise JournalError("cannot seal a partial submission journal")
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "git_commit": expected_commit,
        "target_cluster": target_cluster,
        "journal_path": str(journal_path.resolve()),
        "journal_sha256": _sha256(journal_path),
        "roles": list(ROLE_ORDER),
        "job_count": len(ROLE_ORDER),
    }
    _exclusive_write(
        seal_path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    _validate_complete(
        journal_path,
        seal_path,
        expected_commit=expected_commit,
        target_cluster=target_cluster,
    )


def inspect(
    journal_path: Path,
    seal_path: Path,
    *,
    expected_commit: str,
    target_cluster: str,
) -> str:
    if not journal_path.exists() and not seal_path.exists():
        return "ABSENT"
    if not journal_path.exists() or not seal_path.exists():
        raise JournalError("partial submission journal exists; refusing to resubmit")
    _validate_complete(
        journal_path,
        seal_path,
        expected_commit=expected_commit,
        target_cluster=target_cluster,
    )
    return "COMPLETE"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--seal", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--target-cluster", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("inspect")
    subparsers.add_parser("initialize")
    reserve_parser = subparsers.add_parser("reserve")
    reserve_parser.add_argument("--role", required=True)
    reserve_parser.add_argument("--dependency", required=True)
    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--role", required=True)
    record_parser.add_argument("--job-id", required=True)
    record_parser.add_argument("--dependency", required=True)
    subparsers.add_parser("seal")
    return parser


def main() -> int:
    args = _parser().parse_args()
    common = {
        "expected_commit": args.expected_commit,
        "target_cluster": args.target_cluster,
    }
    if args.command == "inspect":
        print(inspect(args.journal, args.seal, **common))
    elif args.command == "initialize":
        initialize(args.journal, args.seal)
    elif args.command == "reserve":
        reserve(
            args.journal,
            args.seal,
            role=args.role,
            dependency=args.dependency,
            target_cluster=args.target_cluster,
        )
    elif args.command == "record":
        record(
            args.journal,
            args.seal,
            role=args.role,
            job_id=args.job_id,
            dependency=args.dependency,
            target_cluster=args.target_cluster,
        )
    elif args.command == "seal":
        seal(args.journal, args.seal, **common)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except JournalError as error:
        raise SystemExit(f"[DUCA_BURST_JOURNAL][FAIL] {error}") from error
