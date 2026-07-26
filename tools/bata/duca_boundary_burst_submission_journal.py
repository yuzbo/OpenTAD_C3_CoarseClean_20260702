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


SCHEMA = "duca_boundary_burst_submission_journal_v2"
HEADER = ("role", "job_id", "dependency", "cluster")
PENDING_JOB_ID = "PENDING"
INITIAL_ROLE_ORDER = (
    "r0_holdout_map",
    "p0",
    "gate",
)
UNIFORM_ROLE = "two_stage_exact_uniform"
SELECTED_G0_ROLE = "r0_selected_boundary_burst_g0"
AGGREGATE_ROLE = "aggregate"
ROLE_ORDER = (*INITIAL_ROLE_ORDER, UNIFORM_ROLE, SELECTED_G0_ROLE, AGGREGATE_ROLE)
COMPLETE_ROLE_COUNT = len(ROLE_ORDER)
SUBMISSION_MANIFEST_NAME = "submission_manifest.json"
SUBMISSION_MANIFEST_SEAL_NAME = "submission_manifest.sha256"
SUBMISSION_MANIFEST_SCHEMA = "duca_boundary_burst_submission_v2"
GENERATED_SBATCH_FILENAMES = {
    "r0_holdout_map": "r0.sbatch",
    "p0": "p0.sbatch",
    "gate": "gate.sbatch",
    UNIFORM_ROLE: f"{UNIFORM_ROLE}.sbatch",
    SELECTED_G0_ROLE: f"{SELECTED_G0_ROLE}.sbatch",
    AGGREGATE_ROLE: "aggregate.sbatch",
}


class JournalError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _submission_manifest_binding(journal_path: Path) -> dict[str, str]:
    run_root = journal_path.expanduser().resolve().parent
    manifest = run_root / SUBMISSION_MANIFEST_NAME
    seal = run_root / SUBMISSION_MANIFEST_SEAL_NAME
    if not manifest.is_file() or not seal.is_file():
        raise JournalError("submission manifest binding is missing")
    try:
        expected_sha256 = seal.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise JournalError("submission manifest seal is unreadable") from error
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or _sha256(manifest) != expected_sha256
    ):
        raise JournalError("submission manifest path/hash drift")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise JournalError("submission manifest is unreadable") from error
    if not isinstance(payload, dict):
        raise JournalError("submission manifest contract drift")
    unsigned = dict(payload)
    recorded_self_hash = unsigned.pop("manifest_sha256", None)
    canonical = hashlib.sha256(
        json.dumps(
            unsigned, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    artifacts = payload.get("generated_sbatch_artifacts")
    if (
        payload.get("schema") != SUBMISSION_MANIFEST_SCHEMA
        or payload.get("ok") is not True
        or payload.get("fail_closed") is not True
        or payload.get("run_root") != str(run_root)
        or recorded_self_hash != canonical
        or not isinstance(artifacts, dict)
        or set(artifacts) != set(GENERATED_SBATCH_FILENAMES)
    ):
        raise JournalError("submission manifest contract drift")
    for role, filename in GENERATED_SBATCH_FILENAMES.items():
        record = artifacts.get(role)
        expected_path = (run_root / "submission" / filename).resolve()
        if not isinstance(record, dict):
            raise JournalError(f"submission artifact binding is missing for {role}")
        artifact = Path(str(record.get("path", ""))).expanduser().resolve()
        if (
            artifact != expected_path
            or not artifact.is_file()
            or record.get("sha256") != _sha256(artifact)
        ):
            raise JournalError(f"submission artifact path/hash drift for {role}")
    return {
        "run_root": str(run_root),
        "submission_manifest_path": str(manifest),
        "submission_manifest_sha256": expected_sha256,
    }


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
    if role in (UNIFORM_ROLE, SELECTED_G0_ROLE):
        return f"afterok:{completed['gate']}"
    if role == AGGREGATE_ROLE:
        return (
            f"afterok:{completed[UNIFORM_ROLE]}:{completed[SELECTED_G0_ROLE]}"
        )
    raise JournalError(f"unknown submission role: {role}")


def _role_matches_position(role: str, index: int) -> bool:
    return index < COMPLETE_ROLE_COUNT and role == ROLE_ORDER[index]


def _read_rows(path: Path, *, target_cluster: str) -> list[dict[str, str]]:
    if not path.is_file():
        raise JournalError(f"submission journal is missing: {path}")
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != HEADER:
            raise JournalError("submission journal header is invalid")
        rows = list(reader)
    if len(rows) > COMPLETE_ROLE_COUNT:
        raise JournalError("submission journal contains too many rows")

    completed: dict[str, str] = {}
    observed_ids: set[str] = set()
    for index, row in enumerate(rows):
        if None in row or any(row.get(field, "") == "" for field in HEADER):
            raise JournalError("submission journal contains a malformed row")
        expected_role = row["role"]
        if not _role_matches_position(expected_role, index):
            raise JournalError(
                f"submission journal role order drift at position {index}: "
                f"got {expected_role}"
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
    _submission_manifest_binding(journal_path)
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
    if len(rows) >= COMPLETE_ROLE_COUNT or not _role_matches_position(
        role, len(rows)
    ):
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
    if len(rows) != COMPLETE_ROLE_COUNT or any(
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
    roles = [row["role"] for row in rows]
    submission_binding = _submission_manifest_binding(journal_path)
    expected = {
        "schema": SCHEMA,
        "complete": True,
        "git_commit": expected_commit,
        "target_cluster": target_cluster,
        "journal_path": str(journal_path.resolve()),
        "journal_sha256": _sha256(journal_path),
        "roles": roles,
        "main_official60_roles": [UNIFORM_ROLE, SELECTED_G0_ROLE],
        "selected_g0_runtime_routing": "sealed_frontend_decision",
        "job_count": COMPLETE_ROLE_COUNT,
        **submission_binding,
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
    if len(rows) != COMPLETE_ROLE_COUNT or any(
        row["job_id"] == PENDING_JOB_ID for row in rows
    ):
        raise JournalError("cannot seal a partial submission journal")
    roles = [row["role"] for row in rows]
    submission_binding = _submission_manifest_binding(journal_path)
    payload = {
        "schema": SCHEMA,
        "complete": True,
        "git_commit": expected_commit,
        "target_cluster": target_cluster,
        "journal_path": str(journal_path.resolve()),
        "journal_sha256": _sha256(journal_path),
        "roles": roles,
        "main_official60_roles": [UNIFORM_ROLE, SELECTED_G0_ROLE],
        "selected_g0_runtime_routing": "sealed_frontend_decision",
        "job_count": COMPLETE_ROLE_COUNT,
        **submission_binding,
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
