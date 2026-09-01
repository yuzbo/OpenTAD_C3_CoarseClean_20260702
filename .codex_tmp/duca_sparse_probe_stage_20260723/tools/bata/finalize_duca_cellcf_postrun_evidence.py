from __future__ import annotations

import argparse
import ctypes
import csv
from functools import wraps
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence


VARIANTS = ("uniform", "transition_beta0", "cellcf")
JOB_KEYS = (
    "convergence_uniform",
    "convergence_transition_beta0",
    "convergence_cellcf",
    "convergence_summary",
    "training_cost",
    "completion",
)
FORMAL_JOB_KEYS = (
    "uniform",
    "transition_beta0",
    "cellcf",
    "aggregate",
    "cost",
    "completion",
)
INTENT_SCHEMA = "duca_cellcf_postrun_submission_intent_v1"
MANIFEST_SCHEMA = "duca_cellcf_postrun_submission_manifest_v1"
RECEIPT_SCHEMA = "duca_cellcf_postrun_slurm_receipt_v1"
CONVERGENCE_SCHEMA = "duca_cellcf_fixed_convergence_trajectory_v1"
TRAINING_COST_SCHEMA = "duca_cellcf_training_cost_summary_v1"
SUPPORTED_TRAINED_COMMIT = "1642f265e48391418a7c8a4a087e33e2b7bf6899"
CANDIDATE_SCHEMA = "duca_cellcf_postrun_evidence_candidate_v2"
FINAL_SCHEMA = "duca_cellcf_postrun_evidence_completion_v4"
RECOVERY_INTENT_SCHEMA = "duca_cellcf_cost_recovery_intent_v1"
RECOVERY_MANIFEST_SCHEMA = "duca_cellcf_cost_recovery_submission_v1"
RECOVERY_FAILURE_SCHEMA = "duca_cellcf_cost_recovery_original_failure_v1"
FileIdentity = tuple[int, int, int, int, int]
SnapshotRecord = tuple[str, FileIdentity, FileIdentity]


class SnapshotRecords(dict[Path, SnapshotRecord]):
    _IN_CLOSE_WRITE = 0x00000008
    _IN_MODIFY = 0x00000002
    _IN_ATTRIB = 0x00000004
    _IN_MOVED_FROM = 0x00000040
    _IN_MOVED_TO = 0x00000080
    _IN_CREATE = 0x00000100
    _IN_DELETE = 0x00000200
    _IN_DELETE_SELF = 0x00000400
    _IN_MOVE_SELF = 0x00000800
    _IN_ONLYDIR = 0x01000000
    _WATCH_MASK = (
        _IN_CLOSE_WRITE
        | _IN_MODIFY
        | _IN_ATTRIB
        | _IN_MOVED_FROM
        | _IN_MOVED_TO
        | _IN_CREATE
        | _IN_DELETE
        | _IN_DELETE_SELF
        | _IN_MOVE_SELF
    )

    def __init__(self) -> None:
        super().__init__()
        self._inotify_fd: int | None = None
        self._watched_directories: dict[Path, FileIdentity] = {}
        self._libc: Any = None
        if sys.platform.startswith("linux"):
            libc = ctypes.CDLL(None, use_errno=True)
            libc.inotify_init1.argtypes = [ctypes.c_int]
            libc.inotify_init1.restype = ctypes.c_int
            libc.inotify_add_watch.argtypes = [
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint32,
            ]
            libc.inotify_add_watch.restype = ctypes.c_int
            descriptor = libc.inotify_init1(
                os.O_NONBLOCK | getattr(os, "O_CLOEXEC", 0)
            )
            if descriptor < 0:
                error = ctypes.get_errno()
                raise OSError(error, "failed to initialize inotify")
            self._libc = libc
            self._inotify_fd = int(descriptor)

    def watch_path(self, path: str | Path, label: str) -> Path:
        source = Path(path).expanduser()
        source_identity = _file_identity(source, label)
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} is missing: {source}") from exc
        _require(
            _file_identity(resolved, label) == source_identity,
            f"{label} resolved identity drifted before monitoring",
        )
        parent_identity = _directory_identity(resolved.parent, label)
        self._watch_directory(resolved.parent, parent_identity, label)
        try:
            observed_resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} disappeared while monitoring began") from exc
        _require(
            observed_resolved == resolved
            and _file_identity(resolved, label) == source_identity
            and _directory_identity(resolved.parent, label) == parent_identity,
            f"{label} path changed while monitoring began",
        )
        self.assert_no_mutations()
        return resolved

    def _watch_directory(
        self,
        directory: Path,
        expected_identity: FileIdentity,
        label: str,
    ) -> None:
        previous = self._watched_directories.get(directory)
        if previous is not None:
            _require(
                previous == expected_identity,
                f"{label} parent directory identity drifted",
            )
            return
        if self._inotify_fd is None:
            self._watched_directories[directory] = expected_identity
            return
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            directory_fd = os.open(directory, flags)
        except OSError as exc:
            raise ValueError(
                f"{label} parent directory cannot be opened safely: {directory}"
            ) from exc
        try:
            opened_identity = _identity_from_stat(os.fstat(directory_fd))
            _require(
                opened_identity == expected_identity,
                f"{label} parent directory changed before monitoring",
            )
            watch_path = f"/proc/self/fd/{directory_fd}"
            watch = self._libc.inotify_add_watch(
                self._inotify_fd,
                os.fsencode(watch_path),
                self._WATCH_MASK | self._IN_ONLYDIR,
            )
            if watch < 0:
                error = ctypes.get_errno()
                raise OSError(
                    error,
                    f"failed to monitor evidence directory: {directory}",
                )
            _require(
                _identity_from_stat(os.fstat(directory_fd)) == expected_identity,
                f"{label} parent directory changed while monitoring began",
            )
        finally:
            os.close(directory_fd)
        _require(
            _directory_identity(directory, label) == expected_identity,
            f"{label} parent directory path changed while monitoring began",
        )
        self._watched_directories[directory] = expected_identity

    def observe(
        self,
        path: Path,
        record: SnapshotRecord,
        label: str,
    ) -> None:
        previous = self.get(path)
        if previous is not None:
            _require(previous == record, f"{label} changed between validations")
        dict.__setitem__(self, path, record)

    @property
    def mutation_monitor_active(self) -> bool:
        return self._inotify_fd is not None

    def assert_no_mutations(self) -> None:
        if self._inotify_fd is None:
            return
        observed = bytearray()
        while True:
            try:
                chunk = os.read(self._inotify_fd, 1024 * 1024)
            except BlockingIOError:
                break
            if not chunk:
                break
            observed.extend(chunk)
        _require(
            not observed,
            "an evidence directory mutated during semantic validation",
        )

    def close(self) -> None:
        if self._inotify_fd is not None:
            os.close(self._inotify_fd)
            self._inotify_fd = None

    def __del__(self) -> None:
        try:
            self.close()
        except OSError:
            pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity_from_stat(value: os.stat_result) -> FileIdentity:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _file_identity(path: Path, label: str) -> FileIdentity:
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} is missing: {path}") from exc
    _require(not stat.S_ISLNK(observed.st_mode), f"{label} must not be a symlink")
    _require(stat.S_ISREG(observed.st_mode), f"{label} is not a regular file")
    return _identity_from_stat(observed)


def _directory_identity(path: Path, label: str) -> FileIdentity:
    try:
        observed = os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise ValueError(f"{label} parent directory is missing: {path}") from exc
    _require(
        not stat.S_ISLNK(observed.st_mode)
        and stat.S_ISDIR(observed.st_mode),
        f"{label} parent is not a stable directory",
    )
    return _identity_from_stat(observed)


def _read_snapshot(
    path: str | Path,
    label: str,
    *,
    records: SnapshotRecords | None = None,
) -> tuple[Path, bytes, str]:
    if records is not None:
        resolved = records.watch_path(path, label)
    else:
        source = Path(path).expanduser()
        _file_identity(source, label)
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} is missing: {source}") from exc
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {resolved}") from exc
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode),
            f"{label} is not a regular file",
        )
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        _identity_from_stat(before) == _identity_from_stat(after),
        f"{label} changed while it was read",
    )
    _require(
        _file_identity(resolved, label) == _identity_from_stat(after),
        f"{label} path identity changed while it was read",
    )
    payload_bytes = b"".join(chunks)
    digest = hashlib.sha256(payload_bytes).hexdigest()
    if records is not None:
        records.observe(
            resolved,
            (
                digest,
                _identity_from_stat(after),
                _directory_identity(resolved.parent, label),
            ),
            label,
        )
    return (
        resolved,
        payload_bytes,
        digest,
    )


def _capture_snapshot_record(
    path: str | Path,
    label: str,
    *,
    records: SnapshotRecords | None = None,
) -> tuple[Path, SnapshotRecord]:
    if records is not None:
        resolved = records.watch_path(path, label)
    else:
        source = Path(path).expanduser()
        _file_identity(source, label)
        try:
            resolved = source.resolve(strict=True)
        except OSError as exc:
            raise ValueError(f"{label} is missing: {source}") from exc
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely: {resolved}") from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = _identity_from_stat(after)
    _require(
        _identity_from_stat(before) == identity,
        f"{label} changed while it was hashed",
    )
    _require(
        _file_identity(resolved, label) == identity,
        f"{label} path identity changed while it was hashed",
    )
    record = (
        digest.hexdigest(),
        identity,
        _directory_identity(resolved.parent, label),
    )
    if records is not None:
        records.observe(resolved, record, label)
    return resolved, record


def _record_snapshot(
    records: SnapshotRecords,
    path: Path,
    digest: str,
    label: str,
) -> None:
    resolved, record = _capture_snapshot_record(
        path,
        label,
        records=records,
    )
    observed_digest = record[0]
    _require(resolved == path, f"{label} resolved path drifted")
    _require(observed_digest == digest, f"{label} changed before it was recorded")
    records.observe(resolved, record, label)


def _verify_snapshot(
    path: Path,
    record: SnapshotRecord,
) -> None:
    resolved, observed_record = _capture_snapshot_record(
        path,
        f"sealed evidence input {path}",
    )
    _require(resolved == path, f"evidence input path changed after validation: {path}")
    _require(
        observed_record[:2] == record[:2],
        f"evidence input changed after snapshot validation: {path}",
    )
    _require(
        observed_record[2] == record[2],
        f"evidence directory changed after snapshot validation: {path.parent}",
    )


def _record_hashed_dependency(
    records: SnapshotRecords,
    record: Mapping[str, Any],
    label: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    _require(isinstance(record, Mapping), f"{label} binding is missing")
    expected_digest = _require_hash(record.get("sha256"), f"{label} bound hash")
    resolved, snapshot_record = _capture_snapshot_record(
        record.get("path", ""),
        label,
        records=records,
    )
    if expected_path is not None:
        _require(resolved == expected_path, f"{label} path mismatch")
    _require(
        snapshot_record[0] == expected_digest,
        f"{label} bound hash mismatch",
    )
    records.observe(resolved, snapshot_record, label)
    return resolved


def _record_payload_path_hash_pairs(
    records: SnapshotRecords,
    payload: Mapping[str, Any],
    label: str,
) -> None:
    for path_key, path_value in sorted(payload.items()):
        if not str(path_key).endswith("_path"):
            continue
        sha_key = f"{str(path_key)[:-5]}_sha256"
        if sha_key not in payload:
            continue
        _record_hashed_dependency(
            records,
            {
                "path": path_value,
                "sha256": payload.get(sha_key),
            },
            f"{label} {path_key}",
        )


def _load_json(
    path: str | Path,
    label: str,
    *,
    records: SnapshotRecords | None = None,
) -> tuple[Path, dict[str, Any], str]:
    resolved, payload_bytes, digest = _read_snapshot(
        path,
        label,
        records=records,
    )
    payload = json.loads(payload_bytes.decode("utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload, digest


def _validate_embedded_hash(
    payload: Mapping[str, Any], key: str, label: str
) -> None:
    observed = payload.get(key)
    unsigned = dict(payload)
    unsigned.pop(key, None)
    _require(
        isinstance(observed, str)
        and re.fullmatch(r"[0-9a-f]{64}", observed) is not None
        and observed == canonical_sha256(unsigned),
        f"{label} canonical hash mismatch",
    )


def _require_hash(value: Any, label: str) -> str:
    normalized = str(value or "")
    _require(
        re.fullmatch(r"[0-9a-f]{64}", normalized) is not None,
        f"{label} is not a SHA256",
    )
    return normalized


def _require_commit(value: str, label: str) -> str:
    _require(
        re.fullmatch(r"[0-9a-f]{40}", value) is not None,
        f"{label} is not a full git commit",
    )
    return value


def _require_under(path: Path, root: Path, label: str) -> None:
    _require(path != root and root in path.parents, f"{label} escaped {root}")


def _default_aggregate_loader(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.duca_cellcf_suite_binding import load_suite_aggregate_binding

    return load_suite_aggregate_binding(**kwargs)


def revalidate_trained_suite_exact(**kwargs: Any) -> Mapping[str, Any]:
    trained_root = Path(kwargs["repo_root"]).expanduser().resolve()
    evidence_root = Path(
        kwargs["evidence_repo_root"]
    ).expanduser().resolve()
    expected_evidence_commit = str(kwargs["expected_evidence_commit"])
    with tempfile.TemporaryDirectory(
        prefix="duca-cellcf-cross-commit-revalidation-"
    ) as output_root_value:
        output_path = Path(output_root_value) / "final_suite_evidence.json"
        command = [
            sys.executable,
            "-m",
            "tools.bata.validate_duca_cellcf_suite",
            "--repo-root",
            str(trained_root),
            "--seed",
            str(kwargs["seed"]),
            "--expected-commit",
            str(kwargs["expected_commit"]),
            "--evidence-repo-root",
            str(evidence_root),
            "--expected-evidence-commit",
            expected_evidence_commit,
            "--gate-json",
            str(kwargs["gate_json"]),
            "--pilot-json",
            str(kwargs["pilot_json"]),
            "--cost-evidence",
            str(kwargs["cost_evidence"]),
            "--require-cost-evidence",
            "--output-json",
            str(output_path),
        ]
        if kwargs.get("require_clean"):
            command.append("--require-clean")
        for variant, path in sorted(kwargs["post_run_evidence"].items()):
            command.extend(["--post-run-evidence", f"{variant}={path}"])
        environment = dict(os.environ)
        environment["PYTHONNOUSERSITE"] = "1"
        environment.pop("PYTHONHOME", None)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(
            command,
            cwd=evidence_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        _require(
            result.returncode == 0,
            "cross-commit trained-suite revalidation failed: "
            f"{result.stderr.strip() or result.stdout.strip()}",
        )
        _, payload, _ = _load_json(
            output_path, "historically regenerated final suite"
        )
        return payload


def _default_convergence_rebuilder(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.summarize_duca_cellcf_convergence import (
        build_convergence_evidence,
    )

    return build_convergence_evidence(**kwargs)


def _default_training_cost_rebuilder(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.summarize_duca_cellcf_training_cost import (
        summarize_training_cost,
    )

    return summarize_training_cost(**kwargs)


def _default_scheduler_validator(**kwargs: Any) -> Mapping[str, Any]:
    from tools.bata.validate_duca_cellcf_slurm_receipt import (
        validate_slurm_receipt,
    )

    return validate_slurm_receipt(**kwargs)


def _default_formal_completion_validator(
    *, job_id: int, job_name: str, cluster: str
) -> Mapping[str, Any]:
    record = _default_scheduler_terminal_reader(
        job_id=job_id,
        job_name=job_name,
        cluster=cluster,
    )
    _require(
        record.get("state") == "COMPLETED"
        and record.get("exit_code") == "0:0",
        "formal completion is not uniquely COMPLETED/0:0",
    )
    return record


def _default_scheduler_terminal_reader(
    *, job_id: int, job_name: str, cluster: str
) -> Mapping[str, Any]:
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-M",
            cluster,
            "-j",
            str(job_id),
            "-n",
            "-P",
            "-o",
            "JobIDRaw,JobName%128,Cluster,State,ExitCode,ElapsedRaw",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    _require(
        result.returncode == 0,
        f"formal completion sacct query failed: {result.stderr.strip()}",
    )
    matches = []
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) >= 6 and fields[0] == str(job_id):
            matches.append(fields[:6])
    _require(
        len(matches) == 1
        and matches[0][0] == str(job_id)
        and matches[0][1] == job_name
        and matches[0][2] == cluster,
        "scheduler job identity is not unique",
    )
    return {
        "ok": True,
        "job_id": job_id,
        "job_name": job_name,
        "cluster": cluster,
        "state": matches[0][3],
        "exit_code": matches[0][4],
        "elapsed_raw_seconds": int(matches[0][5]),
    }


def _default_repository_validator(root: Path, expected_commit: str) -> None:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            raise ValueError(
                f"git validation failed for {root}: {result.stderr.strip()}"
            )
        return result.stdout.strip()

    _require(git("rev-parse", "HEAD") == expected_commit, f"repository commit drift: {root}")
    _require(
        not git("status", "--porcelain", "--untracked-files=normal"),
        f"repository is dirty: {root}",
    )
    ignored = git(
        "ls-files",
        "--others",
        "--ignored",
        "--exclude-standard",
        "--",
        "*.py",
        "*.pth",
        "sitecustomize.py",
        "usercustomize.py",
    )
    _require(not ignored, f"ignored Python source could shadow repository: {root}")


def _load_ledger(
    path: Path,
    *,
    records: SnapshotRecords | None = None,
) -> tuple[list[dict[str, str]], str]:
    _, payload_bytes, digest = _read_snapshot(
        path,
        "post-run submitted ledger",
        records=records,
    )
    decoded = payload_bytes.decode("utf-8")
    with io.StringIO(decoded, newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in rows] == list(JOB_KEYS),
        "post-run ledger does not bind the exact ordered six-job DAG",
    )
    return rows, digest


def _load_formal_completion(
    path: Path,
    *,
    expected_commit: str,
    expected_seed: int,
    expected_profile: str,
    records: SnapshotRecords | None = None,
) -> tuple[dict[str, str], str]:
    _, payload_bytes, digest = _read_snapshot(
        path,
        "formal submitted-job ledger",
        records=records,
    )
    with io.StringIO(payload_bytes.decode("utf-8"), newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in rows] == list(FORMAL_JOB_KEYS),
        "formal submitted-job ledger does not bind the exact six-job DAG",
    )
    for row in rows:
        profile = row.get("training_profile")
        if profile is None and expected_commit == SUPPORTED_TRAINED_COMMIT:
            profile = "exposure132"
        _require(
            row.get("commit") == expected_commit,
            f"formal {row.get('job_key')} commit mismatch",
        )
        _require(
            row.get("seed") == str(expected_seed),
            f"formal {row.get('job_key')} seed mismatch",
        )
        _require(
            profile == expected_profile,
            f"formal {row.get('job_key')} training profile mismatch",
        )
    clusters = {row.get("cluster") for row in rows}
    _require(
        len(clusters) == 1 and None not in clusters and "" not in clusters,
        "formal target cluster is ambiguous",
    )
    ids = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        _require(
            re.fullmatch(r"[1-9][0-9]*", job_id) is not None,
            f"formal {row.get('job_key')} job id is invalid",
        )
        ids[str(row["job_key"])] = job_id
    _require(
        len(set(ids.values())) == len(ids),
        "formal submitted-job ledger contains duplicate job ids",
    )
    expected_dependencies = {
        "uniform": "none",
        "transition_beta0": "none",
        "cellcf": "none",
        "aggregate": (
            "afterok:"
            + ":".join(
                ids[key] for key in ("uniform", "transition_beta0", "cellcf")
            )
        ),
        "cost": f"afterok:{ids['aggregate']}",
        "completion": f"afterok:{ids['aggregate']}:{ids['cost']}",
    }
    for row in rows:
        _require(
            row.get("dependency") == expected_dependencies[str(row["job_key"])],
            f"formal {row.get('job_key')} dependency mismatch",
        )
    completion = dict(rows[-1])
    _require(completion.get("job_key") == "completion", "formal completion row is missing")
    _require(bool(completion.get("job_name")), "formal completion job name is missing")
    return completion, digest


def _validate_cost_recovery_submission(
    *,
    manifest_path: str | Path,
    manifest_sha256: str,
    run_root: Path,
    trained_root: Path,
    trained_commit: str,
    evidence_root: Path,
    evidence_commit: str,
    aggregate_file: Path,
    aggregate_sha256: str,
    final_file: Path,
    cost_evidence_path: Path,
    formal_ledger_path: Path,
    formal_ledger_sha256: str,
    terminal_state_reader: Callable[..., Mapping[str, Any]],
    records: SnapshotRecords,
) -> dict[str, Any]:
    expected_manifest_sha = _require_hash(
        manifest_sha256, "cost recovery manifest hash"
    )
    manifest_file, manifest, observed_manifest_sha = _load_json(
        manifest_path,
        "cost recovery submission manifest",
        records=records,
    )
    _require(
        observed_manifest_sha == expected_manifest_sha,
        "cost recovery submission manifest hash mismatch",
    )
    _validate_embedded_hash(
        manifest,
        "artifact_sha256",
        "cost recovery submission manifest",
    )
    _record_snapshot(
        records,
        manifest_file,
        observed_manifest_sha,
        "cost recovery submission manifest",
    )
    recovery_root = manifest_file.parent
    _require_under(
        recovery_root,
        run_root,
        "cost recovery root",
    )
    _require(
        manifest.get("schema") == RECOVERY_MANIFEST_SCHEMA
        and manifest.get("ok") is True
        and manifest.get("status") == "SUBMITTED_HELD_VERIFIED"
        and manifest.get("task") == "offline_temporal_action_detection",
        "cost recovery submission manifest identity/status mismatch",
    )
    expected_manifest = {
        "trained_git_commit": trained_commit,
        "cost_producer_evidence_commit": evidence_commit,
        "aggregate_evidence_path": str(aggregate_file),
        "aggregate_evidence_sha256": aggregate_sha256,
        "cost_evidence_path": str(cost_evidence_path),
        "final_suite_evidence_path": str(final_file),
    }
    for key, value in expected_manifest.items():
        _require(
            manifest.get(key) == value,
            f"cost recovery submission manifest mismatch: {key}",
        )

    intent_path, intent, intent_sha = _load_json(
        manifest.get("submission_intent_path", ""),
        "cost recovery submission intent",
        records=records,
    )
    _require(
        intent_path == recovery_root / "submission_intent.json"
        and manifest.get("submission_intent_sha256") == intent_sha,
        "cost recovery intent path/hash mismatch",
    )
    _validate_embedded_hash(
        intent, "artifact_sha256", "cost recovery submission intent"
    )
    _record_snapshot(
        records,
        intent_path,
        intent_sha,
        "cost recovery submission intent",
    )
    expected_intent = {
        "schema": RECOVERY_INTENT_SCHEMA,
        "status": "INTENT_RECORDED",
        "task": "offline_temporal_action_detection",
        "formal_run_root": str(run_root),
        "recovery_root": str(recovery_root),
        "trained_repository": str(trained_root),
        "trained_git_commit": trained_commit,
        "cost_producer_repository": str(evidence_root),
        "cost_producer_evidence_commit": evidence_commit,
        "aggregate_evidence_path": str(aggregate_file),
        "aggregate_evidence_sha256": aggregate_sha256,
        "original_formal_ledger_path": str(formal_ledger_path),
        "original_formal_ledger_sha256": formal_ledger_sha256,
        "cost_root": str(recovery_root / "cost"),
        "cost_evidence_path": str(cost_evidence_path),
        "final_suite_evidence_path": str(final_file),
    }
    for key, value in expected_intent.items():
        _require(
            intent.get(key) == value,
            f"cost recovery intent mismatch: {key}",
        )
    _require(
        "do not rerun" in str(intent.get("recovery_scope", "")),
        "cost recovery scope does not preserve the completed training arms",
    )

    original_failure_path, original_failure, original_failure_sha = _load_json(
        manifest.get("original_failure_receipt_path", ""),
        "original cost failure receipt",
        records=records,
    )
    _require(
        original_failure_path
        == recovery_root / "original_failure_receipt.json"
        and manifest.get("original_failure_receipt_sha256")
        == original_failure_sha
        and intent.get("original_failure_receipt_path")
        == str(original_failure_path)
        and intent.get("original_failure_receipt_sha256")
        == original_failure_sha,
        "original cost failure receipt path/hash mismatch",
    )
    _validate_embedded_hash(
        original_failure,
        "artifact_sha256",
        "original cost failure receipt",
    )
    _record_snapshot(
        records,
        original_failure_path,
        original_failure_sha,
        "original cost failure receipt",
    )
    _require(
        original_failure.get("schema") == RECOVERY_FAILURE_SCHEMA
        and original_failure.get("ok") is True
        and original_failure.get("original_formal_ledger_path")
        == str(formal_ledger_path)
        and original_failure.get("original_formal_ledger_sha256")
        == formal_ledger_sha256,
        "original cost failure receipt identity mismatch",
    )
    scheduler_query = _record_hashed_dependency(
        records,
        {
            "path": original_failure.get("scheduler_query_path"),
            "sha256": original_failure.get("scheduler_query_sha256"),
        },
        "original cost failure scheduler query",
        expected_path=recovery_root
        / "receipts"
        / "original_terminal_jobs.sacct",
    )

    _, formal_bytes, reopened_formal_sha = _read_snapshot(
        formal_ledger_path,
        "original formal submitted-job ledger",
        records=records,
    )
    _require(
        reopened_formal_sha == formal_ledger_sha256,
        "original formal submitted-job ledger changed during recovery validation",
    )
    with io.StringIO(formal_bytes.decode("utf-8"), newline="") as handle:
        formal_rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in formal_rows]
        == list(FORMAL_JOB_KEYS),
        "original formal ledger no longer binds the exact six-job DAG",
    )
    original_jobs = {
        key: next(
            row for row in formal_rows if row.get("job_key") == key
        )
        for key in ("cost", "completion")
    }
    frozen_original = {
        key: original_failure.get(key)
        for key in ("cost", "completion")
    }
    for key in ("cost", "completion"):
        record = frozen_original[key]
        row = original_jobs[key]
        _require(
            isinstance(record, Mapping)
            and int(record.get("job_id", -1)) == int(row["job_id"])
            and record.get("job_name") == row["job_name"]
            and record.get("cluster") == row["cluster"],
            f"original {key} failure receipt does not match the formal ledger",
        )
        observed_state = str(record.get("state", ""))
        if key == "cost":
            _require(
                observed_state == "FAILED"
                and record.get("exit_code") == "1:0"
                and int(record.get("elapsed_raw_seconds", -1)) > 0,
                "original cost frozen terminal state mismatch",
            )
        else:
            _require(
                re.fullmatch(
                    r"CANCELLED(?: by [1-9][0-9]*)?",
                    observed_state,
                )
                is not None
                and record.get("exit_code") == "0:0"
                and int(record.get("elapsed_raw_seconds", -1)) == 0,
                "original completion frozen terminal state mismatch",
            )
        live = terminal_state_reader(
            job_id=int(row["job_id"]),
            job_name=row["job_name"],
            cluster=row["cluster"],
        )
        _require(
            live.get("ok") is True
            and live.get("state") == record.get("state")
            and live.get("exit_code") == record.get("exit_code")
            and int(live.get("elapsed_raw_seconds", -1))
            == int(record.get("elapsed_raw_seconds", -2)),
            f"original {key} scheduler state no longer matches recovery evidence",
        )
        frozen_original[key] = {
            "frozen": dict(record),
            "live": dict(live),
        }

    ledger_path = Path(str(manifest.get("jobs_ledger_path", ""))).resolve()
    _require(
        ledger_path == recovery_root / "jobs.submitted.tsv",
        "cost recovery ledger path mismatch",
    )
    rows, ledger_sha = _load_recovery_ledger(
        ledger_path,
        records=records,
    )
    _require(
        manifest.get("jobs_ledger_sha256") == ledger_sha,
        "cost recovery ledger hash mismatch",
    )
    jobs = manifest.get("jobs")
    _require(
        isinstance(jobs, list)
        and [record.get("job_key") for record in jobs]
        == ["cost", "completion"],
        "cost recovery manifest job set/order mismatch",
    )
    manifest_jobs = {str(record["job_key"]): record for record in jobs}
    ids = {str(row["job_key"]): str(row["job_id"]) for row in rows}
    _require(
        rows[0].get("dependency") == "none"
        and rows[1].get("dependency") == f"afterok:{ids['cost']}",
        "cost recovery dependency chain mismatch",
    )
    recovery_terminal = {}
    for row in rows:
        key = str(row["job_key"])
        record = manifest_jobs[key]
        expected_row = {
            "trained_commit": trained_commit,
            "cost_producer_evidence_commit": evidence_commit,
            "submission_intent_sha256": intent_sha,
            "original_formal_ledger_sha256": formal_ledger_sha256,
        }
        for field, value in expected_row.items():
            _require(
                row.get(field) == value,
                f"cost recovery {key} ledger mismatch: {field}",
            )
        for field in (
            "job_id",
            "job_name",
            "cluster",
            "dependency",
            "sbatch_file",
            "sbatch_sha256",
            "raw_sbatch_response",
            "scheduler_script",
            "scheduler_script_sha256",
        ):
            manifest_value = record.get(field)
            if field == "job_id":
                manifest_value = str(manifest_value)
            _require(
                row.get(field) == manifest_value,
                f"cost recovery {key} manifest/ledger mismatch: {field}",
            )
        job_file = _record_hashed_dependency(
            records,
            {
                "path": row.get("sbatch_file"),
                "sha256": row.get("sbatch_sha256"),
            },
            f"cost recovery {key} sbatch",
            expected_path=recovery_root / "jobs" / f"{key}.sbatch",
        )
        _record_hashed_dependency(
            records,
            {
                "path": row.get("scheduler_receipt"),
                "sha256": row.get("scheduler_receipt_sha256"),
            },
            f"cost recovery {key} scheduler receipt",
            expected_path=recovery_root
            / "receipts"
            / f"{key}.scheduler.txt",
        )
        scheduler_script = _record_hashed_dependency(
            records,
            {
                "path": row.get("scheduler_script"),
                "sha256": row.get("scheduler_script_sha256"),
            },
            f"cost recovery {key} scheduler-owned script",
            expected_path=recovery_root
            / "receipts"
            / f"{key}.scheduler.sbatch",
        )
        _require(
            row.get("scheduler_script_sha256")
            == row.get("sbatch_sha256"),
            f"cost recovery {key} scheduler-owned script differs",
        )
        _require(
            str(row.get("raw_sbatch_response"))
            == f"{row['job_id']};{row['cluster']}",
            f"cost recovery {key} raw sbatch response mismatch",
        )
        live = terminal_state_reader(
            job_id=int(row["job_id"]),
            job_name=row["job_name"],
            cluster=row["cluster"],
        )
        _require(
            live.get("ok") is True
            and live.get("state") == "COMPLETED"
            and live.get("exit_code") == "0:0",
            f"cost recovery {key} is not COMPLETED/0:0",
        )
        recovery_terminal[key] = {
            "job_file": str(job_file),
            "scheduler_owned_script": str(scheduler_script),
            "scheduler": dict(live),
        }
    original_terminal_ids = {
        str(original_jobs["cost"]["job_id"]),
        str(original_jobs["completion"]["job_id"]),
    }
    _require(
        set(ids.values()).isdisjoint(original_terminal_ids),
        "recovery job illegally reuses an original terminal job id",
    )
    return {
        "schema": "duca_cellcf_cost_recovery_validation_v1",
        "ok": True,
        "status": "complete_via_cost_recovery",
        "original_formal_dag_complete": False,
        "cost_producer_evidence_commit": evidence_commit,
        "manifest": {
            "path": str(manifest_file),
            "sha256": observed_manifest_sha,
        },
        "intent": {"path": str(intent_path), "sha256": intent_sha},
        "ledger": {"path": str(ledger_path), "sha256": ledger_sha},
        "original_failure_receipt": {
            "path": str(original_failure_path),
            "sha256": original_failure_sha,
            "scheduler_query_path": str(scheduler_query),
        },
        "original_terminal_jobs": frozen_original,
        "recovery_terminal_jobs": recovery_terminal,
    }


def _load_recovery_ledger(
    path: Path,
    *,
    records: SnapshotRecords | None = None,
) -> tuple[list[dict[str, str]], str]:
    _, payload_bytes, digest = _read_snapshot(
        path,
        "cost recovery submitted-job ledger",
        records=records,
    )
    with io.StringIO(payload_bytes.decode("utf-8"), newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    _require(
        [row.get("job_key") for row in rows] == ["cost", "completion"],
        "cost recovery ledger does not bind the exact two-job DAG",
    )
    return rows, digest


def _expected_dependencies(rows: Sequence[Mapping[str, str]]) -> dict[str, str]:
    ids = {}
    for row in rows:
        job_id = str(row.get("job_id") or "")
        _require(
            re.fullmatch(r"[1-9][0-9]*", job_id) is not None,
            f"{row.get('job_key')} has an invalid job id",
        )
        ids[str(row["job_key"])] = job_id
    _require(len(set(ids.values())) == len(ids), "post-run job ids are not unique")
    return {
        "convergence_uniform": "none",
        "convergence_transition_beta0": "none",
        "convergence_cellcf": "none",
        "convergence_summary": (
            "afterok:"
            + ":".join(
                ids[key]
                for key in (
                    "convergence_uniform",
                    "convergence_transition_beta0",
                    "convergence_cellcf",
                )
            )
        ),
        "training_cost": "none",
        "completion": (
            f"afterok:{ids['convergence_summary']}:{ids['training_cost']}"
        ),
    }


def _validate_receipt(
    path_value: str,
    expected_sha: str,
    *,
    expected_status: str,
    row: Mapping[str, str],
    intent_path: Path,
    intent_sha: str,
    trained_commit: str,
    evidence_commit: str,
    aggregate_sha: str,
    records: SnapshotRecords | None = None,
    submitted_path: Path | None = None,
    submitted_sha: str | None = None,
) -> tuple[Path, dict[str, Any], str]:
    path, payload, observed_sha = _load_json(
        path_value,
        f"{row['job_key']} {expected_status} receipt",
        records=records,
    )
    _require(
        observed_sha == expected_sha,
        f"{row['job_key']} receipt hash mismatch",
    )
    _validate_embedded_hash(payload, "artifact_sha256", f"{row['job_key']} receipt")
    expected = {
        "schema": RECEIPT_SCHEMA,
        "status": expected_status,
        "task": "offline_temporal_action_detection",
        "job_key": row["job_key"],
        "job_id": int(row["job_id"]),
        "job_name": row["job_name"],
        "cluster": row["cluster"],
        "dependency": None if row["dependency"] == "none" else row["dependency"],
        "submission_token": row["submission_token"],
        "job_file": str(Path(row["job_file"]).resolve()),
        "job_file_sha256": row["job_file_sha256"],
        "trained_git_commit": trained_commit,
        "evidence_git_commit": evidence_commit,
        "aggregate_suite_evidence_sha256": aggregate_sha,
        "submission_intent": str(intent_path),
        "submission_intent_sha256": intent_sha,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{row['job_key']} receipt mismatch: {key}")
    raw = str(payload.get("raw_sbatch_response") or "").strip()
    _require(
        raw.splitlines()[0] == f"{row['job_id']};{row['cluster']}",
        f"{row['job_key']} receipt raw sbatch response mismatch",
    )
    if expected_status == "SUBMITTED_UNVERIFIED":
        _require(payload.get("scheduler_validation") is None, "unverified receipt has validation")
        _require(payload.get("submitted_receipt") is None, "unverified receipt is self-linked")
    else:
        validation = payload.get("scheduler_validation")
        _require(
            isinstance(validation, Mapping)
            and validation.get("ok") is True
            and int(validation.get("job_id", -1)) == int(row["job_id"]),
            f"{row['job_key']} verified receipt lacks scheduler proof",
        )
        _require(
            validation.get("scheduler_script_verified") is True,
            f"{row['job_key']} verified receipt lacks scheduler script proof",
        )
        _require(
            validation.get("submission_command_held_verified") is True
            and validation.get("current_user_hold_verified") is True,
            f"{row['job_key']} verified receipt lacks frozen held-job proof",
        )
        _require(
            submitted_path is not None
            and payload.get("submitted_receipt") == str(submitted_path)
            and payload.get("submitted_receipt_sha256") == submitted_sha,
            f"{row['job_key']} verified receipt lost its submitted receipt",
        )
    return path, payload, observed_sha


def _with_snapshot_records(
    function: Callable[..., dict[str, Any]],
) -> Callable[..., dict[str, Any]]:
    @wraps(function)
    def wrapped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        snapshot_records = SnapshotRecords()
        kwargs["_snapshot_records"] = snapshot_records
        try:
            return function(*args, **kwargs)
        finally:
            snapshot_records.close()

    return wrapped


@_with_snapshot_records
def finalize_postrun_evidence(
    *,
    run_root: str | Path,
    control_root: str | Path,
    trained_repo_root: str | Path,
    trained_commit: str,
    evidence_repo_root: str | Path,
    evidence_commit: str,
    aggregate_path: str | Path,
    aggregate_sha256: str,
    final_suite_path: str | Path,
    final_suite_sha256: str,
    cost_recovery_manifest_path: str | Path | None = None,
    cost_recovery_manifest_sha256: str | None = None,
    require_postrun_completed: bool = True,
    aggregate_loader: Callable[..., Mapping[str, Any]] = _default_aggregate_loader,
    final_suite_revalidator: Callable[..., Mapping[str, Any]] = (
        revalidate_trained_suite_exact
    ),
    convergence_rebuilder: Callable[..., Mapping[str, Any]] = (
        _default_convergence_rebuilder
    ),
    training_cost_rebuilder: Callable[..., Mapping[str, Any]] = (
        _default_training_cost_rebuilder
    ),
    scheduler_validator: Callable[..., Mapping[str, Any]] = (
        _default_scheduler_validator
    ),
    formal_completion_validator: Callable[..., Mapping[str, Any]] = (
        _default_formal_completion_validator
    ),
    postrun_terminal_validator: Callable[..., Mapping[str, Any]] = (
        _default_formal_completion_validator
    ),
    scheduler_terminal_reader: Callable[..., Mapping[str, Any]] = (
        _default_scheduler_terminal_reader
    ),
    repository_validator: Callable[[Path, str], None] = (
        _default_repository_validator
    ),
    require_linux_mutation_monitor: bool = True,
    _snapshot_records: SnapshotRecords | None = None,
) -> dict[str, Any]:
    trained_commit = _require_commit(trained_commit, "trained commit")
    evidence_commit = _require_commit(evidence_commit, "evidence commit")
    _require(
        trained_commit == SUPPORTED_TRAINED_COMMIT,
        "unsupported trained commit for this frozen post-run protocol",
    )
    _require(
        evidence_commit != trained_commit,
        "trained and evidence commits must be distinct",
    )
    aggregate_sha256 = _require_hash(aggregate_sha256, "aggregate evidence hash")
    final_suite_sha256 = _require_hash(final_suite_sha256, "final suite hash")
    _require(
        (cost_recovery_manifest_path is None)
        == (cost_recovery_manifest_sha256 is None),
        "cost recovery manifest path and hash are required together",
    )
    run_root_path = Path(run_root).expanduser().resolve()
    control_root_path = Path(control_root).expanduser().resolve()
    trained_root = Path(trained_repo_root).expanduser().resolve()
    evidence_root = Path(evidence_repo_root).expanduser().resolve()
    _require(run_root_path.is_dir(), "formal run root is missing")
    _require(control_root_path.is_dir(), "post-run control root is missing")
    _require_under(control_root_path, run_root_path, "post-run control root")
    _require(trained_root.is_dir(), "trained repository is missing")
    _require(evidence_root.is_dir(), "evidence repository is missing")
    repository_validator(trained_root, trained_commit)
    repository_validator(evidence_root, evidence_commit)
    _require(_snapshot_records is not None, "snapshot monitor was not initialized")
    snapshot_records = _snapshot_records
    _require(
        not require_linux_mutation_monitor
        or snapshot_records.mutation_monitor_active,
        "formal evidence sealing requires the Linux mutation monitor",
    )

    aggregate_file, aggregate_payload, aggregate_observed_sha = _load_json(
        aggregate_path,
        "aggregate suite evidence",
        records=snapshot_records,
    )
    final_file, final_payload, final_observed_sha = _load_json(
        final_suite_path,
        "final suite evidence",
        records=snapshot_records,
    )
    _require(
        aggregate_observed_sha == aggregate_sha256,
        "aggregate suite evidence hash mismatch",
    )
    _require(
        final_observed_sha == final_suite_sha256,
        "final suite evidence hash mismatch",
    )
    _record_snapshot(
        snapshot_records,
        aggregate_file,
        aggregate_observed_sha,
        "aggregate suite evidence",
    )
    _record_snapshot(
        snapshot_records,
        final_file,
        final_observed_sha,
        "final suite evidence",
    )
    post_run_paths = {
        variant: run_root_path / "logs" / variant / "post_run_evidence.json"
        for variant in VARIANTS
    }
    post_run_payloads: dict[str, dict[str, Any]] = {}
    for variant, path in post_run_paths.items():
        resolved, payload, digest = _load_json(
            path,
            f"{variant} post-run evidence",
            records=snapshot_records,
        )
        _require(resolved == path, f"{variant} post-run path drift")
        post_run_payloads[variant] = payload
        _record_snapshot(
            snapshot_records,
            resolved,
            digest,
            f"{variant} post-run evidence",
        )
    for label, record in (
        ("aggregate real-loader gate", aggregate_payload.get("real_loader_gate")),
        ("aggregate DDP pilot", aggregate_payload.get("ddp_pilot")),
    ):
        if record is not None:
            _record_hashed_dependency(
                snapshot_records,
                record,
                label,
            )
    completed_runs = aggregate_payload.get("completed_runs")
    if isinstance(completed_runs, Mapping):
        for variant in VARIANTS:
            record = completed_runs.get(variant)
            if record is not None:
                _record_hashed_dependency(
                    snapshot_records,
                    record,
                    f"{variant} aggregate post-run dependency",
                    expected_path=post_run_paths[variant],
                )
    reference_data = aggregate_payload.get("reference_data_artifacts")
    if isinstance(reference_data, Mapping):
        _record_payload_path_hash_pairs(
            snapshot_records,
            reference_data,
            "aggregate reference data",
        )
    variant_records = aggregate_payload.get("variants")
    if isinstance(variant_records, list):
        for record in variant_records:
            if not isinstance(record, Mapping):
                continue
            validation = record.get("validation")
            if not isinstance(validation, Mapping):
                continue
            config_value = validation.get("config")
            config_sha = record.get("config_sha256")
            if config_value is None or config_sha is None:
                continue
            relative_config = record.get("config")
            expected_config = (
                trained_root / str(relative_config)
                if relative_config is not None
                else Path(str(config_value))
            ).resolve()
            _record_hashed_dependency(
                snapshot_records,
                {"path": config_value, "sha256": config_sha},
                f"{record.get('name')} trained config",
                expected_path=expected_config,
            )
    for variant, payload in post_run_payloads.items():
        _record_payload_path_hash_pairs(
            snapshot_records,
            payload,
            f"{variant} post-run dependency",
        )
    aggregate_binding = aggregate_loader(
        path=aggregate_file,
        expected_sha256=aggregate_sha256,
        expected_commit=trained_commit,
        expected_profile="exposure132",
        post_run_paths=post_run_paths,
    )
    _require(aggregate_binding.get("seed") == aggregate_payload.get("seed"), "aggregate seed drift")
    formal_ledger_path = run_root_path / "jobs.submitted.tsv"
    formal_completion, formal_ledger_sha = _load_formal_completion(
        formal_ledger_path,
        expected_commit=trained_commit,
        expected_seed=int(aggregate_binding["seed"]),
        expected_profile="exposure132",
        records=snapshot_records,
    )
    _record_snapshot(
        snapshot_records,
        formal_ledger_path.resolve(),
        formal_ledger_sha,
        "formal submitted-job ledger",
    )
    cost_record = final_payload.get("cost_evidence")
    final_profile = final_payload.get("training_profile")
    if final_profile is None and trained_commit == SUPPORTED_TRAINED_COMMIT:
        final_profile = "exposure132"
    _require(
        final_payload.get("schema") == "duca_cellcf_suite_manifest_v1"
        and final_payload.get("ok") is True
        and final_payload.get("status") == "complete"
        and final_payload.get("task") == "offline_temporal_action_detection"
        and final_payload.get("git_commit") == trained_commit
        and final_profile == "exposure132"
        and final_payload.get("seed") == aggregate_binding["seed"]
        and final_payload.get("cost_evidence_required") is True
        and isinstance(cost_record, Mapping)
        and cost_record.get("validated") is True,
        "final suite evidence has invalid completion semantics",
    )
    cost_evidence_path: Path | None = None
    for label, record in (
        ("real-loader gate", aggregate_binding.get("real_loader_gate")),
        ("DDP pilot", aggregate_binding.get("ddp_pilot")),
        ("trained-checkpoint cost evidence", cost_record),
    ):
        _require(isinstance(record, Mapping), f"{label} binding is missing")
        resolved, _, digest = _read_snapshot(
            record.get("path", ""),
            label,
            records=snapshot_records,
        )
        expected_digest = record.get("sha256")
        if expected_digest is not None:
            _require(
                _require_hash(expected_digest, f"{label} bound hash") == digest,
                f"{label} bound hash mismatch",
            )
        _record_snapshot(snapshot_records, resolved, digest, label)
        if label == "trained-checkpoint cost evidence":
            cost_evidence_path = resolved
    _require(cost_evidence_path is not None, "trained-checkpoint cost path is missing")
    loaded_cost_path, cost_payload, cost_payload_sha = _load_json(
        cost_evidence_path,
        "trained-checkpoint cost evidence",
        records=snapshot_records,
    )
    _require(
        loaded_cost_path == cost_evidence_path
        and snapshot_records[cost_evidence_path][0] == cost_payload_sha,
        "trained-checkpoint cost evidence changed before dependency enumeration",
    )
    _require(
        cost_payload.get("cost_producer_evidence_commit")
        == evidence_commit,
        "trained-checkpoint cost producer differs from the sealing commit",
    )
    cost_recovery = None
    if cost_recovery_manifest_path is None:
        formal_completion_scheduler = formal_completion_validator(
            job_id=int(formal_completion["job_id"]),
            job_name=formal_completion["job_name"],
            cluster=formal_completion["cluster"],
        )
        _require(
            formal_completion_scheduler.get("ok") is True,
            "formal completion scheduler revalidation failed",
        )
    else:
        cost_recovery = _validate_cost_recovery_submission(
            manifest_path=cost_recovery_manifest_path,
            manifest_sha256=str(cost_recovery_manifest_sha256),
            run_root=run_root_path,
            trained_root=trained_root,
            trained_commit=trained_commit,
            evidence_root=evidence_root,
            evidence_commit=evidence_commit,
            aggregate_file=aggregate_file,
            aggregate_sha256=aggregate_sha256,
            final_file=final_file,
            cost_evidence_path=cost_evidence_path,
            formal_ledger_path=formal_ledger_path,
            formal_ledger_sha256=formal_ledger_sha,
            terminal_state_reader=scheduler_terminal_reader,
            records=snapshot_records,
        )
        formal_completion_scheduler = {
            "ok": True,
            "mode": "cost_recovery",
            "status": "complete_via_cost_recovery",
            "original_formal_dag_complete": False,
            "original_completion_job_id": int(
                formal_completion["job_id"]
            ),
            "recovery": cost_recovery,
        }
    profile_artifacts = cost_payload.get("profile_artifacts")
    if isinstance(profile_artifacts, Mapping):
        for group, profile_records in profile_artifacts.items():
            if not isinstance(profile_records, list):
                continue
            for index, profile_record in enumerate(profile_records):
                if not isinstance(profile_record, Mapping):
                    continue
                profile_label = f"{group} cost profile {index}"
                profile_path = _record_hashed_dependency(
                    snapshot_records,
                    profile_record,
                    profile_label,
                )
                loaded_path, profile_payload, profile_sha = _load_json(
                    profile_path,
                    profile_label,
                    records=snapshot_records,
                )
                _require(
                    loaded_path == profile_path
                    and snapshot_records[profile_path][0] == profile_sha,
                    f"{profile_label} changed before dependency enumeration",
                )
                _record_payload_path_hash_pairs(
                    snapshot_records,
                    profile_payload,
                    profile_label,
                )
                profile_config_path = profile_payload.get("config_path")
                profile_config_sha = profile_payload.get(
                    "profile_config_sha256"
                )
                if (
                    profile_config_path is not None
                    or profile_config_sha is not None
                ):
                    _record_hashed_dependency(
                        snapshot_records,
                        {
                            "path": profile_config_path,
                            "sha256": profile_config_sha,
                        },
                        f"{profile_label} config",
                    )
    regenerated_final = final_suite_revalidator(
        repo_root=trained_root,
        evidence_repo_root=evidence_root,
        expected_evidence_commit=evidence_commit,
        seed=int(aggregate_binding["seed"]),
        expected_commit=trained_commit,
        require_clean=True,
        gate_json=aggregate_binding["real_loader_gate"]["path"],
        pilot_json=aggregate_binding["ddp_pilot"]["path"],
        post_run_evidence=post_run_paths,
        cost_evidence=cost_record["path"],
        require_cost_evidence=True,
    )
    _require(regenerated_final == final_payload, "final suite evidence is not reproducible")

    intent_path, intent, intent_sha = _load_json(
        control_root_path / "submission_intent.json",
        "post-run submission intent",
        records=snapshot_records,
    )
    _record_snapshot(
        snapshot_records,
        intent_path,
        intent_sha,
        "post-run submission intent",
    )
    _validate_embedded_hash(intent, "artifact_sha256", "post-run submission intent")
    postrun_output_root = Path(str(intent.get("postrun_output_root") or "")).resolve()
    _require(
        postrun_output_root == control_root_path / "artifacts",
        "post-run output root is not the versioned control artifact root",
    )
    intent_cluster = str(intent.get("target_cluster") or "")
    _require(
        re.fullmatch(r"[A-Za-z0-9._-]+", intent_cluster) is not None,
        "submission intent target cluster is invalid",
    )
    expected_intent = {
        "schema": INTENT_SCHEMA,
        "status": "INTENT_RECORDED",
        "task": "offline_temporal_action_detection",
        "formal_run_root": str(run_root_path),
        "trained_repository": str(trained_root),
        "trained_git_commit": trained_commit,
        "evidence_repository": str(evidence_root),
        "evidence_git_commit": evidence_commit,
        "target_cluster": intent_cluster,
        "aggregate_suite_evidence_path": str(aggregate_file),
        "aggregate_suite_evidence_sha256": aggregate_sha256,
        "final_suite_evidence_path": str(final_file),
        "final_suite_evidence_sha256": final_suite_sha256,
        "postrun_output_root": str(postrun_output_root),
    }
    if cost_recovery is not None:
        expected_intent.update(
            {
                "cost_recovery_manifest_path": cost_recovery[
                    "manifest"
                ]["path"],
                "cost_recovery_manifest_sha256": cost_recovery[
                    "manifest"
                ]["sha256"],
            }
        )
    for key, value in expected_intent.items():
        _require(intent.get(key) == value, f"submission intent mismatch: {key}")
    intent_jobs = intent.get("jobs")
    _require(
        isinstance(intent_jobs, list)
        and [record.get("job_key") for record in intent_jobs] == list(JOB_KEYS),
        "submission intent job set/order mismatch",
    )

    ledger_path = control_root_path / "jobs.submitted.tsv"
    _require(ledger_path.is_file(), "post-run submitted ledger is missing")
    rows, ledger_sha = _load_ledger(
        ledger_path,
        records=snapshot_records,
    )
    expected_dependencies = _expected_dependencies(rows)
    clusters = {row.get("cluster") for row in rows}
    _require(len(clusters) == 1 and None not in clusters, "post-run cluster identity is ambiguous")
    target_cluster = next(iter(clusters))
    _require(
        expected_intent["target_cluster"] == target_cluster,
        "submission intent target cluster mismatch",
    )
    _require(
        formal_completion["cluster"] == target_cluster,
        "formal and post-run jobs target different clusters",
    )
    intent_by_key = {str(record["job_key"]): record for record in intent_jobs}
    scheduler_records = []
    postrun_terminal_records = []
    receipt_records = []
    for row in rows:
        key = row["job_key"]
        _require(row["dependency"] == expected_dependencies[key], f"{key} dependency mismatch")
        expected_row = {
            "trained_commit": trained_commit,
            "evidence_commit": evidence_commit,
            "aggregate_sha256": aggregate_sha256,
            "submission_intent_sha256": intent_sha,
        }
        for field, value in expected_row.items():
            _require(row.get(field) == value, f"{key} ledger mismatch: {field}")
        intent_job = intent_by_key[key]
        for field in (
            "job_name",
            "submission_token",
            "job_file",
            "job_file_sha256",
        ):
            _require(row.get(field) == str(intent_job.get(field)), f"{key} intent mismatch: {field}")
        job_file, _, job_file_observed_sha = _read_snapshot(
            row["job_file"],
            f"{key} bound job file",
            records=snapshot_records,
        )
        _require(
            job_file == control_root_path / "jobs" / f"{key}.sbatch"
            and job_file_observed_sha == row["job_file_sha256"],
            f"{key} bound job file changed",
        )
        _record_snapshot(
            snapshot_records,
            job_file,
            job_file_observed_sha,
            f"{key} bound job file",
        )
        _require(
            Path(row["submitted_receipt"]).resolve()
            == control_root_path / "receipts" / f"{key}.submitted.json"
            and Path(row["verified_receipt"]).resolve()
            == control_root_path / "receipts" / f"{key}.verified.json",
            f"{key} receipt path mismatch",
        )
        submitted_path, _, submitted_observed_sha = _validate_receipt(
            row["submitted_receipt"],
            row["submitted_receipt_sha256"],
            expected_status="SUBMITTED_UNVERIFIED",
            row=row,
            intent_path=intent_path,
            intent_sha=intent_sha,
            trained_commit=trained_commit,
            evidence_commit=evidence_commit,
            aggregate_sha=aggregate_sha256,
            records=snapshot_records,
        )
        _record_snapshot(
            snapshot_records,
            submitted_path,
            submitted_observed_sha,
            f"{key} submitted receipt",
        )
        verified_path, _, verified_observed_sha = _validate_receipt(
            row["verified_receipt"],
            row["verified_receipt_sha256"],
            expected_status="VERIFIED",
            row=row,
            intent_path=intent_path,
            intent_sha=intent_sha,
            trained_commit=trained_commit,
            evidence_commit=evidence_commit,
            aggregate_sha=aggregate_sha256,
            records=snapshot_records,
            submitted_path=submitted_path,
            submitted_sha=row["submitted_receipt_sha256"],
        )
        _record_snapshot(
            snapshot_records,
            verified_path,
            verified_observed_sha,
            f"{key} verified receipt",
        )
        scheduler = scheduler_validator(
            job_id=int(row["job_id"]),
            job_name=row["job_name"],
            token=row["submission_token"],
            cluster=row["cluster"],
            job_file=job_file,
            job_file_sha256=row["job_file_sha256"],
            dependency="" if row["dependency"] == "none" else row["dependency"],
            require_scheduler_script=False,
            submitted_with_hold=True,
            require_current_user_hold=False,
        )
        _require(
            scheduler.get("ok") is True
            and scheduler.get("submission_command_held_verified") is True,
            f"{key} scheduler revalidation failed",
        )
        scheduler_records.append(dict(scheduler))
        if require_postrun_completed:
            terminal = postrun_terminal_validator(
                job_id=int(row["job_id"]),
                job_name=row["job_name"],
                cluster=row["cluster"],
            )
            _require(
                terminal.get("ok") is True
                and terminal.get("state") == "COMPLETED"
                and terminal.get("exit_code") == "0:0",
                f"{key} is not externally sealed as COMPLETED/0:0",
            )
            postrun_terminal_records.append(dict(terminal))
        receipt_records.append(
            {
                "job_key": key,
                "submitted": {
                    "path": str(submitted_path),
                    "sha256": row["submitted_receipt_sha256"],
                },
                "verified": {
                    "path": str(verified_path),
                    "sha256": row["verified_receipt_sha256"],
                },
                "submission_time_scheduler_script_verified": True,
            }
        )

    manifest_path, manifest, manifest_sha = _load_json(
        control_root_path / "submission_manifest.json",
        "post-run submission manifest",
        records=snapshot_records,
    )
    _record_snapshot(
        snapshot_records,
        manifest_path,
        manifest_sha,
        "post-run submission manifest",
    )
    _validate_embedded_hash(manifest, "artifact_sha256", "post-run submission manifest")
    expected_manifest = {
        "schema": MANIFEST_SCHEMA,
        "ok": True,
        "task": "offline_temporal_action_detection",
        "training_profile": "exposure132",
        "formal_run_root": str(run_root_path),
        "trained_repository": str(trained_root),
        "trained_git_commit": trained_commit,
        "evidence_repository": str(evidence_root),
        "evidence_git_commit": evidence_commit,
        "target_cluster": target_cluster,
        "aggregate_suite_evidence_path": str(aggregate_file),
        "aggregate_suite_evidence_sha256": aggregate_sha256,
        "final_suite_evidence_path": str(final_file),
        "final_suite_evidence_sha256": final_suite_sha256,
        "postrun_output_root": str(postrun_output_root),
        "submission_intent_path": str(intent_path),
        "submission_intent_sha256": intent_sha,
        "formal_completion_job_id": int(
            formal_completion["job_id"]
        ),
        "jobs_ledger_path": str(ledger_path),
        "jobs_ledger_sha256": ledger_sha,
        "jobs": rows,
    }
    if cost_recovery is not None:
        expected_manifest.update(
            {
                "cost_recovery_manifest_path": cost_recovery[
                    "manifest"
                ]["path"],
                "cost_recovery_manifest_sha256": cost_recovery[
                    "manifest"
                ]["sha256"],
                "completion_mode": "cost_recovery",
                "original_formal_dag_complete": False,
            }
        )
    for key, value in expected_manifest.items():
        _require(manifest.get(key) == value, f"submission manifest mismatch: {key}")

    convergence_path, convergence, convergence_sha = _load_json(
        postrun_output_root / "convergence" / "fixed_trajectory.json",
        "fixed convergence trajectory",
        records=snapshot_records,
    )
    training_cost_path, training_cost, training_cost_sha = _load_json(
        postrun_output_root / "training_cost" / "training_cost_summary.json",
        "training cost summary",
        records=snapshot_records,
    )
    _record_snapshot(
        snapshot_records,
        convergence_path,
        convergence_sha,
        "fixed convergence trajectory",
    )
    _record_snapshot(
        snapshot_records,
        training_cost_path,
        training_cost_sha,
        "training cost summary",
    )
    for payload, schema, label in (
        (convergence, CONVERGENCE_SCHEMA, "fixed convergence trajectory"),
        (training_cost, TRAINING_COST_SCHEMA, "training cost summary"),
    ):
        _require(
            payload.get("schema") == schema
            and payload.get("ok") is True
            and payload.get("task") == "offline_temporal_action_detection"
            and payload.get("git_commit") == trained_commit
            and payload.get("evidence_git_commit") == evidence_commit,
            f"{label} identity/status mismatch",
        )
        _validate_embedded_hash(payload, "artifact_sha256", label)
    _require(
        training_cost.get("training_profile") == "exposure132",
        "training cost profile mismatch",
    )
    _require(
        convergence.get("variants") == list(VARIANTS)
        and convergence.get("fixed_epochs") == [59, 89, 131]
        and convergence.get("primary_epoch") == 131
        and convergence.get("primary_state_key") == "state_dict_ema",
        "convergence trajectory protocol mismatch",
    )

    convergence_root = postrun_output_root / "convergence"
    evaluation_paths = {}
    variant_receipt_paths = {
        variant: convergence_root / variant / "variant_complete.json"
        for variant in VARIANTS
    }
    slurm_cost_paths = {
        variant: (
            postrun_output_root
            / "training_cost"
            / f"{variant}.slurm_cost.json"
        )
        for variant in VARIANTS
    }
    for variant in VARIANTS:
        evaluation_paths[(variant, 59)] = (
            convergence_root / variant / "epoch_59" / "evaluation.json"
        )
        evaluation_paths[(variant, 89)] = (
            convergence_root / variant / "epoch_89" / "evaluation.json"
        )
        evaluation_paths[(variant, 131)] = (
            run_root_path / "logs" / variant / "terminal_evaluation.json"
        )
    for label, path in (
        [
            (f"{variant} trajectory receipt", path)
            for variant, path in variant_receipt_paths.items()
        ]
        + [
            (f"{variant} epoch-{epoch} evaluation", path)
            for (variant, epoch), path in evaluation_paths.items()
        ]
        + [
            (f"{variant} Slurm cost evidence", path)
            for variant, path in slurm_cost_paths.items()
        ]
    ):
        resolved, payload, digest = _load_json(
            path,
            label,
            records=snapshot_records,
        )
        _record_snapshot(snapshot_records, resolved, digest, label)
        _record_payload_path_hash_pairs(
            snapshot_records,
            payload,
            label,
        )
        checkpoint_value = payload.get("checkpoint_path")
        if checkpoint_value is not None:
            sidecar_path, sidecar, sidecar_sha = _load_json(
                f"{Path(str(checkpoint_value)).expanduser().resolve()}.metadata.json",
                f"{label} checkpoint sidecar",
                records=snapshot_records,
            )
            _validate_embedded_hash(
                sidecar,
                "sidecar_sha256",
                f"{label} checkpoint sidecar",
            )
            _record_snapshot(
                snapshot_records,
                sidecar_path,
                sidecar_sha,
                f"{label} checkpoint sidecar",
            )
    rebuilt_convergence = convergence_rebuilder(
        expected_commit=trained_commit,
        expected_evidence_commit=evidence_commit,
        suite_aggregate_path=aggregate_file,
        suite_aggregate_sha256=aggregate_sha256,
        post_run_paths=post_run_paths,
        variant_receipt_paths=variant_receipt_paths,
        evaluation_paths=evaluation_paths,
    )
    _require(
        rebuilt_convergence == convergence,
        "fixed convergence trajectory is not reproducible",
    )
    rebuilt_training_cost = training_cost_rebuilder(
        expected_commit=trained_commit,
        expected_evidence_commit=evidence_commit,
        suite_aggregate_path=aggregate_file,
        suite_aggregate_sha256=aggregate_sha256,
        post_run_paths=post_run_paths,
        slurm_cost_paths=slurm_cost_paths,
    )
    _require(
        rebuilt_training_cost == training_cost,
        "training cost summary is not reproducible",
    )
    for row in convergence.get("rows", []):
        metrics = row.get("metrics", row)
        for key, value in metrics.items():
            if key == "variant" or isinstance(value, str):
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _require(math.isfinite(float(value)), "convergence contains non-finite metrics")

    completion_row = next(
        row for row in rows if row["job_key"] == "completion"
    )
    completion_scheduler = next(
        record
        for record in scheduler_records
        if int(record["job_id"]) == int(completion_row["job_id"])
    )
    candidate_binding = None
    if require_postrun_completed:
        candidate_path, candidate, candidate_sha = _load_json(
            control_root_path / "postrun_evidence_candidate.json",
            "post-run completion candidate",
            records=snapshot_records,
        )
        _validate_embedded_hash(
            candidate, "artifact_sha256", "post-run completion candidate"
        )
        _require(
            candidate.get("schema") == CANDIDATE_SCHEMA
            and candidate.get("ok") is False
            and candidate.get("status") == "pending_external_seal"
            and candidate.get("requires_external_seal") is True
            and candidate.get("trained_git_commit") == trained_commit
            and candidate.get("evidence_git_commit") == evidence_commit
            and candidate.get("jobs_ledger", {}).get("sha256") == ledger_sha,
            "post-run completion candidate identity/status mismatch",
        )
        candidate_completion = candidate.get("postrun_completion_job")
        _require(
            isinstance(candidate_completion, Mapping)
            and int(candidate_completion.get("job_id", -1))
            == int(completion_row["job_id"])
            and candidate_completion.get("job_name")
            == completion_row["job_name"]
            and candidate_completion.get("state_at_evidence_write")
            in {"RUNNING", "COMPLETING"}
            and candidate_completion.get("exit_code") is None,
            "post-run completion candidate was not written by the active completion job",
        )
        _require(
            candidate.get("postrun_terminal_revalidation") == []
            and candidate.get("candidate_evidence") is None,
            "post-run completion candidate contains forbidden terminal proof",
        )
        _require(
            candidate.get("aggregate_suite_evidence", {}).get("sha256")
            == aggregate_sha256
            and candidate.get("final_suite_evidence", {}).get("sha256")
            == final_suite_sha256
            and candidate.get("submission_intent", {}).get("sha256")
            == intent_sha
            and candidate.get("submission_manifest", {}).get("sha256")
            == manifest_sha
            and candidate.get("artifacts", {})
            .get("convergence", {})
            .get("sha256")
            == convergence_sha
            and candidate.get("artifacts", {})
            .get("training_cost", {})
            .get("sha256")
            == training_cost_sha,
            "post-run completion candidate artifact binding mismatch",
        )
        _record_snapshot(
            snapshot_records,
            candidate_path,
            candidate_sha,
            "post-run completion candidate",
        )
        candidate_binding = {
            "path": str(candidate_path),
            "sha256": candidate_sha,
        }

    snapshot_records.assert_no_mutations()
    for path, record in snapshot_records.items():
        _verify_snapshot(path, record)
    snapshot_records.assert_no_mutations()

    payload = {
        "schema": (
            FINAL_SCHEMA if require_postrun_completed else CANDIDATE_SCHEMA
        ),
        "ok": bool(require_postrun_completed),
        "status": (
            "complete"
            if require_postrun_completed
            else "pending_external_seal"
        ),
        "requires_external_seal": not require_postrun_completed,
        "task": "offline_temporal_action_detection",
        "training_profile": "exposure132",
        "trained_git_commit": trained_commit,
        "evidence_git_commit": evidence_commit,
        "cost_producer_evidence_commit": evidence_commit,
        "completion_mode": (
            "cost_recovery"
            if cost_recovery is not None
            else "original_formal_dag"
        ),
        "original_formal_dag_complete": cost_recovery is None,
        "aggregate_suite_evidence": {
            "path": str(aggregate_file),
            "sha256": aggregate_sha256,
        },
        "final_suite_evidence": {
            "path": str(final_file),
            "sha256": final_suite_sha256,
        },
        "submission_intent": {
            "path": str(intent_path),
            "sha256": intent_sha,
        },
        "submission_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha,
        },
        "jobs_ledger": {"path": str(ledger_path), "sha256": ledger_sha},
        "receipts": receipt_records,
        "scheduler_revalidation": scheduler_records,
        "postrun_terminal_revalidation": postrun_terminal_records,
        "postrun_completion_job": {
            "job_id": int(completion_row["job_id"]),
            "job_name": completion_row["job_name"],
            "state_at_evidence_write": completion_scheduler["state"],
            "exit_code": (
                "0:0" if require_postrun_completed else None
            ),
        },
        "formal_completion_scheduler_revalidation": dict(
            formal_completion_scheduler
        ),
        "cost_recovery": cost_recovery,
        "candidate_evidence": candidate_binding,
        "artifacts": {
            "convergence": {
                "path": str(convergence_path),
                "sha256": convergence_sha,
            },
            "training_cost": {
                "path": str(training_cost_path),
                "sha256": training_cost_sha,
            },
        },
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    return payload


def _exclusive_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    directory_fd = os.open(
        target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def validate_seal_execution_context(
    *,
    candidate: bool,
    slurm_job_id: str | None,
) -> int | None:
    normalized = str(slurm_job_id or "").strip()
    if candidate:
        _require(
            re.fullmatch(r"[1-9][0-9]*", normalized) is not None,
            "candidate evidence must be written by its Slurm completion job",
        )
        return int(normalized)
    _require(
        not normalized,
        "final evidence seal must run outside every Slurm allocation",
    )
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--control-root", required=True)
    parser.add_argument("--trained-repo-root", required=True)
    parser.add_argument("--trained-commit", required=True)
    parser.add_argument("--evidence-repo-root", required=True)
    parser.add_argument("--evidence-commit", required=True)
    parser.add_argument("--aggregate", required=True)
    parser.add_argument("--aggregate-sha256", required=True)
    parser.add_argument("--final-suite", required=True)
    parser.add_argument("--final-suite-sha256", required=True)
    parser.add_argument("--cost-recovery-manifest")
    parser.add_argument("--cost-recovery-manifest-sha256")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidate", action="store_true")
    args = parser.parse_args(argv)
    control_root = Path(args.control_root).expanduser().resolve()
    expected_output = control_root / (
        "postrun_evidence_candidate.json"
        if args.candidate
        else "postrun_evidence_complete.json"
    )
    _require(
        Path(args.output).expanduser().resolve() == expected_output,
        "post-run output path does not match candidate/final mode",
    )
    active_slurm_job_id = validate_seal_execution_context(
        candidate=args.candidate,
        slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    )
    payload = finalize_postrun_evidence(
        run_root=args.run_root,
        control_root=args.control_root,
        trained_repo_root=args.trained_repo_root,
        trained_commit=args.trained_commit,
        evidence_repo_root=args.evidence_repo_root,
        evidence_commit=args.evidence_commit,
        aggregate_path=args.aggregate,
        aggregate_sha256=args.aggregate_sha256,
        final_suite_path=args.final_suite,
        final_suite_sha256=args.final_suite_sha256,
        cost_recovery_manifest_path=args.cost_recovery_manifest,
        cost_recovery_manifest_sha256=(
            args.cost_recovery_manifest_sha256
        ),
        require_postrun_completed=not args.candidate,
        require_linux_mutation_monitor=True,
    )
    if args.candidate:
        _require(
            payload.get("postrun_completion_job", {}).get("job_id")
            == active_slurm_job_id,
            "candidate evidence was written by the wrong Slurm job",
        )
    _exclusive_write_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
