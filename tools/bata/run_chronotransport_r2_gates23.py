#!/usr/bin/env python3
"""Formal repository-owned replay plus Gate-2/Gate-3 adjudication CLI."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.gates23 import (
    adjudicate_gates23,
    _path_without_symlink_components,
    build_gates23_terminal_marker,
    load_exact_canonical_json,
    run_registered_gates23_replay,
    validate_gates23_report,
    validate_stage_b_phase_markers_static,
)
from opentad.models.chronotransport.filesystem import (
    exclusive_file_lock,
    open_bound_directory,
    open_bound_regular_file,
    path_exists_no_follow,
    publish_bytes_exclusive,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.registration import (
    FORMAL_OUTPUT_BASE,
    validate_pre_gate1_registration,
)


GATES23_REGISTERED_SOURCE_PATHS = (
    "opentad/models/chronotransport/gates23.py",
    "tools/bata/chronotransport_r2_gates23_replay_factory.py",
    "tools/bata/run_chronotransport_r2_gates23.py",
    "tests/test_chronotransport_r2_gates23.py",
)


class FormalStopped(RuntimeError):
    pass


def _file_sha256(path: Path) -> str:
    with open_bound_regular_file(path, label=f"Gate2/3 hash input {path}") as bound:
        return bound.size_and_sha256()[1]


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish one immutable artifact atomically and fail if the name exists."""

    publish_bytes_exclusive(path, payload, label="formal Gate2/3 publication")


def _publish_or_validate_exact(path: Path, payload: bytes, *, label: str) -> bool:
    """Publish once, or resume only from exactly recomputed immutable bytes."""

    existed = path_exists_no_follow(path, label=label)
    try:
        publish_bytes_exclusive(
            path,
            payload,
            label=label,
            allow_existing_exact=True,
        )
    except FileExistsError as error:
        raise ValueError(
            f"existing {label} bytes differ from exact recomputation"
        ) from error
    return not existed


def _validate_recoverable_publication_state(outputs: Mapping[str, Path]) -> None:
    """Allow exact replay/report recovery, but never reopen a terminal run."""

    terminal = _path_without_symlink_components(
        outputs["terminal"], label="formal Gate2/3 terminal", allow_missing=True
    )
    if path_exists_no_follow(terminal, label="formal Gate2/3 terminal"):
        raise FileExistsError(
            "formal Gate2/3 terminal already exists for this immutable R"
        )
    for name in ("replay", "report"):
        artifact = _path_without_symlink_components(
            outputs[name],
            label=f"formal Gate2/3 {name}",
            allow_missing=True,
        )
        if path_exists_no_follow(artifact, label=f"formal Gate2/3 {name}"):
            with open_bound_regular_file(
                artifact, label=f"formal Gate2/3 {name}"
            ):
                pass


@contextmanager
def _exclusive_run_lock(root: Path):
    """Hold the sole formal writer lock for one immutable registration R."""

    lock_path = root / "run.lock"
    with exclusive_file_lock(
        lock_path,
        label="formal Gate2/3 run lock",
        payload=f"pid={os.getpid()}\n".encode("ascii"),
    ):
        yield lock_path


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise ValueError("formal Gate2/3 cannot derive registration commit R from HEAD")
    return commit


def _registration_relative_path(path: Path) -> str:
    resolved = _path_without_symlink_components(
        path, label="formal Gate2/3 registration artifact"
    )
    repository_root = _path_without_symlink_components(
        ROOT, label="formal Gate2/3 repository root"
    )
    try:
        return resolved.relative_to(repository_root).as_posix()
    except ValueError as error:
        raise ValueError("formal Gate2/3 registration must be tracked inside repository") from error


def _assert_gates23_sources_registered(
    registration: Mapping[str, Any], *, repository_root: Path = ROOT
) -> None:
    """Fail closed until the new Gate2/3 surface is hash-bound by registration R."""

    sources = registration.get("source_files") if isinstance(registration, Mapping) else None
    if not isinstance(sources, Mapping):
        raise ValueError("registration source_files mapping is missing")
    for relative in GATES23_REGISTERED_SOURCE_PATHS:
        if relative not in sources:
            raise ValueError(
                f"Gate2/3 source is absent from registration R: {relative}"
            )
        with open_bound_directory(
            repository_root, label="formal Gate2/3 repository root"
        ) as root:
            with root.open_regular(
                relative, label=f"registered Gate2/3 source {relative}"
            ) as source:
                source_sha256 = source.size_and_sha256()[1]
        if source_sha256 != sources[relative]:
            raise ValueError(
                f"Gate2/3 source bytes differ from registration R: {relative}"
            )


def _resolve_outputs(registration_commit: str) -> dict[str, Path]:
    base = _path_without_symlink_components(
        FORMAL_OUTPUT_BASE,
        label="formal Gate2/3 output base",
        allow_missing=True,
    )
    root = _path_without_symlink_components(
        base / registration_commit / "shared" / "gates23",
        label="formal Gate2/3 output root",
        allow_missing=True,
    )
    try:
        root.relative_to(base)
    except ValueError as error:
        raise ValueError("formal Gate2/3 output root escapes the fixed base") from error
    return {
        "root": root,
        "report": root / "gates23_report.json",
        "terminal": root / "terminal_marker.json",
        "replay": root / "gates23_replay.json",
    }


def _write_terminal(
    outputs: Mapping[str, Path],
    *,
    report: Mapping[str, Any],
    replay_artifact: Mapping[str, Any],
    registration: Mapping[str, Any],
    gate1_unlock: Mapping[str, Any],
    phase_marker_paths: Mapping[int, Path | str],
    gate1_unlock_path: Path,
    repository_root: Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    marker = build_gates23_terminal_marker(
        report=report,
        replay_artifact=replay_artifact,
        registration=registration,
        gate1_unlock=gate1_unlock,
        phase_marker_paths=phase_marker_paths,
        gate1_unlock_path=gate1_unlock_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
        report_path=outputs["report"],
    )
    _atomic_write(outputs["terminal"], canonical_json_bytes(marker) + b"\n")
    return marker


def _validate_gate1(
    path: Path,
    *,
    registration: Mapping[str, Any],
    repository_root: Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    from opentad.models.chronotransport.gate1_unlock import (
        build_gate1_unlock_artifact,
        validate_gate1_unlock_artifact,
    )

    artifact = load_exact_canonical_json(path, label="Gate1 unlock")
    rebuilt = build_gate1_unlock_artifact(
        artifact.get("gate1_input", {}),
        repository_root=str(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    if rebuilt != artifact:
        raise ValueError("Gate1 unlock is not an exact recomputable artifact")
    if (
        artifact.get("registration_sha256")
        != registration.get("registration_sha256")
    ):
        raise ValueError("Gate1 unlock registration identity mismatch")
    if artifact.get("status") != "PASS" or artifact.get("oracle_headroom") is not True:
        raise FormalStopped("Gate1 did not unlock Stage B/Gate2")
    return validate_gate1_unlock_artifact(
        artifact,
        registration=registration,
        repository_root=str(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def _install_stop_handlers() -> None:
    def stop(signum, _frame):
        raise FormalStopped(f"received stop signal {signum}")

    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is not None:
            signal.signal(signum, stop)


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration_commit = _git_head(ROOT)
    outputs = _resolve_outputs(registration_commit)
    with _exclusive_run_lock(outputs["root"]):
        return _run_locked(args, registration_commit=registration_commit, outputs=outputs)


def _run_locked(
    args: argparse.Namespace,
    *,
    registration_commit: str,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    replay_argument = _path_without_symlink_components(
        args.replay_artifact,
        label="formal Gate2/3 replay output argument",
        allow_missing=True,
    )
    if replay_argument != outputs["replay"]:
        raise ValueError(
            "formal Gate2/3 replay must be the canonical R/shared/gates23/gates23_replay.json"
        )
    _validate_recoverable_publication_state(outputs)
    _install_stop_handlers()
    registration_raw = load_exact_canonical_json(
        args.registration, label="pre-Gate1 registration"
    )
    registration = validate_pre_gate1_registration(registration_raw)
    _assert_gates23_sources_registered(registration, repository_root=ROOT)
    registration_relpath = _registration_relative_path(args.registration)
    registration = validate_pre_gate1_registration(
        registration_raw,
        repository_root=ROOT,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    gate1 = _validate_gate1(
        args.gate1_unlock,
        registration=registration,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    marker_paths = {
        3407: args.phase_marker_3407,
        3408: args.phase_marker_3408,
        3409: args.phase_marker_3409,
    }
    validate_stage_b_phase_markers_static(
        marker_paths,
        registration=registration,
        gate1_unlock=gate1,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    replay = run_registered_gates23_replay(
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    _publish_or_validate_exact(
        outputs["replay"],
        canonical_json_bytes(replay) + b"\n",
        label="Gate2/3 replay",
    )
    report = adjudicate_gates23(
        replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    validate_gates23_report(
        report,
        replay_artifact=replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    _publish_or_validate_exact(
        outputs["report"],
        canonical_json_bytes(report) + b"\n",
        label="Gate2/3 report",
    )
    marker = _write_terminal(
        outputs,
        report=report,
        replay_artifact=replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=marker_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    result = {
        "status": report["status"],
        "report_path": str(outputs["report"]),
        "report_sha256": report["artifact_sha256"],
        "terminal_marker_path": str(outputs["terminal"]),
        "terminal_state": marker["terminal_state"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repository-owned CT-P3R-3S-r2 Gate2/Gate3 replay/adjudication"
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate1-unlock", type=Path, required=True)
    parser.add_argument("--phase-marker-3407", type=Path, required=True)
    parser.add_argument("--phase-marker-3408", type=Path, required=True)
    parser.add_argument("--phase-marker-3409", type=Path, required=True)
    parser.add_argument("--replay-artifact", type=Path, required=True)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
