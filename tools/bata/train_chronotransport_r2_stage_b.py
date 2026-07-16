#!/usr/bin/env python3
"""Fixed repository-owned CT-P3R-3S-r2 Stage-B formal entrypoint."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.formal_stage_b import (
    _file_sha256,
    build_r2_stage_b_phase_completion_marker,
    build_fit_schedule_constant_artifact,
    logical_risk_predictor_state_sha256,
    run_r2_stage_b_training,
    validate_r2_stage_b_phase_completion_marker,
)
from opentad.models.chronotransport.environment import (
    observe_formal_slurm_environment,
    observed_environment_to_provenance,
)
from opentad.models.chronotransport.gate1_unlock import validate_gate1_unlock_artifact
from opentad.models.chronotransport.protocol import canonical_json_bytes, canonical_sha256
from opentad.models.chronotransport.registration import (
    FORMAL_OUTPUT_BASE,
    validate_pre_gate1_registration,
)
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from tools.bata.chronotransport_r2_stage_b_factory import (
    build_repository_stage_b_components,
)


def _load_json(path: Path) -> Any:
    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)


def _path_without_symlink_components(
    path: Path | str, *, label: str, allow_missing: bool = False
) -> Path:
    """Return an absolute lexical path after checking every existing component."""

    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            if allow_missing:
                break
            raise FileNotFoundError(current) from None
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}")
    return absolute


def _reuse_exact_regular_file(path: Path, payload: bytes, *, label: str) -> bool:
    """Accept an interrupted publication only when the existing inode is exact."""

    try:
        before = os.lstat(path)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return False
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{label} changed identity while being verified")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            existing = stream.read()
        after = os.lstat(path)
        if after.st_dev != opened.st_dev or after.st_ino != opened.st_ino:
            raise RuntimeError(f"{label} changed identity while being verified")
    finally:
        os.close(descriptor)
    if existing != payload:
        raise FileExistsError(f"{label} already exists with different bytes: {path}")
    return True


def _read_regular_file_bytes(path: Path | str, *, label: str) -> tuple[Path, bytes]:
    """Read one exact regular inode without following a symlink or replacement."""

    exact = _path_without_symlink_components(path, label=label)
    before = os.lstat(exact)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(exact, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_dev != before.st_dev
            or opened.st_ino != before.st_ino
        ):
            raise RuntimeError(f"{label} changed identity while being read")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            payload = stream.read()
        after = os.lstat(exact)
        if after.st_dev != opened.st_dev or after.st_ino != opened.st_ino:
            raise RuntimeError(f"{label} changed identity while being read")
    finally:
        os.close(descriptor)
    return exact, payload


def _load_torch_regular_file(path: Path | str, *, label: str) -> tuple[Path, Any]:
    exact, payload = _read_regular_file_bytes(path, label=label)
    try:
        value = torch.load(io.BytesIO(payload), map_location="cpu")
    except Exception as error:
        raise ValueError(f"{label} is not a valid torch checkpoint") from error
    return exact, value


def _regular_file_exists(path: Path | str, *, label: str) -> bool:
    exact = _path_without_symlink_components(path, label=label, allow_missing=True)
    try:
        metadata = os.lstat(exact)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular non-symlink file")
    return True


def _atomic_write(path: Path, payload: bytes) -> None:
    """Publish one formal Stage-B completion artifact without replacement."""

    path = _path_without_symlink_components(
        path, label="formal Stage-B output path", allow_missing=True
    )
    parent = _path_without_symlink_components(
        path.parent, label="formal Stage-B output parent", allow_missing=True
    )
    parent.mkdir(parents=True, exist_ok=True)
    parent = _path_without_symlink_components(
        parent, label="formal Stage-B output parent"
    )
    path = _path_without_symlink_components(
        path, label="formal Stage-B output path", allow_missing=True
    )
    if _reuse_exact_regular_file(
        path, payload, label="formal Stage-B completion artifact"
    ):
        return
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=parent
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            if not _reuse_exact_regular_file(
                path, payload, label="formal Stage-B completion artifact"
            ):
                raise error
        else:
            try:
                directory_fd = os.open(parent, os.O_RDONLY)
            except OSError:
                directory_fd = None
            if directory_fd is not None:
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextmanager
def _exclusive_run_lock(path: Path):
    """Allow only one formal Stage-B writer for one registered seed root."""

    path = _path_without_symlink_components(
        path, label="formal Stage-B run lock", allow_missing=True
    )
    parent = _path_without_symlink_components(
        path.parent, label="formal Stage-B run-lock parent", allow_missing=True
    )
    parent.mkdir(parents=True, exist_ok=True)
    _path_without_symlink_components(
        parent, label="formal Stage-B run-lock parent"
    )
    path = _path_without_symlink_components(
        path, label="formal Stage-B run lock", allow_missing=True
    )
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    identity = os.fstat(descriptor)
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        try:
            current = os.lstat(path)
        except FileNotFoundError:
            current = None
        if (
            current is not None
            and current.st_dev == identity.st_dev
            and current.st_ino == identity.st_ino
        ):
            path.unlink()


def _validate_unlock(
    unlock: Any,
    *,
    registration: Mapping[str, Any],
    repository_root: str | Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, Any]:
    return validate_gate1_unlock_artifact(
        unlock,
        registration=registration,
        repository_root=str(repository_root),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )


def _formal_guard(required_environment: Mapping[str, Any]) -> dict[str, Any]:
    return observe_formal_slurm_environment(required_environment)


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    commit = completed.stdout.strip()
    if completed.returncode != 0 or len(commit) != 40:
        raise ValueError("formal Stage B cannot derive registration commit R from HEAD")
    return commit


def _load_formal_registration(path: Path):
    resolved = _path_without_symlink_components(
        path, label="formal Stage-B registration", allow_missing=True
    )
    try:
        relative = resolved.relative_to(
            _path_without_symlink_components(
                ROOT, label="formal Stage-B repository root"
            )
        ).as_posix()
    except ValueError as error:
        raise ValueError("formal Stage-B registration must be tracked inside repository root") from error
    registration_commit = _git_head(ROOT)
    registration = validate_pre_gate1_registration(
        _load_json(resolved),
        repository_root=ROOT,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=relative,
    )
    return registration, registration_commit, relative


def _resolve_formal_stage_b_paths(
    args: argparse.Namespace,
    *,
    registration: Mapping[str, Any],
    registration_commit: str,
    repository_root: Path = ROOT,
) -> dict[str, Path]:
    """Resolve the only four writable Stage-B artifacts under ``R/seed``."""

    output_root = registration.get("output_root")
    if not isinstance(output_root, Mapping) or output_root.get("base") != FORMAL_OUTPUT_BASE:
        raise ValueError("formal Stage-B output base differs from the fixed registration base")
    base = _path_without_symlink_components(
        FORMAL_OUTPUT_BASE,
        label="formal Stage-B output base",
        allow_missing=True,
    )
    seed_root = _path_without_symlink_components(
        base / registration_commit / str(args.seed),
        label="formal Stage-B canonical R/seed root",
        allow_missing=True,
    )
    try:
        seed_root.relative_to(base)
    except ValueError as error:
        raise ValueError("formal Stage-B canonical R/seed root escapes output base") from error

    names = ("output", "ledger", "fit_baseline", "phase_marker")
    resolved = {
        name: _path_without_symlink_components(
            getattr(args, name),
            label=f"formal Stage-B {name}",
            allow_missing=True,
        )
        for name in names
    }
    for name, path in resolved.items():
        if path.parent != seed_root:
            raise ValueError(
                f"formal Stage-B {name} must be a direct child of canonical R/seed root"
            )
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("formal Stage-B output artifacts must be pairwise distinct")

    repository = _path_without_symlink_components(
        repository_root,
        label="formal Stage-B repository root",
        allow_missing=True,
    )
    for path in resolved.values():
        try:
            path.relative_to(repository)
        except ValueError:
            pass
        else:
            raise ValueError("formal Stage-B outputs must remain outside the repository")

    input_names = (
        "registration",
        "gate1_unlock",
        "manifest",
        "media_registry",
        "config_identity",
        "exposure_artifact",
        "checkpoint",
        "resume",
    )
    inputs = {
        _path_without_symlink_components(
            value,
            label=f"formal Stage-B {name} input",
            allow_missing=True,
        )
        for name in input_names
        if (value := getattr(args, name, None)) is not None
    }
    if set(resolved.values()) & inputs:
        raise ValueError("formal Stage-B output artifacts must differ from every input")
    return resolved


def _registered_provenance(
    registration: Mapping[str, Any],
    unlock: Mapping[str, Any],
    unlock_path: Path,
    registration_commit: str,
    observed_environment: Mapping[str, Any],
) -> dict[str, Any]:
    observed = observed_environment_to_provenance(
        observed_environment,
        required_environment=registration["environment"],
    )
    return {
        "registration_sha256": registration["registration_sha256"],
        "registration_commit": registration_commit,
        "spec_commit": registration["spec"]["commit"],
        "spec_sha256": registration["spec"]["sha256"],
        "implementation_commit": registration["implementation_commit"],
        "source_files_sha256": canonical_sha256(registration["source_files"]),
        "upstream_commits_sha256": canonical_sha256(registration["upstream_commits"]),
        "split_hashes_sha256": canonical_sha256(
            registration["window_manifest"]["artifact"]["split_hashes"]
        ),
        "action_library_sha256": registration["candidate_library"]["library_sha256"],
        "environment_sha256": registration["environment"]["environment_sha256"],
        **observed,
        "cost_plan_sha256": canonical_sha256(registration["profiler"]),
        "gate1_unlock_payload_sha256": canonical_sha256(unlock),
        "gate1_unlock_file_sha256": _file_sha256(unlock_path),
        "gate1_status": "PASS",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    registration, registration_commit, registration_relpath = (
        _load_formal_registration(args.registration)
    )
    outputs = _resolve_formal_stage_b_paths(
        args,
        registration=registration,
        registration_commit=registration_commit,
    )
    lock_path = outputs["output"].parent / ".chronotransport-r2-stage-b.lock"
    with _exclusive_run_lock(lock_path):
        return _run_locked(
            args,
            registration=registration,
            registration_commit=registration_commit,
            registration_relpath=registration_relpath,
            outputs=outputs,
        )


def _run_locked(
    args: argparse.Namespace,
    *,
    registration: Mapping[str, Any],
    registration_commit: str,
    registration_relpath: str,
    outputs: Mapping[str, Path],
) -> dict[str, Any]:
    observed_environment = _formal_guard(registration["environment"])
    unlock = _validate_unlock(
        _load_json(args.gate1_unlock),
        registration=registration,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    exposure = _load_json(args.exposure_artifact)
    library_rows = registration["candidate_library"]["candidates"]
    library_by_name = {row["name"]: row for row in library_rows}
    candidate_action_sha256_by_name = {
        name: library_by_name[name]["action_sha256"]
        for name in R2_NON_DENSE_NAMES
    }
    checkpoint, checkpoint_bytes = _read_regular_file_bytes(
        args.checkpoint, label="formal Stage-B dense checkpoint"
    )
    registered_checkpoint = registration["dense_checkpoint"]
    if (
        len(checkpoint_bytes) != registered_checkpoint["bytes"]
        or hashlib.sha256(checkpoint_bytes).hexdigest()
        != registered_checkpoint["sha256"]
    ):
        raise ValueError("dense checkpoint file bytes/SHA do not match registration")
    components = build_repository_stage_b_components(
        registration=registration,
        manifest_path=args.manifest,
        media_registry_path=args.media_registry,
        config_identity_path=args.config_identity,
        exposure_artifact=exposure,
        seed=args.seed,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    registered_provenance = _registered_provenance(
        registration,
        unlock,
        args.gate1_unlock,
        registration_commit,
        observed_environment,
    )
    checkpoint_exists = _regular_file_exists(
        outputs["output"], label="formal Stage-B trained checkpoint"
    )
    ledger_exists = _regular_file_exists(
        outputs["ledger"], label="formal Stage-B external ledger"
    )
    baseline_exists = _regular_file_exists(
        outputs["fit_baseline"], label="formal Stage-B fit-only baseline"
    )
    marker_exists = _regular_file_exists(
        outputs["phase_marker"], label="formal Stage-B phase marker"
    )
    phase_surface_exists = baseline_exists or marker_exists
    if marker_exists and not baseline_exists:
        raise RuntimeError(
            "formal Stage-B phase marker exists without its fit-only baseline"
        )
    if checkpoint_exists and not ledger_exists:
        raise RuntimeError(
            "formal Stage-B checkpoint exists without its external ledger"
        )
    reuse_training_pair = checkpoint_exists and ledger_exists
    if phase_surface_exists and not reuse_training_pair:
        raise RuntimeError(
            "formal Stage-B phase artifact exists without a complete training pair"
        )

    result: dict[str, Any] | None = None
    if not reuse_training_pair:
        result = run_r2_stage_b_training(
            model=components.model,
            batches=components.batches,
            replay_step=components.replay_step,
            preflight=lambda model: components.candidate_order_probe(),
            seed=args.seed,
            exposure_artifact=components.exposure_artifact,
            dense_checkpoint_path=checkpoint,
            dense_checkpoint_sha256=registered_checkpoint["sha256"],
            dense_checkpoint_use_ema=components.dense_checkpoint_use_ema,
            manifest_sha256=components.manifest["manifest_sha256"],
            library_sha256=registration["candidate_library"]["library_sha256"],
            config_sha256=components.config_sha256,
            output_checkpoint=outputs["output"],
            ledger_path=outputs["ledger"],
            resume_from=args.resume,
            checkpoint_frequency=args.checkpoint_frequency,
            registered_provenance=registered_provenance,
        )
        if result["status"] != "TRAINING_COMPLETE_BASELINE_PENDING":
            raise RuntimeError(
                "formal Stage-B training did not reach its baseline-pending state"
            )
    _, trained = _load_torch_regular_file(
        outputs["output"], label="formal Stage-B trained checkpoint"
    )
    if not isinstance(trained, Mapping) or not isinstance(
        trained.get("state_dict_ema"), Mapping
    ):
        raise ValueError("formal Stage-B checkpoint lacks a state_dict_ema mapping")
    components.model.load_state_dict(trained["state_dict_ema"], strict=True)
    predictor_state_sha256 = logical_risk_predictor_state_sha256(
        components.model, trained["state_dict_ema"]
    )
    baseline_rows = components.fit_baseline_rows()
    baseline = build_fit_schedule_constant_artifact(
        baseline_rows,
        seed=args.seed,
        fit_window_ids=components.batches.window_ids,
        candidate_action_sha256_by_name=candidate_action_sha256_by_name,
        provenance={
            "registration_sha256": registration["registration_sha256"],
            "manifest_sha256": components.manifest["manifest_sha256"],
            "library_sha256": registration["candidate_library"]["library_sha256"],
            "trained_checkpoint_sha256": _file_sha256(outputs["output"]),
            "predictor_state_sha256": predictor_state_sha256,
        },
    )
    phase_context = {
        "registration_sha256": registration["registration_sha256"],
        "registration_commit": registration_commit,
        "seed": args.seed,
        "model": components.model,
        "batches": components.batches,
        "exposure_artifact": components.exposure_artifact,
        "dense_checkpoint_path": checkpoint,
        "dense_checkpoint_sha256": registered_checkpoint["sha256"],
        "dense_checkpoint_use_ema": components.dense_checkpoint_use_ema,
        "registered_provenance": registered_provenance,
        "checkpoint_path": outputs["output"],
        "ledger_path": outputs["ledger"],
        "candidate_action_sha256_by_name": candidate_action_sha256_by_name,
        "manifest_sha256": components.manifest["manifest_sha256"],
        "library_sha256": registration["candidate_library"]["library_sha256"],
        "config_sha256": components.config_sha256,
    }
    baseline_bytes = canonical_json_bytes(baseline) + b"\n"
    handle, temporary_baseline = tempfile.mkstemp(
        prefix=".fit-baseline-preflight.",
        suffix=".json",
        dir=outputs["fit_baseline"].parent,
    )
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(baseline_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        preflight_marker = build_r2_stage_b_phase_completion_marker(
            **phase_context,
            fit_baseline_path=Path(temporary_baseline),
        )
    finally:
        if os.path.exists(temporary_baseline):
            os.unlink(temporary_baseline)

    if result is None:
        metadata = trained.get("meta")
        if not isinstance(metadata, Mapping):
            raise ValueError("formal Stage-B checkpoint lacks metadata")
        result = {
            "status": metadata["status"],
            "seed": metadata["seed"],
            "counters": dict(metadata["counters"]),
            "successful_update_cursor": metadata["successful_update_cursor"],
            "checkpoint_path": str(outputs["output"]),
            "checkpoint_sha256": preflight_marker["trained_checkpoint"][
                "exact_bytes_sha256"
            ],
            "ledger_path": str(outputs["ledger"]),
            "ledger_sha256": preflight_marker["ledger"]["exact_bytes_sha256"],
            "reused_existing_training_pair": True,
        }

    _atomic_write(outputs["fit_baseline"], baseline_bytes)
    phase_marker = build_r2_stage_b_phase_completion_marker(
        **phase_context,
        fit_baseline_path=outputs["fit_baseline"],
    )
    _atomic_write(outputs["phase_marker"], canonical_json_bytes(phase_marker) + b"\n")
    validate_r2_stage_b_phase_completion_marker(
        outputs["phase_marker"],
        **phase_context,
        fit_baseline_path=outputs["fit_baseline"],
    )
    result["training_status"] = result["status"]
    result["status"] = "PHASE_COMPLETE"
    result["fit_baseline_path"] = str(outputs["fit_baseline"])
    result["fit_baseline_sha256"] = baseline["artifact_sha256"]
    result["phase_marker_path"] = str(outputs["phase_marker"])
    result["phase_marker_sha256"] = phase_marker["artifact_sha256"]
    print(json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed repository-owned CT-P3R-3S-r2 formal Stage B"
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate1-unlock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-registry", type=Path, required=True)
    parser.add_argument("--config-identity", type=Path, required=True)
    parser.add_argument("--exposure-artifact", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3407, 3408, 3409), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--fit-baseline", type=Path, required=True)
    parser.add_argument("--phase-marker", type=Path, required=True)
    parser.add_argument(
        "--resume",
        type=Path,
        help="Only a formal periodic *.stepN.pth prefix; its matching .jsonl is required.",
    )
    parser.add_argument("--checkpoint-frequency", type=int, default=1)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
