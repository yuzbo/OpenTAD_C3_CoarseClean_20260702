#!/usr/bin/env python3
"""Formal paired 4,200-success ChronoTransport Stage-C entrypoint."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping

import torch


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.environment import (
    observe_formal_slurm_environment,
    observed_environment_to_provenance,
)
from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    exclusive_file_lock,
    load_bound_json,
    load_bound_torch,
    path_exists_no_follow,
    publish_bytes_exclusive,
    read_bound_bytes,
    secure_lexical_path,
)
from opentad.models.chronotransport.formal_stage_c import (
    STAGE_C_COMPLETION_SCHEMA,
    STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    StageCInvalidImplementationError,
    build_paired_stage_c_state,
    build_stage_c_completion_marker,
    load_paired_stage_c_checkpoint,
    run_paired_stage_c_training,
    validate_paired_stage_c_checkpoint,
)
from opentad.models.chronotransport.gate1_unlock import (
    validate_gate1_unlock_artifact,
)
from opentad.models.chronotransport.gates23 import validate_gate3_unlock_artifact
from opentad.models.chronotransport.protocol import (
    canonical_json_bytes,
    canonical_sha256,
)
from opentad.models.chronotransport.registration import (
    FORMAL_OUTPUT_BASE,
    validate_pre_gate1_registration,
)
from tools.bata.chronotransport_r2_stage_c_factory import (
    build_repository_stage_c_components,
)


_PERIODIC_NAME = re.compile(r"^stage_c_paired\.step([1-9][0-9]*)\.pth$")


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _, value, _, _ = load_bound_json(path, label=label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON mapping")
    return dict(value)


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    if completed.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", value):
        raise ValueError("formal Stage C cannot resolve registration commit R")
    return value


def _load_formal_registration(path: Path):
    exact = secure_lexical_path(path, label="formal Stage-C registration")
    root = secure_lexical_path(ROOT, label="formal Stage-C repository root")
    try:
        relative = exact.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("formal Stage-C registration must be repository-local") from error
    commit = _git_head(ROOT)
    registration = validate_pre_gate1_registration(
        _load_json(exact, label="formal Stage-C registration"),
        repository_root=ROOT,
        context_mode="formal",
        registration_commit=commit,
        registration_relpath=relative,
    )
    return registration, commit, relative


def _phase_paths(args: argparse.Namespace) -> dict[int, Path]:
    return {
        seed: secure_lexical_path(
            getattr(args, f"phase_marker_{seed}"),
            label=f"formal Stage-C Stage-B phase marker {seed}",
        )
        for seed in (3407, 3408, 3409)
    }


def _resolve_outputs(
    args: argparse.Namespace,
    *,
    registration_commit: str,
) -> dict[str, Path]:
    base = secure_lexical_path(
        FORMAL_OUTPUT_BASE,
        label="formal Stage-C output base",
        allow_missing=True,
    )
    root = secure_lexical_path(
        base / registration_commit / str(args.seed) / "stage_c",
        label="formal Stage-C canonical seed root",
        allow_missing=True,
    )
    expected = {
        "output": root / "stage_c_paired_complete.pth",
        "ledger": root / "stage_c_paired_ledger.jsonl",
        "terminal": root / "stage_c_paired_terminal.json",
    }
    resolved = {
        name: secure_lexical_path(
            getattr(args, name),
            label=f"formal Stage-C {name}",
            allow_missing=True,
        )
        for name in expected
    }
    if resolved != expected:
        raise ValueError("formal Stage-C outputs must use the canonical R/seed/stage_c paths")
    if len(set(resolved.values())) != len(resolved):
        raise ValueError("formal Stage-C output paths must be distinct")
    repository = secure_lexical_path(ROOT, label="formal Stage-C repository root")
    for path in resolved.values():
        try:
            path.relative_to(repository)
        except ValueError:
            pass
        else:
            raise ValueError("formal Stage-C outputs must remain outside the repository")
    if args.resume is not None:
        resume = secure_lexical_path(
            args.resume,
            label="formal Stage-C resume checkpoint",
        )
        match = _PERIODIC_NAME.fullmatch(resume.name)
        if resume.parent != root or match is None or int(match.group(1)) % 70 != 0:
            raise ValueError("Stage-C resume must be a canonical epoch checkpoint")
        resolved["resume"] = resume
    return resolved


def _ledger_bytes(checkpoint: Mapping[str, Any]) -> bytes:
    return b"".join(
        canonical_json_bytes(row) + b"\n" for row in checkpoint["paired_trace"]
    )


def _torch_bytes(value: Any) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _periodic_path(final_output: Path, cursor: int) -> Path:
    return final_output.with_name(f"stage_c_paired.step{cursor}.pth")


def _publish_checkpoint_pair(path: Path, checkpoint: Mapping[str, Any]) -> None:
    sidecar = path.with_suffix(".jsonl")
    checkpoint_bytes = _torch_bytes(checkpoint)
    ledger_bytes = _ledger_bytes(checkpoint)
    publish_bytes_exclusive(
        path,
        checkpoint_bytes,
        label="formal Stage-C periodic checkpoint",
    )
    publish_bytes_exclusive(
        sidecar,
        ledger_bytes,
        label="formal Stage-C periodic ledger",
        allow_existing_exact=True,
    )


def _load_resume(
    path: Path,
    *,
    state,
    seed: int,
    fit_window_ids,
    provenance,
) -> None:
    _, checkpoint, _, _ = load_bound_torch(
        path, label="formal Stage-C resume checkpoint"
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("formal Stage-C resume checkpoint must be a mapping")
    validate_paired_stage_c_checkpoint(
        checkpoint,
        expected_seed=seed,
        expected_fit_window_ids=fit_window_ids,
        expected_provenance=provenance,
        formal=True,
        require_complete=False,
        expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    )
    expected_sidecar = _ledger_bytes(checkpoint)
    sidecar = path.with_suffix(".jsonl")
    if path_exists_no_follow(sidecar, label="formal Stage-C resume ledger"):
        _, actual_sidecar, _ = read_bound_bytes(
            sidecar, label="formal Stage-C resume ledger"
        )
        if actual_sidecar != expected_sidecar:
            raise ValueError("formal Stage-C resume ledger differs from checkpoint trace")
    else:
        # A crash between the two exclusive publications is recoverable.  The
        # checkpoint is authoritative and deterministically reconstructs the
        # missing sidecar; no successful update is replayed to repair it.
        publish_bytes_exclusive(
            sidecar,
            expected_sidecar,
            label="formal Stage-C recovered resume ledger",
        )
    load_paired_stage_c_checkpoint(
        state,
        checkpoint,
        expected_seed=seed,
        expected_fit_window_ids=fit_window_ids,
        expected_provenance=provenance,
        formal=True,
        expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    )


def _provenance(
    *,
    registration: Mapping[str, Any],
    registration_commit: str,
    gate1: Mapping[str, Any],
    gates23_replay: Mapping[str, Any],
    gates23_report: Mapping[str, Any],
    phase_markers: Mapping[int, Mapping[str, Any]],
    components,
    observed_environment: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "registration_sha256": registration["registration_sha256"],
        "registration_commit": registration_commit,
        "implementation_commit": registration["implementation_commit"],
        "spec_commit": registration["spec"]["commit"],
        "spec_sha256": registration["spec"]["sha256"],
        "source_files_sha256": canonical_sha256(registration["source_files"]),
        "upstream_commits_sha256": canonical_sha256(registration["upstream_commits"]),
        "manifest_sha256": components.manifest["manifest_sha256"],
        "config_sha256": components.config_sha256,
        "library_sha256": registration["candidate_library"]["library_sha256"],
        "cost_profile_sha256": components.cost_profile_sha256,
        "gate1_unlock_sha256": gate1["artifact_sha256"],
        "gates23_replay_sha256": gates23_replay["artifact_sha256"],
        "gates23_report_sha256": gates23_report["artifact_sha256"],
        "phase_marker_sha256_by_seed": {
            str(seed): phase_markers[seed]["artifact_sha256"]
            for seed in (3407, 3408, 3409)
        },
        "stage_b_checkpoint": dict(components.stage_b_checkpoint_identity),
        "dense_checkpoint": dict(components.dense_checkpoint_identity),
        **observed_environment_to_provenance(
            observed_environment,
            required_environment=registration["environment"],
        ),
    }


def _publish_final(
    *,
    outputs: Mapping[str, Path],
    checkpoint: Mapping[str, Any],
    seed: int,
    fit_window_ids,
    provenance,
) -> dict[str, Any]:
    output = outputs["output"]
    if path_exists_no_follow(output, label="formal Stage-C final checkpoint"):
        _, existing, checkpoint_bytes, checkpoint_sha = load_bound_torch(
            output, label="formal Stage-C final checkpoint"
        )
        if not isinstance(existing, Mapping):
            raise ValueError("formal Stage-C final checkpoint must be a mapping")
        validate_paired_stage_c_checkpoint(
            existing,
            expected_seed=seed,
            expected_fit_window_ids=fit_window_ids,
            expected_provenance=provenance,
            formal=True,
            require_complete=True,
            expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
        )
        checkpoint = existing
    else:
        checkpoint_bytes = _torch_bytes(checkpoint)
        publish_bytes_exclusive(
            output,
            checkpoint_bytes,
            label="formal Stage-C final checkpoint",
        )
        checkpoint_sha = hashlib.sha256(checkpoint_bytes).hexdigest()
    ledger_bytes = _ledger_bytes(checkpoint)
    publish_bytes_exclusive(
        outputs["ledger"],
        ledger_bytes,
        label="formal Stage-C final ledger",
        allow_existing_exact=True,
    )
    ledger_sha = hashlib.sha256(ledger_bytes).hexdigest()
    marker = build_stage_c_completion_marker(
        checkpoint,
        checkpoint_path=str(output),
        checkpoint_file_sha256=checkpoint_sha,
        ledger_path=str(outputs["ledger"]),
        ledger_file_sha256=ledger_sha,
    )
    publish_bytes_exclusive(
        outputs["terminal"],
        canonical_json_bytes(marker) + b"\n",
        label="formal Stage-C terminal",
        allow_existing_exact=True,
    )
    return marker


def _prepare(
    args: argparse.Namespace,
    *,
    entrypoint_relative: str = "tools/bata/train_chronotransport_r2_stage_c.py",
):
    registration, registration_commit, registration_relpath = (
        _load_formal_registration(args.registration)
    )
    outputs = _resolve_outputs(args, registration_commit=registration_commit)
    observed = observe_formal_slurm_environment(registration["environment"])
    gate1 = _load_json(args.gate1_unlock, label="formal Stage-C Gate1 unlock")
    gate1 = validate_gate1_unlock_artifact(
        gate1,
        registration=registration,
        repository_root=str(ROOT),
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    gates23_replay = _load_json(
        args.gates23_replay, label="formal Stage-C Gates2/3 replay"
    )
    gates23_report = _load_json(
        args.gates23_report, label="formal Stage-C Gates2/3 report"
    )
    phase_paths = _phase_paths(args)
    phase_markers = {
        seed: _load_json(path, label=f"formal Stage-C phase marker {seed}")
        for seed, path in phase_paths.items()
    }
    gates23_report = validate_gate3_unlock_artifact(
        gates23_report,
        replay_artifact=gates23_replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=phase_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    components = build_repository_stage_c_components(
        registration=registration,
        manifest_path=args.manifest,
        media_registry_path=args.media_registry,
        config_identity_path=args.config_identity,
        seed=args.seed,
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
        gate1_unlock=gate1,
        stage_b_phase_marker=phase_markers[args.seed],
    )
    audit_formal_python_runtime(
        repository_root=ROOT,
        registered_sources=registration["source_files"],
        entrypoint_relative=entrypoint_relative,
    )
    provenance = _provenance(
        registration=registration,
        registration_commit=registration_commit,
        gate1=gate1,
        gates23_replay=gates23_replay,
        gates23_report=gates23_report,
        phase_markers=phase_markers,
        components=components,
        observed_environment=observed,
    )
    state = build_paired_stage_c_state(
        components.ct_model, components.matched_model
    )
    for model in (components.ct_model, components.matched_model):
        runtime = next(
            module
            for module in model.modules()
            if module.__class__.__name__ == "ChronoTransportRuntime"
        )
        if (
            runtime.scheduler.registered_cost_profile_sha256
            != components.cost_profile_sha256
            or runtime.cost_is_measured is not True
            or runtime.nonlinear_cost_ready is not True
        ):
            raise RuntimeError("formal Stage-C direct cost provenance is not installed")
    return (
        registration,
        registration_commit,
        outputs,
        components,
        provenance,
        state,
    )


def run(
    args: argparse.Namespace,
    *,
    entrypoint_relative: str = "tools/bata/train_chronotransport_r2_stage_c.py",
) -> dict[str, Any]:
    prepared = _prepare(args, entrypoint_relative=entrypoint_relative)
    registration, registration_commit, outputs, components, provenance, state = prepared
    try:
        if args.precheck_only:
            result = {
                "schema": "chronotransport-r2-stage-c-precheck-v1",
                "status": "PRECHECK_OK",
                "registration_sha256": registration["registration_sha256"],
                "registration_commit": registration_commit,
                "seed": args.seed,
                "fit_windows": len(components.fit_window_ids),
                "cost_profile_sha256": components.cost_profile_sha256,
                "side_effect_free": True,
            }
            print(json.dumps(result, sort_keys=True))
            return result

        lock_path = outputs["output"].parent / ".chronotransport-r2-stage-c.lock"
        with exclusive_file_lock(
            lock_path,
            label="formal Stage-C run lock",
            payload=f"pid={os.getpid()}\n".encode("ascii"),
        ):
            if "resume" in outputs:
                _load_resume(
                    outputs["resume"],
                    state=state,
                    seed=args.seed,
                    fit_window_ids=components.fit_window_ids,
                    provenance=provenance,
                )

            def checkpoint_sink(cursor: int, checkpoint: Mapping[str, Any]) -> None:
                _publish_checkpoint_pair(
                    _periodic_path(outputs["output"], cursor), checkpoint
                )

            result = run_paired_stage_c_training(
                state,
                materialize_batch=components.materialize_batch,
                fit_window_ids=components.fit_window_ids,
                seed=args.seed,
                provenance=provenance,
                formal=True,
                checkpoint_sink=checkpoint_sink,
            )
            marker = _publish_final(
                outputs=outputs,
                checkpoint=result["checkpoint"],
                seed=args.seed,
                fit_window_ids=components.fit_window_ids,
                provenance=provenance,
            )
            summary = {
                key: value for key, value in result.items() if key != "checkpoint"
            }
            summary.update(
                terminal_path=str(outputs["terminal"]),
                terminal_sha256=marker["artifact_sha256"],
            )
            print(json.dumps(summary, sort_keys=True))
            return summary
    finally:
        components.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run fixed formal paired CT/matched-dense Stage C"
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate1-unlock", type=Path, required=True)
    parser.add_argument("--gates23-replay", type=Path, required=True)
    parser.add_argument("--gates23-report", type=Path, required=True)
    parser.add_argument("--phase-marker-3407", type=Path, required=True)
    parser.add_argument("--phase-marker-3408", type=Path, required=True)
    parser.add_argument("--phase-marker-3409", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--media-registry", type=Path, required=True)
    parser.add_argument("--config-identity", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(3407, 3408, 3409), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--terminal", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--precheck-only", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
