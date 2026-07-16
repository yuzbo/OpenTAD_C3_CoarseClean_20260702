#!/usr/bin/env python3
"""Fail-closed precheck for the registered ChronoTransport r2 Gate-1 chain."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.adjudication import validate_gate1_record_artifact
from opentad.models.chronotransport.environment import (
    observe_formal_slurm_environment,
)
from opentad.models.chronotransport.full_stack_profiler import (
    validate_full_stack_profile_artifact,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes, canonical_sha256
from opentad.models.chronotransport.registration import (
    resolve_gate1_output_root,
    validate_pre_gate1_registration,
)


GATE1_INPUT_FILENAME = "gate1_input.json"
GATE1_RESULT_FILENAME = "gate1_result.json"
GATE1_TERMINAL_FILENAME = "gate1_terminal.json"


def _path_without_symlink_components(
    path: str | Path, *, label: str, allow_missing: bool = False
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


def _load_json_no_duplicates(path: Path) -> Any:
    def reject(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject)


def load_exact_registration(path: str | Path) -> dict[str, object]:
    path = _path_without_symlink_components(path, label="registration artifact")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError("registration file must be a regular non-symlink file")
    registration = _load_json_no_duplicates(path)
    if not isinstance(registration, dict):
        raise TypeError("registration file root must be an object")
    if path.read_bytes() != canonical_json_bytes(registration) + b"\n":
        raise ValueError("registration file bytes are not exact canonical bytes")
    return registration


def _validate_gate1_input_payload(
    path: Path,
    *,
    registration: dict[str, object],
    repository_root: Path,
    registration_commit: str,
    registration_relpath: str,
) -> dict[str, object]:
    path = _path_without_symlink_components(path, label="Gate 1 input")
    if not stat.S_ISREG(os.lstat(path).st_mode):
        raise ValueError("Gate 1 input must be a regular exact artifact")
    payload = _load_json_no_duplicates(path)
    if path.read_bytes() != canonical_json_bytes(payload) + b"\n":
        raise ValueError("Gate 1 input file bytes are not exact canonical bytes")
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "registration",
        "calibration",
        "evaluation",
        "full_stack_profile",
    }:
        raise ValueError("Gate 1 input artifact fields mismatch")
    if payload["schema"] != "chronotransport-r2-gate1-input-v3":
        raise ValueError("unsupported Gate 1 input schema")
    if payload["registration"] != registration:
        raise ValueError("Gate 1 input does not embed the exact formal registration")
    context = {
        "repository_root": str(repository_root),
        "registration_commit": registration_commit,
        "registration_relpath": registration_relpath,
    }
    profile = validate_full_stack_profile_artifact(
        payload["full_stack_profile"], registration=registration, **context
    )
    calibration = validate_gate1_record_artifact(
        payload["calibration"],
        registration=registration,
        expected_split="calibration",
        **context,
    )
    evaluation = validate_gate1_record_artifact(
        payload["evaluation"],
        registration=registration,
        expected_split="evaluation",
        **context,
    )
    return {
        "gate1_input_sha256": canonical_sha256(payload),
        "profile_sha256": profile["profile_sha256"],
        "calibration_artifact_sha256": calibration["artifact_sha256"],
        "evaluation_artifact_sha256": evaluation["artifact_sha256"],
    }


def _validate_precheck(
    *,
    registration_path: str | Path,
    repository_root: str | Path,
    registration_commit: str,
    output_root: str | Path,
    gate1_input_path: str | Path | None = None,
    gate1_output_path: str | Path | None = None,
    terminal_marker_path: str | Path | None = None,
    allowed_output_root: str | Path = "/data/run01/sczc063/yuzibo",
    observe_environment: bool,
) -> dict[str, object]:
    registration_path = _path_without_symlink_components(
        registration_path, label="registration artifact"
    )
    repository_root = _path_without_symlink_components(
        repository_root, label="repository root"
    )
    registration = load_exact_registration(registration_path)
    try:
        registration_relpath = registration_path.relative_to(repository_root).as_posix()
    except ValueError as exc:
        raise ValueError("registration artifact must be inside the repository at R") from exc
    validated = validate_pre_gate1_registration(
        registration,
        repository_root=repository_root,
        context_mode="formal",
        registration_commit=registration_commit,
        registration_relpath=registration_relpath,
    )
    output = _path_without_symlink_components(
        output_root, label="Gate 1 output root", allow_missing=True
    )
    allowed = _path_without_symlink_components(
        allowed_output_root, label="allowed output root", allow_missing=True
    )
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("resolved output root escapes the registered allowed root") from exc

    registered_output = _path_without_symlink_components(
        resolve_gate1_output_root(validated, registration_commit),
        label="R-derived Gate 1 output root",
        allow_missing=True,
    )
    output_spec = validated.get("output_root")
    if isinstance(output_spec, dict) and set(output_spec) == {"base", "template"}:
        if output_spec["template"] != "{base}/{registration_commit}/shared/gate1":
            raise ValueError("registration output root template mismatch")
        registered_base = _path_without_symlink_components(
            output_spec["base"],
            label="registered output base",
            allow_missing=True,
        )
        expected_registered_output = _path_without_symlink_components(
            registered_base / registration_commit / "shared" / "gate1",
            label="registered Gate 1 output root",
            allow_missing=True,
        )
        if registered_output != expected_registered_output:
            raise ValueError("R-derived output root was altered by path resolution")
    else:
        # The formal validator always returns the complete registration. This
        # fallback only keeps isolated validator mocks useful in focused tests.
        expected_registered_output = registered_output
    if output != expected_registered_output:
        raise ValueError("resolved output root differs from R-derived registration template")
    supplied = (gate1_input_path, gate1_output_path, terminal_marker_path)
    if any(item is not None for item in supplied) and not all(
        item is not None for item in supplied
    ):
        raise ValueError("Gate 1 input/result/terminal paths must be supplied together")
    canonical_paths: dict[str, Path] = {}
    if all(item is not None for item in supplied):
        expected = {
            "input": output / GATE1_INPUT_FILENAME,
            "result": output / GATE1_RESULT_FILENAME,
            "terminal": output / GATE1_TERMINAL_FILENAME,
        }
        actual = {
            "input": Path(gate1_input_path),
            "result": Path(gate1_output_path),
            "terminal": Path(terminal_marker_path),
        }
        for name in expected:
            candidate = _path_without_symlink_components(
                actual[name],
                label=f"Gate 1 {name}",
                allow_missing=True,
            )
            if candidate != expected[name]:
                raise ValueError(f"Gate 1 {name} must use its fixed canonical filename")
            canonical_paths[name] = candidate
        if len(set(canonical_paths.values())) != 3:
            raise ValueError("Gate 1 input/result/terminal artifacts must be distinct")
        if canonical_paths["input"] == registration_path:
            raise ValueError("Gate 1 input and registration artifacts must be distinct")
        for name in ("result", "terminal"):
            if canonical_paths[name].exists():
                raise ValueError(
                    f"Gate 1 {name} already exists; formal Gate 1 has no resume mode"
                )
    report = {
        "schema": (
            "chronotransport-r2-precheck-report-v2"
            if observe_environment
            else "chronotransport-r2-precheck-report-test-fixture-v1"
        ),
        "registration_sha256": validated["registration_sha256"],
        "implementation_commit": validated["implementation_commit"],
        "registration_commit": registration_commit,
        "resolved_output_root": str(output),
        "status": "PRECHECK_OK",
    }
    if observe_environment:
        report["observed_environment"] = observe_formal_slurm_environment(
            validated["environment"]
        )
    if canonical_paths:
        report.update(
            _validate_gate1_input_payload(
                canonical_paths["input"],
                registration=validated,
                repository_root=repository_root,
                registration_commit=registration_commit,
                registration_relpath=registration_relpath,
            )
        )
    return report


def validate_precheck(
    *,
    registration_path: str | Path,
    repository_root: str | Path,
    registration_commit: str,
    output_root: str | Path,
    gate1_input_path: str | Path | None = None,
    gate1_output_path: str | Path | None = None,
    terminal_marker_path: str | Path | None = None,
    allowed_output_root: str | Path = "/data/run01/sczc063/yuzibo",
) -> dict[str, object]:
    """Validate static inputs and record the live Slurm allocation identity."""

    return _validate_precheck(
        registration_path=registration_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        output_root=output_root,
        gate1_input_path=gate1_input_path,
        gate1_output_path=gate1_output_path,
        terminal_marker_path=terminal_marker_path,
        allowed_output_root=allowed_output_root,
        observe_environment=True,
    )


def validate_precheck_for_test_only(
    *,
    registration_path: str | Path,
    repository_root: str | Path,
    registration_commit: str,
    output_root: str | Path,
    gate1_input_path: str | Path | None = None,
    gate1_output_path: str | Path | None = None,
    terminal_marker_path: str | Path | None = None,
    allowed_output_root: str | Path = "/data/run01/sczc063/yuzibo",
) -> dict[str, object]:
    """Run filesystem/registration checks without claiming a formal GPU precheck."""

    return _validate_precheck(
        registration_path=registration_path,
        repository_root=repository_root,
        registration_commit=registration_commit,
        output_root=output_root,
        gate1_input_path=gate1_input_path,
        gate1_output_path=gate1_output_path,
        terminal_marker_path=terminal_marker_path,
        allowed_output_root=allowed_output_root,
        observe_environment=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--registration-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate1-input", type=Path)
    parser.add_argument("--gate1-output", type=Path)
    parser.add_argument("--terminal-marker", type=Path)
    parser.add_argument(
        "--allowed-output-root",
        type=Path,
        default=Path("/data/run01/sczc063/yuzibo"),
    )
    args = parser.parse_args()
    report = validate_precheck(
        registration_path=args.registration,
        repository_root=args.repository_root,
        registration_commit=args.registration_commit,
        output_root=args.output_root,
        gate1_input_path=args.gate1_input,
        gate1_output_path=args.gate1_output,
        terminal_marker_path=args.terminal_marker,
        allowed_output_root=args.allowed_output_root,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
