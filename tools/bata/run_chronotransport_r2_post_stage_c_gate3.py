#!/usr/bin/env python3
"""Formal post-Stage-C replay, recalibration, and Gate-3 adjudication."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.filesystem import (
    exclusive_file_lock,
    open_bound_directory,
    path_exists_no_follow,
    publish_bytes_exclusive,
    read_bound_bytes,
    secure_lexical_path,
)
from opentad.models.chronotransport.gates23 import (
    R2_SEEDS,
    _finite_nonnegative,
    _validate_costs,
    load_exact_canonical_json,
    validate_gate3_unlock_artifact,
    validate_stage_b_phase_markers_static,
)
from opentad.models.chronotransport.post_stage_c import (
    adjudicate_post_stage_c_gate3,
    build_post_stage_c_gate3_terminal,
    build_post_stage_c_gate3_unlock,
    validate_post_stage_c_gate3_report,
    validate_post_stage_c_replay_artifact,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.registration import FORMAL_OUTPUT_BASE
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from tools.bata.chronotransport_r2_post_stage_c_factory import (
    build_registered_post_stage_c_replay_artifact,
    validate_completed_stage_c_population,
)
from tools.bata.run_chronotransport_r2_gates23 import _validate_gate1
from tools.bata.train_chronotransport_r2_stage_c import _load_formal_registration


POST_STAGE_C_REGISTERED_SOURCE_PATHS = (
    "opentad/models/chronotransport/post_stage_c.py",
    "tools/bata/chronotransport_r2_post_stage_c_factory.py",
    "tools/bata/run_chronotransport_r2_post_stage_c_gate3.py",
    "tools/bata/validate_chronotransport_r2_post_stage_c_gate3.py",
    "tests/test_chronotransport_r2_gates23.py",
)


@dataclass(frozen=True)
class PreparedPostStageC:
    registration: dict[str, Any]
    registration_commit: str
    registration_relpath: str
    gate1: dict[str, Any]
    gates23_replay: dict[str, Any]
    gates23_report: dict[str, Any]
    phase_marker_paths: dict[int, Path]
    fit_baseline_constants_by_seed: dict[str, dict[str, float]]
    candidate_cost_p50: dict[str, float]
    budget: float
    outputs: dict[str, Path]


def _assert_sources_registered(registration: Mapping[str, Any]) -> None:
    sources = registration.get("source_files")
    if not isinstance(sources, Mapping):
        raise ValueError("post-Stage-C registration source_files mapping is missing")
    with open_bound_directory(ROOT, label="post-Stage-C repository root") as root:
        for relative in POST_STAGE_C_REGISTERED_SOURCE_PATHS:
            expected = sources.get(relative)
            if not isinstance(expected, str):
                raise ValueError(f"post-Stage-C source is unregistered: {relative}")
            with root.open_regular(
                relative, label=f"registered post-Stage-C source {relative}"
            ) as source:
                actual = source.size_and_sha256()[1]
            if actual != expected:
                raise ValueError(
                    f"post-Stage-C source differs from registration R: {relative}"
                )


def _phase_paths(args: argparse.Namespace) -> dict[int, Path]:
    return {
        seed: secure_lexical_path(
            getattr(args, f"phase_marker_{seed}"),
            label=f"post-Stage-C phase marker {seed}",
        )
        for seed in R2_SEEDS
    }


def _resolve_outputs(registration_commit: str) -> dict[str, Path]:
    base = secure_lexical_path(
        FORMAL_OUTPUT_BASE,
        label="post-Stage-C output base",
        allow_missing=True,
    )
    root = secure_lexical_path(
        base / registration_commit / "shared" / "post_stage_c_gate3",
        label="post-Stage-C output root",
        allow_missing=True,
    )
    try:
        root.relative_to(base)
    except ValueError as error:
        raise ValueError("post-Stage-C output root escapes fixed base") from error
    outputs = {
        "root": root,
        "replay": root / "post_stage_c_replay.json",
        "report": root / "post_stage_c_gate3_report.json",
        "unlock": root / "post_stage_c_gate3_unlock.json",
        "terminal": root / "terminal_marker.json",
    }
    repository = secure_lexical_path(ROOT, label="post-Stage-C repository root")
    for path in outputs.values():
        try:
            path.relative_to(repository)
        except ValueError:
            continue
        raise ValueError("post-Stage-C outputs must remain outside the repository")
    return outputs


def _require_canonical_inputs(
    args: argparse.Namespace, *, registration_commit: str
) -> None:
    base = secure_lexical_path(
        FORMAL_OUTPUT_BASE,
        label="post-Stage-C canonical input base",
    )
    root = base / registration_commit / "shared"
    expected = {
        "gate1_unlock": root / "gate1" / "gate1_result.json",
        "gates23_replay": root / "gates23" / "gates23_replay.json",
        "gates23_report": root / "gates23" / "gates23_report.json",
    }
    for name, path in expected.items():
        actual = secure_lexical_path(
            getattr(args, name), label=f"post-Stage-C {name}"
        )
        if actual != path:
            raise ValueError(f"post-Stage-C {name} must use its canonical R path")


def _gate1_cost_contract(gate1: Mapping[str, Any]) -> tuple[dict[str, float], float]:
    result = gate1.get("gate1_result")
    if not isinstance(result, Mapping):
        raise ValueError("post-Stage-C Gate3 requires the exact Gate1 result")
    raw = result.get("candidate_cost_p50")
    if not isinstance(raw, Mapping):
        raise ValueError("post-Stage-C Gate3 lacks measured candidate p50 costs")
    costs = _validate_costs(
        {name: raw[name] for name in (*R2_NON_DENSE_NAMES, "dense")}
    )
    budget = _finite_nonnegative(result.get("budget"), "post-Stage-C Gate1 B*")
    if (
        result.get("budget_source") != "measured_p50:periodic4_transport"
        or budget != costs["periodic4_transport"]
    ):
        raise ValueError("post-Stage-C B* must equal measured periodic4 p50")
    return costs, budget


def _prepare_inputs(args: argparse.Namespace) -> PreparedPostStageC:
    registration, commit, relpath = _load_formal_registration(args.registration)
    _assert_sources_registered(registration)
    _require_canonical_inputs(args, registration_commit=commit)
    gate1 = _validate_gate1(
        args.gate1_unlock,
        registration=registration,
        repository_root=ROOT,
        registration_commit=commit,
        registration_relpath=relpath,
    )
    replay = load_exact_canonical_json(
        args.gates23_replay, label="pre-Stage-C Gates2/3 replay"
    )
    report = load_exact_canonical_json(
        args.gates23_report, label="pre-Stage-C Gates2/3 report"
    )
    phase_paths = _phase_paths(args)
    report = validate_gate3_unlock_artifact(
        report,
        replay_artifact=replay,
        registration=registration,
        gate1_unlock=gate1,
        phase_marker_paths=phase_paths,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=commit,
        registration_relpath=relpath,
    )
    _, baselines = validate_stage_b_phase_markers_static(
        phase_paths,
        registration=registration,
        gate1_unlock=gate1,
        gate1_unlock_path=args.gate1_unlock,
        repository_root=ROOT,
        registration_commit=commit,
        registration_relpath=relpath,
    )
    costs, budget = _gate1_cost_contract(gate1)
    return PreparedPostStageC(
        registration=registration,
        registration_commit=commit,
        registration_relpath=relpath,
        gate1=gate1,
        gates23_replay=replay,
        gates23_report=report,
        phase_marker_paths=phase_paths,
        fit_baseline_constants_by_seed=baselines,
        candidate_cost_p50=costs,
        budget=budget,
        outputs=_resolve_outputs(commit),
    )


def _validate_publication_state(outputs: Mapping[str, Path]) -> None:
    if path_exists_no_follow(outputs["terminal"], label="post-Stage-C terminal"):
        raise FileExistsError("post-Stage-C Gate3 terminal already exists for R")
    for name in ("replay", "report", "unlock"):
        path = outputs[name]
        if path_exists_no_follow(path, label=f"post-Stage-C {name}"):
            read_bound_bytes(path, label=f"existing post-Stage-C {name}")


def _publish_or_validate_exact(path: Path, value: Mapping[str, Any], *, label: str) -> None:
    try:
        publish_bytes_exclusive(
            path,
            canonical_json_bytes(value) + b"\n",
            label=label,
            allow_existing_exact=True,
        )
    except FileExistsError as error:
        raise ValueError(f"existing {label} differs from exact recomputation") from error


def _file_sha256(path: Path, *, label: str) -> str:
    return read_bound_bytes(path, label=label)[2]


def _publish_result(
    prepared: PreparedPostStageC,
    *,
    replay: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    outputs = prepared.outputs
    _publish_or_validate_exact(outputs["replay"], replay, label="post-Stage-C replay")
    _publish_or_validate_exact(outputs["report"], report, label="post-Stage-C report")
    unlock = None
    if report["status"] == "PASS":
        unlock = build_post_stage_c_gate3_unlock(report, replay)
        _publish_or_validate_exact(
            outputs["unlock"], unlock, label="post-Stage-C unlock"
        )
    elif path_exists_no_follow(outputs["unlock"], label="post-Stage-C unlock"):
        raise ValueError("FAIL post-Stage-C Gate3 cannot coexist with an unlock")
    terminal = build_post_stage_c_gate3_terminal(
        report=report,
        replay=replay,
        replay_path=str(outputs["replay"]),
        replay_file_sha256=_file_sha256(
            outputs["replay"], label="post-Stage-C replay"
        ),
        report_path=str(outputs["report"]),
        report_file_sha256=_file_sha256(
            outputs["report"], label="post-Stage-C report"
        ),
        unlock=unlock,
        unlock_path=str(outputs["unlock"]) if unlock is not None else None,
        unlock_file_sha256=(
            _file_sha256(outputs["unlock"], label="post-Stage-C unlock")
            if unlock is not None
            else None
        ),
    )
    publish_bytes_exclusive(
        outputs["terminal"],
        canonical_json_bytes(terminal) + b"\n",
        label="post-Stage-C terminal",
    )
    return {
        "schema": "chronotransport-r2-post-stage-c-gate3-run-result-v1",
        "status": report["status"],
        "registration_sha256": prepared.registration["registration_sha256"],
        "registration_commit": prepared.registration_commit,
        "report_path": str(outputs["report"]),
        "report_sha256": report["artifact_sha256"],
        "unlock_path": str(outputs["unlock"]) if unlock is not None else None,
        "terminal_path": str(outputs["terminal"]),
        "terminal_sha256": terminal["artifact_sha256"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare_inputs(args)
    if args.precheck_only:
        bindings = validate_completed_stage_c_population(
            registration_path=args.registration,
            registration=prepared.registration,
            registration_commit=prepared.registration_commit,
            gate1_unlock_path=args.gate1_unlock,
            gates23_replay_path=args.gates23_replay,
            gates23_report_path=args.gates23_report,
            phase_marker_paths=prepared.phase_marker_paths,
        )
        result = {
            "schema": "chronotransport-r2-post-stage-c-gate3-precheck-v1",
            "status": "PRECHECK_OK",
            "registration_sha256": prepared.registration["registration_sha256"],
            "registration_commit": prepared.registration_commit,
            "validated_stage_c_seeds": list(R2_SEEDS),
            "stage_c_bindings": bindings,
            "side_effect_free": True,
        }
        print(json.dumps(result, sort_keys=True))
        return result

    outputs = prepared.outputs
    with exclusive_file_lock(
        outputs["root"] / ".post-stage-c-gate3.run.lock",
        label="post-Stage-C Gate3 run lock",
        payload=f"pid={os.getpid()}\n".encode("ascii"),
    ):
        _validate_publication_state(outputs)
        replay = build_registered_post_stage_c_replay_artifact(
            registration_path=args.registration,
            registration=prepared.registration,
            registration_commit=prepared.registration_commit,
            gate1_unlock=prepared.gate1,
            gate1_unlock_path=args.gate1_unlock,
            pre_stage_c_gates23_replay=prepared.gates23_replay,
            gates23_replay_path=args.gates23_replay,
            pre_stage_c_gates23_report=prepared.gates23_report,
            gates23_report_path=args.gates23_report,
            phase_marker_paths=prepared.phase_marker_paths,
        )
        replay = validate_post_stage_c_replay_artifact(replay)
        report = adjudicate_post_stage_c_gate3(
            replay,
            candidate_cost_p50=prepared.candidate_cost_p50,
            budget=prepared.budget,
            fit_baseline_constants_by_seed=(
                prepared.fit_baseline_constants_by_seed
            ),
        )
        report = validate_post_stage_c_gate3_report(
            report,
            replay=replay,
            candidate_cost_p50=prepared.candidate_cost_p50,
            budget=prepared.budget,
            fit_baseline_constants_by_seed=(
                prepared.fit_baseline_constants_by_seed
            ),
        )
        result = _publish_result(prepared, replay=replay, report=report)
    print(json.dumps(result, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run formal post-Stage-C replay and Gate-3 recalibration"
    )
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate1-unlock", type=Path, required=True)
    parser.add_argument("--gates23-replay", type=Path, required=True)
    parser.add_argument("--gates23-report", type=Path, required=True)
    parser.add_argument("--phase-marker-3407", type=Path, required=True)
    parser.add_argument("--phase-marker-3408", type=Path, required=True)
    parser.add_argument("--phase-marker-3409", type=Path, required=True)
    parser.add_argument("--precheck-only", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    if result["status"] == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
