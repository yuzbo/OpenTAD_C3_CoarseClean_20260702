#!/usr/bin/env python3
"""Run resumable formal Gate-4 seed shards and final adjudication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.filesystem import (
    audit_formal_python_runtime,
    exclusive_file_lock,
    path_exists_no_follow,
    publish_bytes_exclusive,
    read_bound_bytes,
    secure_lexical_path,
)
from opentad.models.chronotransport.environment import validate_observed_environment
from opentad.models.chronotransport.formal_gate4 import (
    adjudicate_formal_gate4,
    build_gate4_terminal,
    validate_gate4_seed_shard,
)
from opentad.models.chronotransport.gates23 import (
    SCHEDULER_EPSILON,
    load_exact_canonical_json,
)
from opentad.models.chronotransport.post_stage_c import (
    build_post_stage_c_gate3_terminal,
    build_post_stage_c_gate3_unlock,
    validate_post_stage_c_gate3_report,
    validate_post_stage_c_replay_artifact,
)
from opentad.models.chronotransport.protocol import canonical_json_bytes
from opentad.models.chronotransport.registration import FORMAL_OUTPUT_BASE
from tools.bata.chronotransport_r2_gate4_factory import (
    build_formal_gate4_evidence,
    build_formal_gate4_seed_shard,
    precheck_formal_gate4_seed,
)
from tools.bata.run_chronotransport_r2_post_stage_c_gate3 import (
    _prepare_inputs as prepare_pre_stage_c_inputs,
)


def _canonical_paths(registration_commit: str) -> dict[str, Any]:
    base = secure_lexical_path(
        FORMAL_OUTPUT_BASE, label="formal Gate4 output base", allow_missing=True
    )
    shared = secure_lexical_path(
        base / registration_commit / "shared" / "gate4",
        label="formal Gate4 shared output root",
        allow_missing=True,
    )
    seeds = {
        seed: secure_lexical_path(
            base / registration_commit / str(seed) / "gate4" / "gate4_seed_shard.json",
            label=f"formal Gate4 seed {seed} shard",
            allow_missing=True,
        )
        for seed in (3407, 3408, 3409)
    }
    return {
        "shared": shared,
        "seeds": seeds,
        "timing": shared / "gate4_timing_evidence.json",
        "metric": shared / "gate4_metric_evidence.json",
        "regret": shared / "gate4_regret_evidence.json",
        "report": shared / "gate4_report.json",
        "terminal": shared / "terminal_marker.json",
    }


def _load_post_stage_c(prepared, args: argparse.Namespace):
    canonical_root = secure_lexical_path(
        Path(FORMAL_OUTPUT_BASE)
        / prepared.registration_commit
        / "shared"
        / "post_stage_c_gate3",
        label="formal Gate4 canonical post-Stage-C root",
    )
    expected = {
        "post_stage_c_replay": canonical_root / "post_stage_c_replay.json",
        "post_stage_c_report": canonical_root / "post_stage_c_gate3_report.json",
        "post_stage_c_unlock": canonical_root / "post_stage_c_gate3_unlock.json",
        "post_stage_c_terminal": canonical_root / "terminal_marker.json",
    }
    for field, path in expected.items():
        if secure_lexical_path(getattr(args, field), label=f"formal Gate4 {field}") != path:
            raise ValueError(f"formal Gate4 {field} must use its canonical R path")
    replay = validate_post_stage_c_replay_artifact(
        load_exact_canonical_json(
            expected["post_stage_c_replay"],
            label="formal Gate4 post-Stage-C replay",
        )
    )
    report = validate_post_stage_c_gate3_report(
        load_exact_canonical_json(
            expected["post_stage_c_report"],
            label="formal Gate4 post-Stage-C report",
        ),
        replay=replay,
        candidate_cost_p50=prepared.candidate_cost_p50,
        budget=prepared.budget,
        fit_baseline_constants_by_seed=prepared.fit_baseline_constants_by_seed,
    )
    unlock = load_exact_canonical_json(
        expected["post_stage_c_unlock"], label="formal Gate4 post-Stage-C unlock"
    )
    if unlock != build_post_stage_c_gate3_unlock(report, replay):
        raise ValueError("formal Gate4 post-Stage-C unlock differs from recomputation")
    terminal = load_exact_canonical_json(
        expected["post_stage_c_terminal"], label="formal Gate4 post-Stage-C terminal"
    )
    expected_terminal = build_post_stage_c_gate3_terminal(
        report=report,
        replay=replay,
        replay_path=str(expected["post_stage_c_replay"]),
        replay_file_sha256=read_bound_bytes(
            expected["post_stage_c_replay"], label="formal Gate4 post replay bytes"
        )[2],
        report_path=str(expected["post_stage_c_report"]),
        report_file_sha256=read_bound_bytes(
            expected["post_stage_c_report"], label="formal Gate4 post report bytes"
        )[2],
        unlock=unlock,
        unlock_path=str(expected["post_stage_c_unlock"]),
        unlock_file_sha256=read_bound_bytes(
            expected["post_stage_c_unlock"], label="formal Gate4 post unlock bytes"
        )[2],
    )
    if terminal != expected_terminal or terminal["status"] != "SUCCESS":
        raise ValueError("formal Gate4 requires the exact successful post-Stage-C terminal")
    return replay, report, unlock


def _publish_exact(path: Path, value: Mapping[str, Any], *, label: str) -> str:
    payload = canonical_json_bytes(value) + b"\n"
    publish_bytes_exclusive(path, payload, label=label, allow_existing_exact=True)
    return hashlib.sha256(payload).hexdigest()


def _load_seed_shards(paths: Mapping[int, Path]) -> dict[str, dict[str, Any]]:
    return {
        str(seed): validate_gate4_seed_shard(
            load_exact_canonical_json(path, label=f"formal Gate4 seed shard {seed}")
        )
        for seed, path in paths.items()
    }


def _prepare(args: argparse.Namespace):
    prepared = prepare_pre_stage_c_inputs(args)
    audit_formal_python_runtime(
        repository_root=ROOT,
        registered_sources=prepared.registration["source_files"],
        entrypoint_relative="tools/bata/run_chronotransport_r2_gate4.py",
    )
    replay, report, unlock = _load_post_stage_c(prepared, args)
    return prepared, replay, report, unlock, _canonical_paths(
        prepared.registration_commit
    )


def _run_seed(args: argparse.Namespace, prepared, replay, report, unlock, paths):
    seed = int(args.seed)
    if seed not in (3407, 3408, 3409):
        raise ValueError("formal Gate4 seed must be 3407, 3408, or 3409")
    kwargs = {
        "registration_path": args.registration,
        "registration": prepared.registration,
        "registration_commit": prepared.registration_commit,
        "registration_relpath": prepared.registration_relpath,
        "gate1_unlock": prepared.gate1,
        "gate1_unlock_path": args.gate1_unlock,
        "pre_stage_c_gates23_replay_path": args.gates23_replay,
        "pre_stage_c_gates23_report_path": args.gates23_report,
        "phase_marker_paths": prepared.phase_marker_paths,
        "post_stage_c_replay": replay,
        "post_stage_c_report": report,
        "post_stage_c_unlock": unlock,
        "seed": seed,
    }
    if args.precheck_only:
        result = precheck_formal_gate4_seed(**kwargs)
        print(json.dumps(result, sort_keys=True))
        return result
    output = paths["seeds"][seed]
    if path_exists_no_follow(output, label=f"formal Gate4 seed {seed} shard"):
        shard = validate_gate4_seed_shard(
            load_exact_canonical_json(output, label=f"existing Gate4 seed shard {seed}")
        )
        expected_scheduler = {
            "budget": float(prepared.gate1["gate1_result"]["budget"]),
            "epsilon": float(SCHEDULER_EPSILON),
            "calibration_frozen_static": str(
                prepared.gate1["gate1_result"]["calibration_frozen_static"]
            ),
            "q_conf_by_seed": {
                str(item): float(unlock["q_conf_by_seed"][str(item)])
                for item in (3407, 3408, 3409)
            },
            "gate1_unlock_artifact_sha256": prepared.gate1["artifact_sha256"],
            "calibration_sha256": unlock["post_stage_c_gate3_report_sha256"],
        }
        if (
            shard["seed"] != seed
            or shard["registration_sha256"]
            != prepared.registration["registration_sha256"]
            or shard["registration_commit"] != prepared.registration_commit
            or shard["population_artifact_sha256"]
            != prepared.registration["gate4_population"]["artifact"]["artifact_sha256"]
            or shard["post_stage_c_gate3_unlock_sha256"] != unlock["artifact_sha256"]
            or shard["stage_c_binding"] != unlock["stage_c_bindings"][str(seed)]
            or shard["scheduler_contract"] != expected_scheduler
        ):
            raise ValueError("existing formal Gate4 seed shard belongs to another chain")
        validate_observed_environment(
            shard["observed_environment"],
            required_environment=prepared.registration["environment"],
        )
    else:
        shard = build_formal_gate4_seed_shard(**kwargs)
        _publish_exact(output, shard, label=f"formal Gate4 seed {seed} shard")
    result = {
        "status": "SEED_COMPLETE",
        "seed": seed,
        "registration_sha256": prepared.registration["registration_sha256"],
        "seed_shard_artifact_sha256": shard["artifact_sha256"],
        "seed_shard_path": str(output),
        "gate4_adjudicated": False,
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _finalize(prepared, replay, report, unlock, paths):
    shards = _load_seed_shards(paths["seeds"])
    evidence = build_formal_gate4_evidence(
        registration=prepared.registration,
        repository_root=ROOT,
        registration_commit=prepared.registration_commit,
        registration_relpath=prepared.registration_relpath,
        post_stage_c_unlock=unlock,
        post_stage_c_report=report,
        post_stage_c_replay=replay,
        gate1_unlock=prepared.gate1,
        seed_shards=shards,
    )
    file_hashes = {
        name: _publish_exact(
            paths[name], evidence[name], label=f"formal Gate4 {name} evidence"
        )
        for name in ("timing", "metric", "regret")
    }
    gate4_report = adjudicate_formal_gate4(
        timing_evidence=evidence["timing"],
        metric_evidence=evidence["metric"],
        regret_evidence=evidence["regret"],
        registration=prepared.registration,
        population=prepared.registration["gate4_population"]["artifact"],
        post_stage_c_unlock=unlock,
        post_stage_c_report=report,
        post_stage_c_replay=replay,
        gate1_unlock=prepared.gate1,
        seed_shards=shards,
        repository_root=str(ROOT),
        registration_commit=prepared.registration_commit,
        registration_relpath=prepared.registration_relpath,
    )
    report_file_sha = _publish_exact(
        paths["report"], gate4_report, label="formal Gate4 report"
    )
    terminal = build_gate4_terminal(
        report=gate4_report,
        timing_path=str(paths["timing"]),
        timing_file_sha256=file_hashes["timing"],
        metric_path=str(paths["metric"]),
        metric_file_sha256=file_hashes["metric"],
        regret_path=str(paths["regret"]),
        regret_file_sha256=file_hashes["regret"],
        report_path=str(paths["report"]),
        report_file_sha256=report_file_sha,
    )
    _publish_exact(paths["terminal"], terminal, label="formal Gate4 terminal")
    print(json.dumps(terminal, sort_keys=True))
    return terminal


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.finalize) == (args.seed is not None):
        raise ValueError("choose exactly one of --seed or --finalize")
    if args.precheck_only and args.finalize:
        raise ValueError("Gate4 finalize has no precheck-only mode")
    prepared, replay, report, unlock, paths = _prepare(args)
    lock = (
        paths["shared"] / ".gate4.finalize.lock"
        if args.finalize
        else paths["seeds"][int(args.seed)].parent / ".gate4.seed.lock"
    )
    with exclusive_file_lock(
        lock,
        label="formal Gate4 execution lock",
        payload=f"pid={os.getpid()}\n".encode("ascii"),
    ):
        if args.finalize:
            return _finalize(prepared, replay, report, unlock, paths)
        return _run_seed(args, prepared, replay, report, unlock, paths)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--gate1-unlock", type=Path, required=True)
    parser.add_argument("--gates23-replay", type=Path, required=True)
    parser.add_argument("--gates23-report", type=Path, required=True)
    parser.add_argument("--phase-marker-3407", type=Path, required=True)
    parser.add_argument("--phase-marker-3408", type=Path, required=True)
    parser.add_argument("--phase-marker-3409", type=Path, required=True)
    parser.add_argument("--post-stage-c-replay", type=Path, required=True)
    parser.add_argument("--post-stage-c-report", type=Path, required=True)
    parser.add_argument("--post-stage-c-unlock", type=Path, required=True)
    parser.add_argument("--post-stage-c-terminal", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--seed", type=int)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--precheck-only", action="store_true")
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    if result.get("status") == "FAIL":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
