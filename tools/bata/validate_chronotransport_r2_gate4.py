#!/usr/bin/env python3
"""Recompute and validate the complete formal Gate-4 artifact chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.formal_gate4 import (
    adjudicate_formal_gate4,
    build_gate4_terminal,
)
from opentad.models.chronotransport.gates23 import load_exact_canonical_json
from opentad.models.chronotransport.filesystem import read_bound_bytes
from tools.bata.chronotransport_r2_gate4_factory import build_formal_gate4_evidence
from tools.bata.run_chronotransport_r2_gate4 import (
    _load_seed_shards,
    _prepare,
    build_parser,
)


def run(args) -> dict[str, Any]:
    if not args.finalize or args.seed is not None or args.precheck_only:
        raise ValueError("formal Gate4 validator requires --finalize")
    prepared, replay, post_report, unlock, paths = _prepare(args)
    shards = _load_seed_shards(paths["seeds"])
    rebuilt_evidence = build_formal_gate4_evidence(
        registration=prepared.registration,
        repository_root=ROOT,
        registration_commit=prepared.registration_commit,
        registration_relpath=prepared.registration_relpath,
        post_stage_c_unlock=unlock,
        post_stage_c_report=post_report,
        post_stage_c_replay=replay,
        gate1_unlock=prepared.gate1,
        seed_shards=shards,
    )
    persisted = {
        name: load_exact_canonical_json(
            paths[name], label=f"persisted formal Gate4 {name} evidence"
        )
        for name in ("timing", "metric", "regret")
    }
    if persisted != rebuilt_evidence:
        raise ValueError("persisted formal Gate4 evidence differs from seed shards")
    expected_report = adjudicate_formal_gate4(
        timing_evidence=persisted["timing"],
        metric_evidence=persisted["metric"],
        regret_evidence=persisted["regret"],
        registration=prepared.registration,
        population=prepared.registration["gate4_population"]["artifact"],
        post_stage_c_unlock=unlock,
        post_stage_c_report=post_report,
        post_stage_c_replay=replay,
        gate1_unlock=prepared.gate1,
        seed_shards=shards,
        repository_root=str(ROOT),
        registration_commit=prepared.registration_commit,
        registration_relpath=prepared.registration_relpath,
    )
    report = load_exact_canonical_json(paths["report"], label="formal Gate4 report")
    if report != expected_report:
        raise ValueError("formal Gate4 report differs from exact recomputation")
    terminal = load_exact_canonical_json(
        paths["terminal"], label="formal Gate4 terminal"
    )
    expected_terminal = build_gate4_terminal(
        report=report,
        timing_path=str(paths["timing"]),
        timing_file_sha256=read_bound_bytes(paths["timing"], label="Gate4 timing")[2],
        metric_path=str(paths["metric"]),
        metric_file_sha256=read_bound_bytes(paths["metric"], label="Gate4 metric")[2],
        regret_path=str(paths["regret"]),
        regret_file_sha256=read_bound_bytes(paths["regret"], label="Gate4 regret")[2],
        report_path=str(paths["report"]),
        report_file_sha256=read_bound_bytes(paths["report"], label="Gate4 report")[2],
    )
    if terminal != expected_terminal:
        raise ValueError("formal Gate4 terminal differs from exact recomputation")
    result = {
        "schema": "chronotransport-r2-gate4-validation-v1",
        "status": "VALID",
        "gate4_status": report["status"],
        "registration_sha256": prepared.registration["registration_sha256"],
        "registration_commit": prepared.registration_commit,
        "report_artifact_sha256": report["artifact_sha256"],
        "terminal_artifact_sha256": terminal["artifact_sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
