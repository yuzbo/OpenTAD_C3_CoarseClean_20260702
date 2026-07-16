#!/usr/bin/env python3
"""Revalidate the completed formal post-Stage-C Gate-3 artifact chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.gates23 import R2_SEEDS, load_exact_canonical_json
from opentad.models.chronotransport.post_stage_c import (
    build_post_stage_c_gate3_terminal,
    validate_post_stage_c_gate3_report,
    validate_post_stage_c_gate3_unlock,
    validate_post_stage_c_replay_artifact,
)
from opentad.models.chronotransport.scheduler import R2_NON_DENSE_NAMES
from tools.bata.chronotransport_r2_post_stage_c_factory import (
    validate_completed_stage_c_population,
)
from tools.bata.run_chronotransport_r2_post_stage_c_gate3 import (
    _file_sha256,
    _prepare_inputs,
    build_parser,
)
from opentad.models.chronotransport.filesystem import path_exists_no_follow


_ENTRYPOINT = "tools/bata/validate_chronotransport_r2_post_stage_c_gate3.py"


def _validate_replay_context(
    replay: Mapping[str, Any],
    *,
    prepared,
    stage_c_bindings: Mapping[str, Mapping[str, str]],
) -> None:
    manifest = prepared.registration["window_manifest"]["artifact"]
    library = {
        str(row["name"]): str(row["action_sha256"])
        for row in prepared.registration["candidate_library"]["candidates"]
    }
    expected_actions = {name: library[name] for name in R2_NON_DENSE_NAMES}
    expected_splits = {
        split: list(map(str, manifest["splits"][split]))
        for split in ("calibration", "evaluation")
    }
    if (
        replay["registration_sha256"]
        != prepared.registration["registration_sha256"]
        or replay["registration_commit"] != prepared.registration_commit
        or replay["gate1_unlock_artifact_sha256"]
        != prepared.gate1["artifact_sha256"]
        or replay["pre_stage_c_gates23_report_sha256"]
        != prepared.gates23_report["artifact_sha256"]
        or replay["manifest_sha256"] != manifest["manifest_sha256"]
        or replay["library_sha256"]
        != prepared.registration["candidate_library"]["library_sha256"]
        or replay["candidate_action_sha256_by_name"] != expected_actions
        or replay["split_window_ids"] != expected_splits
        or replay["stage_c_bindings"] != dict(stage_c_bindings)
    ):
        raise ValueError("post-Stage-C replay differs from registered input chain")


def run(args) -> dict[str, Any]:
    if args.precheck_only:
        raise ValueError("post-Stage-C completion validator forbids precheck mode")
    prepared = _prepare_inputs(args)
    outputs = prepared.outputs
    replay = validate_post_stage_c_replay_artifact(
        load_exact_canonical_json(outputs["replay"], label="post-Stage-C replay")
    )
    stage_c_bindings = validate_completed_stage_c_population(
        registration_path=args.registration,
        registration=prepared.registration,
        registration_commit=prepared.registration_commit,
        gate1_unlock_path=args.gate1_unlock,
        gates23_replay_path=args.gates23_replay,
        gates23_report_path=args.gates23_report,
        phase_marker_paths=prepared.phase_marker_paths,
        entrypoint_relative=_ENTRYPOINT,
    )
    _validate_replay_context(
        replay, prepared=prepared, stage_c_bindings=stage_c_bindings
    )
    report = validate_post_stage_c_gate3_report(
        load_exact_canonical_json(outputs["report"], label="post-Stage-C report"),
        replay=replay,
        candidate_cost_p50=prepared.candidate_cost_p50,
        budget=prepared.budget,
        fit_baseline_constants_by_seed=prepared.fit_baseline_constants_by_seed,
    )
    unlock = None
    if report["status"] == "PASS":
        unlock = validate_post_stage_c_gate3_unlock(
            load_exact_canonical_json(
                outputs["unlock"], label="post-Stage-C unlock"
            ),
            report=report,
            replay=replay,
        )
    elif path_exists_no_follow(outputs["unlock"], label="post-Stage-C unlock"):
        raise ValueError("FAIL post-Stage-C report must not have an unlock")
    terminal = load_exact_canonical_json(
        outputs["terminal"], label="post-Stage-C terminal"
    )
    expected_terminal = build_post_stage_c_gate3_terminal(
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
    if terminal != expected_terminal:
        raise ValueError("post-Stage-C terminal differs from exact recomputation")
    result = {
        "schema": "chronotransport-r2-post-stage-c-gate3-validation-v1",
        "status": "VALID",
        "gate3_status": report["status"],
        "registration_sha256": prepared.registration["registration_sha256"],
        "registration_commit": prepared.registration_commit,
        "validated_stage_c_seeds": list(R2_SEEDS),
        "report_sha256": report["artifact_sha256"],
        "terminal_sha256": terminal["artifact_sha256"],
    }
    print(json.dumps(result, sort_keys=True))
    return result


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
