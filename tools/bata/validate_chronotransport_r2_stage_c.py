#!/usr/bin/env python3
"""Revalidate a completed formal paired Stage-C artifact chain."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Mapping


ROOT = Path(os.path.abspath(__file__)).parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.models.chronotransport.filesystem import (
    load_bound_json,
    load_bound_torch,
    read_bound_bytes,
)
from opentad.models.chronotransport.formal_stage_c import (
    STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
    build_stage_c_completion_marker,
    validate_paired_stage_c_checkpoint,
)
from tools.bata.train_chronotransport_r2_stage_c import (
    _ledger_bytes,
    _prepare,
    build_parser,
)


def run(args):
    if args.resume is not None or args.precheck_only:
        raise ValueError("Stage-C completion validation forbids resume/precheck flags")
    registration, commit, outputs, components, provenance, _state = _prepare(
        args,
        entrypoint_relative="tools/bata/validate_chronotransport_r2_stage_c.py",
    )
    try:
        _, checkpoint, _, checkpoint_sha = load_bound_torch(
            outputs["output"], label="formal Stage-C completed checkpoint"
        )
        if not isinstance(checkpoint, Mapping):
            raise ValueError("formal Stage-C completed checkpoint must be a mapping")
        validate_paired_stage_c_checkpoint(
            checkpoint,
            expected_seed=args.seed,
            expected_fit_window_ids=components.fit_window_ids,
            expected_provenance=provenance,
            formal=True,
            require_complete=True,
            expected_total_successful_updates=STAGE_C_TOTAL_SUCCESSFUL_UPDATES,
        )
        _, ledger_bytes, ledger_sha = read_bound_bytes(
            outputs["ledger"], label="formal Stage-C completed ledger"
        )
        if ledger_bytes != _ledger_bytes(checkpoint):
            raise ValueError("formal Stage-C completed ledger differs from checkpoint")
        _, terminal, _, _ = load_bound_json(
            outputs["terminal"], label="formal Stage-C completed terminal"
        )
        expected = build_stage_c_completion_marker(
            checkpoint,
            checkpoint_path=str(outputs["output"]),
            checkpoint_file_sha256=checkpoint_sha,
            ledger_path=str(outputs["ledger"]),
            ledger_file_sha256=ledger_sha,
        )
        if not isinstance(terminal, Mapping) or dict(terminal) != expected:
            raise ValueError("formal Stage-C terminal differs from exact recomputation")
        result = {
            "schema": "chronotransport-r2-stage-c-validation-v1",
            "status": "VALID",
            "registration_sha256": registration["registration_sha256"],
            "registration_commit": commit,
            "seed": args.seed,
            "successful_updates_per_arm": checkpoint["successful_updates"],
            "window_exposures_per_arm": 2 * checkpoint["successful_updates"],
            "terminal_sha256": terminal["artifact_sha256"],
        }
        print(json.dumps(result, sort_keys=True))
        return result
    finally:
        components.close()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
