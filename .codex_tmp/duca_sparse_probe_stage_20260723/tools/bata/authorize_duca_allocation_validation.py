from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from tools.bata.export_duca_allocation_ceiling_inputs import (
    sha256,
    write_json_exclusive,
)


def authorize_validation(
    *,
    training_suite_evidence_json: str | Path,
    expected_commit: str,
    decision: str,
    output_json: str | Path,
) -> dict[str, Any]:
    if decision != "GO":
        raise ValueError("validation authorization requires an explicit GO decision")
    if re.fullmatch(r"[0-9a-f]{40}", expected_commit) is None:
        raise ValueError("expected commit must be a full lowercase Git SHA")
    evidence_path = Path(training_suite_evidence_json).resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if not isinstance(evidence, Mapping):
        raise ValueError("training-suite evidence must be an object")
    if (
        evidence.get("schema_version")
        != "duca_allocation_training_suite_evidence_v1"
        or evidence.get("status")
        != "training_side_ceiling_complete_human_go_kill_required"
    ):
        raise ValueError("training-suite evidence is not eligible for validation GO")
    if evidence.get("git_commit") != expected_commit:
        raise ValueError("training-suite evidence commit mismatch")
    contract = evidence.get("decision_contract")
    if not isinstance(contract, Mapping):
        raise ValueError("training-suite decision contract is missing")
    if (
        contract.get("validation_subset_consumed") is not False
        or contract.get("selector_training_authorized") is not False
        or contract.get("paper_claim_allowed") is not False
    ):
        raise ValueError("training-suite evidence weakens the pre-validation contract")
    result = {
        "schema_version": "duca_allocation_validation_authorization_v1",
        "decision": "GO",
        "git_commit": expected_commit,
        "training_suite_evidence_json": str(evidence_path),
        "training_suite_evidence_json_sha256": sha256(evidence_path),
        "checkpoint_sha256": evidence.get("checkpoint_sha256"),
        "pretrain_sha256": evidence.get("pretrain_sha256"),
        "contract": {
            "single_use": True,
            "authorizes_validation_export": True,
            "authorizes_validation_replay": True,
            "authorizes_selector_training": False,
            "authorizes_paper_claim": False,
        },
    }
    write_json_exclusive(output_json, result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Issue a single-use human GO receipt for DUCA validation."
    )
    parser.add_argument("--training-suite-evidence-json", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--decision", required=True, choices=["GO", "HOLD", "KILL"])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = authorize_validation(
        training_suite_evidence_json=args.training_suite_evidence_json,
        expected_commit=args.expected_commit,
        decision=args.decision,
        output_json=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
