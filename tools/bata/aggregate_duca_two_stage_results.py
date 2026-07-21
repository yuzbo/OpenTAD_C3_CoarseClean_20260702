from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from tools.bata.select_duca_frontend_checkpoint import sha256_file


EXPECTED_VARIANTS = (
    "two_stage_exact_uniform",
    "two_stage_scratch",
    "two_stage_pretrained_joint",
    "two_stage_pretrained_frozen",
)


def aggregate_results(
    *,
    expected_commit: str,
    frontend_decision_path: str | Path,
    frontend_decision_sha256: str,
    gate_suite_path: str | Path,
    gate_suite_sha256: str,
    completion_paths: Sequence[str | Path],
    output_path: str | Path,
) -> dict:
    decision = Path(frontend_decision_path).expanduser().resolve()
    gate = Path(gate_suite_path).expanduser().resolve()
    if not decision.is_file() or sha256_file(decision) != frontend_decision_sha256:
        raise RuntimeError("frontend decision drift")
    if not gate.is_file() or sha256_file(gate) != gate_suite_sha256:
        raise RuntimeError("two-stage gate suite drift")
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("git_commit") != expected_commit or gate_payload.get("ok") is not True:
        raise RuntimeError("two-stage gate suite commit/status mismatch")

    rows = []
    for raw_path in completion_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "duca_two_stage_curriculum_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("frontend_decision_sha256") != frontend_decision_sha256
            or payload.get("gate_suite_sha256") != gate_suite_sha256
        ):
            raise RuntimeError(f"invalid two-stage completion: {path}")
        evaluation = Path(payload["evaluation_path"]).resolve()
        checkpoint = Path(payload["checkpoint_path"]).resolve()
        if (
            not evaluation.is_file()
            or sha256_file(evaluation) != payload["evaluation_sha256"]
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != payload["checkpoint_sha256"]
        ):
            raise RuntimeError(f"two-stage completion artifact drift: {path}")
        rows.append(
            {
                "variant": payload["variant"],
                "terminal_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "metrics": payload["metrics"],
                "completion_path": str(path),
                "completion_sha256": sha256_file(path),
            }
        )
    if {row["variant"] for row in rows} != set(EXPECTED_VARIANTS):
        raise RuntimeError("two-stage result set does not cover the four frozen arms")
    rows.sort(key=lambda row: EXPECTED_VARIANTS.index(row["variant"]))
    payload = {
        "schema": "duca_two_stage_curriculum_result_suite_v1",
        "ok": True,
        "status": "four_arm_terminal_ema_results_sealed",
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "frontend_decision_path": str(decision),
        "frontend_decision_sha256": frontend_decision_sha256,
        "gate_suite_path": str(gate),
        "gate_suite_sha256": gate_suite_sha256,
        "seed": 3407,
        "successful_optimizer_updates_per_arm": 6000,
        "detector_extra_updates_outside_official60": 0,
        "test_subset_used_once_for_terminal_metrics": True,
        "results": rows,
        "paper_claim_allowed": False,
    }
    output = Path(output_path).expanduser().resolve()
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--frontend-decision", required=True)
    parser.add_argument("--frontend-decision-sha256", required=True)
    parser.add_argument("--gate-suite", required=True)
    parser.add_argument("--gate-suite-sha256", required=True)
    parser.add_argument("--completion", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    result = aggregate_results(
        expected_commit=args.expected_commit,
        frontend_decision_path=args.frontend_decision,
        frontend_decision_sha256=args.frontend_decision_sha256,
        gate_suite_path=args.gate_suite,
        gate_suite_sha256=args.gate_suite_sha256,
        completion_paths=args.completion,
        output_path=args.output_json,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
