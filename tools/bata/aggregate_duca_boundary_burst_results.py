from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.bata.select_duca_frontend_checkpoint import sha256_file


EXPECTED_VARIANTS = (
    "two_stage_exact_uniform",
    "gaussian_matched_g0",
    "boundary_burst_r2q3_g0",
    "boundary_burst_r4q5_g0",
)


def _average_map(metrics: Mapping[str, Any], *, label: str) -> float:
    value = metrics.get("average_mAP")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{label} average_mAP is missing or non-numeric")
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError(f"{label} average_mAP is non-finite")
    return value


def aggregate(
    *,
    expected_commit: str,
    decision_path: str | Path,
    decision_sha256: str,
    gate_path: str | Path,
    gate_sha256: str,
    completion_paths: Sequence[str | Path],
    completion_sha256s: Sequence[str],
    output_path: str | Path,
) -> dict:
    decision = Path(decision_path).expanduser().resolve()
    gate = Path(gate_path).expanduser().resolve()
    if not decision.is_file() or sha256_file(decision) != decision_sha256:
        raise RuntimeError("boundary-burst frontend decision drift")
    if not gate.is_file() or sha256_file(gate) != gate_sha256:
        raise RuntimeError("boundary-burst full-model gate drift")
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    if (
        decision_payload.get("schema") != "duca_boundary_burst_frontend_decision_v1"
        or decision_payload.get("ok") is not True
        or decision_payload.get("git_commit") != expected_commit
    ):
        raise RuntimeError("boundary-burst decision commit/status mismatch")
    gate_payload = json.loads(gate.read_text(encoding="utf-8"))
    if gate_payload.get("ok") is not True or gate_payload.get("git_commit") != expected_commit:
        raise RuntimeError("boundary-burst gate commit/status mismatch")
    if len(completion_paths) != len(completion_sha256s):
        raise RuntimeError("every completion requires an upstream SHA256 seal")

    rows = []
    for raw, expected_completion_sha256 in zip(
        completion_paths, completion_sha256s
    ):
        path = Path(raw).expanduser().resolve()
        if (
            not path.is_file()
            or len(str(expected_completion_sha256)) != 64
            or sha256_file(path) != str(expected_completion_sha256)
        ):
            raise RuntimeError(f"boundary-burst completion seal drift: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            payload.get("schema") != "duca_two_stage_curriculum_completion_v1"
            or payload.get("ok") is not True
            or payload.get("git_commit") != expected_commit
            or payload.get("frontend_decision_sha256") != decision_sha256
            or payload.get("gate_suite_sha256") != gate_sha256
        ):
            raise RuntimeError(f"invalid boundary-burst completion: {path}")
        checkpoint = Path(payload["checkpoint_path"]).resolve()
        evaluation = Path(payload["evaluation_path"]).resolve()
        if (
            not checkpoint.is_file()
            or sha256_file(checkpoint) != payload["checkpoint_sha256"]
            or not evaluation.is_file()
            or sha256_file(evaluation) != payload["evaluation_sha256"]
        ):
            raise RuntimeError(f"boundary-burst completion artifact drift: {path}")
        evaluation_payload = json.loads(evaluation.read_text(encoding="utf-8"))
        if evaluation_payload.get("schema_version") != "duca_selected_axis_terminal_evaluation_v1":
            raise RuntimeError(f"boundary-burst evaluation schema mismatch: {evaluation}")
        evaluation_metrics = evaluation_payload.get("metrics")
        if not isinstance(evaluation_metrics, Mapping):
            raise RuntimeError(f"boundary-burst evaluation metrics missing: {evaluation}")
        if payload.get("metrics") != evaluation_metrics:
            raise RuntimeError(f"boundary-burst copied completion metrics mismatch: {path}")
        average_map = _average_map(
            evaluation_metrics, label=f"boundary-burst evaluation {payload['variant']}"
        )
        rows.append(
            {
                "variant": payload["variant"],
                "terminal_epoch": 59,
                "checkpoint_state_key": "state_dict_ema",
                "average_mAP": average_map,
                "metrics": dict(evaluation_metrics),
                "evaluation_path": str(evaluation),
                "evaluation_sha256": payload["evaluation_sha256"],
                "completion_path": str(path),
                "completion_sha256": expected_completion_sha256,
            }
        )
    if {row["variant"] for row in rows} != set(EXPECTED_VARIANTS):
        raise RuntimeError("result set does not cover uniform/Gaussian/R2Q3/R4Q5")
    rows.sort(key=lambda row: EXPECTED_VARIANTS.index(row["variant"]))
    payload = {
        "schema": "duca_boundary_burst_terminal_suite_v1",
        "ok": True,
        "status": "matched_four_arm_terminal_ema_results_sealed",
        "task": "offline_temporal_action_detection",
        "git_commit": expected_commit,
        "seed": 3407,
        "successful_optimizer_updates_per_arm": 6000,
        "test_subset_used_once_for_terminal_metrics": True,
        "frontend_decision_path": str(decision),
        "frontend_decision_sha256": decision_sha256,
        "gate_path": str(gate),
        "gate_sha256": gate_sha256,
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
    parser.add_argument("--decision", required=True)
    parser.add_argument("--decision-sha256", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--gate-sha256", required=True)
    parser.add_argument("--completion", action="append", required=True)
    parser.add_argument("--completion-sha256", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    payload = aggregate(
        expected_commit=args.expected_commit,
        decision_path=args.decision,
        decision_sha256=args.decision_sha256,
        gate_path=args.gate,
        gate_sha256=args.gate_sha256,
        completion_paths=args.completion,
        completion_sha256s=args.completion_sha256,
        output_path=args.output_json,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
