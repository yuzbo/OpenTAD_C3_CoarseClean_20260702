"""Seal one label-free full held-out prediction and its measured execution cost."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.prepare_duca_h65_system_multibudget_exposure import sha256_file


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def seal(
    *,
    prediction_path: Path,
    cost_path: Path,
    checkpoint_path: Path,
    config_path: Path,
    calibration_path: Path,
    held_out_ids_path: Path,
    inference_annotation_path: Path,
    output_path: Path,
    expected_commit: str,
    arm: str,
    seed: int,
    budget_view: str,
) -> dict[str, Any]:
    observed_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, encoding="utf-8"
    ).strip()
    if observed_commit != expected_commit:
        raise SystemExit("prediction checkout differs from the frozen commit")
    status = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=ROOT,
        text=True,
        encoding="utf-8",
    ).strip()
    if status:
        raise SystemExit("prediction sealing requires a clean exact-commit checkout")
    for path in (
        prediction_path,
        cost_path,
        checkpoint_path,
        config_path,
        calibration_path,
        held_out_ids_path,
        inference_annotation_path,
    ):
        if not path.is_file():
            raise SystemExit(f"prediction sealing input missing: {path}")
    expected_ids = {
        line for line in held_out_ids_path.read_text(encoding="utf-8").splitlines() if line
    }
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if len(expected_ids) != 211 or int(calibration["held_out_video_count"]) != 211:
        raise SystemExit("prediction sealing is frozen to the admitted 211-video population")
    inference_report = calibration.get("held_out_inference_annotation", {})
    if sha256_file(inference_annotation_path) != inference_report.get("sha256"):
        raise SystemExit("prediction inference annotation differs from PRE_RUN")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = prediction.get("results")
    if not isinstance(results, dict) or set(results) != expected_ids:
        raise SystemExit("prediction video IDs differ from the admitted held-out manifest")
    cost = json.loads(cost_path.read_text(encoding="utf-8"))
    if (
        cost.get("schema_version")
        != "duca_h65_system_multibudget_execution_cost_v2"
        or cost.get("held_out_semantics_read") is not False
        or int(cost.get("video_count", -1)) != 211
        or set(cost.get("per_video", {})) != expected_ids
        or cost.get("prediction_sha256") != sha256_file(prediction_path)
        or not isinstance(cost.get("per_video_component_ms"), dict)
        or cost.get("full_population_wall_ms") is None
    ):
        raise SystemExit("execution cost does not bind the same 211-video prediction")
    manifest = calibration["held_out_manifest"]
    if int(cost.get("window_count", -1)) != int(manifest["executed_window_count"]):
        raise SystemExit("execution cost does not cover every official sliding-window input")
    if budget_view in {"control_k384", "candidate_k384"}:
        if cost.get("requested_budget_window_counts") != {
            "384": int(manifest["executed_window_count"])
        }:
            raise SystemExit("forced K384 prediction contains another requested budget")
        expected_cost = int(manifest["control_actual_observations"])
    elif budget_view == "candidate_mixed":
        if cost.get("requested_budget_window_counts") != manifest[
            "executed_window_budget_counts"
        ]:
            raise SystemExit("mixed prediction budget counts differ from the frozen manifest")
        expected_cost = int(manifest["mixed_actual_observations"])
    else:
        raise SystemExit("budget view must be control_k384, candidate_k384, or candidate_mixed")
    expected_arm = "control" if budget_view == "control_k384" else "candidate"
    if arm != expected_arm:
        raise SystemExit("arm and budget view describe different prediction identities")
    if int(cost.get("total_actual_observations", -1)) != expected_cost:
        raise SystemExit("measured actual observations differ from the frozen label-free replay")
    payload = {
        "schema_version": "duca_h65_system_multibudget_prediction_seal_v1",
        "git_commit": observed_commit,
        "arm": arm,
        "seed": int(seed),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_epoch": 59,
        "checkpoint_state_key": "state_dict_ema",
        "config_path": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "calibration_sha256": sha256_file(calibration_path),
        "budget_view": budget_view,
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "prediction_video_count": len(results),
        "prediction_count": sum(len(value) for value in results.values()),
        "execution_cost_path": str(cost_path.resolve()),
        "execution_cost_sha256": sha256_file(cost_path),
        "total_actual_observations": int(cost["total_actual_observations"]),
        "total_execution_slots": int(cost["total_execution_slots"]),
        "held_out_video_ids_sha256": sha256_file(held_out_ids_path),
        "held_out_inference_annotation_sha256": sha256_file(
            inference_annotation_path
        ),
        "held_out_labels_or_segments_read": False,
    }
    _atomic_json(output_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--cost", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--held-out-ids", type=Path, required=True)
    parser.add_argument("--inference-annotation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--arm", choices=("control", "candidate"), required=True)
    parser.add_argument("--seed", type=int, choices=(3407, 3408, 3409), required=True)
    parser.add_argument(
        "--budget-view",
        choices=("control_k384", "candidate_k384", "candidate_mixed"),
        required=True,
    )
    args = parser.parse_args()
    payload = seal(
        prediction_path=args.prediction.expanduser().resolve(),
        cost_path=args.cost.expanduser().resolve(),
        checkpoint_path=args.checkpoint.expanduser().resolve(),
        config_path=args.config.expanduser().resolve(),
        calibration_path=args.calibration.expanduser().resolve(),
        held_out_ids_path=args.held_out_ids.expanduser().resolve(),
        inference_annotation_path=args.inference_annotation.expanduser().resolve(),
        output_path=args.output.expanduser().resolve(),
        expected_commit=args.expected_commit,
        arm=args.arm,
        seed=args.seed,
        budget_view=args.budget_view,
    )
    print(
        "SEALED "
        f"{payload['arm']} seed={payload['seed']} view={payload['budget_view']} "
        f"videos={payload['prediction_video_count']} predictions={payload['prediction_count']}"
    )


if __name__ == "__main__":
    main()
