"""Evaluation only for the complete, matched cells available at the user stop.

This diagnostic does not admit or complete either original three-seed matrix.
Its cell selection is frozen before any metric-bearing annotation is opened.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.continuous_roi_s2_v3_full200_compute import (
    atomic_publish_json, canonical_sha256, require_clean_commit, sha256_file,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_infer import (
    _cell_from_seal, _load_identity_hashes, run_bound_cell_inference,
)
from tools.bata.continuous_roi_s2_v3_full200_compute_train import validate_full_data_manifest
from tools.bata.zoomtoken_full200_matrix_spec import get_matrix_spec, validate_matrix_cell

TRAINING_COMMIT = "21aa2945b934a0dba469a517c224efe9b30d3967"
MATCHED_SEEDS = {"d2s": (4407, 4408), "patad": (4407,)}
CONFIG_PREFIX = {
    "D160": "continuous_roi_s2_v3_d160",
    "G96": "continuous_roi_s2_v3_g96",
    "D2S-U128-B128": "continuous_roi_d2s_v3_u128_burst128",
    "PATAD-U128-B128": "continuous_roi_patad_v3_u128",
}


def validate_selected_cells(rows, matrix_kind: str) -> None:
    spec = get_matrix_spec(matrix_kind)
    expected = {(arm, seed) for arm in spec.arms for seed in MATCHED_SEEDS[matrix_kind]}
    observed = [(row["arm"], int(row["seed"])) for row in rows]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("diagnostic must contain every frozen matched cell, without replacements")


def validate_training_state(checkpoint: Mapping, receipt: Mapping, identity: Mapping) -> dict:
    if (receipt.get("complete") is not True or receipt.get("epochs") != 60
            or receipt.get("successful_updates") != 6000
            or receipt.get("training_identity_count") != 200
            or receipt.get("checkpoint_state") != "epoch_59_state_dict_ema_update_6000"):
        raise ValueError("diagnostic rejects partial training and intermediate checkpoints")
    if (checkpoint.get("epoch") != 59 or checkpoint.get("successful_updates") != 6000
            or checkpoint.get("identity_hashes") != identity
            or not checkpoint.get("state_dict_ema")
            or checkpoint.get("sample_order_trace_sha256") != receipt["sample_order_trace_sha256"]):
        raise ValueError("checkpoint does not corroborate the training terminal receipt")
    states = checkpoint.get("optimizer", {}).get("state", {})
    steps = [float(state["step"]) for state in states.values() if "step" in state]
    if not steps or len(steps) != len(states) or set(steps) != {6000.0}:
        raise ValueError(f"actual AdamW state is not 6000 updates: {sorted(set(steps))}")
    audit = receipt.get("update_audit", {})
    if any(audit.get(key) != 6000 for key in ("consumed_batches", "ema_updates", "scheduler_advances")):
        raise ValueError("training, EMA, or scheduler counters differ from 6000")
    return {"training_videos": 200, "epochs": 60, "successful_updates": 6000,
            "optimizer_parameter_states": len(steps), "optimizer_step_min": min(steps),
            "optimizer_step_max": max(steps), "checkpoint_state": "state_dict_ema"}


def verify_sources(args) -> None:
    require_clean_commit(args.expected_commit, ROOT)
    require_clean_commit(TRAINING_COMMIT, args.training_source)
    subprocess.run(
        ["git", "diff", "--quiet", TRAINING_COMMIT, args.expected_commit,
         "--", "opentad", "configs/adatad/thumos"], cwd=ROOT, check=True,
    )


def validate_manifest_files(manifest) -> None:
    evaluation = manifest["evaluation"]
    for path, expected in (
        (evaluation["heldout_inference_annotation"], evaluation["heldout_inference_annotation_sha256"]),
        (manifest["class_map"]["path"], manifest["class_map"]["sha256"]),
    ):
        if sha256_file(path) != expected:
            raise ValueError("frozen inference annotation or class map changed")
    training = manifest["training"]
    database = json.loads(Path(training["training_only_annotation"]).read_text())["database"]
    identities = {video for video, row in database.items() if row["subset"] == "training"}
    if identities != set(training["identity_order"]):
        raise ValueError("training annotation does not cover the frozen 200-video population")


def run_cell(args, manifest, cell, *, precheck_only=False):
    label = f"{cell['arm']}_seed{cell['seed']}"
    call = argparse.Namespace(
        arm=cell["arm"], seed=cell["seed"], expected_commit=args.expected_commit,
        identity_hashes=Path(cell["identity_hashes_path"]),
        work_dir=args.output_root / ("precheck_work" if precheck_only else "inference_work") / label,
        output=args.output_root / "predictions" / f"{label}.json",
    )
    return run_bound_cell_inference(call, manifest=manifest, cell=cell, precheck_only=precheck_only)


def precheck(args, manifest, spec) -> None:
    import torch

    plan_path = args.output_root / "control" / "diagnostic_plan.json"
    if plan_path.exists():
        raise FileExistsError("use a fresh diagnostic namespace; never overwrite an admitted run")
    rows = []
    for arm in spec.arms:
        for seed in MATCHED_SEEDS[spec.key]:
            cell_dir = args.training_root / "work_dirs" / f"{arm}_seed{seed}"
            checkpoint = cell_dir / "checkpoint" / "epoch_59.pth"
            terminal = cell_dir / "training_terminal_receipt.json"
            receipt = json.loads(terminal.read_text())
            checked = dict(receipt)
            if canonical_sha256({k: v for k, v in checked.items() if k != "receipt_sha256"}) != receipt["receipt_sha256"]:
                raise ValueError("training receipt self-hash differs")
            if receipt["arm"] != arm or receipt["seed"] != seed or receipt["protocol_id"] != spec.protocol_id:
                raise ValueError("training receipt belongs to a different cell")
            config = args.training_source / "configs" / "adatad" / "thumos" / f"{CONFIG_PREFIX[arm]}_seed{seed}.py"
            identity_path = args.training_root / "control" / f"identity_hashes_{arm}_seed{seed}.json"
            identity = _load_identity_hashes(identity_path)
            if (identity["code_sha256"] != hashlib.sha256(TRAINING_COMMIT.encode("ascii")).hexdigest()
                    or identity["config_sha256"] != sha256_file(config)
                    or identity["annotation_sha256"] != manifest["annotation"]["sha256"]
                    or identity["class_map_sha256"] != manifest["class_map"]["sha256"]
                    or identity["media_manifest_sha256"] != manifest["media"]["records_sha256"]):
                raise ValueError("training identity does not bind the frozen source and population")
            validate_matrix_cell(config, arm=arm, seed=seed, spec=spec)
            if receipt["checkpoint_path"] != checkpoint.as_posix() or sha256_file(checkpoint) != receipt["checkpoint_sha256"]:
                raise ValueError("final checkpoint differs from its terminal receipt")
            payload = torch.load(checkpoint, map_location="cpu")
            evidence = validate_training_state(payload, receipt, identity)
            del payload
            row = {
                "arm": arm, "seed": seed, "config_path": config.as_posix(),
                "config_sha256": identity["config_sha256"],
                "checkpoint_path": checkpoint.as_posix(), "checkpoint_sha256": receipt["checkpoint_sha256"],
                "training_terminal_receipt_path": terminal.as_posix(),
                "training_terminal_receipt_sha256": sha256_file(terminal),
                "identity_hashes_path": identity_path.as_posix(), "training_evidence": evidence,
            }
            row["precheck"] = run_cell(args, manifest, row, precheck_only=True)
            rows.append(row)
            print(f"PRECHECK PASS {arm} seed {seed}: {evidence}", flush=True)
    validate_selected_cells(rows, spec.key)
    plan = {
        "schema_version": "zoomtoken_user_stop_diagnostic_v1", "diagnostic_only": True,
        "reason": "user cancelled remaining training on 2026-09-08; evaluate complete matched seeds only",
        "training_protocol_id": spec.protocol_id, "matrix_kind": spec.key,
        "training_commit": TRAINING_COMMIT, "execution_commit": args.expected_commit,
        "training_root": args.training_root.as_posix(), "training_source": args.training_source.as_posix(),
        "population_manifest_sha256": manifest["manifest_sha256"],
        "training_videos": 200, "evaluation_videos": 211, "evaluation_windows": 792,
        "original_matrix_cells": 9, "evaluated_cells": len(rows),
        "matched_seeds": list(MATCHED_SEEDS[spec.key]), "rows": rows,
        "full_three_seed_matrix_complete": False, "paper_admission": False,
    }
    atomic_publish_json(plan_path, plan)
    print(f"PRECHECK COMPLETE: {plan_path}", flush=True)


def evaluate(args, manifest, spec) -> None:
    from tools.bata.continuous_roi_s2_v3_full200_compute_eval import (
        VideoOccurrence, _json_safe, _official_point_metrics, _read_complete_ground_truth,
        evaluate_slot_metrics, load_complete_prediction_bundle,
    )

    plan_path = args.output_root / "control" / "diagnostic_plan.json"
    plan = json.loads(plan_path.read_text())
    if (plan["execution_commit"] != args.expected_commit or plan["training_commit"] != TRAINING_COMMIT
            or plan["training_root"] != args.training_root.as_posix()
            or plan["population_manifest_sha256"] != manifest["manifest_sha256"]
            or plan["matrix_kind"] != spec.key or plan["diagnostic_only"] is not True):
        raise ValueError("PRECHECK does not admit this diagnostic run")
    validate_selected_cells(plan["rows"], spec.key)
    marker_path = args.output_root / "control" / "diagnostic_gt_open.json"
    if marker_path.exists():
        raise FileExistsError("diagnostic GT has already been opened; do not rerun inference")
    # No metric-bearing GT is read until every selected cell has full predictions.
    for row in plan["rows"]:
        row = _cell_from_seal(plan, arm=row["arm"], seed=row["seed"])
        run_cell(args, manifest, row)
        print(f"PREDICTIONS COMPLETE {row['arm']} seed {row['seed']}", flush=True)
    bundles = []
    for row in plan["rows"]:
        path = args.output_root / "predictions" / f"{row['arm']}_seed{row['seed']}.json"
        bundles.append(load_complete_prediction_bundle(
            path, expected_arm=row["arm"], expected_seed=row["seed"],
            expected_population_manifest_sha256=manifest["manifest_sha256"],
            expected_video_order=manifest["evaluation"]["video_order"], class_map=manifest["class_map"]["classes"],
        ))
    annotation = manifest["annotation"]["path"]
    if sha256_file(annotation) != manifest["annotation"]["sha256"]:
        raise ValueError("metric annotation differs from the frozen training protocol")
    atomic_publish_json(marker_path, {
        "schema_version": "zoomtoken_user_stop_diagnostic_gt_open_v1", "diagnostic_only": True,
        "plan_path": plan_path.as_posix(), "annotation_path": annotation,
        "annotation_sha256": manifest["annotation"]["sha256"],
        "bundles": [{"arm": b.arm, "seed": b.seed, "bundle_sha256": b.bundle_sha256} for b in bundles],
    })
    ground_truth = _read_complete_ground_truth(
        annotation_path=annotation, expected_video_order=manifest["evaluation"]["video_order"],
        class_map=manifest["class_map"]["classes"],
    )
    results = []
    for bundle in bundles:
        official = _official_point_metrics(
            annotation_path=annotation, ground_truth=ground_truth, bundle=bundle,
            class_map=manifest["class_map"]["classes"],
        )
        metrics = evaluate_slot_metrics(
            ground_truth, bundle.predictions,
            occurrences=tuple(VideoOccurrence(v, v) for v in bundle.video_order),
            class_count=len(manifest["class_map"]["classes"]),
            q1=float.fromhex(manifest["short_q1"]["q1_float64_hex"]),
        )
        result = {"arm": bundle.arm, "seed": bundle.seed, "official_metrics_0_to_1": official,
                  "diagnostic_metrics": _json_safe(asdict(metrics))}
        atomic_publish_json(args.output_root / "metrics" / f"{bundle.arm}_seed{bundle.seed}.json", result)
        results.append(result)
        print(f"METRICS {bundle.arm} seed {bundle.seed}: {official}", flush=True)
    atomic_publish_json(args.output_root / "diagnostic_results.json", {
        "diagnostic_only": True, "complete": True, "matrix_kind": spec.key,
        "full_three_seed_matrix_complete": False, "paper_admission": False,
        "training_videos_per_cell": 200, "epochs_per_cell": 60, "updates_per_cell": 6000,
        "evaluation_videos_per_cell": 211, "evaluation_windows_per_cell": 792,
        "plan_path": plan_path.as_posix(), "matched_seeds": list(MATCHED_SEEDS[spec.key]),
        "original_matrix_cells": 9, "evaluated_cells": len(results), "cells": results,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("precheck", "evaluate"))
    parser.add_argument("--training-root", type=Path, required=True)
    parser.add_argument("--training-source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    for key in ("training_root", "training_source", "output_root"):
        setattr(args, key, getattr(args, key).resolve())
    spec = get_matrix_spec()
    if spec.key not in MATCHED_SEEDS:
        raise ValueError("only the two user-stopped matrices are admitted")
    if "SLURM_JOB_ID" not in os.environ:
        raise RuntimeError("PRECHECK and evaluation require Slurm GPU allocation")
    if args.output_root == args.training_root or args.training_root in args.output_root.parents:
        raise ValueError("diagnostics must not write into the original training run")
    verify_sources(args)
    manifest = validate_full_data_manifest(args.training_root / "manifest" / "full_data_manifest.json")
    validate_manifest_files(manifest)
    (precheck if args.mode == "precheck" else evaluate)(args, manifest, spec)


if __name__ == "__main__":
    main()
