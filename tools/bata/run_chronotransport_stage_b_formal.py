#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from mmengine.config import Config

from opentad.datasets import build_dataset
from opentad.models.chronotransport.formal_stage_b import (
    build_split_manifest,
    calibrate_stage_b_records,
    compact_stage_b_record,
    save_calibrated_stage_b_checkpoint,
    summarize_stage_b_evaluation,
    validate_split_manifest,
)
from opentad.models.chronotransport.replay import (
    canonical_record_line,
    paired_detector_losses,
)
from tools.bata.check_chronotransport_checkpoint import _strip_ddp_prefix
from tools.bata.chronotransport_opentad_factory import (
    _chronotransport_runtime,
    dataset_video_ids,
    make_replay_batch_source,
    stage_b_factory,
)
from tools.bata.train_chronotransport_stage_b import run_training


DEFAULT_SCHEDULES = (
    "periodic2_transport",
    "periodic4_transport",
    "periodic8_transport",
    "periodic2_hold",
    "hold_only",
    "transport_only",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_record_line(record) + "\n" for record in records),
        encoding="utf-8",
    )


def _evaluate(
    model,
    batch_source,
    *,
    split: str,
    schedules: tuple[str, ...],
    calibration_offset: float,
) -> tuple[list[dict], list[dict]]:
    model.eval()
    runtime = _chronotransport_runtime(model)
    runtime.capture_replay_signals = True
    rows = []
    compact = []
    with torch.no_grad():
        for batch in batch_source(0):
            sample_id = str(batch["sample_id"])
            forward_batch = dict(batch)
            forward_batch.pop("sample_id", None)
            forward_batch.pop("split", None)
            for schedule in schedules:
                result = paired_detector_losses(
                    model,
                    forward_batch,
                    counterfactual_schedule=schedule,
                    track_counterfactual_grad=False,
                )
                signals = runtime.latest_signals
                executed = runtime.latest_schedule
                if signals is None or executed is None:
                    raise RuntimeError("formal Stage-B evaluation requires signals and schedule")
                upper = runtime.risk_predictor(
                    signals,
                    executed.actions.unsqueeze(1),
                ).reshape(-1)
                if upper.numel() != 1:
                    raise RuntimeError("formal Stage-B v1 evaluation requires batch_size=1")
                upper_value = float(upper.detach().cpu())
                prediction = upper_value - float(calibration_offset)
                if result.dense_features is None or result.counterfactual_features is None:
                    raise RuntimeError("formal Stage-B evaluation requires ephemeral features")
                feature_mse = float(
                    F.mse_loss(
                        result.counterfactual_features.float(),
                        result.dense_features.float(),
                    ).detach().cpu()
                )
                regret = float(result.regret.detach().cpu())
                row = {
                    "sample_id": sample_id,
                    "split": split,
                    "schedule": schedule,
                    "predicted_risk": prediction,
                    "upper_risk": upper_value,
                    "regret": regret,
                    "feature_mse": feature_mse,
                }
                summary = dict(runtime.latest_summary or {})
                cost = {
                    key: int(summary[key])
                    for key in ("recompute_rows", "transport_rows", "hold_rows")
                    if key in summary
                }
                rows.append(row)
                compact.append(
                    compact_stage_b_record(
                        sample_id=sample_id,
                        split=split,
                        schedule=schedule,
                        predicted_risk=prediction,
                        upper_risk=upper_value,
                        regret=regret,
                        feature_mse=feature_mse,
                        dense_loss=float(result.dense_total.detach().cpu()),
                        counterfactual_loss=float(result.counterfactual_total.detach().cpu()),
                        cost=cost,
                    )
                )
    return rows, compact


def run(args) -> dict[str, object]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("formal Stage B requires exactly one visible CUDA device")
    if int(args.epochs) <= 0:
        raise ValueError("formal Stage-B epochs must be positive")
    schedules = tuple(value.strip() for value in args.schedules.split(",") if value.strip())
    if schedules != DEFAULT_SCHEDULES:
        raise ValueError(f"formal Stage-B v1 schedules must be {DEFAULT_SCHEDULES}")

    cfg = Config.fromfile(args.config)
    all_dataset = build_dataset(cfg.dataset.train)
    all_ids = dataset_video_ids(all_dataset)
    manifest_path = Path(args.split_manifest)
    if manifest_path.exists():
        manifest = validate_split_manifest(
            json.loads(manifest_path.read_text(encoding="utf-8")),
            expected_video_ids=all_ids,
        )
        if int(manifest["seed"]) != int(args.seed):
            raise ValueError("existing split manifest seed does not match formal run")
    else:
        manifest = build_split_manifest(all_ids, seed=args.seed)
        _write_json(manifest_path, manifest)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trained_checkpoint = output_root / "stage_b_trained.pth"
    calibrated_checkpoint = output_root / "stage_b_calibrated.pth"
    metrics_path = output_root / "train_metrics.jsonl"
    calibration_ledger = output_root / "calibration_ledger.jsonl"
    evaluation_ledger = output_root / "evaluation_ledger.jsonl"
    report_path = output_root / "formal_stage_b_report.json"

    os.environ["CHRONOTRANSPORT_CONFIG"] = str(args.config)
    os.environ["CHRONOTRANSPORT_CHECKPOINT"] = str(args.resume or args.checkpoint)
    os.environ["CHRONOTRANSPORT_REPLAY_SPLIT"] = "fit"
    os.environ["CHRONOTRANSPORT_ACTIVE_SPLIT"] = "fit"
    os.environ["CHRONOTRANSPORT_SPLIT_MANIFEST"] = str(manifest_path)
    os.environ["CHRONOTRANSPORT_RESTARTABLE_BATCHES"] = "1"
    os.environ["CHRONOTRANSPORT_STAGE_B_SCHEDULES"] = ",".join(schedules)
    os.environ["CHRONOTRANSPORT_SEED"] = str(args.seed)
    model, fit_batches, loss_step, optimizer_factory = stage_b_factory()
    total_steps = len(manifest["splits"]["fit"]) * int(args.epochs)
    train_result = run_training(
        model=model,
        batch_source=fit_batches,
        loss_step=loss_step,
        optimizer_factory=optimizer_factory,
        output=trained_checkpoint,
        steps=total_steps,
        resume=args.resume,
        metrics_path=metrics_path,
        checkpoint_interval=args.checkpoint_interval,
        ema_decay=args.ema_decay,
        split_hashes=manifest["split_hashes"],
        seed=args.seed,
        max_grad_norm=1.0,
    )

    trained = torch.load(trained_checkpoint, map_location="cpu")
    model.load_state_dict(_strip_ddp_prefix(trained["state_dict_ema"]), strict=True)
    runtime = _chronotransport_runtime(model)
    runtime.risk_predictor.set_calibration_offset(0.0)
    device = torch.device("cuda:0")
    calibration_batches = make_replay_batch_source(
        config_path=args.config,
        video_ids=manifest["splits"]["calibration"],
        split="calibration",
        device=device,
        seed=args.seed + 1000,
    )
    calibration_rows, calibration_compact = _evaluate(
        model,
        calibration_batches,
        split="calibration",
        schedules=schedules,
        calibration_offset=0.0,
    )
    calibration = calibrate_stage_b_records(
        calibration_rows,
        coverage=args.coverage,
    )
    runtime.risk_predictor.set_calibration_offset(calibration["offset"])
    evaluation_batches = make_replay_batch_source(
        config_path=args.config,
        video_ids=manifest["splits"]["evaluation"],
        split="evaluation",
        device=device,
        seed=args.seed + 2000,
    )
    evaluation_rows, evaluation_compact = _evaluate(
        model,
        evaluation_batches,
        split="evaluation",
        schedules=schedules,
        calibration_offset=calibration["offset"],
    )
    evaluation = summarize_stage_b_evaluation(
        evaluation_rows,
        coverage_target=args.coverage,
        min_spearman=args.min_spearman,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.seed,
    )
    _write_records(calibration_ledger, calibration_compact)
    _write_records(evaluation_ledger, evaluation_compact)
    save_calibrated_stage_b_checkpoint(
        trained_checkpoint,
        calibrated_checkpoint,
        calibration_offset=calibration["offset"],
        split_hashes=manifest["split_hashes"],
        p3_gate_status=evaluation["status"],
    )
    report = {
        "schema_version": "chronotransport_formal_stage_b_v1",
        "status": "COMPLETE",
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "schedules": list(schedules),
        "split_manifest": str(manifest_path),
        "split_hashes": manifest["split_hashes"],
        "train": train_result,
        "calibration": calibration,
        "evaluation": evaluation,
        "artifacts": {
            "trained_checkpoint": str(trained_checkpoint),
            "calibrated_checkpoint": str(calibrated_checkpoint),
            "train_metrics": str(metrics_path),
            "calibration_ledger": str(calibration_ledger),
            "evaluation_ledger": str(evaluation_ledger),
        },
        "claim_flags": {
            "deploy": False,
            "metric": False,
            "latency": False,
            "paper": False,
        },
    }
    _write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--coverage", type=float, default=0.9)
    parser.add_argument("--min-spearman", type=float, default=0.2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--schedules", default=",".join(DEFAULT_SCHEDULES))
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
