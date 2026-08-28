#!/usr/bin/env python
"""Read-only pre-run validation for the frozen DUCA semantic-budget experiment."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from mmengine.config import Config

from tools.bata.duca_frontend_initialization import sha256_file


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if sha256_file(args.checkpoint) != args.checkpoint_sha256.lower():
        raise RuntimeError("Stage-1 checkpoint SHA256 mismatch")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != 29 or "state_dict_ema" not in checkpoint:
        raise RuntimeError("Stage-1 checkpoint must expose epoch-29 state_dict_ema")

    cfg = Config.fromfile(args.config)
    if not bool(cfg.model.offline_window_table) or not bool(cfg.model.freeze_frame_selector):
        raise RuntimeError("config must freeze and bypass the online selector during detector training")
    if int(cfg.model.projection.max_seq_len) != 384:
        raise RuntimeError("detector axis must remain 384")
    if int(cfg.solver.train.batch_size) != 2:
        raise RuntimeError("formal logical batch must contain two videos")
    if int(cfg.solver.train.sampler_seed) != 3407:
        raise RuntimeError("formal training video order must use sampler seed 3407")
    if int(cfg.workflow.expected_train_batches_per_epoch) != 100:
        raise RuntimeError("formal training must execute 100 updates per epoch")
    if int(cfg.workflow.expected_successful_optimizer_updates) != 6000:
        raise RuntimeError("formal training must execute 6000 successful updates")

    rows = []
    manifest_path = Path(args.table).with_suffix(Path(args.table).suffix + ".manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    expected_nonce = "DUCA-DYNAMIC-BUDGET-WINDOW-CONTRACT-CORRECTION-v005-20260828"
    if manifest.get("control_permutation_nonce") != expected_nonce:
        raise RuntimeError("window-table control permutation nonce does not match the frozen contract")
    with open(args.table, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    grouped = defaultdict(list)
    observed_budgets = set()
    row_keys = set()
    for row in rows:
        key = (str(row["split"]), str(row["video_name"]))
        row_key = (
            *key,
            int(row["window_start_frame"]),
            int(row["window_end_frame"]),
            int(row["window_index"]),
            int(row["window_count"]),
        )
        if row_key in row_keys:
            raise RuntimeError(f"duplicate full window identity {row_key}")
        row_keys.add(row_key)
        grouped[key].append(row)
        for budget in (256, 384, 512):
            positions = row["positions_by_budget"][str(budget)]
            if len(positions) != budget or positions != sorted(set(positions)):
                raise RuntimeError(f"row {key}/{row['window_index']} violates exact K={budget}")
        observed_budgets.update(
            (int(row["semantic_budget"]), int(row["permuted_control_budget"]))
        )
    split_videos = defaultdict(set)
    for (split, video_name), video_rows in grouped.items():
        split_videos[split].add(video_name)
        ordered = sorted(video_rows, key=lambda row: int(row["window_index"]))
        if [int(row["window_index"]) for row in ordered] != list(range(len(ordered))):
            raise RuntimeError(f"non-contiguous windows for {(split, video_name)}")
        if any(int(row["window_count"]) != len(ordered) for row in ordered):
            raise RuntimeError(f"window_count mismatch for {(split, video_name)}")
        semantic = [int(row["semantic_budget"]) for row in ordered]
        control = [int(row["permuted_control_budget"]) for row in ordered]
        if sum(semantic) != 384 * len(ordered) or sorted(semantic) != sorted(control):
            raise RuntimeError(f"budget matching failed for {(split, video_name)}")
    if len(split_videos["training"]) != 200 or len(split_videos["validation"]) != 211:
        raise RuntimeError(
            "full table must contain exactly 200 training and 211 validation videos"
        )
    if observed_budgets != {256, 384, 512}:
        raise RuntimeError("full table must exercise all three real backbone K buckets")
    print(
        json.dumps(
            {
                "status": "PRE_RUN_TABLE_READY",
                "row_count": len(rows),
                "training_videos": len(split_videos["training"]),
                "validation_videos": len(split_videos["validation"]),
                "budgets": sorted(observed_budgets),
                "checkpoint_epoch": 29,
                "checkpoint_state_key": "state_dict_ema",
                "checkpoint_sha256": args.checkpoint_sha256.lower(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
