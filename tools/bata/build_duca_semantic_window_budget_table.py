#!/usr/bin/env python
"""Build the frozen H65 semantic-window budget and acquisition table."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from tools.bata.duca_frontend_initialization import (
    initialize_model_from_checkpoint,
    sha256_file,
)


BUDGETS = (256, 384, 512)
CONTROL_PERMUTATION_NONCE = (
    "DUCA-DYNAMIC-BUDGET-WINDOW-CONTRACT-CORRECTION-v005-20260828"
)


def percentile_rank_with_median_ties(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("percentile rank requires a non-empty 1-D sequence")
    if values.size == 1:
        return np.zeros(1, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    cursor = 0
    while cursor < values.size:
        end = cursor + 1
        while end < values.size and values[order[end]] == values[order[cursor]]:
            end += 1
        median_rank = 0.5 * (cursor + end - 1)
        ranks[order[cursor:end]] = median_rank / float(values.size - 1)
        cursor = end
    return ranks


def semantic_and_permuted_budgets(rows, permutation_nonce=CONTROL_PERMUTATION_NONCE):
    count = len(rows)
    if count <= 0:
        raise ValueError("video must contain at least one window")
    boundary = percentile_rank_with_median_ties([row["boundary_evidence"] for row in rows])
    uncertainty = percentile_rank_with_median_ties([row["uncertainty_evidence"] for row in rows])
    score = 0.5 * (boundary + uncertainty)
    quota = 0 if count == 1 else (1 if count == 2 else count // 3)
    high_order = sorted(
        range(count),
        key=lambda index: (
            -float(score[index]),
            -float(boundary[index]),
            -float(uncertainty[index]),
            int(rows[index]["window_start_frame"]),
        ),
    )
    low_order = sorted(
        range(count),
        key=lambda index: (
            float(score[index]),
            float(boundary[index]),
            float(uncertainty[index]),
            int(rows[index]["window_start_frame"]),
        ),
    )
    semantic = [384] * count
    for index in high_order[:quota]:
        semantic[index] = 512
    for index in low_order:
        if semantic[index] == 384:
            semantic[index] = 256
            if semantic.count(256) == quota:
                break
    if sum(semantic) != 384 * count:
        raise RuntimeError("semantic budget assignment must preserve the exact within-video mean")

    permutation_order = sorted(
        range(count),
        key=lambda index: hashlib.sha256(
            (
                f"{permutation_nonce}|{rows[index]['split']}|{rows[index]['video_name']}|"
                f"{int(rows[index]['window_start_frame'])}"
            ).encode("utf-8")
        ).digest(),
    )
    permuted = [None] * count
    for index, budget in zip(permutation_order, sorted(semantic)):
        permuted[index] = int(budget)
    for index, row in enumerate(rows):
        row["boundary_percentile_rank"] = float(boundary[index])
        row["uncertainty_percentile_rank"] = float(uncertainty[index])
        row["semantic_score"] = float(score[index])
        row["semantic_budget"] = int(semantic[index])
        row["permuted_control_budget"] = int(permuted[index])
    if sorted(semantic) != sorted(permuted):
        raise RuntimeError("permuted control must preserve the per-video K multiset")
    return rows


def _dataset_config(args, split):
    return dict(
        type="DucaVideoGroupedThumosSlidingDataset",
        ann_file=args.annotation,
        subset_name=split,
        block_list=None,
        class_map=args.class_map,
        data_path=args.data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        ioa_thresh=0.75,
        include_background_windows=True,
        stateless_seed=args.seed,
        group_by_video=False,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=[
                    "video_name",
                    "window_start_frame",
                    "window_end_frame",
                    "window_size",
                    "duca_split",
                    "duca_window_index",
                    "duca_window_count",
                ],
            ),
        ],
    )


def _row_from_selector(selector, inputs, masks, meta):
    positions_by_budget = {}
    evidence = None
    for budget in BUDGETS:
        output = selector.forward_test(inputs=inputs, masks=masks, metas=[meta], budget=budget)
        state = output["selector_outputs"]
        positions = state["grid"].selected_positions[0]
        positions = positions[positions >= 0].detach().cpu().long().tolist()
        if len(positions) != budget or positions != sorted(set(positions)):
            raise RuntimeError(f"selector failed exact sorted unique K={budget}")
        positions_by_budget[str(budget)] = [int(value) for value in positions]
        if evidence is None:
            valid = state.get("valid_mask", masks)[0].bool()
            p_action = state["p_action"][0].float().clamp(1.0e-6, 1.0 - 1.0e-6)
            entropy = -(p_action * torch.log2(p_action) + (1.0 - p_action) * torch.log2(1.0 - p_action))
            boundary = torch.sigmoid(state["boundary_logits"][0].float())
            evidence = (
                float(boundary[valid].mean().item()),
                float(entropy[valid].mean().item()),
                int(valid.long().sum().item()),
            )
    return positions_by_budget, evidence


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--pretrain", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    observed_sha = sha256_file(args.checkpoint)
    if observed_sha != args.checkpoint_sha256.lower():
        raise RuntimeError("frozen H65 Stage-1 checkpoint SHA256 mismatch")
    cfg = Config.fromfile(args.stage1_config)
    cfg.model.backbone.custom.pretrain = str(Path(args.pretrain).expanduser().resolve())
    model = build_detector(cfg.model)
    initialize_model_from_checkpoint(
        model,
        dict(
            enabled=True,
            checkpoint_path=args.checkpoint,
            checkpoint_sha256=args.checkpoint_sha256,
            state_key="state_dict_ema",
            expected_checkpoint_epoch=29,
            reset_state_keys=[],
        ),
    )
    selector = model.frame_selector.to(args.device).eval()
    selector.inference_policy_alpha = 1.0
    for parameter in selector.parameters():
        parameter.requires_grad = False

    all_rows = []
    with torch.inference_mode():
        for split in ("training", "validation"):
            dataset = build_dataset(_dataset_config(args, split))
            loader = build_dataloader(
                dataset,
                batch_size=1,
                rank=0,
                world_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=2,
            )
            for batch in loader:
                inputs = batch["inputs"].to(args.device)
                masks = batch["masks"].to(args.device)
                meta = dict(batch["metas"][0])
                positions, evidence = _row_from_selector(selector, inputs, masks, meta)
                boundary, uncertainty, valid_len = evidence
                all_rows.append(
                    dict(
                        schema_version="duca_semantic_window_budget_table_v1",
                        split=str(split),
                        video_name=str(meta["video_name"]),
                        window_index=int(meta["duca_window_index"]),
                        window_count=int(meta["duca_window_count"]),
                        window_start_frame=int(meta["window_start_frame"]),
                        window_end_frame=int(meta["window_end_frame"]),
                        valid_len=int(valid_len),
                        boundary_evidence=float(boundary),
                        uncertainty_evidence=float(uncertainty),
                        positions_by_budget=positions,
                    )
                )

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["split"], row["video_name"])].append(row)
    finalized = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: row["window_index"])
        if [row["window_index"] for row in rows] != list(range(len(rows))):
            raise RuntimeError(f"non-contiguous window indices for {key}")
        finalized.extend(semantic_and_permuted_budgets(rows))

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        for row in finalized:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    manifest = {
        "schema_version": "duca_semantic_window_budget_manifest_v1",
        "table_path": str(output),
        "table_sha256": sha256_file(output),
        "stage1_config": str(Path(args.stage1_config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": observed_sha,
        "checkpoint_epoch": 29,
        "checkpoint_state_key": "state_dict_ema",
        "videomae_pretrain": str(Path(args.pretrain).resolve()),
        "seed": int(args.seed),
        "control_permutation_nonce": CONTROL_PERMUTATION_NONCE,
        "budgets": list(BUDGETS),
        "row_count": len(finalized),
        "video_count": len(grouped),
        "uses_gt": False,
        "uses_teacher": False,
        "uses_prediction_cache": False,
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
