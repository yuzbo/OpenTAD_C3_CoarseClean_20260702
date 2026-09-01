# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.backbones.et_trc_videomae import TemporalLowRankJVP
from opentad.utils import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="ZT-DIAG-2025-01 Taylor Residual Manifold Diagnostic")
    parser.add_argument("config", type=str, help="path to base config")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint")
    parser.add_argument("--output", type=str, default="diagnostics/zt_diag_2025_01_receipt.json")
    parser.add_argument("--max-batches", type=int, default=20, help="max validation batches for diagnosis")
    parser.add_argument("--stride-k", type=int, default=4, help="sampling stride for Anchor frames")
    return parser.parse_args()


def run_diagnostic(args) -> Dict[str, Any]:
    cfg = Config.fromfile(args.config)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger = setup_logger("TaylorDiagnostic", distributed_rank=0)
    
    logger.info("Building dataset and model for ZT-DIAG-2025-01 diagnostic...")
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
        batch_size=1,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **cfg.solver.val,
    )
    
    detector = build_detector(cfg.model).to(device)
    if args.checkpoint and os.path.isfile(args.checkpoint):
        logger.info(f"Loading weights from {args.checkpoint}...")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        state_dict = ckpt.get("state_dict_ema", ckpt.get("state_dict", ckpt))
        detector.load_state_dict(state_dict, strict=False)
    detector.eval()
    
    backbone = getattr(detector, "backbone", None)
    if backbone is None:
        raise RuntimeError("Detector does not contain a backbone")
    
    # Analyze the Transformer Blocks in VideoMAE
    blocks = getattr(backbone, "blocks", None)
    if blocks is None and hasattr(backbone, "backbone"):
        blocks = getattr(backbone.backbone, "blocks", None)
        
    num_layers = len(blocks) if blocks is not None else 12
    layer_errors_0order = [[] for _ in range(num_layers)]
    layer_errors_taylor = [[] for _ in range(num_layers)]
    layer_cosine_0order = [[] for _ in range(num_layers)]
    layer_cosine_taylor = [[] for _ in range(num_layers)]
    
    k = args.stride_k
    # Use the model's own jacobian_approx if present, else create TemporalLowRankJVP
    jvp_operators = []
    for l_idx in range(num_layers):
        if blocks is not None and hasattr(blocks[l_idx], "jacobian_approx"):
            jvp_operators.append(blocks[l_idx].jacobian_approx.to(device))
        else:
            jvp_operators.append(TemporalLowRankJVP(embed_dims=384, rank=64).to(device))
    
    logger.info(f"Evaluating Taylor vs 0-Order Residual approximations over {args.max_batches} batches (k={k})...")
    
    batch_count = 0
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            batch_count += 1
            imgs = data_dict["imgs"].to(device)  # (B, C, T, H, W)
            
            # Forward through patch embedding
            patch_embed = getattr(backbone, "patch_embed", getattr(getattr(backbone, "backbone", None), "patch_embed", None))
            pos_embed = getattr(backbone, "pos_embed", getattr(getattr(backbone, "backbone", None), "pos_embed", None))
            if patch_embed is None:
                continue
                
            x = patch_embed(imgs)[0]
            if pos_embed is not None:
                x = x + pos_embed.type_as(x)
            
            B, N, C = x.shape
            spatial_tokens = (160 // 16) * (160 // 16)  # 100
            tubelet_count = N // spatial_tokens
            
            x_curr = x
            for l_idx in range(min(num_layers, len(blocks))):
                block = blocks[l_idx]
                
                # Ground truth dense residual
                norm1 = getattr(block, "norm1", nn.LayerNorm(C).to(device))
                attn = getattr(block, "attn", None)
                norm2 = getattr(block, "norm2", nn.LayerNorm(C).to(device))
                mlp = getattr(block, "mlp", None)
                
                if attn is None or mlp is None:
                    continue
                
                attn_out = attn(norm1(x_curr))
                mid = x_curr + attn_out
                mlp_out = mlp(norm2(mid))
                delta_gt = attn_out + mlp_out  # (B, N, C)
                
                # Reshape to (B, T, S, C)
                x_reshaped = x_curr.view(B, tubelet_count, spatial_tokens, C)
                delta_gt_reshaped = delta_gt.view(B, tubelet_count, spatial_tokens, C)
                
                # Anchor frames: 0, k, 2k, ...
                anchor_indices = list(range(0, tubelet_count, k))
                if (tubelet_count - 1) not in anchor_indices:
                    anchor_indices.append(tubelet_count - 1)
                num_anchors = len(anchor_indices)
                
                anchor_map = [
                    min(range(num_anchors), key=lambda i: abs(anchor_indices[i] - t))
                    for t in range(tubelet_count)
                ]
                
                # 0-Order direct carryover
                delta_0order = torch.stack([delta_gt_reshaped[:, anchor_indices[anchor_map[t]]] for t in range(tubelet_count)], dim=1)
                x_anchor_expanded = torch.stack([x_reshaped[:, anchor_indices[anchor_map[t]]] for t in range(tubelet_count)], dim=1)
                
                # 1-Order Taylor: Delta_a + J*(h_i - h_a)
                delta_h = x_reshaped - x_anchor_expanded  # (B, T, S, C)
                j_delta = jvp_operators[l_idx](delta_h)
                delta_taylor = delta_0order + j_delta
                
                # Compute relative Frobenius norm error
                norm_gt = torch.norm(delta_gt_reshaped, p="fro").item() + 1e-6
                err_0 = torch.norm(delta_0order - delta_gt_reshaped, p="fro").item() / norm_gt
                err_t = torch.norm(delta_taylor - delta_gt_reshaped, p="fro").item() / norm_gt
                
                # Cosine similarity
                cos_0 = F.cosine_similarity(delta_0order.flatten(), delta_gt_reshaped.flatten(), dim=0).item()
                cos_t = F.cosine_similarity(delta_taylor.flatten(), delta_gt_reshaped.flatten(), dim=0).item()
                
                layer_errors_0order[l_idx].append(err_0)
                layer_errors_taylor[l_idx].append(err_t)
                layer_cosine_0order[l_idx].append(cos_0)
                layer_cosine_taylor[l_idx].append(cos_t)
                
                x_curr = x_curr + delta_gt
                
    receipt = {
        "protocol_id": "ZT-DIAG-2025-01-TAYLOR-RESIDUAL-MANIFOLD-v001",
        "evaluated_batches": batch_count,
        "stride_k": k,
        "layers": [],
    }
    
    for l_idx in range(num_layers):
        mean_err_0 = float(sum(layer_errors_0order[l_idx]) / max(len(layer_errors_0order[l_idx]), 1))
        mean_err_t = float(sum(layer_errors_taylor[l_idx]) / max(len(layer_errors_taylor[l_idx]), 1))
        mean_cos_0 = float(sum(layer_cosine_0order[l_idx]) / max(len(layer_cosine_0order[l_idx]), 1))
        mean_cos_t = float(sum(layer_cosine_taylor[l_idx]) / max(len(layer_cosine_taylor[l_idx]), 1))
        receipt["layers"].append({
            "layer_index": l_idx,
            "relative_error_0order_copy": round(mean_err_0, 4),
            "relative_error_1order_taylor": round(mean_err_t, 4),
            "cosine_similarity_0order_copy": round(mean_cos_0, 4),
            "cosine_similarity_1order_taylor": round(mean_cos_t, 4),
            "error_reduction_pct": round((mean_err_0 - mean_err_t) / max(mean_err_0, 1e-6) * 100, 2),
        })
        
    logger.info(f"[DIAGNOSTIC SUMMARY] Layer 0 Error: 0-Order={receipt['layers'][0]['relative_error_0order_copy']}, Taylor={receipt['layers'][0]['relative_error_1order_taylor']}")
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote diagnostic receipt to {out_path}")
    return receipt


if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
