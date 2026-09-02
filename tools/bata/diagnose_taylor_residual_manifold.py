# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from opentad.datasets import build_dataloader, build_dataset
from opentad.models import build_detector
from opentad.models.backbones.et_trc_videomae import ETTRCVisionTransformerAdapter  # noqa: F401
from opentad.utils import setup_logger


def _find_trc_backbone(module: nn.Module, depth: int = 0, max_depth: int = 6):
    if depth > max_depth:
        return None
    if hasattr(module, "patch_embed") and hasattr(module, "blocks"):
        return module
    for attr in ("backbone", "model"):
        child = getattr(module, attr, None)
        if child is not None and child is not module:
            result = _find_trc_backbone(child, depth + 1, max_depth=max_depth)
            if result is not None:
                return result
    return None


def _extract_inputs(data_dict: Dict[str, Any]) -> torch.Tensor:
    inputs = data_dict.get("inputs", data_dict.get("imgs", None))
    if isinstance(inputs, dict):
        inputs = inputs.get("inputs", inputs.get("imgs", None))
    if not isinstance(inputs, torch.Tensor):
        raise TypeError("diagnostic batch must contain tensor key 'inputs' or 'imgs'")
    return inputs


def _prepare_inner_backbone_inputs(
    backbone_wrapper: nn.Module,
    inputs: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, int, int]:
    model = getattr(backbone_wrapper, "model", None)
    preprocessor = getattr(model, "data_preprocessor", None)
    if preprocessor is None:
        raise RuntimeError("BackboneWrapper.model lacks the production data_preprocessor")

    inputs = inputs.to(device, non_blocking=True)
    tensor_to_list = getattr(backbone_wrapper, "tensor_to_list", lambda tensor: [t for t in tensor])
    frames, _ = preprocessor.preprocess(
        tensor_to_list(inputs),
        data_samples=None,
        training=False,
    )

    pre_pipeline = getattr(backbone_wrapper, "pre_processing_pipeline", None)
    if pre_pipeline is not None:
        frames = pre_pipeline(dict(frames=frames))["frames"]

    if not isinstance(frames, torch.Tensor) or frames.dim() != 6:
        shape = None if not isinstance(frames, torch.Tensor) else tuple(frames.shape)
        raise RuntimeError(
            "production preprocessing must yield [B, num_segs, C, T, H, W]; "
            f"got {shape}"
        )

    batches, num_segs = map(int, frames.shape[:2])
    frames = frames.flatten(0, 1).contiguous().to(device, non_blocking=True)
    if frames.dim() != 5:
        raise RuntimeError(f"inner backbone input must be [B, C, T, H, W]; got {tuple(frames.shape)}")
    return frames, batches, num_segs


def _add_production_pos_embed(inner_backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
    pos_embed = getattr(inner_backbone, "pos_embed", None)
    if pos_embed is None:
        raise RuntimeError("ET-TRC inner backbone lacks pos_embed")
    if tuple(pos_embed.shape[1:]) != tuple(x.shape[1:]):
        raise RuntimeError(
            "pos_embed shape mismatch after production preprocessing: "
            f"tokens={tuple(x.shape)}, pos_embed={tuple(pos_embed.shape)}"
        )
    return x + pos_embed.to(device=x.device, dtype=x.dtype).clone().detach()


def _infer_patch_geometry(inner_backbone: nn.Module, frames: torch.Tensor, x: torch.Tensor) -> Tuple[int, int, int, int]:
    patch_size = int(getattr(inner_backbone, "patch_size", 16))
    tubelet_size = int(getattr(inner_backbone, "tubelet_size", 2))
    _, _, frame_count, height, width = frames.shape
    if height % patch_size or width % patch_size:
        raise RuntimeError(
            f"input frame size {(height, width)} is not divisible by patch_size={patch_size}"
        )
    h = height // patch_size
    w = width // patch_size
    spatial_tokens = h * w
    if spatial_tokens <= 0 or x.shape[1] % spatial_tokens:
        raise RuntimeError(
            f"cannot derive tubelet geometry from tokens={x.shape[1]} and spatial_tokens={spatial_tokens}"
        )
    tubelet_count = x.shape[1] // spatial_tokens
    expected_tubelets = frame_count // tubelet_size
    if tubelet_count != expected_tubelets:
        raise RuntimeError(
            f"patch token temporal geometry mismatch: tokens imply {tubelet_count} tubelets, "
            f"frames imply {expected_tubelets}"
        )
    return h, w, spatial_tokens, tubelet_count


def _resolve_jvp_operators(blocks: nn.ModuleList, device: torch.device) -> nn.ModuleList:
    jvp_operators = nn.ModuleList()
    missing = []
    for layer_idx, block in enumerate(blocks):
        jvp = getattr(block, "jacobian_approx", None)
        if jvp is None:
            missing.append(layer_idx)
        else:
            jvp_operators.append(jvp.to(device))
    if missing:
        raise RuntimeError(
            "Taylor diagnostic requires trained in-model jacobian_approx modules; "
            f"missing layers {missing}"
        )
    return jvp_operators


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
    logger = setup_logger("TaylorDiagnostic", "diagnostics", distributed_rank=0)
    
    logger.info("Building dataset and model for ZT-DIAG-2025-01 diagnostic...")
    val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
    val_loader = build_dataloader(
        val_dataset,
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
    
    backbone_wrapper = getattr(detector, "backbone", None)
    if backbone_wrapper is None:
        raise RuntimeError("Detector does not contain a backbone")

    inner_backbone = _find_trc_backbone(backbone_wrapper)
    if inner_backbone is None:
        raise RuntimeError(
            f"Cannot find patch_embed+blocks in backbone hierarchy. "
            f"Top-level backbone type: {type(backbone_wrapper).__name__}"
        )
    
    blocks = getattr(inner_backbone, "blocks", None)
    patch_embed = getattr(inner_backbone, "patch_embed", None)
    if blocks is None or len(blocks) == 0:
        raise RuntimeError("ET-TRC inner backbone must expose non-empty blocks")
    if patch_embed is None:
        raise RuntimeError("patch_embed not found on inner backbone")
        
    num_layers = len(blocks)
    layer_errors_0order = [[] for _ in range(num_layers)]
    layer_errors_taylor = [[] for _ in range(num_layers)]
    layer_cosine_0order = [[] for _ in range(num_layers)]
    layer_cosine_taylor = [[] for _ in range(num_layers)]
    
    k = args.stride_k
    jvp_operators = _resolve_jvp_operators(blocks, device)
    
    logger.info(f"Evaluating Taylor vs 0-Order Residual approximations over {args.max_batches} batches (k={k})...")
    
    batch_count = 0
    with torch.no_grad():
        for batch_idx, data_dict in enumerate(val_loader):
            if batch_idx >= args.max_batches:
                break
            batch_count += 1
            inputs = _extract_inputs(data_dict)
            frames, _, _ = _prepare_inner_backbone_inputs(backbone_wrapper, inputs, device)
            
            x = patch_embed(frames)[0]
            x = _add_production_pos_embed(inner_backbone, x)
            
            B, N, C = x.shape
            h, w, spatial_tokens, tubelet_count = _infer_patch_geometry(inner_backbone, frames, x)
            
            x_curr = x
            for l_idx in range(num_layers):
                block = blocks[l_idx]
                if not hasattr(block, "_full_block_residual"):
                    raise RuntimeError(f"ET-TRC layer {l_idx} lacks _full_block_residual")

                delta_gt = block._full_block_residual(x_curr)  # (B, N, C)
                
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
                adapter = getattr(block, "adapter", None)
                if adapter is not None:
                    x_curr = adapter(x_curr, h, w)
                
    receipt = {
        "protocol_id": "ZT-DIAG-2025-01-TAYLOR-RESIDUAL-MANIFOLD-v001",
        "evaluated_batches": batch_count,
        "stride_k": k,
        "production_preprocessing": True,
        "position_embedding": "applied_with_shape_check",
        "jvp_source": "in_model_jacobian_approx",
        "adapter_state_propagated": True,
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
