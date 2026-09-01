# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

PROTOCOL_ID = "ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001"
EMBED_DIM = 384
DEPTH = 12
TUBELETS_PER_CHUNK = 8
SOURCE_BYTES_PER_PIXEL = 1
C_EXEC_CEILING = 0.70

ARMS_COMPUTE_SPEC = {
    "D160": dict(global_res=160, global_patches=(10, 10), global_chunks=48, local_res=0, local_patches=(0, 0), local_chunks=0),
    "G96": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=0, local_patches=(0, 0), local_chunks=0),
    "U128-ALL48-A0": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=128, local_patches=(8, 8), local_chunks=48),
    "U16-UNIFORM-A0": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=128, local_patches=(8, 8), local_chunks=16),
    "BAFDR-K16-LATE": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=128, local_patches=(8, 8), local_chunks=16),
    "BAFDR-K16-NOKD": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=128, local_patches=(8, 8), local_chunks=16),
    "BAFDR-K16-FULL": dict(global_res=96, global_patches=(6, 6), global_chunks=48, local_res=128, local_patches=(8, 8), local_chunks=16),
}


def tokens_per_chunk(patch_hw: tuple[int, int]) -> int:
    return TUBELETS_PER_CHUNK * patch_hw[0] * patch_hw[1]


def videomae_layer_ops(tokens: int, dim: int = EMBED_DIM) -> Dict[str, float]:
    qkv_proj = 4.0 * tokens * dim * dim
    attention = 2.0 * tokens * tokens * dim
    ffn = 8.0 * tokens * dim * dim
    norm_residual = 6.0 * tokens * dim
    return {
        "qkv_and_output_proj_flops": qkv_proj,
        "attention_flops": attention,
        "ffn_flops": ffn,
        "norm_residual_flops": norm_residual,
        "total_flops": qkv_proj + attention + ffn + norm_residual,
    }


def videomae_branch_ops(chunks: int, patch_hw: tuple[int, int]) -> Dict[str, Any]:
    tokens = tokens_per_chunk(patch_hw)
    per_layer = videomae_layer_ops(tokens)
    total = per_layer["total_flops"] * DEPTH * chunks
    return {
        "chunks": chunks,
        "tokens_per_chunk": tokens,
        "total_tokens_evaluated": chunks * tokens,
        "per_layer": per_layer,
        "depth": DEPTH,
        "total_flops": total,
    }


def router_ops(num_chunks: int = 48, channels: int = EMBED_DIM, hidden: int = 128) -> Dict[str, float]:
    layer_norm = float(num_chunks * channels * 5)
    conv3 = float(num_chunks * channels * hidden * 3)
    conv1 = float(num_chunks * hidden * 4)
    score_ops = float(num_chunks * (channels + 16))
    return {
        "layer_norm_flops": layer_norm,
        "conv3_flops": conv3,
        "conv1_flops": conv1,
        "score_and_topk_flops_proxy": score_ops,
        "total_flops": layer_norm + conv3 + conv1 + score_ops,
    }


def residual_projection_ops(local_tubelets: int, channels: int = EMBED_DIM) -> Dict[str, float]:
    proj_local = float(local_tubelets * channels * channels)
    proj_global = float(local_tubelets * channels * channels)
    gate_and_scatter = float(local_tubelets * channels * 4)
    return {
        "proj_local_1x1_flops": proj_local,
        "proj_global_1x1_flops": proj_global,
        "gate_scatter_add_flops": gate_and_scatter,
        "total_flops": proj_local + proj_global + gate_and_scatter,
    }


def local_h2d_bytes(chunks: int, local_res: int) -> int:
    return chunks * TUBELETS_PER_CHUNK * 2 * 3 * local_res * local_res * SOURCE_BYTES_PER_PIXEL


def compute_arm_c_exec(arm_name: str) -> Dict[str, Any]:
    spec = ARMS_COMPUTE_SPEC[arm_name]
    global_ops = videomae_branch_ops(spec["global_chunks"], spec["global_patches"])
    local_ops = videomae_branch_ops(spec["local_chunks"], spec["local_patches"]) if spec["local_chunks"] else {
        "chunks": 0,
        "tokens_per_chunk": 0,
        "total_tokens_evaluated": 0,
        "per_layer": {},
        "depth": DEPTH,
        "total_flops": 0.0,
    }
    router = router_ops() if arm_name.startswith("BAFDR") or arm_name.startswith("U16") else {
        "layer_norm_flops": 0.0,
        "conv3_flops": 0.0,
        "conv1_flops": 0.0,
        "score_and_topk_flops_proxy": 0.0,
        "total_flops": 0.0,
    }
    residual_ops = residual_projection_ops(spec["local_chunks"] * TUBELETS_PER_CHUNK) if spec["local_chunks"] else {
        "proj_local_1x1_flops": 0.0,
        "proj_global_1x1_flops": 0.0,
        "gate_scatter_add_flops": 0.0,
        "total_flops": 0.0,
    }
    total_flops = global_ops["total_flops"] + local_ops["total_flops"] + router["total_flops"] + residual_ops["total_flops"]
    d160_reference = videomae_branch_ops(48, (10, 10))["total_flops"]
    ratio = total_flops / d160_reference
    return {
        "protocol_id": PROTOCOL_ID,
        "arm": arm_name,
        "global": global_ops,
        "local": local_ops,
        "router": router,
        "residual_projection": residual_ops,
        "local_h2d_bytes_uint8": local_h2d_bytes(spec["local_chunks"], spec["local_res"]) if spec["local_chunks"] else 0,
        "total_tokens_evaluated": global_ops["total_tokens_evaluated"] + local_ops["total_tokens_evaluated"],
        "total_flops": total_flops,
        "estimated_gflops": round(total_flops / 1.0e9, 4),
        "c_exec_ratio_vs_d160": round(ratio, 6),
        "satisfies_0_70_ceiling": ratio <= C_EXEC_CEILING,
        "notes": "Static operator ledger for architecture comparison; excludes dataloader wall time and shared detector head cost.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="BA-FDR C_exec operator ledger")
    parser.add_argument("--output", type=str, default="c_exec_summary.json")
    args = parser.parse_args()

    results = {arm: compute_arm_c_exec(arm) for arm in ARMS_COMPUTE_SPEC}
    print(f"{'ARM':<18} | {'Tokens':<8} | {'C_exec':<10} | {'GFLOPs':<10} | {'<=0.70'}")
    print("-" * 66)
    for arm, res in results.items():
        print(
            f"{arm:<18} | {res['total_tokens_evaluated']:<8} | "
            f"{res['c_exec_ratio_vs_d160']:<10.6f} | {res['estimated_gflops']:<10.4f} | "
            f"{res['satisfies_0_70_ceiling']}"
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n[C_exec] Operator ledger written to {output}")


if __name__ == "__main__":
    main()
