# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

ARMS_COMPUTE_SPEC = {
    "D160": {
        "global_res": 160,
        "global_patches": (10, 10),
        "global_chunks": 48,
        "local_res": 0,
        "local_patches": (0, 0),
        "local_chunks": 0,
    },
    "G96": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 0,
        "local_patches": (0, 0),
        "local_chunks": 0,
    },
    "U128-ALL48-A0": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 128,
        "local_patches": (8, 8),
        "local_chunks": 48,
    },
    "U16-UNIFORM-A0": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 128,
        "local_patches": (8, 8),
        "local_chunks": 16,
    },
    "BAFDR-K16-LATE": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 128,
        "local_patches": (8, 8),
        "local_chunks": 16,
    },
    "BAFDR-K16-NOKD": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 128,
        "local_patches": (8, 8),
        "local_chunks": 16,
    },
    "BAFDR-K16-FULL": {
        "global_res": 96,
        "global_patches": (6, 6),
        "global_chunks": 48,
        "local_res": 128,
        "local_patches": (8, 8),
        "local_chunks": 16,
    },
}


def compute_arm_c_exec(arm_name: str) -> Dict[str, Any]:
    spec = ARMS_COMPUTE_SPEC[arm_name]
    tubelets_per_chunk = 8
    
    # Global tokens
    g_hw = spec["global_patches"][0] * spec["global_patches"][1]
    g_tokens_per_chunk = tubelets_per_chunk * g_hw
    g_total_tokens = spec["global_chunks"] * g_tokens_per_chunk

    # Local tokens
    l_hw = spec["local_patches"][0] * spec["local_patches"][1]
    l_tokens_per_chunk = tubelets_per_chunk * l_hw
    l_total_tokens = spec["local_chunks"] * l_tokens_per_chunk

    total_tokens = g_total_tokens + l_total_tokens
    
    # D160 baseline tokens
    d160_tokens = 48 * (8 * 10 * 10)  # 38,400
    ratio = total_tokens / float(d160_tokens)

    # 12-layer VideoMAE FLOP estimation (embed_dim=384, depth=12)
    # Self-attention + FFN per token ~ 24 * d^2 FLOPs
    flops_gflops = (total_tokens * 24 * 384 * 384 * 12) / 1e9

    return {
        "arm": arm_name,
        "global_tokens": g_total_tokens,
        "local_tokens": l_total_tokens,
        "total_tokens_evaluated": total_tokens,
        "c_exec_ratio_vs_d160": round(ratio, 4),
        "estimated_gflops": round(flops_gflops, 2),
        "satisfies_0_70_ceiling": ratio <= 0.70,
    }


def main():
    parser = argparse.ArgumentParser(description="BA-FDR C_exec Calculator")
    parser.add_argument("--output", type=str, default="c_exec_summary.json")
    args = parser.parse_args()

    results = {}
    print(f"{'ARM':<18} | {'Tokens':<8} | {'C_exec Ratio':<12} | {'GFLOPs':<8} | {'<=0.70 Ceiling'}")
    print("-" * 65)
    for arm in ARMS_COMPUTE_SPEC:
        res = compute_arm_c_exec(arm)
        results[arm] = res
        print(f"{arm:<18} | {res['total_tokens_evaluated']:<8} | {res['c_exec_ratio_vs_d160']:<12.4f} | {res['estimated_gflops']:<8.2f} | {res['satisfies_0_70_ceiling']}")

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[C_exec] Analysis written to {args.output}")


if __name__ == "__main__":
    main()
