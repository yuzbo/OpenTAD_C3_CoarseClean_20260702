"""Matched, no-data RACER24 block microbenchmark for the N16R4 evaluator."""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opentad.models.backbones.vit_adapter import Block


def _arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--measurements", type=int, default=200)
    parser.add_argument("--min-speedup", type=float, default=1.08)
    parser.add_argument("--max-memory-ratio", type=float, default=1.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    return parser.parse_args()


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * float(fraction)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _lineage(device):
    spatial = torch.tensor(
        [row * 10 + col for row in range(8) for col in range(8)],
        device=device,
        dtype=torch.long,
    ).repeat(8).view(1, 512)
    tubelets = torch.arange(8, device=device, dtype=torch.long).repeat_interleave(64)
    return tubelets.view(1, 512), spatial


def _stats():
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "ragged_adapter_forward_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
        "executed_adapter_tokens": 0,
        "racer24_block_forward_count": 0,
        "racer24_clip_count": 0,
        "racer24_selected_query_tokens": 0,
    }


def main():
    args = _arguments()
    if args.measurements < 200:
        raise SystemExit("RACER24 requires at least 200 timed measurements")
    if args.warmup < 1:
        raise SystemExit("RACER24 requires a positive warmup")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise SystemExit("RACER24 real-shape profiling requires CUDA")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    block = Block(
        embed_dims=384,
        num_heads=6,
        mlp_ratio=4.0,
        use_adapter=True,
        temporal_size=8,
        with_cp=False,
    ).eval().to(device=device, dtype=dtype)
    carrier = torch.randn(1, 512, 384, device=device, dtype=dtype)
    previous_residual = torch.randn_like(carrier)
    tubelets, spatial = _lineage(device)
    bucket_positions = [torch.arange(512, device=device).view(1, 512)]

    common = dict(
        bucket_positions=bucket_positions,
        tubelet_indices=tubelets,
        spatial_indices=spatial,
        total_tubelets=8,
        grid_height=10,
        grid_width=10,
        count_full_kv_tokens=True,
    )

    def run_control():
        return block.forward_native_ragged(
            carrier,
            packed_stats=_stats(),
            **common,
        )

    def run_candidate():
        return block.forward_native_ragged(
            carrier,
            packed_stats=_stats(),
            racer24_previous_dense_residual=previous_residual,
            **common,
        )

    with torch.inference_mode():
        for index in range(args.warmup):
            (run_control if index % 2 == 0 else run_candidate)()
        torch.cuda.synchronize()

        timings = {"r1_dense": [], "racer24": []}
        for index in range(args.measurements):
            order = (
                (("r1_dense", run_control), ("racer24", run_candidate))
                if index % 2 == 0
                else (("racer24", run_candidate), ("r1_dense", run_control))
            )
            for name, function in order:
                torch.cuda.synchronize()
                started = time.perf_counter()
                output = function()
                torch.cuda.synchronize()
                timings[name].append((time.perf_counter() - started) * 1000.0)
                del output

        memory = {}
        for name, function in (("r1_dense", run_control), ("racer24", run_candidate)):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            output = function()
            torch.cuda.synchronize()
            memory[name] = {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            }
            del output

    latency = {
        name: {
            "p50_ms": _percentile(values, 0.50),
            "p95_ms": _percentile(values, 0.95),
        }
        for name, values in timings.items()
    }
    speedup = latency["r1_dense"]["p50_ms"] / latency["racer24"]["p50_ms"]
    allocated_ratio = (
        memory["racer24"]["peak_allocated_bytes"]
        / memory["r1_dense"]["peak_allocated_bytes"]
    )
    reserved_ratio = (
        memory["racer24"]["peak_reserved_bytes"]
        / memory["r1_dense"]["peak_reserved_bytes"]
    )
    passed = (
        speedup >= args.min_speedup
        and allocated_ratio <= args.max_memory_ratio
        and reserved_ratio <= args.max_memory_ratio
    )
    result = {
        "schema_version": "zoomtoken_racer24_block_profile_v001",
        "shape": {
            "batch": 1,
            "tubelets": 8,
            "tokens_per_tubelet": 64,
            "dense_tokens": 512,
            "selected_per_tubelet": 24,
            "selected_queries": 192,
            "embed_dims": 384,
            "heads": 6,
        },
        "measurements_per_arm": args.measurements,
        "warmup": args.warmup,
        "latency": latency,
        "memory": memory,
        "p50_speedup": speedup,
        "peak_allocated_ratio": allocated_ratio,
        "peak_reserved_ratio": reserved_ratio,
        "gates": {
            "minimum_p50_speedup": args.min_speedup,
            "maximum_peak_memory_ratio": args.max_memory_ratio,
            "passed": passed,
        },
        "scope": "matched_block_only_not_full_stack_tad",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
