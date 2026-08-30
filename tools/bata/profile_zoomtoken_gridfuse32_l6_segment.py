#!/usr/bin/env python3
"""Matched real-shape six-block G0 profiler for GridFuse32-L6."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot summarize an empty timing vector")
    rank = probability * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": _percentile(values, 0.95),
        "mean_ms": statistics.fmean(values),
        "min_ms": min(values),
        "max_ms": max(values),
    }


def _write_exclusive(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _strip_ddp_prefix(state: dict[str, Any]) -> dict[str, Any]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _lineage(torch: Any, device: Any):
    tubelets = (
        torch.arange(8, device=device, dtype=torch.long)
        .repeat_interleave(64)
        .view(1, 512)
    )
    rectangle = (
        torch.arange(8, device=device, dtype=torch.long).view(8, 1) * 10
        + torch.arange(8, device=device, dtype=torch.long).view(1, 8)
    ).reshape(-1)
    spatial = rectangle.repeat(8).view(1, 512)
    positions = [torch.arange(512, device=device, dtype=torch.long).view(1, 512)]
    return tubelets, spatial, positions


def _empty_stats() -> dict[str, int]:
    return {
        "ragged_attention_bucket_call_count": 0,
        "ragged_mlp_bucket_call_count": 0,
        "ragged_adapter_forward_count": 0,
        "executed_attention_tokens": 0,
        "executed_kv_tokens": 0,
        "executed_attention_pairs": 0,
        "executed_mlp_tokens": 0,
        "executed_adapter_tokens": 0,
        "gridfuse_bucket_call_count": 0,
    }


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from mmengine import Config

    from opentad.models import build_detector

    if not torch.cuda.is_available():
        raise RuntimeError("GridFuse32 G0 requires a Slurm-visible CUDA device")
    if int(torch.cuda.device_count()) != 1:
        raise RuntimeError("GridFuse32 G0 requires exactly one visible GPU")
    if int(args.warmup) != 100 or int(args.iterations) < 500:
        raise ValueError("GridFuse32 G0 requires 100 warmups and at least 500 samples per arm")
    expected_commit = str(args.expected_commit).lower()
    if len(expected_commit) != 40 or any(ch not in "0123456789abcdef" for ch in expected_commit):
        raise ValueError("expected commit must be a full lowercase SHA")
    if _git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("GridFuse32 G0 source commit differs from the reviewed candidate")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("GridFuse32 G0 requires a clean source snapshot")
    if not args.checkpoint.is_file() or not args.config.is_file():
        raise FileNotFoundError("GridFuse32 G0 checkpoint or config is missing")

    args.run_root.mkdir(parents=True, exist_ok=False)
    cfg = Config.fromfile(str(args.config))
    route = cfg.model.backbone.backbone.gridfuse32_l6
    if (
        tuple(route.dense_block_indices) != tuple(range(6))
        or tuple(route.fused_block_indices) != tuple(range(6, 12))
        or int(route.native_tokens_per_clip) != 512
        or int(route.merged_tokens_per_clip) != 256
    ):
        raise RuntimeError("GridFuse32 G0 config changed the frozen mechanism")

    model_cfg = cfg.model.copy()
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != 59 or "state_dict_ema" not in checkpoint:
        raise ValueError("GridFuse32 G0 requires the frozen epoch-59 EMA checkpoint")
    model.load_state_dict(_strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True)
    del checkpoint

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = model.to(device).eval()
    wrapper = getattr(model, "backbone", None)
    recognized = getattr(wrapper, "model", None)
    heavy = getattr(recognized, "backbone", None)
    if heavy is None or len(getattr(heavy, "blocks", ())) != 12:
        raise RuntimeError("GridFuse32 G0 could not bind the 12-block VideoMAE backbone")
    blocks = tuple(heavy.blocks[6:12])
    if len(blocks) != 6 or not all(block.use_adapter for block in blocks):
        raise RuntimeError("GridFuse32 G0 requires the same dense Adapter in all six blocks")

    torch.manual_seed(42)
    inputs = torch.randn(1, 512, 384, device=device, dtype=torch.float32)
    tubelets, spatial, positions = _lineage(torch, device)

    def execute(arm: str, stats: dict[str, int] | None = None):
        value = inputs
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            for local_index, block in enumerate(blocks, start=6):
                value = block.forward_native_ragged(
                    value,
                    bucket_positions=positions,
                    tubelet_indices=tubelets,
                    spatial_indices=spatial,
                    total_tubelets=8,
                    grid_height=10,
                    grid_width=10,
                    packed_stats=stats,
                    gridfuse_orientation=(
                        None
                        if arm == "dense"
                        else ("horizontal" if local_index % 2 == 0 else "vertical")
                    ),
                    record_kv_tokens=arm == "dense",
                )
        return value

    dense_ledger = _empty_stats()
    candidate_ledger = _empty_stats()
    execute("dense", dense_ledger)
    execute("candidate", candidate_ledger)
    torch.cuda.synchronize(device)
    expected_dense = {
        "executed_attention_tokens": 6 * 512,
        "executed_kv_tokens": 6 * 512,
        "executed_attention_pairs": 6 * 512 * 512,
        "executed_mlp_tokens": 6 * 512,
        "executed_adapter_tokens": 6 * 512,
    }
    expected_candidate = {
        "executed_attention_tokens": 6 * 256,
        "executed_kv_tokens": 6 * 256,
        "executed_attention_pairs": 6 * 256 * 256,
        "executed_mlp_tokens": 6 * 256,
        "executed_adapter_tokens": 6 * 512,
    }
    for key, expected in expected_dense.items():
        if int(dense_ledger[key]) != expected:
            raise RuntimeError(f"dense G0 ledger mismatch for {key}")
    for key, expected in expected_candidate.items():
        if int(candidate_ledger[key]) != expected:
            raise RuntimeError(f"candidate G0 ledger mismatch for {key}")
    if int(candidate_ledger["gridfuse_bucket_call_count"]) != 6:
        raise RuntimeError("candidate G0 did not execute exactly six fused blocks")

    for index in range(int(args.warmup)):
        order = ("dense", "candidate") if index % 2 == 0 else ("candidate", "dense")
        for arm in order:
            execute(arm)
    torch.cuda.synchronize(device)

    def timed(call: Callable[[], Any]) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = call()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        del output
        return elapsed

    samples = {"dense": [], "candidate": []}
    started = time.time()
    for index in range(int(args.iterations)):
        order = ("dense", "candidate") if index % 2 == 0 else ("candidate", "dense")
        for arm in order:
            samples[arm].append(timed(lambda selected=arm: execute(selected)))

    memory: dict[str, dict[str, float]] = {}
    for arm in ("dense", "candidate"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        output = execute(arm)
        torch.cuda.synchronize(device)
        memory[arm] = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        del output

    timing = {arm: _summary(values) for arm, values in samples.items()}
    speedup = timing["dense"]["p50_ms"] / timing["candidate"]["p50_ms"]
    allocated_ratio = (
        memory["candidate"]["peak_allocated_bytes"]
        / memory["dense"]["peak_allocated_bytes"]
    )
    reserved_ratio = (
        memory["candidate"]["peak_reserved_bytes"]
        / memory["dense"]["peak_reserved_bytes"]
    )
    passed = speedup >= 1.35 and allocated_ratio <= 1.05 and reserved_ratio <= 1.05
    return {
        "schema_version": "zoomtoken_gridfuse32_l6_g0_profile_v001",
        "status": (
            "GRIDFUSE32_L6_G0_PASS_PENDING_G1"
            if passed
            else "STOP_GRIDFUSE32_L6_BEFORE_TRAINING"
        ),
        "gate_passed": passed,
        "source_commit": expected_commit,
        "config": str(args.config.resolve()),
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "sha256": _sha256(args.checkpoint),
            "epoch": 59,
            "state": "state_dict_ema",
        },
        "shape": {
            "batch_size": 1,
            "tubelets_per_clip": 8,
            "tokens_per_tubelet": 64,
            "dense_tokens": 512,
            "candidate_tokens": 256,
            "embed_dims": 384,
            "num_heads": 6,
            "dtype": "fp16_autocast",
            "blocks": list(range(6, 12)),
            "dense_adapter": True,
        },
        "protocol": {
            "warmup_per_arm": int(args.warmup),
            "timed_per_arm": int(args.iterations),
            "alternating_order": True,
            "synchronized": True,
            "candidate_only_compile": False,
        },
        "timing": timing,
        "memory": memory,
        "ledger": {"dense": dense_ledger, "candidate": candidate_ledger},
        "gate": {
            "p50_speedup": speedup,
            "p50_speedup_min": 1.35,
            "allocated_ratio": allocated_ratio,
            "allocated_ratio_max": 1.05,
            "reserved_ratio": reserved_ratio,
            "reserved_ratio_max": 1.05,
            "p95_report_only": True,
        },
        "started_unix_s": started,
        "ended_unix_s": time.time(),
        "slurm": {
            "job_id": __import__("os").environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "training_or_resume_executed": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=500)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    terminal: dict[str, Any]
    try:
        result = profile(args)
        _write_exclusive(args.run_root / "profile.json", result)
        terminal = {
            "schema_version": "zoomtoken_gridfuse32_l6_g0_terminal_v001",
            "status": result["status"],
            "gate_passed": result["gate_passed"],
            "profile": str((args.run_root / "profile.json").resolve()),
        }
        _write_exclusive(args.run_root / "terminal_receipt.json", terminal)
        print(json.dumps(terminal, sort_keys=True))
        return 0 if result["gate_passed"] else 3
    except Exception as exc:
        args.run_root.mkdir(parents=True, exist_ok=True)
        terminal = {
            "schema_version": "zoomtoken_gridfuse32_l6_g0_terminal_v001",
            "status": "GRIDFUSE32_L6_G0_ENGINEERING_OR_PROTOCOL_BLOCKER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        terminal_path = args.run_root / "terminal_receipt.json"
        if not terminal_path.exists():
            _write_exclusive(terminal_path, terminal)
        print(json.dumps(terminal, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
