"""Full-stack CUDA latency, token count, and memory profiling tool for DUCA Evidence Recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from mmengine.config import Config

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from opentad.models.builder import build_detector
from opentad.models.duca.structured_selection import exact_uniform_positions


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(REPO_ROOT),
                stderr=subprocess.DEVNULL,
                text=True,
            )
            .strip()
        )
    except Exception:
        return "unknown"


def profile_model(
    config_path: str,
    num_warmup: int = 50,
    num_iter: int = 200,
    device_str: str = "cuda:0",
) -> Dict[str, Any]:
    """Profile latency, memory, and token saving for a single model config."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is strictly required for formal hardware profiling. CPU fallback is forbidden.")
    if num_warmup < 0 or num_iter <= 0:
        raise ValueError("profiling requires num_warmup >= 0 and num_iter > 0")

    device = torch.device(device_str)
    is_cuda = True

    cfg_path = Path(config_path).resolve()
    cfg = Config.fromfile(str(cfg_path))

    model = build_detector(cfg.model).to(device)
    model.eval()


    # Synthetic 768-window batch (batch_size=1)
    B, C, T, H, W = 1, 3, 768, 160, 160
    # Inputs formatted as [1, 1, 3, 768, 160, 160]
    inputs = torch.randn(B, 1, C, T, H, W, device=device)
    masks = torch.ones(B, T, dtype=torch.bool, device=device)
    metas = [
        {
            "video_name": "profile_sample_0",
            "fps": 25.0,
            "duration": 30.0,
            "snippet_stride": 4,
            "window_size": T,
            "offset_frames": 0,
        }
    ]
    frame_selector_cfg = cfg.model.get("frame_selector", {})
    budget = int(frame_selector_cfg.get("budget", 384))
    if bool(frame_selector_cfg.get("use_h65_selection", False)):
        h65_positions = exact_uniform_positions(T, budget, device="cpu").tolist()
        metas[0]["bata_selected_dense_indices"] = h65_positions
        metas[0]["irregular_selected_positions"] = h65_positions
        metas[0]["selected_dense_indices"] = h65_positions
        metas[0]["irregular_dense_valid_len"] = T
        metas[0]["selected_valid_len"] = budget

    num_classes = int(cfg.model.rpn_head.num_classes)
    ext_cls = [f"class_{idx}" for idx in range(num_classes)]

    if is_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    infer_cfg = getattr(cfg, "inference", None)
    post_cfg = getattr(cfg, "post_processing", None)

    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(
                inputs,
                masks,
                metas,
                return_loss=False,
                infer_cfg=infer_cfg,
                post_cfg=post_cfg,
                ext_cls=ext_cls,
            )
            if is_cuda:
                torch.cuda.synchronize(device)

    latencies_ms = []
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    with torch.no_grad():
        for _ in range(num_iter):
            if is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()

            _ = model(
                inputs,
                masks,
                metas,
                return_loss=False,
                infer_cfg=infer_cfg,
                post_cfg=post_cfg,
                ext_cls=ext_cls,
            )

            if is_cuda:
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)




    latencies_ms.sort()
    p50 = latencies_ms[len(latencies_ms) // 2]
    p95 = latencies_ms[int(len(latencies_ms) * 0.95)]
    mean_lat = sum(latencies_ms) / len(latencies_ms)

    peak_mem_mb = (
        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if is_cuda else 0.0
    )
    peak_reserved_mb = (
        torch.cuda.max_memory_reserved(device) / (1024 * 1024) if is_cuda else 0.0
    )

    return {
        "schema_version": "duca_evidence_recovery_profile_v1",
        "profile_complete": True,
        "config_path": str(cfg_path),
        "config_sha256": _sha256_file(cfg_path),
        "git_commit": _git_commit(),
        "num_warmup": num_warmup,
        "num_iter": num_iter,
        "device": str(device),
        "input_shape": [B, 1, C, T, H, W],
        "batch_size": B,
        "num_classes": num_classes,
        "selected_budget": budget,
        "synthetic_h65_positions": bool(frame_selector_cfg.get("use_h65_selection", False)),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "mean_latency_ms": round(mean_lat, 3),
        "peak_memory_allocated_mb": round(peak_mem_mb, 2),
        "peak_memory_reserved_mb": round(peak_reserved_mb, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile DUCA Evidence Recovery models.")
    parser.add_argument("--config", type=str, required=True, help="Path to model config.")
    parser.add_argument("--output", type=str, default="profile_results.json", help="Output path.")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations.")
    parser.add_argument("--iterations", type=int, default=200, help="Measured iterations.")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device.")
    args = parser.parse_args()

    results = profile_model(
        config_path=args.config,
        num_warmup=args.warmup,
        num_iter=args.iterations,
        device_str=args.device,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
