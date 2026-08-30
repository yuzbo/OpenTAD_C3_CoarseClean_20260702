#!/usr/bin/env python3
"""Conditional same-GPU matched full-stack G2 profiler for GridFuse32-L6."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
    _move_to_device,
    _sample_identity,
    integrate_energy,
)
from tools.bata.spatial_zoom_s1_power import NvidiaSmiPowerSampler  # noqa: E402


ORDER = ("R1", "C", "C", "R1", "C", "R1", "R1", "C")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strip_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    rank = probability * (len(ordered) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("full-stack profiler cannot summarize an empty vector")
    return {
        "count": len(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl_exclusive(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _bind_validation(cfg: Any, args: argparse.Namespace) -> None:
    for split in ("train", "val", "test"):
        target = getattr(cfg.dataset, split)
        target.ann_file = str(args.annotation)
        target.class_map = str(args.class_map)
        target.data_path = str(args.video_root)
        target.subset_name = "validation" if split != "train" else "training"
    cfg.evaluation.ground_truth_filename = str(args.annotation)


def _build_arm(arm: str, args: argparse.Namespace, device: Any):
    import torch
    from mmengine import Config

    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector

    config_path = args.control_config if arm == "R1" else args.candidate_config
    checkpoint_path = args.control_checkpoint if arm == "R1" else args.candidate_checkpoint
    cfg = Config.fromfile(str(config_path))
    _bind_validation(cfg, args)
    cfg.post_processing.sliding_window = True
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    if len(dataset.data_list) != 211:
        raise ValueError("GridFuse32 G2 requires the canonical 211-video validation set")
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        batch_size=1,
        num_workers=0,
    )
    if len(loader) != 792:
        raise ValueError("GridFuse32 G2 requires the canonical 792 loader items")
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != 59 or "state_dict_ema" not in checkpoint:
        raise ValueError(f"GridFuse32 G2 {arm} checkpoint is not epoch-59 EMA")
    model.load_state_dict(_strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True)
    del checkpoint
    return cfg, dataset, loader, model.to(device).eval(), checkpoint_path


def _validate_runtime_ledger(arm: str, model: Any) -> dict[str, Any]:
    wrapper = getattr(model, "backbone", None)
    recognized = getattr(wrapper, "model", None)
    heavy = getattr(recognized, "backbone", None)
    summary = getattr(heavy, "latest_native_packed_summary", None)
    if not isinstance(summary, Mapping):
        raise RuntimeError("GridFuse32 G2 missing the native ragged execution ledger")
    if (
        summary.get("execution_mode") != "true_clip_ragged_no_padding"
        or int(summary.get("window_token_budget", -1)) != 24_576
        or int(summary.get("padded_heavy_tokens_per_window", -1)) != 0
        or int(summary.get("executed_adapter_tokens_all_blocks", -1)) != 294_912
    ):
        raise RuntimeError("GridFuse32 G2 changed the frozen R1 carrier ledger")
    if arm == "C":
        if (
            summary.get("gridfuse_schema_version") != "zoomtoken_gridfuse32_l6_v001"
            or int(summary.get("dense_block_count", -1)) != 6
            or int(summary.get("gridfuse_block_count", -1)) != 6
            or int(summary.get("executed_attention_tokens_all_blocks", -1)) != 221_184
            or int(summary.get("executed_kv_tokens_all_blocks", -1)) != 221_184
            or int(summary.get("executed_mlp_tokens_all_blocks", -1)) != 221_184
            or int(summary.get("attention_pairs_all_blocks", -1)) != 94_371_840
        ):
            raise RuntimeError("GridFuse32 G2 candidate heavy-token ledger changed")
    else:
        if summary.get("gridfuse_schema_version") is not None:
            raise RuntimeError("GridFuse32 G2 control unexpectedly enabled GridFuse")
    return {
        key: summary[key]
        for key in (
            "schema_version",
            "execution_mode",
            "refresh_execution_mode",
            "window_token_budget",
            "padded_heavy_tokens_per_window",
            "attention_pairs_all_blocks",
            "executed_attention_tokens_all_blocks",
            "executed_kv_tokens_all_blocks",
            "executed_mlp_tokens_all_blocks",
            "executed_adapter_tokens_all_blocks",
        )
        if key in summary
    } | {
        "gridfuse_schema_version": summary.get("gridfuse_schema_version"),
    }


def _profile_pass(
    arm: str,
    pass_index: int,
    args: argparse.Namespace,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    from opentad.cores.test_engine import gather_ddp_results

    cfg, dataset, loader, model, checkpoint_path = _build_arm(arm, args, device)
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)

    def forward_once(batch):
        with torch.no_grad(), torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=True
        ):
            return model(
                **batch,
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=external_cls,
            )

    iterator = iter(loader)

    def next_batch():
        nonlocal iterator
        try:
            return next(iterator)
        except StopIteration:
            iterator = iter(loader)
            return next(iterator)

    for _ in range(50):
        forward_once(_move_to_device(next_batch(), device))
    synchronize()
    iterator = iter(loader)

    rows: list[dict[str, Any]] = []
    video_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    route_receipt = None
    for ordinal in range(792):
        synchronize()
        started = time.perf_counter()
        energy_started = time.perf_counter()
        input_started = time.perf_counter()
        cpu_batch = next_batch()
        input_ms = (time.perf_counter() - input_started) * 1000.0
        identity = _sample_identity(cpu_batch, ordinal)
        torch.cuda.reset_peak_memory_stats(device)
        h2d_started = time.perf_counter()
        gpu_batch = _move_to_device(cpu_batch, device)
        synchronize()
        h2d_ms = (time.perf_counter() - h2d_started) * 1000.0
        result = forward_once(gpu_batch)
        synchronize()
        ended = time.perf_counter()
        if not isinstance(result, Mapping):
            raise RuntimeError("GridFuse32 G2 detector returned no result mapping")
        for video_id, detections in result.items():
            video_rows[str(video_id)].extend(detections)
        current_route = _validate_runtime_ledger(arm, model)
        if route_receipt is None:
            route_receipt = current_route
        elif current_route != route_receipt:
            raise RuntimeError("GridFuse32 G2 route ledger changed within a pass")
        rows.append(
            {
                "pass_index": pass_index,
                "arm": arm,
                "ordinal": ordinal,
                **identity,
                "input_pipeline_serial_ms": input_ms,
                "h2d_ms": h2d_ms,
                "decode_to_window_output_wall_ms": (ended - started) * 1000.0,
                "final_video_nms_ms": 0.0,
                "end_to_end_serial_ms": (ended - started) * 1000.0,
                "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                "energy_window_perf_s": [energy_started, ended],
                "gpu_energy_j": None,
            }
        )
        del cpu_batch, gpu_batch, result
    synchronize()
    nms_started = time.perf_counter()
    finalized = gather_ddp_results(1, video_rows, cfg.post_processing)
    synchronize()
    nms_ended = time.perf_counter()
    if not isinstance(finalized, Mapping):
        raise RuntimeError("GridFuse32 G2 official Soft-NMS finalizer failed")
    amortized_nms_ms = (nms_ended - nms_started) * 1000.0 / len(rows)
    for row in rows:
        row["final_video_nms_ms"] = amortized_nms_ms
        row["end_to_end_serial_ms"] += amortized_nms_ms
        row["nms_energy_window_perf_s"] = [nms_started, nms_ended]

    receipt = {
        "pass_index": pass_index,
        "arm": arm,
        "sample_count": len(rows),
        "video_count": len(dataset.data_list),
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "route_ledger": route_receipt,
        "training_or_resume_executed": False,
    }
    del model, loader, dataset
    torch.cuda.empty_cache()
    return rows, receipt


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available() or int(torch.cuda.device_count()) != 1:
        raise RuntimeError("GridFuse32 G2 requires exactly one Slurm-visible GPU")
    if int(os.environ.get("WORLD_SIZE", -1)) != 1 or int(
        os.environ.get("LOCAL_RANK", -1)
    ) != 0:
        raise RuntimeError("GridFuse32 G2 requires torchrun world-size one")
    if dist.is_initialized():
        raise RuntimeError("GridFuse32 G2 requires a fresh process group")
    dist.init_process_group("nccl", rank=0, world_size=1)
    expected_commit = str(args.expected_commit).lower()
    if _git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("GridFuse32 G2 source commit differs from the reviewed candidate")
    if _git("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("GridFuse32 G2 requires a clean source snapshot")
    for path in (
        args.control_config,
        args.candidate_config,
        args.control_checkpoint,
        args.candidate_checkpoint,
        args.annotation,
        args.class_map,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"GridFuse32 G2 required file is missing: {path}")
    if not args.video_root.is_dir():
        raise FileNotFoundError("GridFuse32 G2 canonical video root is missing")
    args.run_root.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    sampler = NvidiaSmiPowerSampler(gpu_id=physical_gpu, interval_ms=20)
    all_rows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    sampler.start()
    time.sleep(sampler.interval_s * 1.5)
    try:
        for pass_index, arm in enumerate(ORDER):
            rows, receipt = _profile_pass(arm, pass_index, args, device)
            all_rows.extend(rows)
            receipts.append(receipt)
    finally:
        time.sleep(sampler.interval_s * 1.5)
        sampler.stop()

    pass_counts = {index: 792 for index in range(len(ORDER))}
    for row in all_rows:
        start, end = row.pop("energy_window_perf_s")
        nms_start, nms_end = row.pop("nms_energy_window_perf_s")
        sample_energy = integrate_energy(sampler.samples, start=start, end=end)
        nms_energy = integrate_energy(sampler.samples, start=nms_start, end=nms_end)
        if sample_energy is None or nms_energy is None:
            raise RuntimeError("GridFuse32 G2 raw power trace has incomplete coverage")
        row["gpu_energy_j"] = sample_energy + nms_energy / pass_counts[row["pass_index"]]

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in all_rows:
        grouped[row["arm"]].append(row)
    summaries = {}
    for arm in ("R1", "C"):
        rows = grouped[arm]
        summaries[arm] = {
            "sample_count": len(rows),
            "pass_count": ORDER.count(arm),
            "end_to_end_serial_ms": _summary(
                [float(row["end_to_end_serial_ms"]) for row in rows]
            ),
            "gross_gpu_energy_j": sum(float(row["gpu_energy_j"]) for row in rows),
            "peak_gpu_allocated_bytes": max(row["peak_gpu_allocated_bytes"] for row in rows),
            "peak_gpu_reserved_bytes": max(row["peak_gpu_reserved_bytes"] for row in rows),
        }
    ratios = {
        "p50": summaries["C"]["end_to_end_serial_ms"]["p50"]
        / summaries["R1"]["end_to_end_serial_ms"]["p50"],
        "gross_energy": summaries["C"]["gross_gpu_energy_j"]
        / summaries["R1"]["gross_gpu_energy_j"],
        "peak_allocated": summaries["C"]["peak_gpu_allocated_bytes"]
        / summaries["R1"]["peak_gpu_allocated_bytes"],
        "peak_reserved": summaries["C"]["peak_gpu_reserved_bytes"]
        / summaries["R1"]["peak_gpu_reserved_bytes"],
    }
    passed = (
        ratios["p50"] <= 0.95
        and ratios["gross_energy"] <= 0.95
        and ratios["peak_allocated"] <= 1.05
        and ratios["peak_reserved"] <= 1.05
    )
    power_origin = sampler.samples[0][0]
    power_rows = [
        {
            "sequence": index,
            "timestamp_ms": (timestamp - power_origin) * 1000.0,
            "power_w": power,
        }
        for index, (timestamp, power) in enumerate(sampler.samples)
    ]
    _write_jsonl_exclusive(args.run_root / "raw_samples.jsonl", all_rows)
    _write_jsonl_exclusive(args.run_root / "raw_power_trace.jsonl", power_rows)
    return {
        "schema_version": "zoomtoken_gridfuse32_l6_g2_fullstack_v001",
        "status": (
            "GRIDFUSE32_L6_G2_FULLSTACK_PASS_PENDING_FRESH_PRO"
            if passed
            else "GRIDFUSE32_L6_G2_VALID_NEGATIVE_PENDING_FRESH_PRO"
        ),
        "gate_passed": passed,
        "source_commit": expected_commit,
        "order": list(ORDER),
        "warmup_samples_per_pass": 50,
        "canonical_video_count": 211,
        "canonical_loader_item_count": 792,
        "scope": "decode_to_soft_nms",
        "arm_summaries": summaries,
        "ratios": ratios,
        "gates": {
            "p50_ratio_max": 0.95,
            "gross_energy_ratio_max": 0.95,
            "peak_allocated_ratio_max": 1.05,
            "peak_reserved_ratio_max": 1.05,
        },
        "pass_receipts": receipts,
        "artifacts": {
            "raw_samples": str((args.run_root / "raw_samples.jsonl").resolve()),
            "raw_power_trace": str((args.run_root / "raw_power_trace.jsonl").resolve()),
        },
        "training_or_resume_executed": False,
        "official_test_opened": False,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-config", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        payload = profile(args)
        _write_json_exclusive(args.run_root / "profile.json", payload)
        terminal = {
            "schema_version": "zoomtoken_gridfuse32_l6_g2_terminal_v001",
            "status": payload["status"],
            "gate_passed": payload["gate_passed"],
            "profile": str((args.run_root / "profile.json").resolve()),
        }
        _write_json_exclusive(args.run_root / "terminal_receipt.json", terminal)
        print(json.dumps(terminal, sort_keys=True))
        return 0 if payload["gate_passed"] else 3
    except Exception as exc:
        args.run_root.mkdir(parents=True, exist_ok=True)
        terminal = {
            "schema_version": "zoomtoken_gridfuse32_l6_g2_terminal_v001",
            "status": "GRIDFUSE32_L6_G2_ENGINEERING_OR_PROTOCOL_BLOCKER",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        path = args.run_root / "terminal_receipt.json"
        if not path.exists():
            _write_json_exclusive(path, terminal)
        print(json.dumps(terminal, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
