#!/usr/bin/env python3
"""Same-GPU final-EMA cost replay for ZoomToken K100 and strict R1.

This entry point never trains or resumes a model.  It replays the canonical
THUMOS14 validation population with the two completed epoch-59 EMA checkpoints
in one Slurm allocation, using a counterbalanced ABBA+BAAB order.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REMOTE_BOUNDARY = Path("/data/run01/sczc063/yuzibo")
PROFILE_ORDER = ("K100", "R1", "R1", "K100", "R1", "K100", "K100", "R1")
WARMUP_WINDOWS = 50
POWER_INTERVAL_MS = 20
EXPECTED_VIDEO_COUNT = 211
EXPECTED_WINDOW_COUNT = 792
EXPECTED_STATE_ENTRIES = 527
EXPECTED_METRICS_PERCENT = {
    "K100": {
        "average_mAP": 68.51,
        "mAP@0.3": 83.61,
        "mAP@0.4": 79.77,
        "mAP@0.5": 71.73,
        "mAP@0.6": 61.19,
        "mAP@0.7": 46.27,
    },
    "R1": {
        "average_mAP": 69.07,
        "mAP@0.3": 84.37,
        "mAP@0.4": 79.93,
        "mAP@0.5": 73.34,
        "mAP@0.6": 61.14,
        "mAP@0.7": 46.57,
    },
}
ARM_SPECS = {
    "K100": {
        "job_id": "1248835",
        "training_revision": "70dcbe1089866f6ee3821176eb41d2dc10ee8d14",
        "config": "configs/adatad/thumos/georoute_official_b_alltoken_prebackbone_seed42_v001.py",
        "official_support": "all_native",
        "tokens_per_tubelet": 100,
        "executed_tokens_per_window": 38400,
    },
    "R1": {
        "job_id": "1249099",
        "training_revision": "9e25c6d38de8c993948025629181470b858682b4",
        "config": "configs/adatad/thumos/georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py",
        "official_support": "strict_rect8x8",
        "tokens_per_tubelet": 64,
        "executed_tokens_per_window": 24576,
    },
}


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def percentile(values: Sequence[float], probability: float) -> float:
    checked = sorted(float(value) for value in values)
    if not checked:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("percentile probability must be in [0, 1]")
    rank = probability * (len(checked) - 1)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    weight = rank - lower
    return checked[lower] * (1.0 - weight) + checked[upper] * weight


def summarize(values: Sequence[float]) -> dict[str, float]:
    checked = [float(value) for value in values]
    if not checked or not all(math.isfinite(value) and value >= 0.0 for value in checked):
        raise ValueError("summary requires finite nonnegative values")
    return {
        "count": len(checked),
        "mean": statistics.fmean(checked),
        "p50": percentile(checked, 0.50),
        "p95": percentile(checked, 0.95),
        "min": min(checked),
        "max": max(checked),
    }


def _interpolate_power(samples: Sequence[tuple[float, float]], timestamp: float) -> float:
    if timestamp <= samples[0][0]:
        return samples[0][1]
    if timestamp >= samples[-1][0]:
        return samples[-1][1]
    for left, right in zip(samples[:-1], samples[1:]):
        if left[0] <= timestamp <= right[0]:
            weight = (timestamp - left[0]) / (right[0] - left[0])
            return left[1] * (1.0 - weight) + right[1] * weight
    raise RuntimeError("power interpolation left its trace")


def integrate_energy(
    samples: Sequence[tuple[float, float]], *, start: float, end: float
) -> float | None:
    checked = sorted((float(t), float(p)) for t, p in samples)
    if (
        len(checked) < 2
        or checked[0][0] > start
        or checked[-1][0] < end
        or end <= start
    ):
        return None
    clipped = [
        (start, _interpolate_power(checked, start)),
        *((t, p) for t, p in checked if start < t < end),
        (end, _interpolate_power(checked, end)),
    ]
    return float(
        sum(
            0.5 * (left[1] + right[1]) * (right[0] - left[0])
            for left, right in zip(clipped[:-1], clipped[1:])
        )
    )


def temporal_iou(first: Sequence[float], second: Sequence[float]) -> float:
    intersection = max(0.0, min(first[1], second[1]) - max(first[0], second[0]))
    union = (first[1] - first[0]) + (second[1] - second[0]) - intersection
    return intersection / union if union > 0.0 else 0.0


def boundary_quality(
    annotation: Mapping[str, Any], predictions: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    ground_truth = []
    for video_id, video in annotation["database"].items():
        if video.get("subset") != "validation":
            continue
        for index, item in enumerate(video.get("annotations", ())):
            if item.get("label") == "Ambiguous":
                continue
            ground_truth.append(
                {
                    "id": f"{video_id}:{index}",
                    "video": str(video_id),
                    "label": str(item["label"]),
                    "segment": tuple(map(float, item["segment"])),
                }
            )
    candidates = []
    for video_id, rows in predictions.items():
        for index, item in enumerate(rows):
            candidates.append(
                {
                    "id": f"{video_id}:{index}",
                    "video": str(video_id),
                    "label": str(item["label"]),
                    "segment": tuple(map(float, item["segment"])),
                    "score": float(item["score"]),
                }
            )
    matched_gt: set[str] = set()
    matches = []
    for prediction in sorted(candidates, key=lambda row: (-row["score"], row["id"])):
        available = [
            (temporal_iou(prediction["segment"], gt["segment"]), gt)
            for gt in ground_truth
            if gt["id"] not in matched_gt
            and gt["video"] == prediction["video"]
            and gt["label"] == prediction["label"]
        ]
        available = [row for row in available if row[0] >= 0.50]
        if not available:
            continue
        overlap, gt = max(available, key=lambda row: (row[0], row[1]["id"]))
        matched_gt.add(gt["id"])
        matches.append((prediction, gt, overlap))
    start_errors, end_errors = [], []
    short_start, short_end = [], []
    short_ids = {
        gt["id"] for gt in ground_truth if 0.0 < gt["segment"][1] - gt["segment"][0] <= 5.0
    }
    short_hits_070 = 0
    for prediction, gt, overlap in matches:
        duration = gt["segment"][1] - gt["segment"][0]
        start = abs(prediction["segment"][0] - gt["segment"][0]) / duration
        end = abs(prediction["segment"][1] - gt["segment"][1]) / duration
        start_errors.append(start)
        end_errors.append(end)
        if gt["id"] in short_ids:
            short_start.append(start)
            short_end.append(end)
            short_hits_070 += int(overlap >= 0.70)
    return {
        "matching": "score_greedy_same_class_tiou_at_least_0.50",
        "ground_truth_count": len(ground_truth),
        "matched_count": len(matches),
        "mean_abs_start_error_normalized": statistics.fmean(start_errors) if start_errors else None,
        "mean_abs_end_error_normalized": statistics.fmean(end_errors) if end_errors else None,
        "short_action": {
            "definition_seconds": "0 < duration <= 5.0",
            "ground_truth_count": len(short_ids),
            "matched_count": len(short_start),
            "recall_at_tiou_0.70": short_hits_070 / len(short_ids) if short_ids else None,
            "mean_abs_start_error_normalized": statistics.fmean(short_start) if short_start else None,
            "mean_abs_end_error_normalized": statistics.fmean(short_end) if short_end else None,
        },
    }


def _git_identity(root: Path, expected_commit: str) -> None:
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if head != expected_commit or dirty:
        raise RuntimeError("BPNS cost replay requires its exact clean execution commit")


def _strip_ddp_prefix(state: Mapping[str, Any]) -> dict[str, Any]:
    keys = list(state)
    if keys and all(str(key).startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state.items()}
    return dict(state)


def _bind_test_config(cfg: Any, args: argparse.Namespace) -> None:
    cfg.dataset.test.ann_file = str(args.annotation)
    cfg.dataset.test.class_map = str(args.class_map)
    cfg.dataset.test.data_path = str(args.video_root)
    cfg.dataset.test.subset_name = "validation"
    cfg.dataset.test.test_mode = True
    cfg.evaluation.ground_truth_filename = str(args.annotation)
    cfg.evaluation.subset = "validation"
    cfg.post_processing.sliding_window = True


def _population_manifest(dataset: Any) -> tuple[list[str], set[str]]:
    manifest = []
    videos = set()
    for ordinal, row in enumerate(dataset.data_list):
        if not isinstance(row, (tuple, list)) or len(row) < 4 or len(row[3]) == 0:
            raise ValueError(f"validation population row {ordinal} is malformed")
        video_id = str(row[0])
        videos.add(video_id)
        manifest.append(f"{video_id}:{int(row[3][0])}")
    if len(videos) != EXPECTED_VIDEO_COUNT or len(manifest) != EXPECTED_WINDOW_COUNT:
        raise ValueError(
            f"expected 40 validation videos/136 windows, observed {len(videos)}/{len(manifest)}"
        )
    if len(set(manifest)) != len(manifest):
        raise ValueError("validation population contains duplicate physical windows")
    return manifest, videos


def _checkpoint_preflight(torch: Any, path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint.get("state_dict_ema")
    if int(checkpoint.get("epoch", -1)) != 59 or not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is not an epoch-59 EMA artifact: {path}")
    if len(state) != EXPECTED_STATE_ENTRIES:
        raise ValueError(f"checkpoint EMA parameter count differs from 527: {path}")
    return {"epoch": 59, "ema_parameter_count": len(state), "checkpoint_role": checkpoint.get("checkpoint_role")}


def precheck(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from mmengine.config import Config
    from opentad.datasets import build_dataset

    _git_identity(ROOT, args.expected_commit)
    if args.result_root.exists():
        raise FileExistsError(f"formal BPNS result root already exists: {args.result_root}")
    required = (args.annotation, args.class_map, args.video_root, args.k100_checkpoint, args.r1_checkpoint)
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not str(args.result_root.resolve()).startswith(str(REMOTE_BOUNDARY.resolve()) + os.sep):
        raise ValueError("BPNS result root leaves the shared remote write boundary")
    manifests = []
    checkpoint_receipts = {}
    for arm in ("K100", "R1"):
        spec = ARM_SPECS[arm]
        cfg = Config.fromfile(str(ROOT / spec["config"]))
        _bind_test_config(cfg, args)
        manifest, videos = _population_manifest(build_dataset(copy.deepcopy(cfg.dataset.test)))
        manifests.append(manifest)
        checkpoint_receipts[arm] = _checkpoint_preflight(
            torch, args.k100_checkpoint if arm == "K100" else args.r1_checkpoint
        )
        if len(videos) != EXPECTED_VIDEO_COUNT:
            raise RuntimeError("validation video count changed during preflight")
    if manifests[0] != manifests[1]:
        raise ValueError("K100 and R1 do not share an ordered validation population")
    return {
        "status": "PRECHECK_READY",
        "execution_commit": args.expected_commit,
        "profile_order": list(PROFILE_ORDER),
        "video_count": EXPECTED_VIDEO_COUNT,
        "window_count": EXPECTED_WINDOW_COUNT,
        "checkpoints": checkpoint_receipts,
        "reads_validation_metrics": False,
        "trains_or_resumes": False,
    }


def _move_to_device(value: Any, device: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class CudaMethodTimer:
    def __init__(self, torch_module: Any, target: Any, method: str) -> None:
        self.torch = torch_module
        self.target = target
        self.method = method
        self.original = getattr(target, method)
        self.pairs: list[tuple[Any, Any]] = []

        def wrapped(*args, **kwargs):
            start = self.torch.cuda.Event(enable_timing=True)
            end = self.torch.cuda.Event(enable_timing=True)
            start.record()
            result = self.original(*args, **kwargs)
            end.record()
            self.pairs.append((start, end))
            return result

        setattr(target, method, wrapped)

    def reset(self) -> None:
        self.pairs.clear()

    def elapsed(self) -> float:
        return float(sum(start.elapsed_time(end) for start, end in self.pairs))

    def close(self) -> None:
        setattr(self.target, self.method, self.original)
        self.reset()


class WallMethodTimer:
    def __init__(self, target: Any, method: str, synchronize) -> None:
        self.target = target
        self.method = method
        self.original = getattr(target, method)
        self.synchronize = synchronize
        self.values: list[float] = []

        def wrapped(*args, **kwargs):
            self.synchronize()
            started = time.perf_counter()
            result = self.original(*args, **kwargs)
            self.synchronize()
            self.values.append((time.perf_counter() - started) * 1000.0)
            return result

        setattr(target, method, wrapped)

    def reset(self) -> None:
        self.values.clear()

    def elapsed(self) -> float:
        return float(sum(self.values))

    def close(self) -> None:
        setattr(self.target, self.method, self.original)
        self.reset()


def _evaluate_predictions(cfg: Any, predictions: Mapping[str, Any]) -> dict[str, float]:
    from opentad.evaluations import build_evaluator

    evaluator = build_evaluator(dict(prediction_filename={"results": predictions}, **cfg.evaluation))
    return {key: float(value) for key, value in evaluator.evaluate().items()}


def _assert_metric_parity(arm: str, metrics: Mapping[str, float]) -> None:
    expected = EXPECTED_METRICS_PERCENT[arm]
    for key, target in expected.items():
        observed = 100.0 * float(metrics[key])
        if abs(observed - target) > 0.015:
            raise RuntimeError(
                f"{arm} final-EMA replay differs from its historical result: {key}={observed:.6f}, expected {target:.2f}"
            )


def _sample_identity(cpu_batch: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    metas = cpu_batch.get("metas")
    if not isinstance(metas, list) or len(metas) != 1:
        raise ValueError("BPNS profile requires batch-one metadata")
    meta = metas[0]
    start = meta.get("window_start_frame", meta.get("window_start"))
    if start is None:
        raise ValueError(f"validation window {ordinal} has no start-frame identity")
    video_id = str(meta["video_name"])
    return {"video_id": video_id, "physical_window_id": f"{video_id}:{int(start)}", "window_ordinal": ordinal}


def _measure(fn, synchronize) -> tuple[Any, float]:
    synchronize()
    started = time.perf_counter()
    result = fn()
    synchronize()
    return result, (time.perf_counter() - started) * 1000.0


def _profile_one_pass(
    *,
    torch: Any,
    arm: str,
    pass_index: int,
    checkpoint_path: Path,
    args: argparse.Namespace,
    device: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], list[tuple[float, float]], tuple[float, float]]:
    from mmengine.config import Config
    from torch.nn.parallel import DistributedDataParallel
    from opentad.cores.test_engine import gather_ddp_results
    from opentad.datasets import build_dataloader, build_dataset
    from opentad.models import build_detector
    from opentad.utils import set_seed

    spec = ARM_SPECS[arm]
    cfg = Config.fromfile(str(ROOT / spec["config"]))
    _bind_test_config(cfg, args)
    dataset = build_dataset(copy.deepcopy(cfg.dataset.test))
    manifest, videos = _population_manifest(dataset)
    loader = build_dataloader(dataset, rank=0, world_size=1, shuffle=False, drop_last=False, batch_size=1, num_workers=0)
    if len(loader) != EXPECTED_WINDOW_COUNT:
        raise ValueError("BPNS profile loader changed the frozen population")

    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = None
    model = build_detector(model_cfg)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if int(checkpoint.get("epoch", -1)) != 59 or "state_dict_ema" not in checkpoint:
        raise ValueError(f"{arm} replay requires epoch-59 EMA")
    model.load_state_dict(_strip_ddp_prefix(checkpoint["state_dict_ema"]), strict=True)
    del checkpoint
    model = model.to(device).eval()
    ddp_model = DistributedDataParallel(model, device_ids=[0], output_device=0)
    external_cls = dataset.class_map
    synchronize = lambda: torch.cuda.synchronize(device)
    set_seed(42)

    wrapper = model.backbone
    heavy = wrapper.model.backbone
    forward_timer = CudaMethodTimer(torch, model, "forward_test")
    backbone_timer = CudaMethodTimer(torch, wrapper, "forward")
    heavy_timer = CudaMethodTimer(torch, heavy, "forward_native_ragged")
    post_timer = WallMethodTimer(model, "post_processing", synchronize)

    def forward_once(batch):
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
            return ddp_model(
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

    for _ in range(WARMUP_WINDOWS):
        forward_once(_move_to_device(next_batch(), device))
    synchronize()
    iterator = iter(loader)

    samples = []
    energy_windows = []
    video_rows: dict[str, list[dict[str, Any]]] = {}
    audit_receipt = None
    final_window = None
    try:
        for ordinal in range(EXPECTED_WINDOW_COUNT):
            synchronize()
            energy_start = time.monotonic_ns() / 1e9
            continuous_start = time.perf_counter()
            cpu_batch, input_ms = _measure(next_batch, synchronize)
            identity = _sample_identity(cpu_batch, ordinal)
            if identity["physical_window_id"] != manifest[ordinal]:
                raise ValueError("runtime loader order differs from the frozen population")
            torch.cuda.reset_peak_memory_stats(device)
            gpu_batch, h2d_ms = _measure(lambda: _move_to_device(cpu_batch, device), synchronize)
            for timer in (forward_timer, backbone_timer, heavy_timer, post_timer):
                timer.reset()
            result, detector_wall_ms = _measure(lambda: forward_once(gpu_batch), synchronize)
            if not isinstance(result, Mapping):
                raise ValueError("detector returned no result mapping")
            for video_id, rows in result.items():
                video_rows.setdefault(str(video_id), []).extend(rows)
            energy_end = time.monotonic_ns() / 1e9
            continuous_ms = (time.perf_counter() - continuous_start) * 1000.0
            audit = dict(wrapper.latest_georoute_audit or {})
            packed = dict(audit.get("packed") or {})
            observed_audit = {
                "official_support": audit.get("official_support"),
                "selection_application": audit.get("selection_application"),
                "native_materialization_before_heavy": audit.get("native_materialization_before_heavy"),
                "selected_tokens_per_tubelet": audit.get("selected_tokens_per_tubelet"),
                "physical_tokens_per_window": audit.get("physical_tokens_per_window"),
                "executed_patch_tokens_per_window": packed.get("executed_patch_tokens_per_window"),
                "padded_heavy_tokens_per_window": packed.get("padded_heavy_tokens_per_window"),
                "heavy_backbone_forward_count": audit.get("heavy_backbone_forward_count"),
            }
            expected_audit = {
                "official_support": spec["official_support"],
                "selection_application": "pre_heavy_videomae",
                "native_materialization_before_heavy": True,
                "selected_tokens_per_tubelet": spec["tokens_per_tubelet"],
                "physical_tokens_per_window": spec["executed_tokens_per_window"],
                "executed_patch_tokens_per_window": spec["executed_tokens_per_window"],
                "padded_heavy_tokens_per_window": 0,
                "heavy_backbone_forward_count": 1,
            }
            if observed_audit != expected_audit:
                raise RuntimeError(f"{arm} runtime audit changed: {observed_audit}")
            if audit_receipt is None:
                audit_receipt = observed_audit
            elif audit_receipt != observed_audit:
                raise RuntimeError(f"{arm} runtime audit changed within one pass")
            samples.append(
                {
                    "arm": arm,
                    "pass_index": pass_index,
                    **identity,
                    "input_pipeline_serial_ms": input_ms,
                    "h2d_ms": h2d_ms,
                    "detector_forward_wall_ms": detector_wall_ms,
                    "model_forward_cuda_ms": forward_timer.elapsed(),
                    "backbone_wrapper_cuda_ms": backbone_timer.elapsed(),
                    "heavy_backbone_cuda_ms": heavy_timer.elapsed(),
                    "postprocess_wall_ms": post_timer.elapsed(),
                    "final_video_nms_ms": 0.0,
                    "decode_to_window_output_wall_ms": continuous_ms,
                    "end_to_end_serial_ms": continuous_ms,
                    "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024**2),
                    "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024**2),
                    "gpu_energy_j": None,
                }
            )
            energy_windows.append((energy_start, energy_end))
            del cpu_batch, gpu_batch, result
        synchronize()
        nms_start = time.monotonic_ns() / 1e9
        finalized = gather_ddp_results(1, video_rows, cfg.post_processing)
        synchronize()
        nms_end = time.monotonic_ns() / 1e9
        final_window = (nms_start, nms_end)
        if not isinstance(finalized, Mapping) or not set(map(str, finalized)).issubset(videos):
            raise ValueError("final NMS returned identities outside validation")
        amortized_nms_ms = (nms_end - nms_start) * 1000.0 / len(samples)
        for sample in samples:
            sample["final_video_nms_ms"] = amortized_nms_ms
            sample["end_to_end_serial_ms"] += amortized_nms_ms
        metrics = _evaluate_predictions(cfg, finalized)
        _assert_metric_parity(arm, metrics)
    finally:
        for timer in (post_timer, heavy_timer, backbone_timer, forward_timer):
            timer.close()

    if final_window is None or audit_receipt is None:
        raise RuntimeError("BPNS profile pass did not reach final NMS")
    predictions = {"results": dict(finalized)}
    receipt = {
        "arm": arm,
        "pass_index": pass_index,
        "job_id": spec["job_id"],
        "training_revision": spec["training_revision"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": 59,
        "checkpoint_state": "state_dict_ema",
        "video_count": len(videos),
        "window_count": len(manifest),
        "metrics": metrics,
        "runtime_audit": audit_receipt,
    }
    del ddp_model, model, loader, dataset
    torch.cuda.empty_cache()
    return samples, receipt, predictions, energy_windows, final_window


def _gpu_identity(torch: Any) -> dict[str, Any]:
    selector = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not selector or "," in selector or torch.cuda.device_count() != 1:
        raise RuntimeError("BPNS profile requires exactly one Slurm-visible GPU")
    query = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            selector,
            "--query-gpu=uuid,name,driver_version,power.limit",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    fields = [field.strip() for field in query.stdout.strip().split(",")]
    if len(fields) != 4 or not fields[0].startswith("GPU-"):
        raise RuntimeError("could not bind the Slurm-visible GPU UUID")
    props = torch.cuda.get_device_properties(0)
    return {
        "node": platform.node(),
        "gpu_uuid": fields[0],
        "gpu_name": fields[1],
        "driver_version": fields[2],
        "power_limit_w": float(fields[3]),
        "total_memory_bytes": int(props.total_memory),
        "compute_capability": [int(props.major), int(props.minor)],
        "cuda_visible_devices": selector,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def _load_power_trace(path: Path) -> list[tuple[float, float]]:
    samples = []
    previous = -1
    with path.open("r", encoding="utf-8") as handle:
        for expected_sequence, line in enumerate(handle):
            row = json.loads(line)
            sequence = int(row["sequence"])
            monotonic_ns = int(row["monotonic_ns"])
            power_w = float(row["power_w"])
            if sequence != expected_sequence or monotonic_ns <= previous or power_w < 0.0:
                raise ValueError("power trace violates ordering or finiteness")
            previous = monotonic_ns
            samples.append((monotonic_ns / 1e9, power_w))
    if len(samples) < 2:
        raise ValueError("power sidecar produced too few samples")
    return samples


class PowerSidecar:
    def __init__(self, *, gpu_uuid: str, scratch: Path, sidecar_cpu: int, allocated_cpus: Sequence[int]) -> None:
        self.gpu_uuid = gpu_uuid
        self.scratch = scratch
        self.sidecar_cpu = int(sidecar_cpu)
        self.allocated_cpus = tuple(map(int, allocated_cpus))
        self.trace = scratch / "power.jsonl"
        self.ready = scratch / "ready.json"
        self.process = None
        self.samples: list[tuple[float, float]] = []

    def start(self) -> None:
        self.scratch.mkdir(parents=True, exist_ok=False)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "sidecar",
                "--gpu-uuid",
                self.gpu_uuid,
                "--trace",
                str(self.trace),
                "--ready",
                str(self.ready),
                "--sidecar-cpu",
                str(self.sidecar_cpu),
                "--allocated-cpus",
                ",".join(map(str, self.allocated_cpus)),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20.0
        while not self.ready.is_file():
            if self.process.poll() is not None:
                raise RuntimeError(f"power sidecar exited early: {self.process.stderr.read().strip()}")
            if time.monotonic() >= deadline:
                self.process.kill()
                raise RuntimeError("power sidecar did not become ready")
            time.sleep(0.02)

    def stop(self) -> None:
        if self.process is None:
            raise RuntimeError("power sidecar was not started")
        if self.process.poll() is None:
            self.process.terminate()
        try:
            code = self.process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5.0)
            raise RuntimeError("power sidecar required SIGKILL")
        stderr = self.process.stderr.read().strip() if self.process.stderr else ""
        if code != 0:
            raise RuntimeError(f"power sidecar failed with {code}: {stderr}")
        self.samples = _load_power_trace(self.trace)


def run_power_sidecar(args: argparse.Namespace) -> int:
    allocated = tuple(sorted(int(value) for value in args.allocated_cpus.split(",") if value))
    if len(allocated) != 5 or args.sidecar_cpu not in allocated:
        raise ValueError("power sidecar requires one CPU from the five-CPU allocation")
    os.sched_setaffinity(0, {args.sidecar_cpu})
    if tuple(sorted(os.sched_getaffinity(0))) != (args.sidecar_cpu,):
        raise RuntimeError("power sidecar could not bind its reserved CPU")
    stopped = False

    def request_stop(_signal, _frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    process = subprocess.Popen(
        [
            "nvidia-smi",
            "-i",
            args.gpu_uuid,
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            f"--loop-ms={POWER_INTERVAL_MS}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    sequence = 0
    try:
        with args.trace.open("x", encoding="utf-8") as trace:
            while not stopped:
                line = process.stdout.readline()
                if not line:
                    if process.poll() is not None:
                        raise RuntimeError(process.stderr.read().strip() or "nvidia-smi exited")
                    continue
                power = float(line.strip())
                row = {"sequence": sequence, "monotonic_ns": time.monotonic_ns(), "power_w": power}
                trace.write(json.dumps(row, sort_keys=True) + "\n")
                trace.flush()
                if sequence == 0:
                    _atomic_json(
                        args.ready,
                        {
                            "gpu_uuid": args.gpu_uuid,
                            "sidecar_cpu": args.sidecar_cpu,
                            "interval_ms": POWER_INTERVAL_MS,
                        },
                    )
                sequence += 1
    finally:
        if process.poll() is None:
            process.terminate()
        process.wait(timeout=5.0)
    return 0


def _short_action_annotation(annotation: Mapping[str, Any]) -> dict[str, Any]:
    filtered = copy.deepcopy(annotation)
    for video in filtered["database"].values():
        video["annotations"] = [
            row
            for row in video.get("annotations", ())
            if row.get("label") != "Ambiguous"
            and 0.0 < float(row["segment"][1]) - float(row["segment"][0]) <= 5.0
        ]
    return filtered


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from mmengine.config import Config

    preflight = precheck(args)
    if not os.environ.get("SLURM_JOB_ID") or int(os.environ.get("SLURM_CPUS_PER_TASK", -1)) != 5:
        raise RuntimeError("formal BPNS profile requires one Slurm GPU and five CPUs")
    if (int(os.environ.get("LOCAL_RANK", -1)), int(os.environ.get("RANK", -1)), int(os.environ.get("WORLD_SIZE", -1))) != (0, 0, 1):
        raise RuntimeError("formal BPNS profile requires torchrun world-size one")
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    hardware = _gpu_identity(torch)
    allocated = tuple(sorted(os.sched_getaffinity(0)))
    if len(allocated) != 5:
        raise RuntimeError("formal BPNS profile requires exactly five allocated CPUs")
    sidecar_cpu = allocated[-1]
    detector_cpus = allocated[:-1]
    args.result_root.mkdir(parents=True, exist_ok=False)
    scratch = Path("/tmp") / f"zoomtoken_bpns_cost_job{os.environ['SLURM_JOB_ID']}"
    sidecar = PowerSidecar(
        gpu_uuid=hardware["gpu_uuid"],
        scratch=scratch,
        sidecar_cpu=sidecar_cpu,
        allocated_cpus=allocated,
    )
    sidecar.start()
    os.sched_setaffinity(0, set(detector_cpus))
    time.sleep(0.05)
    all_samples = []
    pass_receipts = []
    predictions_by_arm = {}
    energy_windows_by_pass = []
    nms_windows = []
    try:
        for pass_index, arm in enumerate(PROFILE_ORDER):
            checkpoint = args.k100_checkpoint if arm == "K100" else args.r1_checkpoint
            samples, receipt, predictions, energy_windows, nms_window = _profile_one_pass(
                torch=torch,
                arm=arm,
                pass_index=pass_index,
                checkpoint_path=checkpoint,
                args=args,
                device=device,
            )
            if arm in predictions_by_arm and predictions_by_arm[arm] != predictions:
                raise RuntimeError(f"{arm} repeated replay produced different predictions")
            predictions_by_arm.setdefault(arm, predictions)
            all_samples.extend(samples)
            pass_receipts.append(receipt)
            energy_windows_by_pass.append(energy_windows)
            nms_windows.append(nms_window)
    finally:
        time.sleep(0.05)
        sidecar.stop()

    for pass_index, energy_windows in enumerate(energy_windows_by_pass):
        nms_energy = integrate_energy(sidecar.samples, start=nms_windows[pass_index][0], end=nms_windows[pass_index][1])
        if nms_energy is None:
            raise RuntimeError("power trace does not cover final NMS")
        pass_rows = [row for row in all_samples if row["pass_index"] == pass_index]
        for row, window in zip(pass_rows, energy_windows):
            energy = integrate_energy(sidecar.samples, start=window[0], end=window[1])
            if energy is None:
                raise RuntimeError("power trace does not cover a measured validation window")
            row["gpu_energy_j"] = energy + nms_energy / len(pass_rows)

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    short_annotation_path = args.result_root / "short_action_validation_gt.json"
    _atomic_json(short_annotation_path, _short_action_annotation(annotation))
    arm_quality = {}
    from opentad.evaluations import build_evaluator

    for arm, prediction in predictions_by_arm.items():
        prediction_path = args.result_root / f"{arm.lower()}_predictions.json"
        _atomic_json(prediction_path, prediction)
        cfg = Config.fromfile(str(ROOT / ARM_SPECS[arm]["config"]))
        _bind_test_config(cfg, args)
        short_evaluator = build_evaluator(
            dict(
                prediction_filename=prediction,
                ground_truth_filename=str(short_annotation_path),
                subset="validation",
                tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
            )
        )
        arm_quality[arm] = {
            "boundary": boundary_quality(annotation, prediction["results"]),
            "short_action_mAP": {key: float(value) for key, value in short_evaluator.evaluate().items()},
            "prediction_path": str(prediction_path),
        }

    latency_keys = (
        "input_pipeline_serial_ms",
        "h2d_ms",
        "detector_forward_wall_ms",
        "model_forward_cuda_ms",
        "backbone_wrapper_cuda_ms",
        "heavy_backbone_cuda_ms",
        "postprocess_wall_ms",
        "final_video_nms_ms",
        "decode_to_window_output_wall_ms",
        "end_to_end_serial_ms",
    )
    grouped = defaultdict(list)
    for row in all_samples:
        grouped[row["arm"]].append(row)
    arm_summaries = {}
    for arm in ("K100", "R1"):
        rows = grouped[arm]
        total_seconds = sum(row["end_to_end_serial_ms"] for row in rows) / 1000.0
        gross_energy = sum(float(row["gpu_energy_j"]) for row in rows)
        arm_summaries[arm] = {
            "sample_count": len(rows),
            "pass_count": PROFILE_ORDER.count(arm),
            "latency_ms": {key: summarize([row[key] for row in rows]) for key in latency_keys},
            "throughput_windows_per_second": len(rows) / total_seconds,
            "peak_gpu_allocated_mb": max(row["peak_gpu_allocated_mb"] for row in rows),
            "peak_gpu_reserved_mb": max(row["peak_gpu_reserved_mb"] for row in rows),
            "gross_gpu_energy_j": gross_energy,
            "gpu_energy_j_per_window": gross_energy / len(rows),
            "final_ema_metrics": pass_receipts[next(i for i, row in enumerate(pass_receipts) if row["arm"] == arm)]["metrics"],
            "quality": arm_quality[arm],
            "tokens_per_tubelet": ARM_SPECS[arm]["tokens_per_tubelet"],
            "executed_tokens_per_window": ARM_SPECS[arm]["executed_tokens_per_window"],
        }
    comparison = {
        "r1_over_k100": {
            "native_spatial_input_ratio": 64.0 / 100.0,
            "native_spatial_input_reduction": 0.36,
            "attention_pair_ratio_per_tubelet": (64.0 / 100.0) ** 2,
            "end_to_end_p50_ratio": arm_summaries["R1"]["latency_ms"]["end_to_end_serial_ms"]["p50"] / arm_summaries["K100"]["latency_ms"]["end_to_end_serial_ms"]["p50"],
            "throughput_ratio": arm_summaries["R1"]["throughput_windows_per_second"] / arm_summaries["K100"]["throughput_windows_per_second"],
            "peak_allocated_memory_ratio": arm_summaries["R1"]["peak_gpu_allocated_mb"] / arm_summaries["K100"]["peak_gpu_allocated_mb"],
            "energy_per_window_ratio": arm_summaries["R1"]["gpu_energy_j_per_window"] / arm_summaries["K100"]["gpu_energy_j_per_window"],
            "accuracy_delta_percentage_points": {
                key: 100.0 * (arm_summaries["R1"]["final_ema_metrics"][key] - arm_summaries["K100"]["final_ema_metrics"][key])
                for key in EXPECTED_METRICS_PERCENT["K100"]
            },
        },
        "interpretation_boundary": "native token and attention-pair ratios are structural proxies; latency, memory and energy fields are measured full-stack evidence",
    }
    profile_payload = {
        "schema_version": "zoomtoken_bpns_r1_same_gpu_cost_v001",
        "status": "COMPLETED_FINAL_EMA_REPLAY",
        "execution_commit": args.expected_commit,
        "seed": 42,
        "profile_order": list(PROFILE_ORDER),
        "counterbalancing": "ABBA+BAAB",
        "warmup_windows_per_pass": WARMUP_WINDOWS,
        "power_interval_ms": POWER_INTERVAL_MS,
        "dataset": {"name": "THUMOS14", "split": "validation", "video_count": EXPECTED_VIDEO_COUNT, "window_count": EXPECTED_WINDOW_COUNT, "official_test_opened": False},
        "hardware": hardware,
        "software": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda},
        "cpu_partition": {"allocated": list(allocated), "detector": list(detector_cpus), "sidecar": sidecar_cpu},
        "precheck": preflight,
        "pass_receipts": pass_receipts,
        "arm_summaries": arm_summaries,
        "comparison": comparison,
        "training_or_resume_executed": False,
        "paper_claim_allowed_without_independent_result_to_claim": False,
    }
    _write_jsonl(args.result_root / "cost_samples.jsonl", all_samples)
    power_rows = [
        {"sequence": index, "monotonic_s": timestamp, "power_w": power}
        for index, (timestamp, power) in enumerate(sidecar.samples)
    ]
    _write_jsonl(args.result_root / "power_trace.jsonl", power_rows)
    _atomic_json(args.result_root / "profile.json", profile_payload)
    _atomic_json(
        args.result_root / "terminal_receipt.json",
        {
            "status": profile_payload["status"],
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "execution_commit": args.expected_commit,
            "result_root": str(args.result_root),
            "profile": str(args.result_root / "profile.json"),
            "training_or_resume_executed": False,
            "official_test_opened": False,
        },
    )
    dist.destroy_process_group()
    return profile_payload


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--k100-checkpoint", type=Path, required=True)
    parser.add_argument("--r1-checkpoint", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("precheck", "profile"):
        child = subparsers.add_parser(command)
        _add_common(child)
    sidecar = subparsers.add_parser("sidecar")
    sidecar.add_argument("--gpu-uuid", required=True)
    sidecar.add_argument("--trace", type=Path, required=True)
    sidecar.add_argument("--ready", type=Path, required=True)
    sidecar.add_argument("--sidecar-cpu", type=int, required=True)
    sidecar.add_argument("--allocated-cpus", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "sidecar":
        return run_power_sidecar(args)
    result = precheck(args) if args.command == "precheck" else profile(args)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
