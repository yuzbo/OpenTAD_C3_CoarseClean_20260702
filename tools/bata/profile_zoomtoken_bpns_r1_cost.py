#!/usr/bin/env python3
"""Same-GPU final-EMA cost replay for ZoomToken K100 and strict R1.

This entry point never trains or resumes a model.  It replays the canonical
THUMOS14 validation population with the two completed epoch-59 EMA checkpoints
in one Slurm allocation, using a counterbalanced ABBA+BAAB order.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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
from decimal import Decimal, ROUND_HALF_UP
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
ACCURACY_PARITY_METRIC_KEYS = tuple(EXPECTED_METRICS_PERCENT["K100"])
ACCURACY_PARITY_TOLERANCE_PP = 0.05
ACCURACY_PARITY_REFERENCE = {
    "precision": "reported_2dp",
    "source_revision": "b7357817d81127ab2d713b5471d008ea893efd35",
    "source_path": "tools/bata/profile_zoomtoken_bpns_r1_cost.py",
    "source_symbol": "EXPECTED_METRICS_PERCENT",
    "source_sha256": "80f2ea7991e26886329a46179169295e46e0958e9f8cde698d45a8fdf0eccd4c",
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_identity(payload: Any) -> str:
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _plain_mapping(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    if isinstance(value, Mapping):
        return {str(key): _plain_mapping(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_mapping(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


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
    remote_contains = subprocess.run(
        ["git", "-C", str(root), "branch", "-r", "--contains", head],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if not remote_contains:
        raise RuntimeError("BPNS cost replay requires an execution commit present on a remote ref")


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
        manifest.append(f"{ordinal}:{video_id}:{int(row[3][0])}")
    if len(videos) != EXPECTED_VIDEO_COUNT or len(manifest) != EXPECTED_WINDOW_COUNT:
        raise ValueError(
            f"expected {EXPECTED_VIDEO_COUNT} validation videos/{EXPECTED_WINDOW_COUNT} loader items, "
            f"observed {len(videos)}/{len(manifest)}"
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
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "size_bytes": path.stat().st_size,
        "epoch": 59,
        "ema_parameter_count": len(state),
        "checkpoint_role": checkpoint.get("checkpoint_role"),
    }


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
    config_receipts = {}
    for arm in ("K100", "R1"):
        spec = ARM_SPECS[arm]
        config_path = ROOT / spec["config"]
        cfg = Config.fromfile(str(config_path))
        _bind_test_config(cfg, args)
        manifest, videos = _population_manifest(build_dataset(copy.deepcopy(cfg.dataset.test)))
        manifests.append(manifest)
        config_receipts[arm] = {
            "path": str(config_path),
            "sha256": _sha256_file(config_path),
            "dataset_contract_sha256": _json_identity(_plain_mapping(cfg.dataset.test)),
            "evaluator_contract_sha256": _json_identity(_plain_mapping(cfg.evaluation)),
            "nms_contract_sha256": _json_identity(_plain_mapping(cfg.post_processing)),
        }
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
        "population_manifest_sha256": _json_identity(manifests[0]),
        "annotation": {"path": str(args.annotation), "sha256": _sha256_file(args.annotation)},
        "class_map": {"path": str(args.class_map), "sha256": _sha256_file(args.class_map)},
        "configs": config_receipts,
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


def _accuracy_parity_contract() -> dict[str, Any]:
    return {
        "comparison_unit": "percentage_point",
        "observed_value": "100 * evaluator_raw_fraction_without_rounding",
        "display_rounding": "decimal_round_half_up_to_2dp",
        "tolerance_pp": ACCURACY_PARITY_TOLERANCE_PP,
        "tolerance_inclusive": True,
        "admission_role": "nonblocking_historical_interval_diagnostic",
        "reported_2dp_interval": "[reported-0.005, reported+0.005)",
        "required_metrics": list(ACCURACY_PARITY_METRIC_KEYS),
        "reference": dict(ACCURACY_PARITY_REFERENCE),
    }


def _reported_2dp_interval_diagnosis(observed_pp: Decimal, reported_pp: Decimal) -> dict[str, Any]:
    half_unit = Decimal("0.005")
    tolerance = Decimal(str(ACCURACY_PARITY_TOLERANCE_PP))
    lower = reported_pp - half_unit
    upper = reported_pp + half_unit
    allowed_lower = observed_pp - tolerance
    allowed_upper = observed_pp + tolerance
    if allowed_lower <= lower and allowed_upper >= upper:
        diagnosis = "compatible"
    elif allowed_upper < lower or allowed_lower >= upper:
        diagnosis = "incompatible"
    else:
        diagnosis = "indeterminate"
    if observed_pp < lower:
        minimum_distance = lower - observed_pp
    elif observed_pp >= upper:
        minimum_distance = observed_pp - upper
    else:
        minimum_distance = Decimal("0")
    return {
        "diagnosis": diagnosis,
        "reference_interval_pp": {
            "lower_inclusive": float(lower),
            "upper_exclusive": float(upper),
        },
        "minimum_distance_to_interval_pp": float(minimum_distance),
    }


def _assert_metric_parity(arm: str, metrics: Mapping[str, float]) -> dict[str, Any]:
    expected = EXPECTED_METRICS_PERCENT[arm]
    metric_receipts = {}
    for key in ACCURACY_PARITY_METRIC_KEYS:
        if key not in metrics:
            raise RuntimeError(f"{arm} final-EMA replay is missing required metric {key}")
        raw_fraction = float(metrics[key])
        if not math.isfinite(raw_fraction):
            raise RuntimeError(f"{arm} final-EMA replay metric {key} is not finite")
        target = float(expected[key])
        observed_pp = Decimal(str(raw_fraction)) * Decimal("100")
        target_pp = Decimal(str(target))
        difference_pp = abs(observed_pp - target_pp)
        interval = _reported_2dp_interval_diagnosis(observed_pp, target_pp)
        metric_receipts[key] = {
            "evaluator_raw_fraction": raw_fraction,
            "observed_pp_unrounded": float(observed_pp),
            "display_pp_2dp_half_up": format(
                observed_pp.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
                ".2f",
            ),
            "reference_pp": target,
            "absolute_difference_pp": float(difference_pp),
            **interval,
        }
    diagnoses = {row["diagnosis"] for row in metric_receipts.values()}
    if "incompatible" in diagnoses:
        status = "NONBLOCKING_INCOMPATIBLE_PRESENT"
    elif "indeterminate" in diagnoses:
        status = "NONBLOCKING_INDETERMINATE_PRESENT"
    else:
        status = "NONBLOCKING_COMPATIBLE"
    return {
        "status": status,
        "contract": _accuracy_parity_contract(),
        "metrics": metric_receipts,
    }


def _sample_identity(cpu_batch: Mapping[str, Any], ordinal: int) -> dict[str, Any]:
    metas = cpu_batch.get("metas")
    if not isinstance(metas, list) or len(metas) != 1:
        raise ValueError("BPNS profile requires batch-one metadata")
    meta = metas[0]
    start = meta.get("window_start_frame", meta.get("window_start"))
    if start is None:
        raise ValueError(f"validation window {ordinal} has no start-frame identity")
    video_id = str(meta["video_name"])
    content_window_id = f"{video_id}:{int(start)}"
    return {
        "video_id": video_id,
        "content_window_id": content_window_id,
        "dataset_item_id": f"{ordinal}:{content_window_id}",
        "window_ordinal": ordinal,
    }


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
            if identity["dataset_item_id"] != manifest[ordinal]:
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
        parity_receipt = _assert_metric_parity(arm, metrics)
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
        "accuracy_parity": parity_receipt,
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
            if (
                sequence != expected_sequence
                or monotonic_ns <= previous
                or not math.isfinite(power_w)
                or power_w < 0.0
            ):
                raise ValueError("power trace violates ordering or finiteness")
            previous = monotonic_ns
            samples.append((monotonic_ns / 1e9, power_w))
    if len(samples) < 2:
        raise ValueError("power sidecar produced too few samples")
    return samples


def _power_coverage_summary(
    samples: Sequence[tuple[float, float]], windows: Sequence[tuple[float, float]]
) -> dict[str, Any]:
    checked = [(float(timestamp), float(power)) for timestamp, power in samples]
    if len(checked) < 2:
        raise ValueError("power coverage requires at least two samples")
    if any(
        not math.isfinite(timestamp) or not math.isfinite(power) or power < 0.0
        for timestamp, power in checked
    ):
        raise ValueError("power coverage contains a non-finite or negative sample")
    if any(right[0] <= left[0] for left, right in zip(checked[:-1], checked[1:])):
        raise ValueError("power coverage timestamps are not strictly increasing")
    measured = [(float(start), float(end)) for start, end in windows]
    if not measured or any(
        not math.isfinite(start) or not math.isfinite(end) or end <= start
        for start, end in measured
    ):
        raise ValueError("measurement windows are missing or invalid")
    measurement_start = min(start for start, _ in measured)
    measurement_end = max(end for _, end in measured)
    if checked[0][0] > measurement_start or checked[-1][0] < measurement_end:
        raise RuntimeError("power trace does not fully cover the measured pass")
    covered_samples = [row for row in checked if measurement_start <= row[0] <= measurement_end]
    if not covered_samples:
        raise RuntimeError("power trace contains no samples inside the measured pass")
    return {
        "status": "COMPLETE",
        "sample_count_total": len(checked),
        "sample_count_inside_measurement": len(covered_samples),
        "trace_first_monotonic_s": checked[0][0],
        "trace_last_monotonic_s": checked[-1][0],
        "measurement_first_monotonic_s": measurement_start,
        "measurement_last_monotonic_s": measurement_end,
        "max_trace_gap_ms": 1000.0
        * max(right[0] - left[0] for left, right in zip(checked[:-1], checked[1:])),
        "covers_measurement_start": True,
        "covers_measurement_end": True,
    }


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


def _persist_pass_artifacts(
    result_root: Path,
    pass_receipts: Sequence[dict[str, Any]],
    predictions_by_pass: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if len(pass_receipts) != len(PROFILE_ORDER) or len(predictions_by_pass) != len(PROFILE_ORDER):
        raise ValueError("complete v003 replay requires one artifact pair for each of eight passes")
    artifacts = []
    for receipt, predictions in zip(pass_receipts, predictions_by_pass):
        pass_index = int(receipt["pass_index"])
        arm = str(receipt["arm"])
        if PROFILE_ORDER[pass_index] != arm:
            raise ValueError("pass artifact identity differs from the frozen profile order")
        stem = f"pass_{pass_index:02d}_{arm.lower()}"
        prediction_path = result_root / f"{stem}_predictions.json"
        evaluator_path = result_root / f"{stem}_evaluator_raw_vector.json"
        _atomic_json(prediction_path, predictions)
        _atomic_json(evaluator_path, receipt["metrics"])
        artifact = {
            "pass_index": pass_index,
            "arm": arm,
            "prediction_path": str(prediction_path),
            "prediction_sha256": _sha256_file(prediction_path),
            "evaluator_raw_vector_path": str(evaluator_path),
            "evaluator_raw_vector_sha256": _sha256_file(evaluator_path),
        }
        receipt["artifacts"] = artifact
        artifacts.append(artifact)
    return artifacts


def _summarize_pass(
    rows: Sequence[Mapping[str, Any]],
    receipt: Mapping[str, Any],
    latency_keys: Sequence[str],
    *,
    expected_window_count: int = EXPECTED_WINDOW_COUNT,
) -> dict[str, Any]:
    if len(rows) != expected_window_count:
        raise ValueError(
            f"pass {receipt['pass_index']} has {len(rows)} cost rows; expected {expected_window_count}"
        )
    arm = str(receipt["arm"])
    pass_index = int(receipt["pass_index"])
    if any(str(row["arm"]) != arm or int(row["pass_index"]) != pass_index for row in rows):
        raise ValueError("cost row identity differs from its pass receipt")
    energies = [float(row["gpu_energy_j"]) for row in rows]
    if not all(math.isfinite(value) and value >= 0.0 for value in energies):
        raise ValueError("pass contains missing, negative, or non-finite energy")
    total_seconds = sum(float(row["end_to_end_serial_ms"]) for row in rows) / 1000.0
    if not math.isfinite(total_seconds) or total_seconds <= 0.0:
        raise ValueError("pass duration is invalid")
    gross_energy = sum(energies)
    return {
        "arm": arm,
        "pass_index": pass_index,
        "sample_count": len(rows),
        "latency_ms": {key: summarize([float(row[key]) for row in rows]) for key in latency_keys},
        "throughput_windows_per_second": len(rows) / total_seconds,
        "peak_gpu_allocated_mb": max(float(row["peak_gpu_allocated_mb"]) for row in rows),
        "peak_gpu_reserved_mb": max(float(row["peak_gpu_reserved_mb"]) for row in rows),
        "gross_gpu_energy_j": gross_energy,
        "gpu_energy_j_per_window": gross_energy / len(rows),
        "final_ema_metrics": dict(receipt["metrics"]),
        "accuracy_diagnostic": receipt["accuracy_parity"],
        "power_coverage": receipt["power_coverage"],
    }


def _median_of_four_arm_summary(
    pass_summaries: Sequence[Mapping[str, Any]], arm: str, latency_keys: Sequence[str]
) -> dict[str, Any]:
    selected = [row for row in pass_summaries if row["arm"] == arm]
    if len(selected) != 4:
        raise ValueError(f"{arm} primary estimate requires exactly four complete passes")
    return {
        "primary_estimator": "median_of_four_pass_estimates",
        "pass_indices": [int(row["pass_index"]) for row in selected],
        "sample_count_per_pass": EXPECTED_WINDOW_COUNT,
        "latency_ms": {
            key: {
                statistic: statistics.median(
                    float(row["latency_ms"][key][statistic]) for row in selected
                )
                for statistic in ("mean", "p50", "p95", "min", "max")
            }
            for key in latency_keys
        },
        "throughput_windows_per_second": statistics.median(
            float(row["throughput_windows_per_second"]) for row in selected
        ),
        "peak_gpu_allocated_mb": statistics.median(
            float(row["peak_gpu_allocated_mb"]) for row in selected
        ),
        "peak_gpu_reserved_mb": statistics.median(
            float(row["peak_gpu_reserved_mb"]) for row in selected
        ),
        "worst_pass_peak_gpu_allocated_mb": max(
            float(row["peak_gpu_allocated_mb"]) for row in selected
        ),
        "worst_pass_peak_gpu_reserved_mb": max(
            float(row["peak_gpu_reserved_mb"]) for row in selected
        ),
        "gross_gpu_energy_j": statistics.median(
            float(row["gross_gpu_energy_j"]) for row in selected
        ),
        "gpu_energy_j_per_window": statistics.median(
            float(row["gpu_energy_j_per_window"]) for row in selected
        ),
        "final_ema_metrics": {
            key: statistics.median(float(row["final_ema_metrics"][key]) for row in selected)
            for key in ACCURACY_PARITY_METRIC_KEYS
        },
    }


def _median_numeric_tree(values: Sequence[Any]) -> Any:
    if not values:
        raise ValueError("cannot aggregate an empty quality sequence")
    if all(isinstance(value, Mapping) for value in values):
        keys = set(values[0])
        if any(set(value) != keys for value in values[1:]):
            raise ValueError("pass quality structures differ")
        return {key: _median_numeric_tree([value[key] for value in values]) for key in sorted(keys)}
    if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
        checked = [float(value) for value in values]
        if not all(math.isfinite(value) for value in checked):
            raise ValueError("pass quality contains a non-finite value")
        return statistics.median(checked)
    if all(value is None for value in values):
        return None
    if all(value == values[0] for value in values[1:]):
        return values[0]
    return list(values)


def _classify_complete_replay(
    pass_summaries: Sequence[Mapping[str, Any]],
    arm_summaries: Mapping[str, Mapping[str, Any]],
    prediction_stability: Mapping[str, Mapping[str, Any]],
    arm_quality: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    k100, r1 = arm_summaries["K100"], arm_summaries["R1"]
    p50_reduction = 1.0 - (
        float(r1["latency_ms"]["end_to_end_serial_ms"]["p50"])
        / float(k100["latency_ms"]["end_to_end_serial_ms"]["p50"])
    )
    energy_reduction = 1.0 - (
        float(r1["gross_gpu_energy_j"]) / float(k100["gross_gpu_energy_j"])
    )
    by_arm = {
        arm: [row for row in pass_summaries if row["arm"] == arm]
        for arm in ("K100", "R1")
    }
    worst_accuracy_delta = {
        key: 100.0
        * (
            min(float(row["final_ema_metrics"][key]) for row in by_arm["R1"])
            - max(float(row["final_ema_metrics"][key]) for row in by_arm["K100"])
        )
        for key in ("average_mAP", "mAP@0.7")
    }
    secondary_conflicts = []
    lower_is_better = {
        "full_stack_p95_ms": lambda row: float(row["latency_ms"]["end_to_end_serial_ms"]["p95"]),
        "peak_gpu_allocated_mb": lambda row: float(row["peak_gpu_allocated_mb"]),
        "peak_gpu_reserved_mb": lambda row: float(row["peak_gpu_reserved_mb"]),
    }
    higher_is_better = {
        "throughput_windows_per_second": lambda row: float(row["throughput_windows_per_second"]),
    }
    for name, getter in lower_is_better.items():
        if min(getter(row) for row in by_arm["R1"]) > max(
            getter(row) for row in by_arm["K100"]
        ):
            secondary_conflicts.append(name)
    for name, getter in higher_is_better.items():
        if max(getter(row) for row in by_arm["R1"]) < min(
            getter(row) for row in by_arm["K100"]
        ):
            secondary_conflicts.append(name)
    k_quality = arm_quality["K100"]["metrics"]["boundary"]
    r_quality = arm_quality["R1"]["metrics"]["boundary"]
    quality_reversals = []
    if (
        r_quality["short_action"]["recall_at_tiou_0.70"] is not None
        and k_quality["short_action"]["recall_at_tiou_0.70"] is not None
        and r_quality["short_action"]["recall_at_tiou_0.70"]
        < k_quality["short_action"]["recall_at_tiou_0.70"]
    ):
        quality_reversals.append("short_action_recall_at_tiou_0.70")
    for key in ("mean_abs_start_error_normalized", "mean_abs_end_error_normalized"):
        if r_quality[key] is not None and k_quality[key] is not None and r_quality[key] > k_quality[key]:
            quality_reversals.append(f"overall_boundary_{key}")
    accuracy_ok = all(value >= -0.30 for value in worst_accuracy_delta.values())
    prediction_hash_conflict = any(
        not bool(receipt["all_four_hashes_identical"])
        for receipt in prediction_stability.values()
    )
    if (
        p50_reduction <= 0.02
        or energy_reduction <= 0.02
        or not accuracy_ok
        or quality_reversals
    ):
        decision = "STOP_BPNS_R1_CANDIDATE"
    elif (
        p50_reduction >= 0.05
        and energy_reduction >= 0.05
        and not secondary_conflicts
        and not prediction_hash_conflict
    ):
        decision = "ACCEPT_FOR_RESULT_TO_CLAIM_REVIEW"
    else:
        decision = "REVISE_AND_RETURN_TO_PRO"
    return {
        "decision": decision,
        "p50_reduction": p50_reduction,
        "gross_energy_per_pass_reduction": energy_reduction,
        "worst_case_accuracy_delta_percentage_points": worst_accuracy_delta,
        "accuracy_noninferiority_pass": accuracy_ok,
        "secondary_systematic_conflicts": secondary_conflicts,
        "quality_reversals": quality_reversals,
        "prediction_hash_conflict": prediction_hash_conflict,
        "rule": {
            "accept": "p50>=5%, gross_energy>=5%, worst Avg-mAP and mAP@0.7 deltas>=-0.30pp, no systematic conflict",
            "revise": "positive but borderline benefit or pass/prediction conflict",
            "stop": "p50 or energy<=2%, accuracy noninferiority failure, or boundary/short-action reversal",
        },
    }


def _write_failure_terminal_receipt(args: argparse.Namespace, error: BaseException) -> Path:
    if args.result_root.exists():
        path = args.result_root / "terminal_receipt.json"
    else:
        path = args.result_root.with_name(f"{args.result_root.name}.terminal_receipt.json")
    if path.exists():
        path = path.with_name(f"terminal_failure_receipt_{os.getpid()}.json")
    _atomic_json(
        path,
        {
            "status": "FAILED_PROTOCOL_INVALID",
            "phase": args.command,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "execution_commit": args.expected_commit,
            "result_root": str(args.result_root),
            "training_or_resume_executed": False,
            "official_test_opened": False,
        },
    )
    return path


def profile(args: argparse.Namespace) -> dict[str, Any]:
    import mmengine
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
    predictions_by_pass = []
    energy_windows_by_pass = []
    nms_windows = []
    primary_error = None
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
            all_samples.extend(samples)
            pass_receipts.append(receipt)
            predictions_by_pass.append(predictions)
            energy_windows_by_pass.append(energy_windows)
            nms_windows.append(nms_window)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        time.sleep(0.05)
        try:
            sidecar.stop()
        except Exception:
            if primary_error is None:
                raise

    for pass_index, energy_windows in enumerate(energy_windows_by_pass):
        pass_receipts[pass_index]["power_coverage"] = _power_coverage_summary(
            sidecar.samples,
            [*energy_windows, nms_windows[pass_index]],
        )
        nms_energy = integrate_energy(sidecar.samples, start=nms_windows[pass_index][0], end=nms_windows[pass_index][1])
        if nms_energy is None:
            raise RuntimeError("power trace does not cover final NMS")
        pass_rows = [row for row in all_samples if row["pass_index"] == pass_index]
        for row, window in zip(pass_rows, energy_windows):
            energy = integrate_energy(sidecar.samples, start=window[0], end=window[1])
            if energy is None:
                raise RuntimeError("power trace does not cover a measured validation window")
            row["gpu_energy_j"] = energy + nms_energy / len(pass_rows)

    pass_artifacts = _persist_pass_artifacts(args.result_root, pass_receipts, predictions_by_pass)
    prediction_stability = {}
    for arm in ("K100", "R1"):
        selected = [row for row in pass_artifacts if row["arm"] == arm]
        hashes = [row["prediction_sha256"] for row in selected]
        prediction_stability[arm] = {
            "pass_indices": [row["pass_index"] for row in selected],
            "prediction_sha256": hashes,
            "all_four_hashes_identical": len(set(hashes)) == 1,
        }

    annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
    short_annotation_path = args.result_root / "short_action_validation_gt.json"
    _atomic_json(short_annotation_path, _short_action_annotation(annotation))
    pass_quality = []
    from opentad.evaluations import build_evaluator

    for receipt, prediction in zip(pass_receipts, predictions_by_pass):
        arm = receipt["arm"]
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
        quality = {
            "boundary": boundary_quality(annotation, prediction["results"]),
            "short_action_mAP": {key: float(value) for key, value in short_evaluator.evaluate().items()},
        }
        receipt["quality"] = quality
        pass_quality.append({"arm": arm, "pass_index": receipt["pass_index"], **quality})
    arm_quality = {
        arm: {
            "primary_estimator": "median_of_four_pass_estimates",
            "metrics": _median_numeric_tree(
                [
                    {"boundary": row["boundary"], "short_action_mAP": row["short_action_mAP"]}
                    for row in pass_quality
                    if row["arm"] == arm
                ]
            ),
        }
        for arm in ("K100", "R1")
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
    pass_summaries = []
    for receipt in pass_receipts:
        pass_rows = [
            row for row in all_samples if int(row["pass_index"]) == int(receipt["pass_index"])
        ]
        pass_summaries.append(_summarize_pass(pass_rows, receipt, latency_keys))

    grouped = defaultdict(list)
    for row in all_samples:
        grouped[row["arm"]].append(row)
    descriptive_pooled_arm_summaries = {}
    for arm in ("K100", "R1"):
        rows = grouped[arm]
        total_seconds = sum(row["end_to_end_serial_ms"] for row in rows) / 1000.0
        gross_energy = sum(float(row["gpu_energy_j"]) for row in rows)
        descriptive_pooled_arm_summaries[arm] = {
            "estimator_role": "descriptive_only_not_primary",
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
    arm_summaries = {}
    for arm in ("K100", "R1"):
        arm_summaries[arm] = _median_of_four_arm_summary(pass_summaries, arm, latency_keys)
        arm_summaries[arm].update(
            {
                "quality": arm_quality[arm],
                "tokens_per_tubelet": ARM_SPECS[arm]["tokens_per_tubelet"],
                "executed_tokens_per_window": ARM_SPECS[arm]["executed_tokens_per_window"],
            }
        )
    comparison = {
        "r1_over_k100": {
            "primary_estimator": "ratio_of_arm_median_of_four_pass_estimates",
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
        "interpretation_boundary": "native token and attention-pair ratios are structural proxies; latency, memory and energy fields are measured full-stack evidence; pooled rows are descriptive only",
    }
    scientific_decision = _classify_complete_replay(
        pass_summaries,
        arm_summaries,
        prediction_stability,
        arm_quality,
    )
    pass_diagnoses = {
        row["diagnosis"]
        for receipt in pass_receipts
        for row in receipt["accuracy_parity"]["metrics"].values()
    }
    accuracy_parity = {
        "status": "NONBLOCKING_DIAGNOSTIC_COMPLETE",
        "diagnoses_present": sorted(pass_diagnoses),
        "contract": _accuracy_parity_contract(),
        "passes": [
            {
                "arm": receipt["arm"],
                "pass_index": receipt["pass_index"],
                "metrics": receipt["accuracy_parity"]["metrics"],
            }
            for receipt in pass_receipts
        ],
    }
    profile_payload = {
        "schema_version": "zoomtoken_bpns_r1_identity_gated_full_stack_v003",
        "status": "COMPLETED_FINAL_EMA_REPLAY",
        "execution_commit": args.expected_commit,
        "seed": 42,
        "profile_order": list(PROFILE_ORDER),
        "counterbalancing": "ABBA+BAAB",
        "warmup_windows_per_pass": WARMUP_WINDOWS,
        "power_interval_ms": POWER_INTERVAL_MS,
        "dataset": {"name": "THUMOS14", "split": "validation", "video_count": EXPECTED_VIDEO_COUNT, "window_count": EXPECTED_WINDOW_COUNT, "official_test_opened": False},
        "hardware": hardware,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "mmengine": mmengine.__version__,
            "precision": "torch.autocast_cuda_float16",
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        },
        "cpu_partition": {"allocated": list(allocated), "detector": list(detector_cpus), "sidecar": sidecar_cpu},
        "precheck": preflight,
        "pass_receipts": pass_receipts,
        "pass_artifacts": pass_artifacts,
        "prediction_stability": prediction_stability,
        "pass_summaries": pass_summaries,
        "accuracy_parity": accuracy_parity,
        "arm_summaries": arm_summaries,
        "descriptive_pooled_arm_summaries": descriptive_pooled_arm_summaries,
        "comparison": comparison,
        "scientific_decision": scientific_decision,
        "training_or_resume_executed": False,
        "paper_claim_allowed_without_independent_result_to_claim": False,
    }
    _write_jsonl(args.result_root / "cost_samples.jsonl", all_samples)
    pass_bounds = [
        (
            pass_index,
            PROFILE_ORDER[pass_index],
            min(start for start, _ in [*energy_windows_by_pass[pass_index], nms_windows[pass_index]]),
            max(end for _, end in [*energy_windows_by_pass[pass_index], nms_windows[pass_index]]),
        )
        for pass_index in range(len(PROFILE_ORDER))
    ]
    power_rows = []
    for index, (timestamp, power) in enumerate(sidecar.samples):
        membership = [
            (pass_index, arm)
            for pass_index, arm, start, end in pass_bounds
            if start <= timestamp <= end
        ]
        power_rows.append(
            {
                "sequence": index,
                "monotonic_s": timestamp,
                "power_w": power,
                "pass_index": membership[0][0] if membership else None,
                "arm": membership[0][1] if membership else None,
            }
        )
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
            "accuracy_parity": accuracy_parity,
            "scientific_decision": scientific_decision,
            "command": list(sys.argv),
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
    try:
        result = precheck(args) if args.command == "precheck" else profile(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except BaseException as error:
        _write_failure_terminal_receipt(args, error)
        raise
    finally:
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except ImportError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
