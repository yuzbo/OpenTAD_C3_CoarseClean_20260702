from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path

import torch
from mmengine.config import Config
from torch.cuda.amp import GradScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from opentad.datasets import build_dataloader, build_dataset
from opentad.datasets.builder import collate
from opentad.datasets.transforms.phystime_raw import validate_raw_video_timebase
from opentad.cores import build_scheduler
from opentad.cores.train_engine import _call_after_optimizer_step
from opentad.cores.test_engine import apply_sliding_window_nms
from opentad.evaluations import build_evaluator
from opentad.models import build_detector
from opentad.utils import ModelEma
from tools.bata.audit_phystime_g0_native_geometry import parameter_schema, run_audit
from tools.bata.audit_phystime_g0_native_geometry import SCHEMA_VERSION as G0_SCHEMA_VERSION
from tools.bata.run_phystime_adatad_real_gate import (
    _all_finite_parameters,
    _canonical_sha256,
    _finite_tree,
    _gradient_stats,
    _load_real_sample,
    _optimized_parameters,
    _optimizer_coverage,
    _require,
    _selected_index_checksum,
    _seed_everything,
    _sha256_file,
    _tensor_sha256,
    _validate_evaluators,
)
from tools.bata.validate_phystime_g1a_track import SCHEMA_VERSION as CONTRACT_SCHEMA_VERSION


GATE_CONFIGS = {
    "selected_axis": ROOT / "configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py",
}
SCHEMA_VERSION = "phystime_g1a_real_gate_v3"
OPTIMIZER_STEPS = 3
GRADIENT_NAMES = (
    "adapter_gradient",
    "projection_gradient",
    "classification_gradient",
    "regression_gradient",
)


def _is_hex_digest(value, lengths):
    if not isinstance(value, str) or len(value) not in set(lengths):
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _require_sha256(value, label):
    _require(_is_hex_digest(value, {64}), f"{label} must be a full SHA-256 digest")


def _require_sha256_sequence(value, label, expected_count):
    _require(
        isinstance(value, list) and len(value) == int(expected_count),
        f"{label} must contain exactly {expected_count} SHA-256 digests",
    )
    for index, digest in enumerate(value):
        _require_sha256(digest, f"{label}[{index}]")


def _state_dict_sha256(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _trainable_parameter_sha256(model):
    digest = hashlib.sha256()
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_parameter_items(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    items = []
    seen = set()
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter_id = id(parameter)
            _require(
                parameter_id in names,
                "optimizer contains a parameter that is not owned by the model",
            )
            _require(
                parameter_id not in seen,
                f"optimizer contains duplicate parameter {names[parameter_id]}",
            )
            seen.add(parameter_id)
            items.append((names[parameter_id], parameter))
    _require(items, "optimizer contains no model parameters")
    return sorted(items, key=lambda item: item[0])


def _optimizer_parameter_sha256(model, optimizer):
    digest = hashlib.sha256()
    for name, parameter in _optimizer_parameter_items(model, optimizer):
        tensor = parameter.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _optimizer_parameter_contract(model, optimizer):
    names = [name for name, _ in _optimizer_parameter_items(model, optimizer)]
    return {
        "optimizer_expected_parameter_count": len(names),
        "optimizer_parameter_names_sha256": hashlib.sha256(
            json.dumps(names, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _gradient_family_for_parameter(parameter_name):
    if parameter_name.startswith("backbone.") and "adapter" in parameter_name.lower():
        return "adapter_gradient"
    if parameter_name.startswith("projection."):
        return "projection_gradient"
    if parameter_name.startswith(("rpn_head.cls_convs.", "rpn_head.cls_head.")):
        return "classification_gradient"
    if parameter_name.startswith(
        ("rpn_head.reg_convs.", "rpn_head.reg_head.", "rpn_head.scale.")
    ):
        return "regression_gradient"
    return None


def _snapshot_optimized_parameters(model, optimizer):
    optimized_ids = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if id(parameter) in optimized_ids
    }


def _parameter_delta_report(model, snapshot):
    deltas = []
    changed_names = []
    for name, parameter in model.named_parameters():
        if name not in snapshot:
            continue
        delta = (parameter.detach().cpu().float() - snapshot[name].float()).abs()
        delta_l1 = float(delta.sum().item())
        delta_max = float(delta.max().item()) if delta.numel() else 0.0
        deltas.append((delta_l1, delta_max))
        if delta_l1 > 0.0:
            changed_names.append(name)
    return {
        "trainable_parameter_delta_l1": float(sum(value[0] for value in deltas)),
        "trainable_parameter_delta_max": float(max((value[1] for value in deltas), default=0.0)),
        "changed_trainable_parameter_count": len(changed_names),
        "changed_trainable_parameter_names_sha256": hashlib.sha256(
            json.dumps(sorted(changed_names), separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }


def _optimizer_state_step_report(optimizer, model):
    parameter_names = {id(parameter): name for name, parameter in model.named_parameters()}
    steps = []
    state_parameter_names = []
    for parameter, state in optimizer.state.items():
        if "step" not in state:
            continue
        _require(
            id(parameter) in parameter_names,
            "optimizer state contains a parameter that is not owned by the model",
        )
        step = state["step"]
        steps.append(int(step.item()) if torch.is_tensor(step) else int(step))
        state_parameter_names.append(parameter_names[id(parameter)])
    return {
        "optimizer_state_parameter_count": len(steps),
        "optimizer_state_min_step": min(steps) if steps else 0,
        "optimizer_state_max_step": max(steps) if steps else 0,
        "optimizer_state_parameter_names_sha256": hashlib.sha256(
            json.dumps(sorted(state_parameter_names), separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def _optimizer_schema(optimizer, model):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        parameter_names = [names[id(parameter)] for parameter in group["params"]]
        groups.append(
            {
                "lr": float(group["lr"]),
                "weight_decay": float(group.get("weight_decay", 0.0)),
                "parameter_count": len(parameter_names),
                "parameter_names": parameter_names,
            }
        )
    payload = json.dumps(groups, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "groups": groups}


def _validate_gradient_stats(report, label):
    _require(isinstance(report, dict), f"{label} must be a mapping")
    parameter_count = int(report.get("parameter_count", -1))
    finite_count = int(report.get("finite_gradient_count", -1))
    nonzero_count = int(report.get("nonzero_gradient_count", -1))
    gradient_l1 = float(report.get("gradient_l1", float("nan")))
    _require(parameter_count > 0, f"{label} has no covered parameters")
    _require(0 <= finite_count <= parameter_count, f"{label} finite count is invalid")
    _require(0 <= nonzero_count <= finite_count, f"{label} non-zero count is invalid")
    _require(math.isfinite(gradient_l1) and gradient_l1 >= 0.0, f"{label} L1 is invalid")
    _require(
        report.get("all_finite") is (finite_count == parameter_count),
        f"{label} all_finite contradicts its counts",
    )
    _require(
        report.get("nonzero") is (nonzero_count > 0),
        f"{label} nonzero contradicts its counts",
    )
    _require(
        (gradient_l1 > 0.0) is (nonzero_count > 0),
        f"{label} L1 contradicts its non-zero count",
    )


def _validate_assignment_debug(debug, batch_size, label):
    _require(isinstance(debug, dict), f"{label} must be a mapping")
    positive_count = int(debug.get("assignment_num_positive", -1))
    per_sample = debug.get("assignment_positive_per_sample")
    valid_count = int(debug.get("assignment_valid_point_count", -1))
    gt_count = int(debug.get("assignment_gt_count", -1))
    positive_fraction = float(debug.get("assignment_positive_fraction", float("nan")))
    raw_count = int(debug.get("assignment_regression_raw_count", -1))
    raw_positive_count = int(
        debug.get("assignment_regression_raw_positive_count", -1)
    )
    active_location_count = int(
        debug.get("assignment_regression_active_location_count", -1)
    )
    _require(positive_count > 0, f"{label} has no positive assignments")
    _require(
        isinstance(per_sample, list) and len(per_sample) == int(batch_size),
        f"{label} per-sample assignment count does not match the production batch",
    )
    _require(
        all(isinstance(value, int) and value >= 0 for value in per_sample),
        f"{label} per-sample assignment counts are invalid",
    )
    _require(sum(per_sample) == positive_count, f"{label} assignment counts disagree")
    _require(
        valid_count == int(batch_size) * 378,
        f"{label} valid-point count does not match the native Q=378 batch contract",
    )
    _require(valid_count >= positive_count, f"{label} valid-point count is invalid")
    _require(gt_count > 0, f"{label} GT count is invalid")
    _require(
        math.isfinite(positive_fraction)
        and math.isclose(
            positive_fraction,
            positive_count / max(valid_count, 1),
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        ),
        f"{label} assignment fraction is inconsistent",
    )
    _require(raw_count == positive_count * 2, f"{label} raw regression count is invalid")
    _require(
        0 <= raw_positive_count <= raw_count,
        f"{label} raw positive regression count is invalid",
    )
    _require(
        0 <= active_location_count <= positive_count,
        f"{label} active regression location count is invalid",
    )
    _require(
        active_location_count <= raw_positive_count <= 2 * active_location_count,
        f"{label} raw activation counts disagree",
    )


def _aggregate_gradient_step_reports(step_gradients):
    _require(
        isinstance(step_gradients, list) and len(step_gradients) == OPTIMIZER_STEPS,
        "G1a gradient aggregation requires exactly three step reports",
    )
    aggregated = {}
    for gradient_name in GRADIENT_NAMES:
        reports = [step[gradient_name] for step in step_gradients]
        for step_index, report in enumerate(reports):
            _validate_gradient_stats(report, f"step {step_index} {gradient_name}")
        parameter_counts = {int(report["parameter_count"]) for report in reports}
        _require(
            len(parameter_counts) == 1 and next(iter(parameter_counts)) > 0,
            f"{gradient_name} parameter coverage changed across steps",
        )
        aggregated[gradient_name] = {
            "parameter_count": next(iter(parameter_counts)),
            "all_finite": all(report.get("all_finite") is True for report in reports),
            "nonzero": any(report.get("nonzero") is True for report in reports),
            "nonzero_step_count": sum(
                report.get("nonzero") is True for report in reports
            ),
            "gradient_l1_across_steps": float(
                sum(float(report.get("gradient_l1", 0.0)) for report in reports)
            ),
            "per_step_nonzero": [
                report.get("nonzero") is True for report in reports
            ],
        }
    return aggregated


def _directory_inventory(path):
    path = Path(path).resolve()
    _require(path.is_dir(), f"dataset directory not found: {path}")
    files = []
    leaves = []
    total_bytes = 0
    for item in sorted((value for value in path.rglob("*") if value.is_file()), key=lambda value: value.as_posix()):
        relative = item.relative_to(path).as_posix()
        size = int(item.stat().st_size)
        record = {
            "relative_path": relative,
            "size_bytes": size,
            "sha256": _sha256_file(item),
        }
        files.append(record)
        leaves.append(
            hashlib.sha256(
                json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).digest()
        )
        total_bytes += size
    _require(files, f"dataset directory contains no files: {path}")
    while len(leaves) > 1:
        if len(leaves) % 2:
            leaves.append(leaves[-1])
        leaves = [
            hashlib.sha256(leaves[index] + leaves[index + 1]).digest()
            for index in range(0, len(leaves), 2)
        ]
    return {
        "path": str(path),
        "file_count": len(files),
        "total_bytes": total_bytes,
        "files": files,
        "inventory_sha256": leaves[0].hex(),
        "hash_scope": "full_file_content_sha256_merkle_v1",
    }


def build_dataset_manifest(cfg, evaluation_ground_truth):
    class_map_path = Path(cfg.dataset.train.class_map).resolve()
    manifest = {
        "annotation": str(Path(evaluation_ground_truth).resolve()),
        "annotation_sha256": _sha256_file(evaluation_ground_truth),
        "class_map": str(class_map_path),
        "class_map_sha256": _sha256_file(class_map_path),
        "train_videos": _directory_inventory(cfg.dataset.train.data_path),
        "test_videos": _directory_inventory(cfg.dataset.test.data_path),
    }
    digest = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest, digest


def _timebase_tolerances(cfg):
    observed = []
    for split in ("train", "test"):
        dataset_split = getattr(cfg.dataset, split, None)
        if dataset_split is None:
            dataset_split = cfg.dataset[split]
        steps = [
            step
            for step in dataset_split.pipeline
            if step["type"] == "BuildPhysTimeRawFrameGeometry"
        ]
        _require(len(steps) == 1, f"{split} must contain one raw timebase geometry step")
        observed.append(
            {
                "fps_relative_tolerance": float(steps[0]["fps_relative_tolerance"]),
                "duration_relative_tolerance": float(
                    steps[0]["duration_relative_tolerance"]
                ),
                "frame_count_relative_tolerance": float(
                    steps[0]["frame_count_relative_tolerance"]
                ),
            }
        )
    _require(observed[0] == observed[1], "train/test timebase tolerances differ")
    return observed[0]


def audit_dataset_timebases(
    cfg,
    evaluation_ground_truth,
    decoder_probe=None,
    dataset_video_names=None,
):
    annotation_path = Path(evaluation_ground_truth).resolve()
    database = json.loads(annotation_path.read_text(encoding="utf-8")).get("database")
    _require(isinstance(database, dict) and database, "timebase audit annotation database is empty")
    if decoder_probe is None:
        from decord import VideoReader

        def decoder_probe(path):
            reader = VideoReader(str(path), num_threads=1)
            return float(reader.get_avg_fps()), int(len(reader))

    tolerances = _timebase_tolerances(cfg)
    records = []
    split_counts = {}
    directory_file_counts = {}
    unreferenced_video_names = {}
    missing_consumed_video_names = {}
    for split in ("train", "test"):
        dataset_split = getattr(cfg.dataset, split, None)
        if dataset_split is None:
            dataset_split = cfg.dataset[split]
        root = Path(dataset_split.data_path).resolve()
        paths = sorted(
            (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
            key=lambda path: path.as_posix(),
        )
        _require(paths, f"timebase audit found no MP4 videos in {root}")
        path_by_name = {}
        for path in paths:
            _require(
                path.stem not in path_by_name,
                f"timebase audit found duplicate video stem {path.stem}",
            )
            path_by_name[path.stem] = path
        if dataset_video_names is None:
            consumed_names = {
                str(row[0]) for row in build_dataset(dataset_split).data_list
            }
        else:
            _require(
                split in dataset_video_names,
                f"timebase audit lacks an explicit consumed-video set for {split}",
            )
            consumed_names = {str(value) for value in dataset_video_names[split]}
        _require(consumed_names, f"timebase audit dataset consumes no videos for {split}")
        missing = sorted(consumed_names - set(path_by_name))
        unreferenced = sorted(set(path_by_name) - consumed_names)
        missing_consumed_video_names[split] = missing
        unreferenced_video_names[split] = unreferenced
        directory_file_counts[split] = len(path_by_name)
        _require(
            not missing,
            f"timebase audit is missing consumed {split} videos: {missing[:5]}",
        )
        audited_paths = [path_by_name[name] for name in sorted(consumed_names)]
        split_counts[split] = len(audited_paths)
        for path in audited_paths:
            video_name = path.stem
            info = database.get(video_name)
            _require(isinstance(info, dict), f"timebase audit lacks annotation for {video_name}")
            annotation_frames = int(info.get("frame", 0))
            duration = float(info.get("duration", 0.0))
            _require(
                annotation_frames > 0 and duration > 0.0,
                f"timebase audit annotation is invalid for {video_name}",
            )
            decoder_avg_fps, decoder_frames = decoder_probe(path)
            errors = validate_raw_video_timebase(
                annotation_fps=annotation_frames / duration,
                decoder_avg_fps=decoder_avg_fps,
                total_frames=decoder_frames,
                duration=duration,
                **tolerances,
            )
            records.append(
                {
                    "split": split,
                    "video_name": video_name,
                    "annotation_frames": annotation_frames,
                    "decoder_frames": int(decoder_frames),
                    "annotation_duration": duration,
                    "decoder_avg_fps": float(decoder_avg_fps),
                    **errors,
                }
            )
    _require(records, "timebase audit produced no records")
    return {
        "schema_version": "phystime_g1a_full_dataset_timebase_v1",
        "audit_pass": True,
        "audit_scope": "dataset_consumed_videos_only",
        "video_count": len(records),
        "split_counts": split_counts,
        "directory_file_counts": directory_file_counts,
        "unreferenced_file_counts": {
            split: len(names) for split, names in unreferenced_video_names.items()
        },
        "unreferenced_video_names": unreferenced_video_names,
        "unreferenced_records_sha256": _canonical_sha256(unreferenced_video_names),
        "missing_consumed_video_count": sum(
            len(names) for names in missing_consumed_video_names.values()
        ),
        "missing_consumed_video_names": missing_consumed_video_names,
        "audited_video_names_sha256": _canonical_sha256(
            {
                split: [
                    record["video_name"]
                    for record in records
                    if record["split"] == split
                ]
                for split in ("train", "test")
            }
        ),
        "tolerances": tolerances,
        "max_fps_relative_error": max(record["fps_relative_error"] for record in records),
        "max_duration_relative_error": max(
            record["duration_relative_error"] for record in records
        ),
        "max_frame_count_relative_error": max(
            record["frame_count_relative_error"] for record in records
        ),
        "frame_count_mismatch_count": sum(
            record["annotation_frames"] != record["decoder_frames"] for record in records
        ),
        "records_sha256": _canonical_sha256(records),
        "records": records,
    }


def _verify_amp_contract(cfg):
    solver = cfg.solver
    contract = {
        "enabled": bool(solver.get("amp", False)),
        "init_scale": float(solver.get("amp_init_scale", -1.0)),
        "fp16_compress": bool(solver.get("fp16_compress", True)),
        "fail_on_non_finite_grad": bool(solver.get("fail_on_non_finite_grad", False)),
        "max_consecutive_skips": int(solver.get("max_consecutive_amp_skips", -1)),
        "max_total_skips_per_epoch": int(solver.get("max_total_amp_skips_per_epoch", -1)),
    }
    _require(contract["enabled"] is True, "G1a real gate requires AMP")
    _require(contract["init_scale"] == 1024.0, "G1a AMP init scale must be 1024")
    _require(contract["fp16_compress"] is False, "G1a single-GPU gate forbids FP16 compression")
    _require(contract["fail_on_non_finite_grad"] is True, "G1a must fail on non-finite gradients")
    _require(contract["max_consecutive_skips"] == 4, "G1a consecutive AMP skip budget changed")
    _require(contract["max_total_skips_per_epoch"] == 8, "G1a epoch AMP skip budget changed")
    return contract


def _load_tail_sample(cfg, seed):
    _seed_everything(seed)
    dataset = build_dataset(cfg.dataset.test)
    for index, row in enumerate(dataset.data_list):
        info = row[1]
        start = int(info.get("feature_start_idx", 0))
        end = int(info.get("feature_end_idx", start))
        if end - start + 1 >= 768:
            continue
        _seed_everything(seed)
        sample = dataset[index]
        valid = int(sample["masks"].sum().item())
        if 0 < valid < 384:
            return dataset, index, sample
    raise RuntimeError("G1a real gate could not find a partial tail test window")


def _load_train_samples(cfg, seed, requested_index, count=OPTIMIZER_STEPS):
    dataset, first_index, first_sample, first_checksum, _ = _load_real_sample(
        cfg, seed=seed, requested_index=requested_index
    )
    indices = [first_index]
    samples = [first_sample]
    checksums = [first_checksum]
    for index in range(first_index + 1, len(dataset)):
        if len(samples) >= int(count):
            break
        _seed_everything(seed + len(samples))
        sample = dataset[index]
        if "inputs" not in sample or sample["inputs"].ndim != 5:
            continue
        if int(sample["inputs"].shape[2]) != 384 or int(sample["masks"].sum().item()) != 384:
            continue
        checksum, _ = _selected_index_checksum(sample["metas"])
        indices.append(index)
        samples.append(sample)
        checksums.append(checksum)
    _require(len(samples) == int(count), f"G1a gate could not load {count} full real training samples")
    return dataset, indices, samples, checksums


def _batch_selected_index_sha256(batch):
    digests = []
    for meta in batch["metas"]:
        digest, _ = _selected_index_checksum(meta)
        digests.append(digest)
    return hashlib.sha256("|".join(digests).encode("ascii")).hexdigest()


def _batch_target_sha256(batch):
    digest = hashlib.sha256()
    for segments, labels in zip(batch["gt_segments"], batch["gt_labels"]):
        digest.update(_tensor_sha256(segments).encode("ascii"))
        digest.update(_tensor_sha256(labels).encode("ascii"))
    return digest.hexdigest()


def _copy_batch_to_device(batch, device):
    moved = dict(batch)
    for key in ("inputs", "masks", "paired_inputs", "paired_masks"):
        value = batch.get(key)
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=False)
    for key in ("gt_segments", "gt_labels"):
        if key in batch:
            moved[key] = [value.to(device) for value in batch[key]]
    if "metas" in batch:
        moved["metas"] = [dict(meta) for meta in batch["metas"]]
    return moved


def _load_production_train_batches(cfg, seed, count=OPTIMIZER_STEPS):
    _seed_everything(seed)
    dataset = build_dataset(cfg.dataset.train)
    loader = build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=True,
        drop_last=True,
        seed=seed,
        **dict(cfg.solver.train),
    )
    loader.sampler.set_epoch(0)
    loader_contract = {
        "production_train_dataloader": True,
        "production_train_batch_size": int(loader.batch_size),
        "production_train_drop_last": bool(loader.drop_last),
        "production_train_shuffle": bool(loader.sampler.shuffle),
    }
    _require(
        loader_contract
        == {
            "production_train_dataloader": True,
            "production_train_batch_size": int(cfg.solver.train.batch_size),
            "production_train_drop_last": True,
            "production_train_shuffle": True,
        },
        "G1a production DataLoader does not match the formal batch contract",
    )
    batches = []
    for batch in loader:
        expected_batch_size = int(cfg.solver.train.batch_size)
        _require(
            int(batch["inputs"].shape[0]) == expected_batch_size,
            "G1a gate did not receive the formal per-GPU training batch size",
        )
        _require(
            all(int(mask.sum().item()) == 384 for mask in batch["masks"]),
            "G1a production gate requires K=384 valid observations per training sample",
        )
        batches.append(batch)
        if len(batches) == int(count):
            break
    _require(
        len(batches) == int(count),
        f"G1a production loader could not materialize {count} full training batches",
    )
    return dataset, batches, len(loader), loader_contract


def _result_segment_count(results):
    count = 0
    for detections in results.values():
        count += len(detections)
        for detection in detections:
            segment = detection.get("segment")
            _require(segment is not None and len(segment) == 2, "post-processed detection lacks a segment")
            start, end = float(segment[0]), float(segment[1])
            score = float(detection.get("score", float("nan")))
            _require(all(math.isfinite(value) for value in (start, end, score)), "post-processed output is non-finite")
            _require(end >= start, "post-processed segment is reversed")
            _require(detection.get("label") is not None, "post-processed detection lacks a label")
    return count


def _target_sha256(sample):
    segments = _tensor_sha256(sample["gt_segments"])
    labels = _tensor_sha256(sample["gt_labels"])
    return hashlib.sha256(f"{segments}|{labels}".encode("ascii")).hexdigest()


def _run_single_video_production_eval(model, dataset, video_name, cfg, class_map, device, use_amp, seed):
    merged = {}
    window_count = 0
    cfg.post_processing.sliding_window = True
    for index, row in enumerate(dataset.data_list):
        if str(row[0]) != str(video_name):
            continue
        _seed_everything(seed)
        sample = dataset[index]
        batch = collate([sample])
        batch["inputs"] = batch["inputs"].to(device)
        batch["masks"] = batch["masks"].to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                return_loss=False,
                infer_cfg=cfg.inference,
                post_cfg=cfg.post_processing,
                ext_cls=class_map,
            )
        _require(_finite_tree(predictions), "single-video production evaluation produced non-finite predictions")
        for key, detections in predictions.items():
            merged.setdefault(key, []).extend(detections)
        window_count += 1
    _require(window_count > 0, "single-video production evaluation found no matching windows")
    merged = apply_sliding_window_nms(merged, cfg.post_processing)
    detection_count = _result_segment_count(merged)
    _require(detection_count > 0, "single-video production evaluation produced no detections")
    evaluator = build_evaluator(dict(prediction_filename={"results": merged}, **cfg.evaluation))
    metrics = evaluator.evaluate()
    finite_metrics = {key: float(value) for key, value in metrics.items()}
    _require(
        finite_metrics and all(math.isfinite(value) for value in finite_metrics.values()),
        "single-video production evaluator returned non-finite metrics",
    )
    return {
        "window_count": window_count,
        "detection_count": detection_count,
        "metrics": finite_metrics,
    }


def _run_variant(
    name,
    cfg,
    train_batches,
    train_loader_length,
    train_loader_contract,
    tail_sample,
    tail_dataset,
    tail_video_name,
    class_map,
    checkpoint,
    device,
    seed,
):
    canonical_config_sha256 = _canonical_sha256(cfg.to_dict())
    amp_contract = _verify_amp_contract(cfg)
    cfg.model.backbone.custom.pretrain = str(checkpoint)
    _seed_everything(seed)
    model = build_detector(cfg.model).to(device).train()
    _require(bool(cfg.solver.get("ema", False)) is True, f"{name} formal training requires EMA")
    model_ema = ModelEma(model)
    schema = parameter_schema(model)
    initial_state_sha256 = _state_dict_sha256(model)
    optimizer, optimizer_report = _optimizer_coverage(cfg, model)
    optimizer_schema = _optimizer_schema(optimizer, model)
    optimizer_parameter_contract = _optimizer_parameter_contract(model, optimizer)
    initial_optimizer_parameter_sha256 = _optimizer_parameter_sha256(model, optimizer)
    optimizer_base_lrs = [float(group["lr"]) for group in optimizer.param_groups]
    _require(
        optimizer_base_lrs and all(value > 0.0 for value in optimizer_base_lrs),
        f"{name} optimizer base learning rates must be positive before scheduler construction",
    )
    scheduler, _ = build_scheduler(cfg.scheduler, optimizer, int(train_loader_length))
    scheduler_initial_lrs = [float(value) for value in scheduler.get_last_lr()]
    optimized_parameters = _optimized_parameters(optimizer)
    optimized_parameter_snapshot = _snapshot_optimized_parameters(model, optimizer)
    backbone_lengths = []

    def capture_backbone_length(_module, _inputs, output):
        _require(torch.is_tensor(output) and output.ndim == 3, "backbone must emit [B,C,J]")
        backbone_lengths.append(int(output.shape[-1]))

    hook = model.backbone.register_forward_hook(capture_backbone_length)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    scaler = GradScaler(
        enabled=True,
        init_scale=amp_contract["init_scale"],
        growth_interval=int(cfg.solver.get("amp_growth_interval", 2000)),
    )
    step_reports = []
    gradient_step_reports = []
    optimizer_steps_completed = 0
    last_losses = None
    gradient_reports = None
    first_meta = None

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step_index, cpu_batch in enumerate(train_batches):
        batch = _copy_batch_to_device(cpu_batch, device)
        meta = batch["metas"][0]
        if first_meta is None:
            first_meta = dict(meta)
        _require(
            all(item.get("phystime_window_crop_uses_gt") is True for item in batch["metas"]),
            f"{name} train-window crop provenance must disclose standard GT-aware random_trunc",
        )
        _require(
            all(item.get("phystime_subsample_uses_gt") is False for item in batch["metas"]),
            f"{name} within-window irregular subsampling must be GT-independent",
        )
        optimizer.zero_grad()
        learning_rates_before = [float(group["lr"]) for group in optimizer.param_groups]
        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=True):
            losses = model(
                inputs=batch["inputs"],
                masks=batch["masks"],
                metas=batch["metas"],
                gt_segments=batch["gt_segments"],
                gt_labels=batch["gt_labels"],
                return_loss=True,
            )
        _require("cost" in losses and _finite_tree(losses), f"{name} produced invalid losses at step {step_index}")
        scaler.scale(losses["cost"]).backward()
        scaler.unscale_(optimizer)
        _require(
            all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all().item())
                for parameter in optimized_parameters
            ),
            f"{name} produced non-finite gradients at step {step_index}",
        )
        named_parameters = list(model.named_parameters())
        gradient_reports = {
            gradient_name: _gradient_stats(
                named_parameters,
                lambda parameter_name, parameter, expected=gradient_name: parameter.requires_grad
                and _gradient_family_for_parameter(parameter_name) == expected,
            )
            for gradient_name in GRADIENT_NAMES
        }
        assignment_debug = model.rpn_head.collect_debug_state()
        assignment_num_positive = int(
            assignment_debug.get("assignment_num_positive", 0)
        )
        _require(
            assignment_num_positive > 0,
            f"{name} step {step_index} produced no positive assignments",
        )
        _require(
            float(losses["reg_loss"].detach().cpu().item()) > 0.0,
            f"{name} step {step_index} regression loss is zero despite positive assignments",
        )
        _validate_assignment_debug(
            assignment_debug,
            batch_size=int(cfg.solver.train.batch_size),
            label=f"{name} step {step_index} assignment",
        )
        for gradient_name, gradient in gradient_reports.items():
            _validate_gradient_stats(gradient, f"{name} step {step_index} {gradient_name}")
            _require(gradient["all_finite"] is True, f"{name}.{gradient_name} is non-finite")
            if gradient_name != "regression_gradient":
                _require(
                    gradient["nonzero"] is True,
                    f"{name} step {step_index} {gradient_name} is zero",
                )
        gradient_step_reports.append(gradient_reports)
        step_diagnostic = {
            "step": step_index,
            "assignment_num_positive": assignment_num_positive,
            "assignment_valid_point_count": int(
                assignment_debug.get("assignment_valid_point_count", 0)
            ),
            "regression_gradient_nonzero": gradient_reports[
                "regression_gradient"
            ]["nonzero"],
            "regression_active_location_count": int(
                assignment_debug["assignment_regression_active_location_count"]
            ),
            "losses": {
                key: float(value.detach().cpu().item())
                for key, value in losses.items()
            },
        }
        print(
            f"[PhysTime G1a gate] {name} step diagnostic "
            f"{json.dumps(step_diagnostic, sort_keys=True)}",
            flush=True,
        )
        clip_grad_norm = float(torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in optimized_parameters if parameter.grad is not None],
            float(cfg.solver.clip_grad_norm),
            error_if_nonfinite=True,
        ).item())
        _require(math.isfinite(clip_grad_norm) and clip_grad_norm > 0.0, f"{name} clip norm is invalid")
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        _require(scale_after >= scale_before, f"{name} AMP skipped optimizer step {step_index}")
        _require(_all_finite_parameters(optimized_parameters), f"{name} produced non-finite parameters")
        _call_after_optimizer_step(model)
        scheduler.step()
        model_ema.update(model)
        learning_rates_after = [float(group["lr"]) for group in optimizer.param_groups]
        optimizer_step_state = _optimizer_state_step_report(optimizer, model)
        expected_optimizer_step = step_index + 1
        _require(
            optimizer_step_state["optimizer_state_parameter_count"]
            == optimizer_parameter_contract["optimizer_expected_parameter_count"]
            and optimizer_step_state["optimizer_state_min_step"]
            == expected_optimizer_step
            and optimizer_step_state["optimizer_state_max_step"]
            == expected_optimizer_step
            and optimizer_step_state["optimizer_state_parameter_names_sha256"]
            == optimizer_parameter_contract["optimizer_parameter_names_sha256"],
            f"{name} optimizer state is incomplete after step {step_index}",
        )
        optimizer_steps_completed += 1
        last_losses = {
            key: float(value.detach().cpu().item()) for key, value in losses.items()
        }
        step_reports.append(
            {
                "step": step_index,
                "losses": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
                "assignment_debug": {
                    key: assignment_debug[key]
                    for key in (
                        "assignment_num_positive",
                        "assignment_positive_per_sample",
                        "assignment_valid_point_count",
                        "assignment_gt_count",
                        "assignment_positive_fraction",
                        "assignment_regression_raw_count",
                        "assignment_regression_raw_positive_count",
                        "assignment_regression_active_location_count",
                    )
                },
                "gradients": gradient_reports,
                "learning_rates_before": learning_rates_before,
                "learning_rates_after": learning_rates_after,
                "clip_grad_norm": clip_grad_norm,
                "scheduler_last_epoch_after": int(scheduler.last_epoch),
                "optimizer_state_parameter_count_after": optimizer_step_state[
                    "optimizer_state_parameter_count"
                ],
                "optimizer_state_min_step_after": optimizer_step_state[
                    "optimizer_state_min_step"
                ],
                "optimizer_state_max_step_after": optimizer_step_state[
                    "optimizer_state_max_step"
                ],
                "optimizer_state_parameter_names_sha256_after": optimizer_step_state[
                    "optimizer_state_parameter_names_sha256"
                ],
                "ema_updated": True,
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
            }
        )
        del losses, batch
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - started) * 1000.0
    _require(optimizer_steps_completed == OPTIMIZER_STEPS, f"{name} did not complete three optimizer steps")
    gradient_reports = _aggregate_gradient_step_reports(gradient_step_reports)
    for gradient_name, gradient in gradient_reports.items():
        _require(
            gradient["all_finite"] is True,
            f"{name}.{gradient_name} was non-finite across the three-step gate",
        )
        _require(
            gradient["nonzero"] is True,
            f"{name}.{gradient_name} stayed zero across all three gate steps",
        )
    final_state_sha256 = _state_dict_sha256(model)
    final_optimizer_parameter_sha256 = _optimizer_parameter_sha256(model, optimizer)
    parameter_state_changed = (
        final_optimizer_parameter_sha256 != initial_optimizer_parameter_sha256
    )
    parameter_delta = _parameter_delta_report(model, optimized_parameter_snapshot)
    optimizer_state = _optimizer_state_step_report(optimizer, model)
    _require(parameter_state_changed, f"{name} optimizer steps did not change optimized parameters")
    _require(
        parameter_delta["trainable_parameter_delta_l1"] > 0.0
        and parameter_delta["changed_trainable_parameter_count"] > 0,
        f"{name} has no measured trainable-parameter delta",
    )
    _require(
        optimizer_state["optimizer_state_parameter_count"]
        == optimizer_parameter_contract["optimizer_expected_parameter_count"]
        and optimizer_state["optimizer_state_min_step"] == OPTIMIZER_STEPS
        and optimizer_state["optimizer_state_max_step"] == OPTIMIZER_STEPS,
        f"{name} optimizer state did not record all three production updates",
    )
    _require(
        optimizer_state["optimizer_state_parameter_names_sha256"]
        == optimizer_parameter_contract["optimizer_parameter_names_sha256"],
        f"{name} optimizer state does not cover the fixed optimizer parameter set",
    )
    scheduler_positive_lr_observed = any(
        value > 0.0
        for step_report in step_reports
        for value in (
            step_report["learning_rates_before"] + step_report["learning_rates_after"]
        )
    )
    _require(scheduler_positive_lr_observed, f"{name} scheduler never exposed a positive learning rate")

    native_audit = model.collect_native_temporal_geometry_audit()
    _require(native_audit.get("raw_observation_count") == 384, f"{name} K audit failed")
    _require(native_audit.get("native_token_count") == 192, f"{name} J audit failed")
    _require(native_audit.get("query_tensor_count") == 378, f"{name} nominal Q audit failed")
    _require(native_audit.get("feature_interpolation") is False, f"{name} interpolated native features")
    _require(
        native_audit.get("lineage_evidence_level")
        == "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
        f"{name} structural lineage audit is missing",
    )

    head_debug = model.rpn_head.collect_debug_state()
    _require(head_debug.get("physical_grid_actionformer_enabled") is True, f"{name} seconds grid is disabled")
    _require(head_debug.get("physical_grid_actionformer_valid_points", 0) > 0, f"{name} has no valid seconds points")
    _require(
        head_debug.get("physical_grid_actionformer_positions_key") == "phystime_g1a_axis_positions_sec",
        f"{name} used the wrong seconds-axis metadata",
    )

    model.eval()
    inference_batch = _copy_batch_to_device(train_batches[0], device)
    cfg.post_processing.sliding_window = False
    torch.cuda.synchronize(device)
    infer_started = time.perf_counter()
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        predictions = model(
            inputs=inference_batch["inputs"],
            masks=inference_batch["masks"],
            metas=inference_batch["metas"],
            return_loss=False,
            infer_cfg=cfg.inference,
            post_cfg=cfg.post_processing,
            ext_cls=class_map,
        )
    torch.cuda.synchronize(device)
    infer_ms = (time.perf_counter() - infer_started) * 1000.0
    inference_backbone_feature_length = int(backbone_lengths[-1])
    _require(_finite_tree(predictions), f"{name} produced non-finite predictions")
    train_result_count = _result_segment_count(predictions)

    tail_batch = collate([tail_sample])
    _require(
        tail_batch["metas"][0].get("phystime_window_crop_uses_gt") is False,
        f"{name} validation tail window must not use GT for cropping",
    )
    _require(
        tail_batch["metas"][0].get("phystime_subsample_uses_gt") is False,
        f"{name} validation tail subsampling must be GT-independent",
    )
    tail_batch["inputs"] = tail_batch["inputs"].to(device)
    tail_batch["masks"] = tail_batch["masks"].to(device)
    cfg.post_processing.sliding_window = True
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        tail_predictions = model(
            inputs=tail_batch["inputs"],
            masks=tail_batch["masks"],
            metas=tail_batch["metas"],
            return_loss=False,
            infer_cfg=cfg.inference,
            post_cfg=cfg.post_processing,
            ext_cls=class_map,
        )
    _require(_finite_tree(tail_predictions), f"{name} produced non-finite tail-window predictions")
    tail_result_count = _result_segment_count(tail_predictions)
    tail_audit = model.collect_native_temporal_geometry_audit()
    tail_head_debug = model.rpn_head.collect_debug_state()
    _require(tail_audit.get("raw_valid_counts", [384])[0] < 384, f"{name} tail gate is not partial")
    _require(tail_audit.get("invalid_native_features_zeroed") is True, f"{name} tail padding was not zeroed")
    tail_raw_valid = int(tail_audit["raw_valid_counts"][0])
    _require(
        tail_audit.get("padding_repeat_counts", [0])[0] == 384 - tail_raw_valid,
        f"{name} tail padding-repeat provenance mismatch",
    )
    _require(
        tail_audit.get("valid_tokens_may_depend_on_padding_repeats", [False])[0] is True,
        f"{name} tail structural padding influence audit mismatch",
    )
    _require(
        tail_audit.get("candidate_mask_policy") == "semantic_anchor_prefix",
        f"{name} tail candidate-mask policy is not explicit",
    )
    padding_isolation = tail_audit.get("backbone_temporal_padding_isolation", {})
    _require(
        padding_isolation.get("strict_isolation_verified") is True
        and padding_isolation.get("attention_key_value_masked") is True
        and padding_isolation.get("adapter_convolution_masked") is True
        and padding_isolation.get("output_invalid_features_zeroed") is True,
        f"{name} did not isolate tail padding inside the backbone",
    )
    _require(
        tail_audit.get("valid_tokens_depend_on_padding_after_isolation") is False,
        f"{name} valid tail tokens still depend on padding after isolation",
    )
    _require(
        tail_head_debug.get("physical_grid_actionformer_valid_points", 0) > 0,
        f"{name} tail has no effective seconds candidates",
    )

    production_eval = _run_single_video_production_eval(
        model,
        tail_dataset,
        tail_video_name,
        cfg,
        class_map,
        device,
        True,
        seed,
    )
    hook.remove()

    report = {
        "decoded_frame_count": int(train_batches[0]["inputs"][0].shape[2]),
        "raw_valid_count": int(train_batches[0]["masks"][0].sum().item()),
        "backbone_feature_length": int(backbone_lengths[0]),
        "inference_backbone_feature_length": inference_backbone_feature_length,
        "finite_loss": _finite_tree(last_losses),
        "finite_predictions": _finite_tree(predictions),
        "optimizer_coverage": optimizer_report["covered"],
        "optimizer": optimizer_report,
        "optimizer_steps_requested": OPTIMIZER_STEPS,
        "optimizer_steps_completed": optimizer_steps_completed,
        "parameter_state_changed": parameter_state_changed,
        "initial_optimizer_parameter_sha256": initial_optimizer_parameter_sha256,
        "final_optimizer_parameter_sha256": final_optimizer_parameter_sha256,
        **parameter_delta,
        **optimizer_parameter_contract,
        **optimizer_state,
        **train_loader_contract,
        "production_scheduler": True,
        "scheduler_class": scheduler.__class__.__name__,
        "scheduler_initial_lrs": scheduler_initial_lrs,
        "scheduler_positive_lr_observed": scheduler_positive_lr_observed,
        "model_ema_enabled": True,
        "model_ema_updates": optimizer_steps_completed,
        "amp_contract_verified": True,
        "amp_contract": amp_contract,
        **gradient_reports,
        "native_geometry_audit": native_audit,
        "tail_native_geometry_audit": tail_audit,
        "head_geometry_debug": head_debug,
        "tail_head_geometry_debug": tail_head_debug,
        "full_post_processing_executed": True,
        "production_single_video_eval_executed": True,
        "production_single_video_window_count": production_eval["window_count"],
        "production_single_video_detection_count": production_eval["detection_count"],
        "production_single_video_metrics": production_eval["metrics"],
        "prediction_time_unit": str(first_meta.get("prediction_time_unit")),
        "train_window_crop_uses_gt": bool(first_meta["phystime_window_crop_uses_gt"]),
        "train_subsample_uses_gt": bool(first_meta["phystime_subsample_uses_gt"]),
        "tail_window_crop_uses_gt": bool(tail_sample["metas"]["phystime_window_crop_uses_gt"]),
        "tail_subsample_uses_gt": bool(tail_sample["metas"]["phystime_subsample_uses_gt"]),
        "train_result_segment_count": train_result_count,
        "tail_result_segment_count": tail_result_count,
        "losses": step_reports[-1]["losses"],
        "optimizer_step_reports": step_reports,
        "train_forward_backward_ms": train_ms,
        "inference_ms": infer_ms,
        "peak_cuda_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024.0**2)),
        "parameter_schema": schema,
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "optimizer_schema": optimizer_schema,
        "optimizer_base_lrs": optimizer_base_lrs,
        "canonical_config_sha256": canonical_config_sha256,
        "runtime_config_sha256": _canonical_sha256(cfg.to_dict()),
    }
    del optimizer, scheduler, model_ema, model, inference_batch, tail_batch
    gc.collect()
    torch.cuda.empty_cache()
    return report


def validate_gate_report(report):
    _require(report.get("schema_version") == SCHEMA_VERSION, "G1a gate schema mismatch")
    _require(report.get("gate_pass") is True, "G1a gate did not pass")
    _require(report.get("K_raw_observations") == 384, "G1a gate K mismatch")
    _require(report.get("J_native_tubelet_tokens") == 192, "G1a gate J mismatch")
    _require(report.get("Q0_base_candidates") == 192, "G1a gate Q0 mismatch")
    _require(report.get("Q_total_candidates") == 378, "G1a gate Q mismatch")
    _require(report.get("selected_index_checksum_match") is True, "G1a selected indices differ")
    _require(report.get("decoded_input_checksum_match") is True, "G1a decoded inputs differ")
    _require(report.get("target_checksum_match") is True, "G1a supervision targets differ")
    variants = report.get("variants", {})
    _require(set(variants) == set(GATE_CONFIGS), "G1a gate must contain both arms")
    selected_variant = variants["selected_axis"]
    physical_variant = variants["physical_metric"]
    recomputed_parameter_schema_match = (
        selected_variant.get("parameter_schema", {}).get("schema")
        == physical_variant.get("parameter_schema", {}).get("schema")
    )
    recomputed_initial_state_match = (
        selected_variant.get("initial_state_sha256")
        == physical_variant.get("initial_state_sha256")
    )
    recomputed_optimizer_schema_match = (
        selected_variant.get("optimizer_schema")
        == physical_variant.get("optimizer_schema")
    )
    _require(
        report.get("parameter_schema_match") is True
        and recomputed_parameter_schema_match,
        "G1a parameter schemas differ",
    )
    _require(
        report.get("initial_state_match") is True
        and recomputed_initial_state_match,
        "G1a initial model states differ",
    )
    _require(
        report.get("optimizer_schema_match") is True
        and recomputed_optimizer_schema_match,
        "G1a optimizer schemas differ",
    )
    _require(report.get("tree_clean") is True, "G1a gate is not bound to a clean tree")
    _require(report.get("real_g0_pass") is True, "G1a real-data G0 gate did not pass")
    _require(report.get("optimizer_steps") == OPTIMIZER_STEPS, "G1a optimizer-step count mismatch")
    _require(report.get("amp_contract_verified") is True, "G1a AMP contract was not verified")
    timebase_audit = report.get("timebase_audit", {})
    _require(timebase_audit.get("audit_pass") is True, "G1a full-dataset timebase audit failed")
    _require(
        timebase_audit.get("audit_scope") == "dataset_consumed_videos_only",
        "G1a timebase audit used the wrong video scope",
    )
    _require(int(timebase_audit.get("video_count", 0)) > 0, "G1a timebase audit is empty")
    _require(
        int(timebase_audit.get("video_count", 0))
        == sum(int(value) for value in timebase_audit.get("split_counts", {}).values()),
        "G1a timebase split counts do not match the audited total",
    )
    _require(
        timebase_audit.get("missing_consumed_video_count") == 0,
        "G1a timebase audit is missing dataset-consumed videos",
    )
    _require_sha256(timebase_audit.get("records_sha256"), "G1a timebase records")
    _require_sha256(
        timebase_audit.get("audited_video_names_sha256"),
        "G1a audited timebase video names",
    )
    _require_sha256(
        timebase_audit.get("unreferenced_records_sha256"),
        "G1a unreferenced timebase records",
    )
    _require(
        timebase_audit.get("frame_count_mismatch_count") == 0,
        "G1a annotation/decoder frame counts differ",
    )
    for key in (
        "dataset_manifest_sha256",
        "checkpoint_sha256",
        "contract_sha256",
        "static_g0_sha256",
        "tail_selected_index_sha256",
        "tail_decoded_input_sha256",
    ):
        _require_sha256(report.get(key), f"G1a {key}")
    for key in ("selected_index_sha256", "decoded_input_sha256", "target_sha256"):
        _require_sha256_sequence(report.get(key), f"G1a {key}", OPTIMIZER_STEPS)
    _require(
        _is_hex_digest(report.get("git_commit"), {40, 64}),
        "G1a commit must be a full Git object id",
    )
    _require(
        _is_hex_digest(report.get("git_tree"), {40, 64}),
        "G1a tree must be a full Git object id",
    )
    for name, result in variants.items():
        for key, expected in {
            "decoded_frame_count": 384,
            "raw_valid_count": 384,
            "backbone_feature_length": 192,
            "inference_backbone_feature_length": 192,
            "finite_loss": True,
            "finite_predictions": True,
            "optimizer_coverage": True,
            "optimizer_steps_requested": OPTIMIZER_STEPS,
            "optimizer_steps_completed": OPTIMIZER_STEPS,
            "parameter_state_changed": True,
            "production_train_dataloader": True,
            "production_train_batch_size": 2,
            "production_train_drop_last": True,
            "production_train_shuffle": True,
            "production_scheduler": True,
            "scheduler_class": "LinearWarmupCosineAnnealingLR",
            "scheduler_positive_lr_observed": True,
            "model_ema_enabled": True,
            "model_ema_updates": OPTIMIZER_STEPS,
            "amp_contract_verified": True,
            "train_window_crop_uses_gt": True,
            "train_subsample_uses_gt": False,
            "tail_window_crop_uses_gt": False,
            "tail_subsample_uses_gt": False,
        }.items():
            _require(result.get(key) == expected, f"{name}.{key} must be {expected!r}")
        _require_sha256(
            result.get("initial_optimizer_parameter_sha256"),
            f"{name} initial optimizer parameters",
        )
        _require_sha256(
            result.get("final_optimizer_parameter_sha256"),
            f"{name} final optimizer parameters",
        )
        _require(
            result["initial_optimizer_parameter_sha256"]
            != result["final_optimizer_parameter_sha256"],
            f"{name} optimizer parameter digest did not change",
        )
        _require(
            math.isfinite(float(result.get("trainable_parameter_delta_l1", float("nan"))))
            and float(result["trainable_parameter_delta_l1"]) > 0.0,
            f"{name} trainable parameter L1 delta is invalid",
        )
        _require(
            math.isfinite(float(result.get("trainable_parameter_delta_max", float("nan"))))
            and float(result["trainable_parameter_delta_max"]) > 0.0,
            f"{name} trainable parameter max delta is invalid",
        )
        _require(
            int(result.get("changed_trainable_parameter_count", 0)) > 0,
            f"{name} changed no trainable parameters",
        )
        if "changed_trainable_parameter_names_sha256" in result:
            _require_sha256(
                result["changed_trainable_parameter_names_sha256"],
                f"{name} changed trainable parameter names",
            )
        expected_optimizer_parameter_count = int(
            result.get("optimizer_expected_parameter_count", 0)
        )
        _require(
            expected_optimizer_parameter_count > 0,
            f"{name} optimizer expected-parameter count is invalid",
        )
        _require_sha256(
            result.get("optimizer_parameter_names_sha256"),
            f"{name} optimizer parameter names",
        )
        _require_sha256(
            result.get("optimizer_state_parameter_names_sha256"),
            f"{name} optimizer state parameter names",
        )
        _require(
            int(result.get("optimizer_state_parameter_count", 0))
            == expected_optimizer_parameter_count
            and int(result.get("optimizer_state_min_step", 0)) == OPTIMIZER_STEPS
            and int(result.get("optimizer_state_max_step", 0)) == OPTIMIZER_STEPS
            and result["optimizer_state_parameter_names_sha256"]
            == result["optimizer_parameter_names_sha256"],
            f"{name} optimizer state does not prove complete three-step updates",
        )
        optimizer_base_lrs = result.get("optimizer_base_lrs")
        scheduler_initial_lrs = result.get("scheduler_initial_lrs")
        _require(
            isinstance(optimizer_base_lrs, list)
            and optimizer_base_lrs
            and all(math.isfinite(float(value)) and float(value) > 0.0 for value in optimizer_base_lrs),
            f"{name} optimizer base learning rates are invalid",
        )
        _require(
            isinstance(scheduler_initial_lrs, list)
            and len(scheduler_initial_lrs) == len(optimizer_base_lrs)
            and all(math.isfinite(float(value)) and float(value) >= 0.0 for value in scheduler_initial_lrs),
            f"{name} scheduler initial learning rates are invalid",
        )
        step_reports = result.get("optimizer_step_reports")
        _require(
            isinstance(step_reports, list) and len(step_reports) == OPTIMIZER_STEPS,
            f"{name} must contain exactly three optimizer-step reports",
        )
        for step_index, step_report in enumerate(step_reports):
            _require(step_report.get("step") == step_index, f"{name} optimizer steps are not contiguous")
            losses = step_report.get("losses")
            _require(isinstance(losses, dict) and "cost" in losses, f"{name} step {step_index} losses missing")
            _require(
                all(math.isfinite(float(value)) for value in losses.values()),
                f"{name} step {step_index} losses are non-finite",
            )
            assignment_debug = step_report.get("assignment_debug", {})
            _validate_assignment_debug(
                assignment_debug,
                batch_size=int(result["production_train_batch_size"]),
                label=f"{name} step {step_index} assignment",
            )
            _require(
                float(losses.get("reg_loss", 0.0)) > 0.0,
                f"{name} step {step_index} has no regression supervision",
            )
            step_gradients = step_report.get("gradients", {})
            _require(
                set(step_gradients) == set(GRADIENT_NAMES),
                f"{name} step {step_index} gradient diagnostics are incomplete",
            )
            for gradient_name, gradient in step_gradients.items():
                _validate_gradient_stats(
                    gradient, f"{name} step {step_index} {gradient_name}"
                )
                _require(
                    gradient.get("all_finite") is True,
                    f"{name} step {step_index} {gradient_name} is non-finite",
                )
                if gradient_name != "regression_gradient":
                    _require(
                        gradient.get("nonzero") is True,
                        f"{name} step {step_index} {gradient_name} is zero",
                    )
            learning_rates_before = step_report.get("learning_rates_before")
            learning_rates_after = step_report.get("learning_rates_after")
            _require(
                isinstance(learning_rates_before, list)
                and isinstance(learning_rates_after, list)
                and len(learning_rates_before) == len(optimizer_base_lrs)
                and len(learning_rates_after) == len(optimizer_base_lrs)
                and all(
                    math.isfinite(float(value)) and float(value) >= 0.0
                    for value in learning_rates_before + learning_rates_after
                ),
                f"{name} step {step_index} scheduler learning rates are invalid",
            )
            clip_grad_norm = float(step_report.get("clip_grad_norm", float("nan")))
            _require(
                math.isfinite(clip_grad_norm) and clip_grad_norm > 0.0,
                f"{name} step {step_index} clip norm is invalid",
            )
            _require(
                int(step_report.get("scheduler_last_epoch_after", -1)) == step_index + 1,
                f"{name} step {step_index} scheduler order is invalid",
            )
            _require(
                int(step_report.get("optimizer_state_parameter_count_after", 0))
                == expected_optimizer_parameter_count
                and int(step_report.get("optimizer_state_min_step_after", 0))
                == step_index + 1
                and int(step_report.get("optimizer_state_max_step_after", 0))
                == step_index + 1
                and step_report.get("optimizer_state_parameter_names_sha256_after")
                == result["optimizer_parameter_names_sha256"],
                f"{name} step {step_index} optimizer state coverage is invalid",
            )
            _require(
                int(step_report.get("optimizer_state_max_step_after", 0)) == step_index + 1,
                f"{name} step {step_index} optimizer state is invalid",
            )
            _require(
                step_report.get("ema_updated") is True,
                f"{name} step {step_index} did not update EMA",
            )
            scale_before = float(step_report.get("amp_scale_before", float("nan")))
            scale_after = float(step_report.get("amp_scale_after", float("nan")))
            _require(
                math.isfinite(scale_before)
                and math.isfinite(scale_after)
                and scale_before > 0.0
                and scale_after >= scale_before,
                f"{name} step {step_index} AMP scale proves a skipped or invalid update",
            )
        recomputed_gradients = _aggregate_gradient_step_reports(
            [step["gradients"] for step in step_reports]
        )
        for gradient_name, recomputed in recomputed_gradients.items():
            recorded = result.get(gradient_name, {})
            for key in (
                "parameter_count",
                "all_finite",
                "nonzero",
                "nonzero_step_count",
                "per_step_nonzero",
            ):
                _require(
                    recorded.get(key) == recomputed[key],
                    f"{name}.{gradient_name}.{key} disagrees with step evidence",
                )
            _require(
                math.isclose(
                    float(recorded.get("gradient_l1_across_steps", float("nan"))),
                    recomputed["gradient_l1_across_steps"],
                    rel_tol=1.0e-9,
                    abs_tol=1.0e-12,
                ),
                f"{name}.{gradient_name} L1 disagrees with step evidence",
            )
            _require(
                recomputed["all_finite"] is True and recomputed["nonzero"] is True,
                f"{name}.{gradient_name} lacks aggregate finite non-zero evidence",
            )
        scheduler_positive_lr_observed = any(
            float(value) > 0.0
            for step in step_reports
            for value in step["learning_rates_before"] + step["learning_rates_after"]
        )
        _require(
            scheduler_positive_lr_observed
            and result.get("scheduler_positive_lr_observed") is True,
            f"{name} scheduler never exposed a positive learning rate",
        )
        _require_sha256(result.get("initial_state_sha256"), f"{name} initial state")
        _require_sha256(result.get("final_state_sha256"), f"{name} final state")
        _require(
            result["initial_state_sha256"] != result["final_state_sha256"],
            f"{name} state digest did not change",
        )
        audit = result.get("native_geometry_audit", {})
        _require(audit.get("feature_interpolation") is False, f"{name} interpolation audit failed")
        _require(audit.get("query_tensor_count") == 378, f"{name} candidate audit failed")
        _require(
            audit.get("lineage_evidence_level")
            == "exact_patch_inputs_plus_structural_receptive_field_upper_bound",
            f"{name} structural lineage evidence is missing",
        )
        _require(result.get("full_post_processing_executed") is True, f"{name} skipped production inference")
        _require(
            result.get("production_single_video_eval_executed") is True,
            f"{name} skipped production single-video evaluation",
        )
        _require(
            int(result.get("production_single_video_detection_count", 0)) > 0,
            f"{name} production evaluation emitted no detections",
        )
        production_metrics = result.get("production_single_video_metrics", {})
        _require(
            production_metrics
            and all(math.isfinite(float(value)) for value in production_metrics.values()),
            f"{name} production metrics are missing or non-finite",
        )
        _require(result.get("prediction_time_unit") == "seconds", f"{name} output is not canonical seconds")
        _require_sha256(result.get("canonical_config_sha256"), f"{name} canonical config")
        tail_audit = result.get("tail_native_geometry_audit", {})
        tail_raw_valid = int(tail_audit.get("raw_valid_counts", [384])[0])
        _require(tail_raw_valid < 384, f"{name} lacks a tail-window gate")
        _require(tail_audit.get("invalid_native_features_zeroed") is True, f"{name} tail padding can leak")
        _require(
            tail_audit.get("padding_repeat_counts", [0])[0] == 384 - tail_raw_valid,
            f"{name} tail padding-repeat audit is incomplete",
        )
        _require(
            tail_audit.get("valid_tokens_may_depend_on_padding_repeats", [False])[0] is True,
            f"{name} tail structural padding influence audit is missing",
        )
        _require(
            tail_audit.get("candidate_mask_policy") == "semantic_anchor_prefix",
            f"{name} tail candidate-mask policy is not explicit",
        )
        padding_isolation = tail_audit.get("backbone_temporal_padding_isolation", {})
        _require(
            padding_isolation.get("strict_isolation_verified") is True
            and padding_isolation.get("attention_key_value_masked") is True
            and padding_isolation.get("adapter_convolution_masked") is True
            and padding_isolation.get("output_invalid_features_zeroed") is True,
            f"{name} tail padding isolation proof is missing",
        )
        _require(
            tail_audit.get("valid_tokens_depend_on_padding_after_isolation") is False,
            f"{name} valid tail tokens depend on padding after isolation",
        )
        debug = result.get("head_geometry_debug", {})
        _require(debug.get("physical_grid_actionformer_enabled") is True, f"{name} seconds grid debug missing")
        _require(debug.get("physical_grid_actionformer_valid_points", 0) > 0, f"{name} has no seconds candidates")
        _require(
            debug.get("physical_grid_actionformer_axis_start_key") == "phystime_g1a_axis_start_sec",
            f"{name} did not use the explicit seconds-domain start",
        )
        _require(
            debug.get("physical_grid_actionformer_axis_end_key") == "phystime_g1a_axis_end_sec",
            f"{name} did not use the explicit seconds-domain end",
        )
        tail_debug = result.get("tail_head_geometry_debug", {})
        _require(
            tail_debug.get("physical_grid_actionformer_enabled") is True,
            f"{name} tail seconds-grid debug is missing",
        )
        _require(
            tail_debug.get("physical_grid_actionformer_valid_points", 0) > 0,
            f"{name} tail has no effective seconds candidates",
        )
    return True


def _load_bound_artifact(path, *, schema_version, pass_key, expected_pass):
    path = Path(path).resolve()
    _require(path.is_file(), f"required gate artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == schema_version, f"artifact schema mismatch: {path}")
    _require(payload.get(pass_key) is expected_pass, f"artifact {pass_key} mismatch: {path}")
    return path, payload


def run_gate(
    checkpoint,
    contract,
    static_g0,
    device,
    expected_commit,
    expected_tree,
    seed=42,
    sample_index=-1,
):
    checkpoint = Path(checkpoint).resolve()
    _require(checkpoint.is_file(), f"VideoMAE-S checkpoint not found: {checkpoint}")
    _require(device.type == "cuda" and torch.cuda.is_available(), "G1a real gate requires CUDA")
    tree_status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    _require(tree_status.strip() == "", "G1a real gate requires a clean fixed snapshot")
    git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_tree = subprocess.check_output(["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True).strip()
    _require(git_commit == str(expected_commit), "G1a runtime commit differs from submitted commit")
    _require(git_tree == str(expected_tree), "G1a runtime tree differs from submitted tree")
    contract_path, contract_payload = _load_bound_artifact(
        contract,
        schema_version=CONTRACT_SCHEMA_VERSION,
        pass_key="contract_pass",
        expected_pass=True,
    )
    static_g0_path, static_g0_payload = _load_bound_artifact(
        static_g0,
        schema_version=G0_SCHEMA_VERSION,
        pass_key="static_precheck_pass",
        expected_pass=True,
    )
    _require(contract_payload.get("git_commit") == git_commit, "G1a contract commit mismatch")
    _require(contract_payload.get("git_tree") == git_tree, "G1a contract tree mismatch")
    _require(static_g0_payload.get("git_commit") == git_commit, "static G0 commit mismatch")
    _require(static_g0_payload.get("git_tree") == git_tree, "static G0 tree mismatch")
    _require(static_g0_payload.get("gate_pass") is False, "static G0 must not claim a real-data pass")

    g0 = run_audit(GATE_CONFIGS, build_models=True)
    _require(
        static_g0_payload.get("config_sha256") == g0.get("config_sha256"),
        "static G0 config hashes differ from the real gate",
    )
    configs = {name: Config.fromfile(path, lazy_import=False) for name, path in GATE_CONFIGS.items()}
    canonical_config_hashes = {
        name: _canonical_sha256(configs[name].to_dict()) for name in GATE_CONFIGS
    }
    _require(
        canonical_config_hashes == contract_payload.get("config_sha256"),
        "G1a canonical config hashes differ from the static contract",
    )
    evaluation_ground_truth = _validate_evaluators(configs)
    timebase_audit = audit_dataset_timebases(configs["selected_axis"], evaluation_ground_truth)
    _require(
        int(sample_index) == -1,
        "formal G1a gate uses the production shuffled DataLoader and forbids sample-index overrides",
    )
    train_batches = {}
    train_loader_lengths = {}
    train_loader_contracts = {}
    tail_samples = {}
    tail_indices = {}
    selected_checksums = {}
    input_checksums = {}
    target_checksums = {}
    video_names = {}
    tail_video_names = {}
    tail_input_checksums = {}
    tail_selected_checksums = {}
    tail_datasets = {}
    for name, cfg in configs.items():
        _, batches, loader_length, loader_contract = _load_production_train_batches(
            cfg, seed=seed
        )
        tail_dataset, tail_index, tail_sample = _load_tail_sample(cfg, seed)
        tail_selected = torch.as_tensor(
            tail_sample["metas"]["selected_raw_frame_indices"], dtype=torch.int64
        ).numpy()
        train_batches[name] = batches
        train_loader_lengths[name] = loader_length
        train_loader_contracts[name] = loader_contract
        tail_samples[name] = tail_sample
        tail_datasets[name] = tail_dataset
        tail_indices[name] = tail_index
        selected_checksums[name] = [
            _batch_selected_index_sha256(batch) for batch in batches
        ]
        tail_selected_checksums[name] = hashlib.sha256(tail_selected.tobytes()).hexdigest()
        input_checksums[name] = [_tensor_sha256(batch["inputs"]) for batch in batches]
        target_checksums[name] = [_batch_target_sha256(batch) for batch in batches]
        tail_input_checksums[name] = _tensor_sha256(tail_sample["inputs"])
        video_names[name] = [
            [str(meta["video_name"]) for meta in batch["metas"]] for batch in batches
        ]
        tail_video_names[name] = str(tail_sample["metas"]["video_name"])
    _require(
        len(set(train_loader_lengths.values())) == 1,
        "G1a arms constructed different production DataLoader lengths",
    )
    _require(
        len(
            {
                json.dumps(value, sort_keys=True, separators=(",", ":"))
                for value in train_loader_contracts.values()
            }
        )
        == 1,
        "G1a arms constructed different production DataLoader contracts",
    )
    _require(
        len({tuple(tuple(batch) for batch in value) for value in video_names.values()}) == 1,
        "G1a arms decoded different production batches",
    )
    _require(
        len({tuple(value) for value in selected_checksums.values()}) == 1,
        "G1a arms selected different raw frames",
    )
    _require(
        len({tuple(value) for value in input_checksums.values()}) == 1,
        "G1a arms produced different augmented RGB tensors",
    )
    _require(
        len({tuple(value) for value in target_checksums.values()}) == 1,
        "G1a arms produced different GT segments or labels",
    )
    _require(len(set(tail_indices.values())) == 1, "G1a arms resolved different tail windows")
    _require(len(set(tail_video_names.values())) == 1, "G1a arms decoded different tail videos")
    _require(len(set(tail_selected_checksums.values())) == 1, "G1a arms selected different tail frames")
    _require(len(set(tail_input_checksums.values())) == 1, "G1a arms produced different tail RGB tensors")

    variants = {}
    for name in GATE_CONFIGS:
        print(f"[PhysTime G1a gate] running {name}", flush=True)
        variants[name] = _run_variant(
            name,
            configs[name],
            train_batches[name],
            train_loader_lengths[name],
            train_loader_contracts[name],
            tail_samples[name],
            tail_datasets[name],
            tail_video_names[name],
            tail_datasets[name].class_map,
            checkpoint,
            device,
            seed,
        )
    parameter_schema_match = (
        variants["selected_axis"]["parameter_schema"]["schema"]
        == variants["physical_metric"]["parameter_schema"]["schema"]
    )
    initial_state_match = (
        variants["selected_axis"]["initial_state_sha256"]
        == variants["physical_metric"]["initial_state_sha256"]
    )
    optimizer_schema_match = (
        variants["selected_axis"]["optimizer_schema"]
        == variants["physical_metric"]["optimizer_schema"]
    )
    _require(
        all(
            variants[name]["canonical_config_sha256"] == canonical_config_hashes[name]
            for name in GATE_CONFIGS
        ),
        "G1a runtime canonical config hashes changed",
    )
    dataset_manifest, dataset_manifest_sha256 = build_dataset_manifest(
        configs["selected_axis"], evaluation_ground_truth
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "gate_pass": True,
        "real_g0_pass": True,
        "input_source": "raw_thumos_mp4",
        "K_raw_observations": g0["K_raw_observations"],
        "J_native_tubelet_tokens": g0["J_native_tubelet_tokens"],
        "Q0_base_candidates": g0["Q0_base_candidates"],
        "Q_total_candidates": g0["Q_total_candidates"],
        "feature_interpolation": False,
        "selected_index_checksum_match": True,
        "decoded_input_checksum_match": True,
        "target_checksum_match": True,
        "parameter_schema_match": parameter_schema_match,
        "initial_state_match": initial_state_match,
        "optimizer_schema_match": optimizer_schema_match,
        "tree_clean": True,
        "optimizer_steps": OPTIMIZER_STEPS,
        "amp_contract_verified": all(
            result.get("amp_contract_verified") is True for result in variants.values()
        ),
        "selected_index_sha256": next(iter(selected_checksums.values())),
        "decoded_input_sha256": next(iter(input_checksums.values())),
        "target_sha256": next(iter(target_checksums.values())),
        "production_train_batch_count": OPTIMIZER_STEPS,
        "sample_videos": next(iter(video_names.values())),
        "tail_sample_index": next(iter(tail_indices.values())),
        "tail_sample_video": next(iter(tail_video_names.values())),
        "tail_selected_index_sha256": next(iter(tail_selected_checksums.values())),
        "tail_decoded_input_sha256": next(iter(tail_input_checksums.values())),
        "evaluation_ground_truth_filename": evaluation_ground_truth,
        "timebase_audit": timebase_audit,
        "dataset_manifest": dataset_manifest,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "contract": str(contract_path),
        "contract_sha256": _sha256_file(contract_path),
        "static_g0": str(static_g0_path),
        "static_g0_sha256": _sha256_file(static_g0_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": _sha256_file(checkpoint),
        "git_commit": git_commit,
        "git_tree": git_tree,
        "cuda_device": str(device),
        "gpu_name": torch.cuda.get_device_name(device),
        "seed": int(seed),
        "g0_audit": g0,
        "variants": variants,
    }
    validate_gate_report(report)
    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run the matched PhysTime G1a raw-video gate")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--static-g0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-index", type=int, default=-1)
    return parser.parse_args()


def _write(path, report):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    try:
        report = run_gate(
            args.checkpoint,
            args.contract,
            args.static_g0,
            torch.device(args.device),
            args.expected_commit,
            args.expected_tree,
            args.seed,
            args.sample_index,
        )
    except Exception as error:
        report = {
            "schema_version": SCHEMA_VERSION,
            "gate_pass": False,
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        _write(args.output, report)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        raise SystemExit(1) from error
    _write(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
