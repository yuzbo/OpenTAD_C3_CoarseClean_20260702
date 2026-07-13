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

from opentad.datasets import build_dataset
from opentad.datasets.builder import collate
from opentad.datasets.transforms.phystime_raw import validate_raw_video_timebase
from opentad.cores.test_engine import apply_sliding_window_nms
from opentad.evaluations import build_evaluator
from opentad.models import build_detector
from tools.bata.audit_phystime_g0_native_geometry import parameter_schema, run_audit
from tools.bata.audit_phystime_g0_native_geometry import SCHEMA_VERSION as G0_SCHEMA_VERSION
from tools.bata.run_phystime_adatad_real_gate import (
    _all_finite_parameters,
    _canonical_sha256,
    _finite_tree,
    _gradient_stats,
    _load_real_sample,
    _move_batch,
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
SCHEMA_VERSION = "phystime_g1a_real_gate_v2"
OPTIMIZER_STEPS = 3


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
    samples,
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
    schema = parameter_schema(model)
    initial_state_sha256 = _state_dict_sha256(model)
    optimizer, optimizer_report = _optimizer_coverage(cfg, model)
    optimizer_schema = _optimizer_schema(optimizer, model)
    optimized_parameters = _optimized_parameters(optimizer)
    backbone_lengths = []

    def capture_backbone_length(_module, _inputs, output):
        _require(torch.is_tensor(output) and output.ndim == 3, "backbone must emit [B,C,J]")
        backbone_lengths.append(int(output.shape[-1]))

    hook = model.backbone.register_forward_hook(capture_backbone_length)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    scaler = GradScaler(enabled=True, init_scale=amp_contract["init_scale"])
    step_reports = []
    optimizer_steps_completed = 0
    last_losses = None
    gradient_reports = None
    first_meta = None

    torch.cuda.synchronize(device)
    started = time.perf_counter()
    for step_index, sample in enumerate(samples):
        batch = _move_batch(collate([sample]), device)
        meta = batch["metas"][0]
        if first_meta is None:
            first_meta = dict(meta)
        _require(
            meta.get("phystime_window_crop_uses_gt") is True,
            f"{name} train-window crop provenance must disclose standard GT-aware random_trunc",
        )
        _require(
            meta.get("phystime_subsample_uses_gt") is False,
            f"{name} within-window irregular subsampling must be GT-independent",
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=True):
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
            "adapter_gradient": _gradient_stats(
                named_parameters,
                lambda parameter_name, parameter: parameter.requires_grad
                and parameter_name.startswith("backbone.")
                and "adapter" in parameter_name.lower(),
            ),
            "projection_gradient": _gradient_stats(
                named_parameters,
                lambda parameter_name, parameter: parameter.requires_grad
                and parameter_name.startswith("projection."),
            ),
            "classification_gradient": _gradient_stats(
                named_parameters,
                lambda parameter_name, parameter: parameter.requires_grad
                and parameter_name.startswith(("rpn_head.cls_convs.", "rpn_head.cls_head.")),
            ),
            "regression_gradient": _gradient_stats(
                named_parameters,
                lambda parameter_name, parameter: parameter.requires_grad
                and parameter_name.startswith(("rpn_head.reg_convs.", "rpn_head.reg_head.")),
            ),
        }
        for gradient_name, gradient in gradient_reports.items():
            _require(gradient["all_finite"] is True, f"{name}.{gradient_name} is non-finite")
            _require(gradient["nonzero"] is True, f"{name}.{gradient_name} is zero")
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in optimized_parameters if parameter.grad is not None],
            float(cfg.solver.clip_grad_norm),
            error_if_nonfinite=True,
        )
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        _require(scale_after >= scale_before, f"{name} AMP skipped optimizer step {step_index}")
        _require(_all_finite_parameters(optimized_parameters), f"{name} produced non-finite parameters")
        optimizer_steps_completed += 1
        last_losses = losses
        step_reports.append(
            {
                "step": step_index,
                "losses": {key: float(value.detach().cpu().item()) for key, value in losses.items()},
                "amp_scale_before": scale_before,
                "amp_scale_after": scale_after,
            }
        )
        del batch
    torch.cuda.synchronize(device)
    train_ms = (time.perf_counter() - started) * 1000.0
    _require(optimizer_steps_completed == OPTIMIZER_STEPS, f"{name} did not complete three optimizer steps")
    final_state_sha256 = _state_dict_sha256(model)
    parameter_state_changed = final_state_sha256 != initial_state_sha256
    _require(parameter_state_changed, f"{name} optimizer steps did not change model state")

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
    inference_batch = _move_batch(collate([samples[0]]), device)
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
        "decoded_frame_count": int(samples[0]["inputs"].shape[2]),
        "raw_valid_count": int(samples[0]["masks"].sum().item()),
        "backbone_feature_length": int(backbone_lengths[0]),
        "inference_backbone_feature_length": inference_backbone_feature_length,
        "finite_loss": _finite_tree(last_losses),
        "finite_predictions": _finite_tree(predictions),
        "optimizer_coverage": optimizer_report["covered"],
        "optimizer": optimizer_report,
        "optimizer_steps_requested": OPTIMIZER_STEPS,
        "optimizer_steps_completed": optimizer_steps_completed,
        "parameter_state_changed": parameter_state_changed,
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
        "canonical_config_sha256": canonical_config_sha256,
        "runtime_config_sha256": _canonical_sha256(cfg.to_dict()),
    }
    del optimizer, model, inference_batch, tail_batch
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
    _require(report.get("parameter_schema_match") is True, "G1a parameter schemas differ")
    _require(report.get("initial_state_match") is True, "G1a initial model states differ")
    _require(report.get("optimizer_schema_match") is True, "G1a optimizer schemas differ")
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
    variants = report.get("variants", {})
    _require(set(variants) == set(GATE_CONFIGS), "G1a gate must contain both arms")
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
            "amp_contract_verified": True,
            "train_window_crop_uses_gt": True,
            "train_subsample_uses_gt": False,
            "tail_window_crop_uses_gt": False,
            "tail_subsample_uses_gt": False,
        }.items():
            _require(result.get(key) == expected, f"{name}.{key} must be {expected!r}")
        for gradient_name in (
            "adapter_gradient",
            "projection_gradient",
            "classification_gradient",
            "regression_gradient",
        ):
            gradient = result.get(gradient_name, {})
            _require(gradient.get("all_finite") is True, f"{name}.{gradient_name} must be finite")
            _require(gradient.get("nonzero") is True, f"{name}.{gradient_name} must be non-zero")
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
            scale_before = float(step_report.get("amp_scale_before", float("nan")))
            scale_after = float(step_report.get("amp_scale_after", float("nan")))
            _require(
                math.isfinite(scale_before)
                and math.isfinite(scale_after)
                and scale_before > 0.0
                and scale_after >= scale_before,
                f"{name} step {step_index} AMP scale proves a skipped or invalid update",
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
    samples = {}
    tail_samples = {}
    sample_indices = {}
    tail_indices = {}
    selected_checksums = {}
    input_checksums = {}
    target_checksums = {}
    video_names = {}
    tail_video_names = {}
    tail_input_checksums = {}
    tail_selected_checksums = {}
    datasets = {}
    tail_datasets = {}
    for name, cfg in configs.items():
        dataset, resolved_indices, train_samples, checksums = _load_train_samples(
            cfg, seed=seed, requested_index=sample_index
        )
        tail_dataset, tail_index, tail_sample = _load_tail_sample(cfg, seed)
        tail_selected = torch.as_tensor(
            tail_sample["metas"]["selected_raw_frame_indices"], dtype=torch.int64
        ).numpy()
        samples[name] = train_samples
        tail_samples[name] = tail_sample
        datasets[name] = dataset
        tail_datasets[name] = tail_dataset
        sample_indices[name] = resolved_indices
        tail_indices[name] = tail_index
        selected_checksums[name] = checksums
        tail_selected_checksums[name] = hashlib.sha256(tail_selected.tobytes()).hexdigest()
        input_checksums[name] = [_tensor_sha256(sample["inputs"]) for sample in train_samples]
        target_checksums[name] = [_target_sha256(sample) for sample in train_samples]
        tail_input_checksums[name] = _tensor_sha256(tail_sample["inputs"])
        video_names[name] = [str(sample["metas"]["video_name"]) for sample in train_samples]
        tail_video_names[name] = str(tail_sample["metas"]["video_name"])
    _require(len({tuple(value) for value in sample_indices.values()}) == 1, "G1a arms resolved different samples")
    _require(len({tuple(value) for value in video_names.values()}) == 1, "G1a arms decoded different videos")
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
            samples[name],
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
        "sample_indices": next(iter(sample_indices.values())),
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
