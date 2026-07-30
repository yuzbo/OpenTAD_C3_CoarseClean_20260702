#!/usr/bin/env python3
"""Audit official ActionFormer S0 assignment support without training.

This diagnostic binds the official matched S0 configs and epoch-35 EMA
checkpoints, seals 64 deterministic THUMOS14 training-split windows, and
compares the full native FPN query support with the frozen stratified-uniform
K=384 support.  It runs backbone/neck geometry and target assignment only:
there is no head loss, backward pass, optimizer, checkpoint mutation, test GT,
or model selection.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import os
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


SCHEMA_VERSION = "actionformer_s0_assignment_support_audit_v1"
DURATION_BUCKETS = (
    ("lt_1s", 0.0, 1.0),
    ("1_2s", 1.0, 2.0),
    ("2_4s", 2.0, 4.0),
    ("4_8s", 4.0, 8.0),
    ("8_16s", 8.0, 16.0),
    ("16_32s", 16.0, 32.0),
    ("ge_32s", 32.0, math.inf),
)


class AuditError(RuntimeError):
    """Raised when a frozen assignment-audit contract is violated."""


def require(condition, message):
    if not condition:
        raise AuditError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def state_dict_sha256(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(
            json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def atomic_write_json(path, payload):
    path = Path(path).resolve()
    require(not path.exists(), f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def git_identity(root):
    root = Path(root).resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=root, text=True
    ).strip()
    clean = (
        subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=root,
            text=True,
        )
        == ""
    )
    return commit, tree, clean


def duration_bucket(duration):
    duration = float(duration)
    require(math.isfinite(duration) and duration >= 0.0, "invalid GT duration")
    for name, lower, upper in DURATION_BUCKETS:
        if lower <= duration < upper:
            return name
    raise AuditError(f"duration outside frozen buckets: {duration}")


def normalize_checkpoint_state(state):
    normalized = {}
    for key, value in state.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def validate_checkpoint(path, expected_sha256):
    import torch

    path = Path(path).resolve()
    require(path.is_file(), f"missing checkpoint: {path}")
    require(sha256_file(path) == expected_sha256, "checkpoint SHA-256 mismatch")
    checkpoint = torch.load(path, map_location="cpu")
    require(int(checkpoint.get("epoch", -1)) == 35, "checkpoint epoch drift")
    require(
        isinstance(checkpoint.get("state_dict_ema"), dict)
        and checkpoint["state_dict_ema"],
        "checkpoint is missing state_dict_ema",
    )
    state = normalize_checkpoint_state(checkpoint["state_dict_ema"])
    require(
        all(torch.isfinite(value).all() for value in state.values()),
        "checkpoint EMA contains non-finite tensors",
    )
    record = {
        "path": str(path),
        "sha256": expected_sha256,
        "epoch": 35,
        "weights_source": "state_dict_ema",
        "state_dict_sha256": state_dict_sha256(state),
        "keys": sorted(state),
        "shapes": {key: list(state[key].shape) for key in sorted(state)},
        "dtypes": {key: str(state[key].dtype) for key in sorted(state)},
    }
    return state, record


def compare_checkpoint_structure(dense_record, sparse_record):
    require(
        dense_record["keys"] == sparse_record["keys"],
        "dense/sparse EMA key mismatch",
    )
    require(
        dense_record["shapes"] == sparse_record["shapes"],
        "dense/sparse EMA shape mismatch",
    )
    require(
        dense_record["dtypes"] == sparse_record["dtypes"],
        "dense/sparse EMA dtype mismatch",
    )
    return canonical_sha256(
        {
            "keys": dense_record["keys"],
            "shapes": dense_record["shapes"],
            "dtypes": dense_record["dtypes"],
        }
    )


def sample_fingerprint(sample, sequence_index):
    record = {
        "sequence_index": int(sequence_index),
        "video_id": str(sample["video_id"]),
        "fps": float(sample["fps"]),
        "duration": float(sample["duration"]),
        "feat_stride": int(sample["feat_stride"]),
        "feat_num_frames": int(sample["feat_num_frames"]),
        "feats_sha256": tensor_sha256(sample["feats"]),
        "segments_sha256": tensor_sha256(sample["segments"]),
        "labels_sha256": tensor_sha256(sample["labels"]),
        "feature_length": int(sample["feats"].shape[-1]),
        "gt_count": int(sample["segments"].shape[0]),
    }
    record["fingerprint_sha256"] = canonical_sha256(record)
    return record


def seal_training_windows(dataset, *, seed, window_count):
    import torch

    require(window_count == 64, "formal assignment audit requires 64 windows")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    order = torch.randperm(len(dataset), generator=generator).tolist()
    require(len(order) >= window_count, "training split has fewer than 64 videos")
    samples = []
    records = []
    for sequence_index, dataset_index in enumerate(order[:window_count]):
        sample = dataset[int(dataset_index)]
        require(sample["segments"] is not None, "sealed training sample has no GT")
        require(sample["labels"] is not None, "sealed training sample has no labels")
        cloned = {
            key: (
                value.detach().cpu().clone()
                if torch.is_tensor(value)
                else copy.deepcopy(value)
            )
            for key, value in sample.items()
        }
        record = sample_fingerprint(cloned, sequence_index)
        record["dataset_index"] = int(dataset_index)
        records.append(record)
        samples.append(cloned)
    return samples, {
        "schema_version": "actionformer_s0_assignment_sample_seal_v1",
        "split": "validation",
        "dataset_role": "official_thumos14_training_split",
        "sampling": "deterministic_randperm_then_single_crop_per_video",
        "original_training_worker_stream_replayed": False,
        "seed": int(seed),
        "window_count": window_count,
        "records": records,
        "window_sequence_sha256": canonical_sha256(records),
    }


def build_assignment_matrices(points, gt_segments, center_sample, center_radius):
    import torch

    num_points = int(points.shape[0])
    num_gt = int(gt_segments.shape[0])
    if num_gt == 0:
        empty = torch.zeros((num_points, 0), dtype=torch.bool, device=points.device)
        return {
            "candidate": empty,
            "assigned_gt": torch.full(
                (num_points,), -1, dtype=torch.long, device=points.device
            ),
        }
    gt = gt_segments[None].expand(num_points, num_gt, 2)
    left = points[:, 0, None] - gt[:, :, 0]
    right = gt[:, :, 1] - points[:, 0, None]
    offsets = torch.stack((left, right), dim=-1)
    if center_sample == "radius":
        centers = 0.5 * (gt[:, :, 0] + gt[:, :, 1])
        radius = points[:, 3, None] * float(center_radius)
        left_center = points[:, 0, None] - torch.maximum(
            centers - radius, gt[:, :, 0]
        )
        right_center = torch.minimum(centers + radius, gt[:, :, 1]) - points[
            :, 0, None
        ]
        inside = torch.stack((left_center, right_center), dim=-1).min(-1).values > 0
    else:
        inside = offsets.min(-1).values > 0
    max_distance = offsets.max(-1).values
    in_range = torch.logical_and(
        max_distance >= points[:, 1, None],
        max_distance <= points[:, 2, None],
    )
    candidate = inside & in_range
    lengths = (gt_segments[:, 1] - gt_segments[:, 0])[None].repeat(
        num_points, 1
    )
    lengths = lengths.masked_fill(~candidate, float("inf"))
    min_length, min_index = lengths.min(dim=1)
    assigned = torch.where(
        torch.isfinite(min_length),
        min_index,
        torch.full_like(min_index, -1),
    )
    return {"candidate": candidate, "assigned_gt": assigned}


def nearest_distance(values, targets):
    import torch

    if values.numel() == 0:
        return [None for _ in range(int(targets.numel()))]
    return [
        float(torch.min(torch.abs(values - target)).item())
        for target in targets
    ]


def _per_level_offsets(points):
    offsets = []
    cursor = 0
    for level in points:
        offsets.append((cursor, cursor + int(level.shape[0])))
        cursor += int(level.shape[0])
    return offsets


def analyze_sample(
    *,
    points,
    dense_masks,
    selected_masks,
    sample_index,
    gt_segments,
    gt_labels,
    sample_meta,
    center_sample,
    center_radius,
):
    import torch

    concat_points = torch.cat(points, dim=0)
    dense_mask = torch.cat(
        [mask[sample_index, 0] for mask in dense_masks], dim=0
    )
    selected_mask = torch.cat(
        [mask[sample_index, 0] for mask in selected_masks], dim=0
    )
    require(
        torch.all(selected_mask <= dense_mask).item(),
        "selected query outside dense-valid support",
    )
    assignment = build_assignment_matrices(
        concat_points,
        gt_segments,
        center_sample,
        center_radius,
    )
    candidate = assignment["candidate"]
    assigned_gt = assignment["assigned_gt"]
    dense_assigned = assigned_gt >= 0
    sparse_assigned = dense_assigned & selected_mask
    offsets = _per_level_offsets(points)
    level_rows = []
    for level_index, ((start, end), point_level, dense_level, selected_level) in enumerate(
        zip(offsets, points, dense_masks, selected_masks)
    ):
        valid = dense_level[sample_index, 0]
        selected = selected_level[sample_index, 0]
        selected_indices = torch.nonzero(selected, as_tuple=True)[0]
        selected_centers = point_level[selected_indices, 0]
        if selected_indices.numel() > 1:
            gaps = selected_centers[1:] - selected_centers[:-1]
            max_gap = float(gaps.max().item())
        else:
            max_gap = None
        level_rows.append(
            {
                "level": level_index,
                "stride": float(point_level[0, 3].item()),
                "valid_query_count": int(valid.sum().item()),
                "selected_query_count": int(selected.sum().item()),
                "dense_positive_count": int(dense_assigned[start:end].sum().item()),
                "selected_positive_count": int(sparse_assigned[start:end].sum().item()),
                "max_selected_center_gap_feature_grid": max_gap,
            }
        )

    gt_rows = []
    selected_centers = concat_points[selected_mask, 0]
    seconds_per_grid = float(sample_meta["feat_stride"]) / float(sample_meta["fps"])
    for gt_index, (segment, label) in enumerate(zip(gt_segments, gt_labels)):
        duration_grid = float((segment[1] - segment[0]).item())
        duration_sec = duration_grid * seconds_per_grid
        boundary_grid = torch.stack(
            (segment[0], 0.5 * (segment[0] + segment[1]), segment[1])
        )
        boundary_distance_grid = nearest_distance(selected_centers, boundary_grid)
        dense_candidate_count = int(
            (candidate[:, gt_index] & dense_mask).sum().item()
        )
        sparse_candidate_count = int(
            (candidate[:, gt_index] & selected_mask).sum().item()
        )
        dense_assignment_count = int(
            ((assigned_gt == gt_index) & dense_mask).sum().item()
        )
        sparse_assignment_count = int(
            ((assigned_gt == gt_index) & selected_mask).sum().item()
        )
        gt_rows.append(
            {
                "gt_index": gt_index,
                "label": int(label.item()),
                "duration_sec": duration_sec,
                "duration_bucket": duration_bucket(duration_sec),
                "dense_candidate_count": dense_candidate_count,
                "selected_candidate_count": sparse_candidate_count,
                "dense_assignment_count": dense_assignment_count,
                "selected_assignment_count": sparse_assignment_count,
                "selected_nearest_start_sec": (
                    None
                    if boundary_distance_grid[0] is None
                    else boundary_distance_grid[0] * seconds_per_grid
                ),
                "selected_nearest_center_sec": (
                    None
                    if boundary_distance_grid[1] is None
                    else boundary_distance_grid[1] * seconds_per_grid
                ),
                "selected_nearest_end_sec": (
                    None
                    if boundary_distance_grid[2] is None
                    else boundary_distance_grid[2] * seconds_per_grid
                ),
            }
        )

    return {
        "video_id": str(sample_meta["video_id"]),
        "feature_length": int(sample_meta["feats"].shape[-1]),
        "gt_count": int(gt_segments.shape[0]),
        "dense_valid_query_count": int(dense_mask.sum().item()),
        "selected_query_count": int(selected_mask.sum().item()),
        "dense_positive_count": int(dense_assigned[dense_mask].sum().item()),
        "selected_positive_count": int(sparse_assigned.sum().item()),
        "per_level": level_rows,
        "per_gt": gt_rows,
    }


def aggregate_rows(rows):
    require(rows, "assignment audit produced no rows")
    gt_rows = [gt for row in rows for gt in row["per_gt"]]
    level_rows = [level for row in rows for level in row["per_level"]]
    duration = {}
    for name, _, _ in DURATION_BUCKETS:
        bucket = [gt for gt in gt_rows if gt["duration_bucket"] == name]
        duration[name] = {
            "gt_count": len(bucket),
            "dense_gt_without_candidate": sum(
                gt["dense_candidate_count"] == 0 for gt in bucket
            ),
            "selected_gt_without_candidate": sum(
                gt["selected_candidate_count"] == 0 for gt in bucket
            ),
            "dense_gt_without_assignment": sum(
                gt["dense_assignment_count"] == 0 for gt in bucket
            ),
            "selected_gt_without_assignment": sum(
                gt["selected_assignment_count"] == 0 for gt in bucket
            ),
            "selected_candidate_retention": (
                sum(gt["selected_candidate_count"] for gt in bucket)
                / max(sum(gt["dense_candidate_count"] for gt in bucket), 1)
            ),
            "selected_assignment_retention": (
                sum(gt["selected_assignment_count"] for gt in bucket)
                / max(sum(gt["dense_assignment_count"] for gt in bucket), 1)
            ),
        }
    per_level = {}
    for level_index in sorted({row["level"] for row in level_rows}):
        level = [row for row in level_rows if row["level"] == level_index]
        per_level[str(level_index)] = {
            "window_count": len(level),
            "valid_query_count": sum(row["valid_query_count"] for row in level),
            "selected_query_count": sum(
                row["selected_query_count"] for row in level
            ),
            "dense_positive_count": sum(
                row["dense_positive_count"] for row in level
            ),
            "selected_positive_count": sum(
                row["selected_positive_count"] for row in level
            ),
            "selected_positive_retention": (
                sum(row["selected_positive_count"] for row in level)
                / max(sum(row["dense_positive_count"] for row in level), 1)
            ),
            "max_selected_center_gap_feature_grid": max(
                (
                    row["max_selected_center_gap_feature_grid"]
                    for row in level
                    if row["max_selected_center_gap_feature_grid"] is not None
                ),
                default=None,
            ),
        }
    dense_positive = sum(row["dense_positive_count"] for row in rows)
    selected_positive = sum(row["selected_positive_count"] for row in rows)
    return {
        "window_count": len(rows),
        "gt_count": len(gt_rows),
        "dense_valid_query_count": sum(
            row["dense_valid_query_count"] for row in rows
        ),
        "selected_query_count": sum(row["selected_query_count"] for row in rows),
        "dense_positive_count": dense_positive,
        "selected_positive_count": selected_positive,
        "selected_positive_retention": selected_positive / max(dense_positive, 1),
        "dense_gt_without_candidate": sum(
            gt["dense_candidate_count"] == 0 for gt in gt_rows
        ),
        "selected_gt_without_candidate": sum(
            gt["selected_candidate_count"] == 0 for gt in gt_rows
        ),
        "dense_gt_without_assignment": sum(
            gt["dense_assignment_count"] == 0 for gt in gt_rows
        ),
        "selected_gt_without_assignment": sum(
            gt["selected_assignment_count"] == 0 for gt in gt_rows
        ),
        "duration_buckets": duration,
        "per_level": per_level,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-root", required=True)
    parser.add_argument("--dense-config", required=True)
    parser.add_argument("--sparse-config", required=True)
    parser.add_argument("--official-data-root", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--dense-checkpoint-sha256", required=True)
    parser.add_argument("--sparse-checkpoint", required=True)
    parser.add_argument("--sparse-checkpoint-sha256", required=True)
    parser.add_argument("--expected-candidate-commit", required=True)
    parser.add_argument("--expected-candidate-tree", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--num-windows", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    require(args.seed == 1234567891, "formal S0 assignment seed drift")
    require(args.num_windows == 64, "formal S0 assignment window count drift")
    require(
        os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"),
        "formal assignment audit requires a Slurm CUDA allocation",
    )
    import torch

    require(torch.cuda.is_available(), "CUDA is unavailable")
    candidate_root = Path(args.candidate_root).resolve()
    commit, tree, clean = git_identity(candidate_root)
    require(
        commit == args.expected_candidate_commit
        and tree == args.expected_candidate_tree
        and clean,
        "candidate runtime identity/cleanliness mismatch",
    )
    dense_config = Path(args.dense_config).resolve()
    sparse_config = Path(args.sparse_config).resolve()
    data_root = Path(args.official_data_root).resolve()
    for path, description in (
        (dense_config, "dense config"),
        (sparse_config, "sparse config"),
        (data_root / "annotations" / "thumos14.json", "annotation"),
        (data_root / "i3d_features", "feature directory"),
    ):
        require(path.exists(), f"missing {description}: {path}")

    dense_state, dense_checkpoint = validate_checkpoint(
        args.dense_checkpoint,
        args.dense_checkpoint_sha256,
    )
    sparse_state, sparse_checkpoint = validate_checkpoint(
        args.sparse_checkpoint,
        args.sparse_checkpoint_sha256,
    )
    structure_sha256 = compare_checkpoint_structure(
        dense_checkpoint, sparse_checkpoint
    )
    del sparse_state

    previous_path = list(sys.path)
    for key in list(sys.modules):
        if key == "libs" or key.startswith("libs."):
            del sys.modules[key]
    sys.path.insert(0, str(candidate_root))
    try:
        core = importlib.import_module("libs.core")
        datasets = importlib.import_module("libs.datasets")
        modeling = importlib.import_module("libs.modeling")
        cfg = copy.deepcopy(core.load_config(str(sparse_config)))
        require(
            cfg["model"]["sparse_head"]
            == {
                "enabled": True,
                "budget": 384,
                "policy": "stratified_uniform",
                "hash_seed": 1234567891,
                "training_loss_support": "selected_native_grid_queries",
            },
            "frozen sparse-head intervention drift",
        )
        cfg["dataset"]["json_file"] = str(
            data_root / "annotations" / "thumos14.json"
        )
        cfg["dataset"]["feat_folder"] = str(data_root / "i3d_features")
        dataset = datasets.make_dataset(
            cfg["dataset_name"],
            True,
            cfg["train_split"],
            **cfg["dataset"],
        )
        samples, seal = seal_training_windows(
            dataset,
            seed=args.seed,
            window_count=args.num_windows,
        )
        model = modeling.make_meta_arch(cfg["model_name"], **cfg["model"])
        missing, unexpected = model.load_state_dict(dense_state, strict=False)
        require(
            not missing and not unexpected,
            f"dense EMA load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}",
        )
        model = model.to(torch.device("cuda:0"))
        model.train()
        state_before = state_dict_sha256(model.state_dict())
        rows = []
        with torch.no_grad():
            for batch_start in range(0, len(samples), 2):
                cpu_batch = samples[batch_start : batch_start + 2]
                for sample_offset, sample in enumerate(cpu_batch):
                    observed = sample_fingerprint(
                        sample,
                        batch_start + sample_offset,
                    )
                    require(
                        observed["fingerprint_sha256"]
                        == seal["records"][batch_start + sample_offset][
                            "fingerprint_sha256"
                        ],
                        "sealed window changed before model replay",
                    )
                batch = []
                for sample in cpu_batch:
                    replay = copy.deepcopy(sample)
                    replay["feats"] = replay["feats"].to("cuda:0")
                    replay["segments"] = replay["segments"].to("cuda:0")
                    replay["labels"] = replay["labels"].to("cuda:0")
                    batch.append(replay)
                batched_inputs, batched_masks = model.preprocessing(batch)
                features, masks = model.backbone(batched_inputs, batched_masks)
                fpn_features, fpn_masks = model.neck(features, masks)
                points = model.point_generator(fpn_features)
                selected_masks = model.sparse_query_selector(fpn_masks)
                for sample_offset, sample in enumerate(batch):
                    row = analyze_sample(
                        points=points,
                        dense_masks=fpn_masks,
                        selected_masks=selected_masks,
                        sample_index=sample_offset,
                        gt_segments=sample["segments"],
                        gt_labels=sample["labels"],
                        sample_meta=cpu_batch[sample_offset],
                        center_sample=model.train_center_sample,
                        center_radius=model.train_center_sample_radius,
                    )
                    production_cls, _ = model.label_points_single_video(
                        torch.cat(points, dim=0),
                        sample["segments"],
                        sample["labels"],
                    )
                    production_positive = production_cls.sum(dim=-1) > 0
                    dense_mask = torch.cat(
                        [mask[sample_offset, 0] for mask in fpn_masks]
                    )
                    selected_mask = torch.cat(
                        [mask[sample_offset, 0] for mask in selected_masks]
                    )
                    require(
                        row["dense_positive_count"]
                        == int((production_positive & dense_mask).sum().item()),
                        "independent dense positive count differs from production",
                    )
                    require(
                        row["selected_positive_count"]
                        == int(
                            (production_positive & selected_mask).sum().item()
                        ),
                        "independent selected positive count differs from production",
                    )
                    require(
                        row["selected_query_count"]
                        == min(384, row["dense_valid_query_count"]),
                        "K384 exact selected-query count drift",
                    )
                    row["sequence_index"] = batch_start + sample_offset
                    row["sample_fingerprint_sha256"] = seal["records"][
                        batch_start + sample_offset
                    ]["fingerprint_sha256"]
                    rows.append(row)
                del batched_inputs, batched_masks, features, masks, fpn_features
        state_after = state_dict_sha256(model.state_dict())
        require(state_before == state_after, "model state changed during audit")
    finally:
        sys.path[:] = previous_path
        for key in list(sys.modules):
            if key == "libs" or key.startswith("libs."):
                del sys.modules[key]

    output_dir = Path(args.output_dir).resolve()
    require(not output_dir.exists(), f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    seal_path = output_dir / "SAMPLE_SEAL.json"
    rows_path = output_dir / "ASSIGNMENT_SUPPORT_ROWS.json"
    atomic_write_json(seal_path, seal)
    atomic_write_json(rows_path, rows)
    summary = aggregate_rows(rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "tested",
        "validation_pass": True,
        "issues": [],
        "claim_boundary": (
            "diagnostic_only_training_split_assignment_support;"
            "not_original_training_worker_stream;not_test;not_model_selection;"
            "not_paper_main_table;not_efficiency_or_accuracy_claim"
        ),
        "paper_main_table_eligible": False,
        "primary_result_allowed": False,
        "paper_ready": False,
        "new_training": False,
        "loss_computed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "test_gt_used": False,
        "source": {
            "candidate_root": str(candidate_root),
            "candidate_commit": commit,
            "candidate_tree": tree,
            "candidate_clean": clean,
            "dense_config": {
                "path": str(dense_config),
                "sha256": sha256_file(dense_config),
            },
            "sparse_config": {
                "path": str(sparse_config),
                "sha256": sha256_file(sparse_config),
            },
            "checkpoint_structure_sha256": structure_sha256,
            "dense_checkpoint": {
                key: value
                for key, value in dense_checkpoint.items()
                if key not in {"keys", "shapes", "dtypes"}
            },
            "sparse_checkpoint": {
                key: value
                for key, value in sparse_checkpoint.items()
                if key not in {"keys", "shapes", "dtypes"}
            },
        },
        "runtime": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "protocol": {
            "split": "validation",
            "dataset_role": "official_thumos14_training_split",
            "window_count": 64,
            "seed": args.seed,
            "budget": 384,
            "policy": "stratified_uniform",
            "dense_weights_used_for_geometry": True,
            "head_logits_computed": False,
            "sample_seal": {
                "path": str(seal_path),
                "sha256": sha256_file(seal_path),
                "window_sequence_sha256": seal["window_sequence_sha256"],
            },
            "rows": {
                "path": str(rows_path),
                "sha256": sha256_file(rows_path),
            },
        },
        "summary": summary,
        "production_positive_count_crosscheck_pass": True,
        "model_state_immutable": True,
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
    }
    report_path = output_dir / "ASSIGNMENT_SUPPORT_AUDIT_COMPLETE.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
