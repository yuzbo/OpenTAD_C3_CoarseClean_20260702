#!/usr/bin/env python3
"""Fail-closed, no-loss SDPQ support and assignment observability audit.

The formal path seals exactly 64 deterministic training windows before loading
the model.  It then rebuilds the same windows, runs only the backbone,
projection, physical-query construction, and target assignment, and compares
the production targets with an independent NumPy implementation.  It never
computes a loss, calls backward, steps an optimizer, or changes a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_SCHEMA = "sdpq_support_observability_audit_v1"
SEAL_SCHEMA = "sdpq_support_observability_sample_seal_v1"
DURATION_BUCKETS = (
    ("lt_1s", 0.0, 1.0),
    ("1_to_4s", 1.0, 4.0),
    ("4_to_16s", 4.0, 16.0),
    ("ge_16s", 16.0, float("inf")),
)


class SupportAuditError(ValueError):
    """Raised when the sealed audit contract is not satisfied."""


def require(condition, message):
    if not condition:
        raise SupportAuditError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload):
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def array_sha256(array):
    array = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def duration_bucket(duration):
    duration = float(duration)
    for name, lower, upper in DURATION_BUCKETS:
        if lower <= duration < upper:
            return name
    raise SupportAuditError(f"invalid GT duration: {duration}")


def _finite_array(value, name, *, dtype=np.float64):
    array = np.asarray(value, dtype=dtype)
    require(np.isfinite(array).all(), f"{name} contains non-finite values")
    return array


def _geometry_arrays(level_geometry, regression_ranges_sec, width_multiplier):
    require(
        len(level_geometry) == len(regression_ranges_sec) > 0,
        "geometry/range level count mismatch",
    )
    pieces = {
        "centers": [],
        "widths": [],
        "intervals": [],
        "domain_valid": [],
        "model_valid": [],
        "evidence": [],
        "assignable": [],
        "coverage": [],
        "levels": [],
        "lower": [],
        "upper": [],
    }
    level_counts = []
    for level, (geometry, regression_range) in enumerate(
        zip(level_geometry, regression_ranges_sec)
    ):
        centers = _finite_array(geometry["centers_sec"], "centers_sec").reshape(-1)
        widths = _finite_array(geometry["widths_sec"], "widths_sec").reshape(-1)
        intervals = _finite_array(
            geometry["intervals_sec"], "intervals_sec"
        ).reshape(-1, 2)
        require(
            widths.shape == centers.shape and intervals.shape == (centers.size, 2),
            "physical query geometry shape mismatch",
        )
        require(np.all(widths > 0.0), "physical query widths must be positive")
        require(
            np.all(intervals[:, 1] > intervals[:, 0]),
            "physical query intervals must be positive",
        )
        lower, upper = (float(value) for value in regression_range)
        require(
            math.isfinite(lower)
            and (math.isfinite(upper) or upper == float("inf"))
            and 0.0 <= lower <= upper,
            "invalid SDPQ regression range",
        )
        domain_valid = np.asarray(
            geometry.get("domain_valid_mask", geometry["valid_mask"]),
            dtype=np.bool_,
        ).reshape(-1)
        model_valid = np.asarray(geometry["valid_mask"], dtype=np.bool_).reshape(-1)
        evidence = np.asarray(
            geometry.get("evidence_mask", model_valid), dtype=np.bool_
        ).reshape(-1)
        assignable = np.asarray(
            geometry.get("assignment_mask", model_valid), dtype=np.bool_
        ).reshape(-1)
        coverage = _finite_array(
            geometry.get("coverage_sec", np.zeros_like(widths)),
            "coverage_sec",
        ).reshape(-1)
        for name, mask in (
            ("domain_valid_mask", domain_valid),
            ("valid_mask", model_valid),
            ("evidence_mask", evidence),
            ("assignment_mask", assignable),
        ):
            require(mask.shape == centers.shape, f"{name} shape mismatch")
        require(coverage.shape == centers.shape, "coverage shape mismatch")
        require(np.all(coverage >= 0.0), "coverage must be non-negative")
        pieces["centers"].append(centers)
        pieces["widths"].append(widths)
        pieces["intervals"].append(intervals)
        pieces["domain_valid"].append(domain_valid)
        pieces["model_valid"].append(model_valid)
        pieces["evidence"].append(evidence)
        pieces["assignable"].append(assignable)
        pieces["coverage"].append(coverage)
        pieces["levels"].append(np.full(centers.size, level, dtype=np.int64))
        pieces["lower"].append(np.full(centers.size, lower, dtype=np.float64))
        pieces["upper"].append(np.full(centers.size, upper, dtype=np.float64))
        level_counts.append(int(centers.size))
    concatenated = {
        name: np.concatenate(values, axis=0)
        for name, values in pieces.items()
    }
    concatenated["width_reference"] = (
        concatenated["widths"] * float(width_multiplier)
    )
    require(
        np.all(concatenated["width_reference"] > 0.0),
        "width references must be positive",
    )
    concatenated["level_counts"] = level_counts
    return concatenated


def recompute_sdpq_targets(
    level_geometry,
    gt_segments,
    gt_labels,
    *,
    num_classes,
    regression_ranges_sec,
    center_sample_radius,
    width_reference_multiplier,
    max_abs_delta_center,
    min_log_width,
    max_log_width,
):
    """Independently reproduce SDPQ target assignment using NumPy/float64."""
    geometry = _geometry_arrays(
        level_geometry,
        regression_ranges_sec,
        width_reference_multiplier,
    )
    segments = _finite_array(gt_segments, "GT segments").reshape(-1, 2)
    labels = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    require(labels.size == segments.shape[0], "GT label count mismatch")
    require(
        np.all((labels >= 0) & (labels < int(num_classes))),
        "GT label is outside the class range",
    )
    if segments.size:
        require(
            np.all(segments[:, 1] > segments[:, 0]),
            "GT segments must have positive duration",
        )
    require(
        math.isfinite(center_sample_radius) and center_sample_radius >= 0.0,
        "center sample radius must be finite and non-negative",
    )
    num_points = geometry["centers"].size
    num_gt = segments.shape[0]
    cls_target = np.zeros((num_points, int(num_classes)), dtype=np.float64)
    offset_target = np.zeros((num_points, 2), dtype=np.float64)
    segment_target = np.zeros((num_points, 2), dtype=np.float64)
    endpoint_target = np.zeros((num_points, 2), dtype=np.float64)
    assigned_gt = np.full(num_points, -1, dtype=np.int64)
    coverage_ratio = np.clip(
        geometry["coverage"] / np.maximum(geometry["widths"], 1.0e-8),
        0.0,
        1.0,
    )
    if num_gt == 0:
        return {
            "cls_target": cls_target,
            "offset_target": offset_target,
            "segment_target": segment_target,
            "endpoint_target": endpoint_target,
            "assigned_gt": assigned_gt,
            "geometry": geometry,
            "per_gt": [],
            "reservation_collision_count": 0,
            "reserved_match_count": 0,
        }

    gt_center = 0.5 * (segments[:, 0] + segments[:, 1])
    gt_width = np.maximum(segments[:, 1] - segments[:, 0], 1.0e-8)
    delta_center = (
        gt_center[None, :] - geometry["centers"][:, None]
    ) / np.maximum(geometry["widths"][:, None], 1.0e-8)
    delta_log_width = np.log(
        gt_width[None, :]
        / np.maximum(geometry["width_reference"][:, None], 1.0e-8)
    )
    normalized_cost = np.abs(delta_center) + np.abs(delta_log_width)
    level_range_ok = (
        gt_width[None, :] >= geometry["lower"][:, None]
    ) & (gt_width[None, :] <= geometry["upper"][:, None])
    range_fallback = ~level_range_ok.any(axis=0)
    level_range_ok[:, range_fallback] = True

    valid_candidates = geometry["assignable"][:, None] & level_range_ok
    local_candidates = valid_candidates & (
        np.abs(delta_center) <= float(center_sample_radius)
    )
    positive_candidates = local_candidates.copy()
    reserved_owner = np.full(num_points, -1, dtype=np.int64)
    candidate_counts = valid_candidates.sum(axis=0)
    gt_order = np.argsort(candidate_counts, kind="stable")
    reserved_by_gt = np.zeros(num_gt, dtype=np.bool_)
    reservation_collision_count = 0
    for gt_index in gt_order.tolist():
        candidates = np.flatnonzero(
            valid_candidates[:, gt_index] & (reserved_owner < 0)
        )
        if candidates.size == 0:
            if valid_candidates[:, gt_index].any():
                reservation_collision_count += 1
            continue
        costs = normalized_cost[candidates, gt_index]
        chosen = int(candidates[int(np.argmin(costs))])
        positive_candidates[chosen, gt_index] = True
        reserved_owner[chosen] = gt_index
        reserved_by_gt[gt_index] = True

    candidate_cost = np.where(
        positive_candidates,
        normalized_cost,
        np.inf,
    )
    min_gt = np.argmin(candidate_cost, axis=1)
    min_cost = candidate_cost[np.arange(num_points), min_gt]
    positive = np.isfinite(min_cost)
    reserved_positive = reserved_owner >= 0
    positive |= reserved_positive
    min_gt = np.where(reserved_positive, np.maximum(reserved_owner, 0), min_gt)
    assigned_gt[positive] = min_gt[positive]
    cls_target[np.flatnonzero(positive), labels[min_gt[positive]]] = 1.0
    offset_target[positive, 0] = np.clip(
        delta_center[positive, min_gt[positive]],
        -float(max_abs_delta_center),
        float(max_abs_delta_center),
    )
    offset_target[positive, 1] = np.clip(
        delta_log_width[positive, min_gt[positive]],
        float(min_log_width),
        float(max_log_width),
    )
    segment_target[positive] = segments[min_gt[positive]]

    max_right = float(geometry["intervals"][:, 1].max())
    for endpoint_index, endpoint_values in enumerate(
        (segments[:, 0], segments[:, 1])
    ):
        inside_cell = (
            endpoint_values[None, :] >= geometry["intervals"][:, 0, None]
        ) & (
            endpoint_values[None, :] < geometry["intervals"][:, 1, None]
        )
        at_final_edge = (
            endpoint_values[None, :] == geometry["intervals"][:, 1, None]
        ) & (geometry["intervals"][:, 1, None] == max_right)
        endpoint_target[:, endpoint_index] = (
            inside_cell | at_final_edge
        ).any(axis=1)

    per_gt = []
    for gt_index, duration in enumerate(gt_width.tolist()):
        range_mask = level_range_ok[:, gt_index]
        assigned_mask = assigned_gt == gt_index
        assigned_coverage = coverage_ratio[assigned_mask]
        per_gt.append(
            {
                "gt_index": int(gt_index),
                "duration_sec": float(duration),
                "duration_bucket": duration_bucket(duration),
                "regression_range_fallback": bool(range_fallback[gt_index]),
                "domain_candidate_count": int(
                    (geometry["domain_valid"] & range_mask).sum()
                ),
                "model_valid_candidate_count": int(
                    (geometry["model_valid"] & range_mask).sum()
                ),
                "evidence_candidate_count": int(
                    (geometry["evidence"] & range_mask).sum()
                ),
                "assignment_eligible_count": int(
                    valid_candidates[:, gt_index].sum()
                ),
                "local_assignment_eligible_count": int(
                    local_candidates[:, gt_index].sum()
                ),
                "assigned_query_count": int(assigned_mask.sum()),
                "reserved_query": bool(reserved_by_gt[gt_index]),
                "assigned_coverage_ratio_mean": (
                    float(assigned_coverage.mean())
                    if assigned_coverage.size
                    else None
                ),
                "assigned_uncovered_query_count": int(
                    (assigned_mask & (coverage_ratio <= 0.0)).sum()
                ),
            }
        )
    return {
        "cls_target": cls_target,
        "offset_target": offset_target,
        "segment_target": segment_target,
        "endpoint_target": endpoint_target,
        "assigned_gt": assigned_gt,
        "geometry": geometry,
        "per_gt": per_gt,
        "reservation_collision_count": int(reservation_collision_count),
        "reserved_match_count": int(reserved_by_gt.sum()),
    }


def summarize_recomputed_sample(recomputed):
    geometry = recomputed["geometry"]
    assigned_gt = recomputed["assigned_gt"]
    positive = assigned_gt >= 0
    coverage_ratio = np.clip(
        geometry["coverage"] / np.maximum(geometry["widths"], 1.0e-8),
        0.0,
        1.0,
    )
    valid_coverage = coverage_ratio[geometry["model_valid"]]
    return {
        "query_count": int(geometry["centers"].size),
        "domain_valid_query_count": int(geometry["domain_valid"].sum()),
        "model_valid_query_count": int(geometry["model_valid"].sum()),
        "evidence_query_count": int(geometry["evidence"].sum()),
        "assignment_eligible_query_count": int(geometry["assignable"].sum()),
        "zero_evidence_assignment_eligible_query_count": int(
            (geometry["assignable"] & ~geometry["evidence"]).sum()
        ),
        "positive_query_count": int(positive.sum()),
        "positive_uncovered_query_count": int(
            (positive & (coverage_ratio <= 0.0)).sum()
        ),
        "mean_model_valid_support_observability": (
            float(valid_coverage.mean()) if valid_coverage.size else 0.0
        ),
        "reservation_collision_count": recomputed[
            "reservation_collision_count"
        ],
        "reserved_match_count": recomputed["reserved_match_count"],
        "per_level_query_count": list(geometry["level_counts"]),
        "per_gt": recomputed["per_gt"],
    }


def aggregate_rows(rows):
    require(rows, "support audit produced no rows")
    gt_rows = [gt for row in rows for gt in row["per_gt"]]
    summary = {
        "window_count": len(rows),
        "gt_count": len(gt_rows),
        "query_count": sum(row["query_count"] for row in rows),
        "domain_valid_query_count": sum(
            row["domain_valid_query_count"] for row in rows
        ),
        "model_valid_query_count": sum(
            row["model_valid_query_count"] for row in rows
        ),
        "evidence_query_count": sum(row["evidence_query_count"] for row in rows),
        "assignment_eligible_query_count": sum(
            row["assignment_eligible_query_count"] for row in rows
        ),
        "zero_evidence_assignment_eligible_query_count": sum(
            row["zero_evidence_assignment_eligible_query_count"] for row in rows
        ),
        "positive_query_count": sum(
            row["positive_query_count"] for row in rows
        ),
        "positive_uncovered_query_count": sum(
            row["positive_uncovered_query_count"] for row in rows
        ),
        "reservation_collision_count": sum(
            row["reservation_collision_count"] for row in rows
        ),
        "reserved_match_count": sum(
            row["reserved_match_count"] for row in rows
        ),
        "gt_without_domain_candidate": sum(
            gt["domain_candidate_count"] == 0 for gt in gt_rows
        ),
        "gt_without_evidence_candidate": sum(
            gt["evidence_candidate_count"] == 0 for gt in gt_rows
        ),
        "gt_without_assignment_eligible_query": sum(
            gt["assignment_eligible_count"] == 0 for gt in gt_rows
        ),
        "gt_without_local_assignment_eligible_query": sum(
            gt["local_assignment_eligible_count"] == 0 for gt in gt_rows
        ),
        "gt_without_assigned_query": sum(
            gt["assigned_query_count"] == 0 for gt in gt_rows
        ),
        "regression_range_fallback_gt_count": sum(
            gt["regression_range_fallback"] for gt in gt_rows
        ),
        "duration_buckets": {},
    }
    for name, _, _ in DURATION_BUCKETS:
        bucket_rows = [gt for gt in gt_rows if gt["duration_bucket"] == name]
        summary["duration_buckets"][name] = {
            "gt_count": len(bucket_rows),
            "gt_without_evidence_candidate": sum(
                gt["evidence_candidate_count"] == 0 for gt in bucket_rows
            ),
            "gt_without_assignment_eligible_query": sum(
                gt["assignment_eligible_count"] == 0 for gt in bucket_rows
            ),
            "gt_without_assigned_query": sum(
                gt["assigned_query_count"] == 0 for gt in bucket_rows
            ),
            "assigned_uncovered_query_count": sum(
                gt["assigned_uncovered_query_count"] for gt in bucket_rows
            ),
        }
    return summary


def _canonical_meta_value(value):
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {
            str(key): _canonical_meta_value(item)
            for key, item in sorted(value.items())
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_meta_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float):
            require(math.isfinite(value), "metadata contains non-finite value")
        return value
    return repr(value)


def _tensor_sample_sha(value, sample_index):
    if value is None:
        return None
    try:
        import torch
    except ImportError as error:
        raise SupportAuditError("PyTorch is required for formal sample sealing") from error
    require(torch.is_tensor(value), "sealed input is not a tensor")
    sample = value[sample_index].detach().cpu().contiguous()
    return array_sha256(sample.numpy())


def _sample_fingerprint(batch, batch_index, sample_index, sequence_index):
    meta = _canonical_meta_value(batch["metas"][sample_index])
    gt_segments = batch["gt_segments"][sample_index].detach().cpu().numpy()
    gt_labels = batch["gt_labels"][sample_index].detach().cpu().numpy()
    support_fields = {}
    for key in (
        "phystime_native_token_timestamps_sec",
        "phystime_native_token_support_intervals_sec",
        "phystime_native_token_ownership_intervals_sec",
        "irregular_selected_positions",
        "irregular_selected_support_intervals_sec",
    ):
        if key in meta:
            support_fields[key] = meta[key]
    record = {
        "sequence_index": int(sequence_index),
        "batch_index": int(batch_index),
        "sample_index": int(sample_index),
        "video_name": meta.get("video_name", "unknown"),
        "window_start": meta.get(
            "window_start", meta.get("snippet_start", None)
        ),
        "duration": meta.get("duration", None),
        "meta_sha256": canonical_sha256(meta),
        "support_sha256": canonical_sha256(support_fields),
        "gt_segments_sha256": array_sha256(gt_segments),
        "gt_labels_sha256": array_sha256(gt_labels),
        "inputs_sha256": _tensor_sample_sha(batch.get("inputs"), sample_index),
        "masks_sha256": _tensor_sample_sha(batch.get("masks"), sample_index),
        "paired_inputs_sha256": _tensor_sample_sha(
            batch.get("paired_inputs"), sample_index
        ),
        "paired_masks_sha256": _tensor_sample_sha(
            batch.get("paired_masks"), sample_index
        ),
    }
    record["fingerprint_sha256"] = canonical_sha256(record)
    return record


def _set_seed(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_train_loader(cfg):
    from opentad.datasets import build_dataloader, build_dataset

    dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=None))
    solver = dict(cfg.solver.train)
    solver["num_workers"] = 0
    solver.pop("prefetch_factor", None)
    solver.pop("persistent_workers", None)
    return build_dataloader(
        dataset,
        rank=0,
        world_size=1,
        shuffle=False,
        drop_last=False,
        **solver,
    )


def _seal_windows(cfg, *, seed, window_count, seal_path):
    _set_seed(seed)
    loader = _build_train_loader(cfg)
    records = []
    batch_count = 0
    for batch_index, batch in enumerate(loader):
        batch_size = len(batch["metas"])
        require(
            len(records) + batch_size <= window_count,
            "requested window count cuts through a batch; refuse ambiguous slicing",
        )
        for sample_index in range(batch_size):
            records.append(
                _sample_fingerprint(
                    batch,
                    batch_index,
                    sample_index,
                    len(records),
                )
            )
        batch_count += 1
        if len(records) == window_count:
            break
    require(len(records) == window_count, "dataset did not provide exactly 64 windows")
    payload = {
        "schema_version": SEAL_SCHEMA,
        "status": "tested",
        "seed": int(seed),
        "split": "train",
        "shuffle": False,
        "num_workers": 0,
        "window_count": len(records),
        "batch_count": batch_count,
        "records": records,
        "window_sequence_sha256": canonical_sha256(records),
    }
    atomic_write_json(seal_path, payload)
    payload["artifact_sha256"] = sha256_file(seal_path)
    return payload


def _git_identity():
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "rev-parse", "HEAD^{tree}"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    )
    return commit, tree, status == ""


def _normalize_state_dict(state):
    normalized = {}
    for key, value in state.items():
        normalized[key[7:] if key.startswith("module.") else key] = value
    return normalized


def _torch_state_sha256(state):
    import torch

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
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _move_primary_batch(batch, device):
    return {
        "inputs": batch["inputs"].to(device, non_blocking=True),
        "masks": batch["masks"].to(device, non_blocking=True),
        "metas": [dict(meta) for meta in batch["metas"]],
        "gt_segments": [
            item.to(device, non_blocking=True) for item in batch["gt_segments"]
        ],
        "gt_labels": [
            item.to(device, non_blocking=True) for item in batch["gt_labels"]
        ],
    }


def _sample_level_geometry(level_geometry, sample_index):
    sampled = []
    for level in level_geometry:
        sampled.append(
            {
                key: value[sample_index].detach().cpu().numpy()
                for key, value in level.items()
                if key
                in {
                    "centers_sec",
                    "widths_sec",
                    "intervals_sec",
                    "domain_valid_mask",
                    "valid_mask",
                    "evidence_mask",
                    "assignment_mask",
                    "coverage_sec",
                }
            }
        )
    return sampled


def _target_error(independent, production):
    error = {}
    for name, observed in zip(
        ("cls_target", "offset_target", "segment_target", "endpoint_target"),
        production,
    ):
        expected = independent[name]
        observed = observed.detach().cpu().numpy().astype(np.float64)
        require(expected.shape == observed.shape, f"{name} shape mismatch")
        difference = np.abs(expected - observed)
        error[name] = float(difference.max()) if difference.size else 0.0
    return error


def _select_checkpoint_state(checkpoint, weights_source, expected_epoch):
    require(
        isinstance(expected_epoch, int) and expected_epoch >= 0,
        "expected checkpoint epoch must be a non-negative integer",
    )
    observed_epoch = checkpoint.get("epoch")
    require(
        isinstance(observed_epoch, (int, np.integer)),
        "checkpoint epoch is missing or non-integral",
    )
    observed_epoch = int(observed_epoch)
    require(
        observed_epoch == expected_epoch,
        (
            "checkpoint epoch mismatch: "
            f"expected {expected_epoch}, observed {observed_epoch}"
        ),
    )
    require(
        weights_source in {"online", "ema"},
        f"unsupported checkpoint weights source: {weights_source}",
    )
    state_key = "state_dict_ema" if weights_source == "ema" else "state_dict"
    require(
        isinstance(checkpoint.get(state_key), dict) and checkpoint[state_key],
        f"checkpoint is missing {state_key}",
    )
    return observed_epoch, state_key, checkpoint[state_key]


def _run_formal_audit(
    cfg,
    *,
    checkpoint_path,
    weights_source,
    expected_checkpoint_epoch,
    videomae_checkpoint,
    seed,
    window_count,
    seal,
):
    import torch
    from mmengine.config import Config
    from opentad.models import build_detector

    cfg = Config(cfg.to_dict())
    if cfg.model.get("backbone") and cfg.model.backbone.get("custom"):
        cfg.model.backbone.custom.pretrain = str(Path(videomae_checkpoint).resolve())
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_epoch, state_key, checkpoint_state = _select_checkpoint_state(
        checkpoint,
        weights_source,
        expected_checkpoint_epoch,
    )
    state = _normalize_state_dict(checkpoint_state)
    require(
        all(torch.isfinite(value).all() for value in state.values()),
        "checkpoint contains non-finite tensors",
    )
    device = torch.device("cuda:0")
    model = build_detector(cfg.model).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    require(
        not missing and not unexpected,
        f"checkpoint load mismatch: missing={missing[:5]} unexpected={unexpected[:5]}",
    )
    require(
        model.rpn_head.__class__.__name__ == "SupportDecoupledPhysicalQueryHead",
        "formal support audit requires the SDPQ head",
    )
    model.eval()
    state_before = _torch_state_sha256(model.state_dict())
    head = model.rpn_head
    parameters = {
        "num_classes": int(head.num_classes),
        "regression_ranges_sec": [
            [float(value) for value in item]
            for item in head.regression_ranges_sec
        ],
        "center_sample_radius": float(head.center_sample_radius),
        "width_reference_multiplier": float(head.width_reference_multiplier),
        "max_abs_delta_center": float(head.max_abs_delta_center),
        "min_log_width": float(head.min_log_width),
        "max_log_width": float(head.max_log_width),
    }
    _set_seed(seed)
    loader = _build_train_loader(cfg)
    rows = []
    max_target_error = {
        "cls_target": 0.0,
        "offset_target": 0.0,
        "segment_target": 0.0,
        "endpoint_target": 0.0,
    }
    seal_cursor = 0
    amp_enabled = bool(cfg.solver.get("amp", False))
    for batch_index, cpu_batch in enumerate(loader):
        if seal_cursor >= window_count:
            break
        batch_size = len(cpu_batch["metas"])
        require(
            seal_cursor + batch_size <= window_count,
            "replay batch crosses the sealed window boundary",
        )
        for sample_index in range(batch_size):
            observed_fingerprint = _sample_fingerprint(
                cpu_batch,
                batch_index,
                sample_index,
                seal_cursor + sample_index,
            )
            require(
                observed_fingerprint
                == seal["records"][seal_cursor + sample_index],
                "replayed training window differs from the sealed manifest",
            )
        batch = _move_primary_batch(cpu_batch, device)
        with torch.no_grad(), torch.cuda.amp.autocast(
            enabled=amp_enabled,
            dtype=torch.float16,
        ):
            model._validate_metas(batch["metas"], training=True)
            observations = model._extract_observations(
                batch["inputs"], batch["masks"]
            )
            observations, aligned_masks, aligned_metas = (
                model._align_native_temporal_geometry(
                    observations,
                    batch["masks"],
                    batch["metas"],
                )
            )
            _, mask_list, level_geometry = model.projection(
                observations,
                aligned_masks,
                aligned_metas,
            )
            points = head.build_query_points(level_geometry)
            production_targets = head._prepare_targets(
                points,
                level_geometry,
                batch["gt_segments"],
                batch["gt_labels"],
            )
        require(
            all(
                torch.equal(mask, geometry["valid_mask"])
                for mask, geometry in zip(mask_list, level_geometry)
            ),
            "projection mask/geometry contract mismatch",
        )
        for sample_index in range(batch_size):
            sample_geometry = _sample_level_geometry(
                level_geometry,
                sample_index,
            )
            independent = recompute_sdpq_targets(
                sample_geometry,
                batch["gt_segments"][sample_index].detach().cpu().numpy(),
                batch["gt_labels"][sample_index].detach().cpu().numpy(),
                **parameters,
            )
            errors = _target_error(
                independent,
                tuple(target[sample_index] for target in production_targets),
            )
            for key, value in errors.items():
                max_target_error[key] = max(max_target_error[key], value)
            row = summarize_recomputed_sample(independent)
            sealed_record = seal["records"][seal_cursor + sample_index]
            row.update(
                {
                    "sequence_index": sealed_record["sequence_index"],
                    "video_name": sealed_record["video_name"],
                    "window_start": sealed_record["window_start"],
                    "fingerprint_sha256": sealed_record[
                        "fingerprint_sha256"
                    ],
                    "production_target_max_abs_error": errors,
                }
            )
            rows.append(row)
        seal_cursor += batch_size
    require(len(rows) == window_count, "formal replay did not audit 64 windows")
    require(
        max(max_target_error.values()) <= 1.0e-5,
        f"independent assignment differs from production: {max_target_error}",
    )
    state_after = _torch_state_sha256(model.state_dict())
    require(state_before == state_after, "model state changed during no-loss audit")
    return {
        "rows": rows,
        "summary": aggregate_rows(rows),
        "max_production_target_abs_error": max_target_error,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_state_key": state_key,
        "checkpoint_state_dict_sha256": _torch_state_sha256(state),
        "model_state_sha256_before": state_before,
        "model_state_sha256_after": state_after,
        "model_state_immutable": True,
        "parameters": parameters,
        "amp_enabled_for_backbone": amp_enabled,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit exactly 64 sealed SDPQ training windows without loss."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-epoch", type=int, required=True)
    parser.add_argument("--weights-source", choices=("online", "ema"), required=True)
    parser.add_argument("--videomae-checkpoint", required=True)
    parser.add_argument("--expected-runtime-commit", required=True)
    parser.add_argument("--expected-runtime-tree", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-windows", type=int, default=64)
    return parser.parse_args()


def main():
    args = parse_args()
    require(args.seed == 42, "formal support audit seed must be 42")
    require(args.num_windows == 64, "formal support audit requires exactly 64 windows")
    require(
        args.expected_checkpoint_epoch >= 0,
        "expected checkpoint epoch must be non-negative",
    )
    require(
        os.environ.get("SLURM_JOB_ID") and os.environ.get("CUDA_VISIBLE_DEVICES"),
        "formal support audit requires a Slurm-assigned CUDA device",
    )
    import torch
    from mmengine.config import Config

    require(torch.cuda.is_available(), "CUDA is unavailable")
    runtime_commit, runtime_tree, clean = _git_identity()
    require(
        runtime_commit == args.expected_runtime_commit
        and runtime_tree == args.expected_runtime_tree
        and clean,
        "runtime commit/tree/cleanliness mismatch",
    )
    config_path = Path(args.config).resolve()
    checkpoint_path = Path(args.checkpoint).resolve()
    videomae_path = Path(args.videomae_checkpoint).resolve()
    for path, description in (
        (config_path, "config"),
        (checkpoint_path, "checkpoint"),
        (videomae_path, "VideoMAE checkpoint"),
    ):
        require(path.is_file(), f"missing {description}: {path}")
    output_dir = Path(args.output_dir).resolve()
    require(not output_dir.exists(), "output directory already exists")
    output_dir.mkdir(parents=True)
    cfg = Config.fromfile(config_path, lazy_import=False)
    seal_path = output_dir / "SAMPLE_SEAL.json"
    seal = _seal_windows(
        cfg,
        seed=args.seed,
        window_count=args.num_windows,
        seal_path=seal_path,
    )
    audit = _run_formal_audit(
        cfg,
        checkpoint_path=checkpoint_path,
        weights_source=args.weights_source,
        expected_checkpoint_epoch=args.expected_checkpoint_epoch,
        videomae_checkpoint=videomae_path,
        seed=args.seed,
        window_count=args.num_windows,
        seal=seal,
    )
    rows_path = output_dir / "support_observability_rows.json"
    atomic_write_json(rows_path, audit["rows"])
    report = {
        "schema_version": OUTPUT_SCHEMA,
        "status": "tested",
        "validation_pass": True,
        "claim_boundary": "diagnostic_only_no_training_support_observability",
        "new_training": False,
        "loss_computed": False,
        "backward_called": False,
        "optimizer_created": False,
        "optimizer_step_called": False,
        "split": "train",
        "seed": args.seed,
        "window_count": args.num_windows,
        "runtime": {
            "commit": runtime_commit,
            "tree": runtime_tree,
            "clean": clean,
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "cuda_device": torch.cuda.get_device_name(0),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "epoch": audit["checkpoint_epoch"],
            "weights_source": args.weights_source,
            "state_dict_sha256": audit["checkpoint_state_dict_sha256"],
        },
        "videomae_checkpoint": {
            "path": str(videomae_path),
            "sha256": sha256_file(videomae_path),
        },
        "sample_seal": {
            "path": str(seal_path),
            "sha256": sha256_file(seal_path),
            "window_sequence_sha256": seal["window_sequence_sha256"],
        },
        "rows": {
            "path": str(rows_path),
            "sha256": sha256_file(rows_path),
        },
        "parameters": audit["parameters"],
        "summary": audit["summary"],
        "independent_production_cross_check": {
            "max_abs_error": audit["max_production_target_abs_error"],
            "atol": 1.0e-5,
            "pass": True,
        },
        "model_state_immutable": audit["model_state_immutable"],
        "model_state_sha256_before": audit["model_state_sha256_before"],
        "model_state_sha256_after": audit["model_state_sha256_after"],
        "amp_enabled_for_backbone": audit["amp_enabled_for_backbone"],
    }
    report_path = output_dir / "SUPPORT_AUDIT_COMPLETE.json"
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
