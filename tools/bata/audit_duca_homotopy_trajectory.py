"""Audit frozen DUCA scorer logits across a uniform-to-learned alpha grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from opentad.models.duca.structured_selection import (
    exact_uniform_positions,
    global_structured_topk,
)
from opentad.models.duca.transition_only import continuous_policy_logits


SCHEMA_VERSION = "duca_homotopy_trajectory_audit_v1"
_GT_RECALL_RADII = (0.0, 1.0, 2.0, 4.0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    cpu = tensor.detach().cpu().contiguous().clone()
    return bytes(cpu.untyped_storage())


def _sha256_tensor(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _sha256_state_dict(state_dict: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise TypeError(f"state_dict value is not a tensor: {key}")
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(list(value.shape)).encode("ascii"))
        digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def parse_alpha_grid(value: str | Sequence[float]) -> list[float]:
    raw = value.split(",") if isinstance(value, str) else list(value)
    alphas = [float(item) for item in raw]
    _require(len(alphas) >= 2, "alpha grid must contain at least two points")
    _require(all(math.isfinite(alpha) and 0.0 <= alpha <= 1.0 for alpha in alphas), "alpha values must be finite and lie in [0,1]")
    _require(all(left < right for left, right in zip(alphas, alphas[1:])), "alpha grid must be strictly increasing")
    return alphas


def _valid_prefix_length(mask: torch.Tensor, *, sample_id: str) -> int:
    _require(mask.ndim == 1, f"{sample_id}: valid mask row must be one-dimensional")
    valid = mask.bool()
    valid_len = int(valid.long().sum().item())
    _require(valid_len > 0, f"{sample_id}: empty validation rows are forbidden")
    observed = torch.nonzero(valid, as_tuple=False).flatten()
    expected = torch.arange(valid_len, device=observed.device, dtype=observed.dtype)
    _require(torch.equal(observed, expected), f"{sample_id}: valid mask must be a contiguous prefix")
    return valid_len


def selection_geometry(positions: Sequence[int], valid_len: int) -> dict[str, Any]:
    valid_len = int(valid_len)
    selected = [int(position) for position in positions]
    _require(valid_len > 0, "valid_len must be positive")
    _require(selected == sorted(set(selected)), "selected positions must be sorted and unique")
    _require(all(0 <= position < valid_len for position in selected), "selected positions fall outside the valid prefix")
    intervals = [right - left for left, right in zip(selected, selected[1:])]
    if selected:
        holes = [selected[0]]
        holes.extend(max(0, interval - 1) for interval in intervals)
        holes.append(valid_len - 1 - selected[-1])
    else:
        holes = [valid_len]
    longest_run = 0
    current_run = 0
    previous = None
    for position in selected:
        current_run = current_run + 1 if previous is not None and position == previous + 1 else 1
        longest_run = max(longest_run, current_run)
        previous = position
    interval_histogram = Counter(intervals)
    hole_histogram = Counter(holes)
    adjacent_denominator = max(0, len(selected) - 1)
    adjacent_count = sum(interval == 1 for interval in intervals)
    return {
        "selected_count": len(selected),
        "selection_rate": float(len(selected) / valid_len),
        "adjacent_selected_pair_count": int(adjacent_count),
        "adjacent_selected_pair_denominator": int(adjacent_denominator),
        "adjacent_selection_rate": float(adjacent_count / adjacent_denominator) if adjacent_denominator else 0.0,
        "selected_gap_histogram": {str(key): int(interval_histogram[key]) for key in sorted(interval_histogram)},
        "unselected_hole_histogram": {str(key): int(hole_histogram[key]) for key in sorted(hole_histogram)},
        "longest_contiguous_selected_run": int(longest_run),
        "max_hole": int(max(holes)),
    }


def _gt_boundaries(gt_segments: Any, valid_len: int) -> tuple[list[float], int]:
    if gt_segments is None:
        return [], 0
    if torch.is_tensor(gt_segments):
        value = gt_segments.detach().cpu().tolist()
    else:
        value = list(gt_segments)
    boundaries: list[float] = []
    clamped = 0
    for index, segment in enumerate(value):
        _require(isinstance(segment, Sequence) and len(segment) == 2, f"GT segment {index} must contain [start,end]")
        start, end = float(segment[0]), float(segment[1])
        _require(math.isfinite(start) and math.isfinite(end), f"GT segment {index} must be finite")
        _require(end >= start, f"GT segment {index} has end before start")
        for boundary in (start, end):
            clipped = min(max(boundary, 0.0), float(valid_len - 1))
            clamped += int(clipped != boundary)
            boundaries.append(clipped)
    return boundaries, clamped


def _evaluate_gt(positions: Sequence[int], gt_segments: Any, valid_len: int) -> dict[str, Any]:
    boundaries, clamped = _gt_boundaries(gt_segments, valid_len)
    if not boundaries:
        return {
            "boundary_count": 0,
            "clamped_boundary_count": int(clamped),
            "mean_nearest_selected_distance": None,
            "max_nearest_selected_distance": None,
            "boundary_recall": {f"r{int(radius)}": None for radius in _GT_RECALL_RADII},
        }
    selected = [float(position) for position in positions]
    _require(bool(selected), "GT evaluation requires at least one selected position")
    distances = [min(abs(position - boundary) for position in selected) for boundary in boundaries]
    return {
        "boundary_count": len(boundaries),
        "clamped_boundary_count": int(clamped),
        "mean_nearest_selected_distance": _mean(distances),
        "max_nearest_selected_distance": float(max(distances)),
        "boundary_recall": {
            f"r{int(radius)}": float(sum(distance <= radius for distance in distances) / len(distances))
            for radius in _GT_RECALL_RADII
        },
    }


def _pairwise_record(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    left_soft: torch.Tensor,
    right_soft: torch.Tensor,
) -> dict[str, Any]:
    left_positions = [int(value) for value in left["hard_positions"]]
    right_positions = [int(value) for value in right["hard_positions"]]
    left_set, right_set = set(left_positions), set(right_positions)
    intersection = len(left_set & right_set)
    union = len(left_set | right_set)
    hard_swaps = len(left_set) - intersection
    rank_displacement = [abs(left_pos - right_pos) for left_pos, right_pos in zip(left_positions, right_positions)]
    soft_l1 = float((left_soft - right_soft).abs().sum().item())
    return {
        "alpha_from": float(left["alpha"]),
        "alpha_to": float(right["alpha"]),
        "hard_path_changed": bool(hard_swaps > 0),
        "hard_jaccard": float(intersection / union) if union else 1.0,
        "hard_overlap_count": int(intersection),
        "hard_swap_count": int(hard_swaps),
        "hard_symmetric_difference_count": int(2 * hard_swaps),
        "max_rank_aligned_position_jump": int(max(rank_displacement)) if rank_displacement else 0,
        "mean_rank_aligned_position_jump": _mean(rank_displacement),
        "soft_occupancy_l1": soft_l1,
        "soft_occupancy_l1_per_effective_k": float(soft_l1 / max(1, len(left_positions))),
        "soft_changed_without_hard_swap": bool(hard_swaps == 0 and soft_l1 > 1.0e-7),
    }


def _trajectory_summary(paths: Sequence[tuple[int, ...]], alphas: Sequence[float], pairwise: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(len(paths) == len(alphas), "trajectory paths and alpha grid must align")
    runs: list[dict[str, Any]] = []
    run_start = 0
    for index in range(1, len(paths) + 1):
        if index == len(paths) or paths[index] != paths[run_start]:
            run_end = index - 1
            runs.append(
                {
                    "alpha_start": float(alphas[run_start]),
                    "alpha_end": float(alphas[run_end]),
                    "alpha_width": float(alphas[run_end] - alphas[run_start]),
                    "grid_point_count": int(run_end - run_start + 1),
                    "hard_positions": list(paths[run_start]),
                }
            )
            run_start = index
    longest = max(runs, key=lambda item: (item["alpha_width"], item["grid_point_count"], -item["alpha_start"]))
    changes = [record for record in pairwise if bool(record["hard_path_changed"])]
    max_jump = max(pairwise, key=lambda item: (item["hard_swap_count"], item["max_rank_aligned_position_jump"]))
    return {
        "unique_hard_path_count": len(set(paths)),
        "hard_path_change_count": len(changes),
        "first_change_alpha": None if not changes else float(changes[0]["alpha_to"]),
        "last_change_alpha": None if not changes else float(changes[-1]["alpha_to"]),
        "first_change_interval": None if not changes else {"alpha_from": changes[0]["alpha_from"], "alpha_to": changes[0]["alpha_to"]},
        "last_change_interval": None if not changes else {"alpha_from": changes[-1]["alpha_from"], "alpha_to": changes[-1]["alpha_to"]},
        "max_single_step_hard_swaps": int(max_jump["hard_swap_count"]),
        "max_single_step_rank_aligned_jump": int(max_jump["max_rank_aligned_position_jump"]),
        "max_single_step_interval": {"alpha_from": max_jump["alpha_from"], "alpha_to": max_jump["alpha_to"]},
        "longest_hard_path_unchanged_interval": longest,
        "all_hard_path_plateaus": runs,
    }


def audit_tensor_trajectory(
    learned_scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    budget: int,
    max_unselected_hole: int | None,
    alpha_grid: str | Sequence[float],
    temperature: float,
    sample_ids: Sequence[str] | None = None,
    gt_segments: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Audit one frozen score tensor without constructing or running a detector."""

    _require(torch.is_tensor(learned_scores) and learned_scores.ndim == 2, "learned_scores must be a [B,T] tensor")
    _require(learned_scores.is_floating_point(), "learned_scores must be floating point")
    _require(bool(torch.isfinite(learned_scores).all().item()), "learned_scores must be finite")
    _require(torch.is_tensor(valid_mask) and valid_mask.shape == learned_scores.shape, "valid_mask must align with learned_scores")
    batch_size = int(learned_scores.shape[0])
    _require(batch_size > 0 and int(learned_scores.shape[1]) > 0, "trajectory tensor must be non-empty")
    budget = int(budget)
    _require(budget > 0, "budget must be positive")
    temperature = float(temperature)
    _require(math.isfinite(temperature) and temperature > 0.0, "temperature must be finite and positive")
    if max_unselected_hole is not None:
        max_unselected_hole = int(max_unselected_hole)
        _require(max_unselected_hole >= 0, "max_unselected_hole must be non-negative")
    alphas = parse_alpha_grid(alpha_grid)
    if sample_ids is None:
        ordered_ids = [f"sample_{index:06d}" for index in range(batch_size)]
    else:
        ordered_ids = [str(value) for value in sample_ids]
    _require(len(ordered_ids) == batch_size, "sample_ids must align with batch size")
    _require(all(ordered_ids), "sample_ids must be non-empty")
    _require(len(set(ordered_ids)) == len(ordered_ids), "sample_ids must be unique and order-stable")
    if gt_segments is not None:
        _require(len(gt_segments) == batch_size, "gt_segments must align with batch size")

    scores = learned_scores.detach().float()
    valid = valid_mask.detach().bool().to(device=scores.device)
    sample_reports: list[dict[str, Any]] = []
    valid_len_le_budget = 0
    pairwise_flat: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch_index, sample_id in enumerate(ordered_ids):
            valid_len = _valid_prefix_length(valid[batch_index], sample_id=sample_id)
            effective_k = min(budget, valid_len)
            valid_len_le_budget += int(valid_len <= budget)
            row_scores = scores[batch_index : batch_index + 1, :valid_len]
            row_valid = torch.ones_like(row_scores, dtype=torch.bool)
            row_max_hole = valid_len if max_unselected_hole is None else max_unselected_hole
            _require(
                valid_len - effective_k <= (effective_k + 1) * row_max_hole,
                f"{sample_id}: infeasible exact-K/max-hole contract T={valid_len}, K={effective_k}, G={row_max_hole}",
            )
            uniform_positions = exact_uniform_positions(valid_len, effective_k, device=scores.device)
            uniform_set = set(int(value) for value in uniform_positions.tolist())
            alpha_records: list[dict[str, Any]] = []
            soft_rows: list[torch.Tensor] = []
            paths: list[tuple[int, ...]] = []

            for alpha in alphas:
                policy_logits = continuous_policy_logits(
                    row_scores,
                    row_valid,
                    k=effective_k,
                    alpha=alpha,
                )
                decoded = global_structured_topk(
                    policy_logits,
                    k=effective_k,
                    max_unselected_hole=row_max_hole,
                    temperature=temperature,
                    training=True,
                )
                positions = tuple(int(value) for value in decoded.selected_positions[0].tolist())
                soft = decoded.soft_occupancy[0].detach().float()
                _require(bool(torch.isfinite(soft).all().item()), f"{sample_id}: non-finite soft occupancy at alpha={alpha}")
                _require(abs(float(soft.sum().item()) - effective_k) <= 2.0e-4, f"{sample_id}: soft occupancy violates exact-K mass at alpha={alpha}")
                geometry = selection_geometry(positions, valid_len)
                _require(geometry["selected_count"] == effective_k, f"{sample_id}: hard DP violated effective_k at alpha={alpha}")
                _require(geometry["max_hole"] <= row_max_hole, f"{sample_id}: hard DP violated max-hole at alpha={alpha}")
                selected_set = set(positions)
                intersection = len(selected_set & uniform_set)
                union = len(selected_set | uniform_set)
                record: dict[str, Any] = {
                    "alpha": float(alpha),
                    "policy_logits_sha256": _sha256_tensor(policy_logits[0]),
                    "hard_positions": list(positions),
                    "soft_occupancy": [float(value) for value in soft.cpu().tolist()],
                    "soft_occupancy_sum": float(soft.sum().item()),
                    "log_partition": float(decoded.log_partition[0].detach().float().item()),
                    "exact_uniform_overlap_count": int(intersection),
                    "exact_uniform_overlap_rate": float(intersection / effective_k),
                    "exact_uniform_jaccard": float(intersection / union) if union else 1.0,
                    "geometry": geometry,
                }
                if gt_segments is not None:
                    record["gt_evaluation"] = _evaluate_gt(positions, gt_segments[batch_index], valid_len)
                alpha_records.append(record)
                soft_rows.append(soft.cpu())
                paths.append(positions)

            pairwise = [
                _pairwise_record(alpha_records[index], alpha_records[index + 1], left_soft=soft_rows[index], right_soft=soft_rows[index + 1])
                for index in range(len(alphas) - 1)
            ]
            for record in pairwise:
                pairwise_flat.append({"sample_id": sample_id, **record})
            sample_reports.append(
                {
                    "sample_id": sample_id,
                    "valid_len": valid_len,
                    "requested_budget": budget,
                    "effective_k": effective_k,
                    "valid_len_le_budget": bool(valid_len <= budget),
                    "max_unselected_hole": int(row_max_hole),
                    "learned_scorer_logits": [float(value) for value in row_scores[0].cpu().tolist()],
                    "learned_scorer_logits_sha256": _sha256_tensor(row_scores[0]),
                    "exact_uniform_positions": [int(value) for value in uniform_positions.tolist()],
                    "alpha_trajectory": alpha_records,
                    "adjacent_alpha": pairwise,
                    "trajectory_summary": _trajectory_summary(paths, alphas, pairwise),
                    "gt_role": "evaluation_only_not_selection_input" if gt_segments is not None else "absent",
                }
            )

    aggregate_by_interval = []
    for pair_index in range(len(alphas) - 1):
        rows = [sample["adjacent_alpha"][pair_index] for sample in sample_reports]
        aggregate_by_interval.append(
            {
                "alpha_from": float(alphas[pair_index]),
                "alpha_to": float(alphas[pair_index + 1]),
                "mean_hard_jaccard": _mean([row["hard_jaccard"] for row in rows]),
                "mean_hard_swap_count": _mean([row["hard_swap_count"] for row in rows]),
                "max_hard_swap_count": int(max(row["hard_swap_count"] for row in rows)),
                "mean_soft_occupancy_l1": _mean([row["soft_occupancy_l1"] for row in rows]),
                "soft_changed_without_hard_swap_count": int(sum(row["soft_changed_without_hard_swap"] for row in rows)),
            }
        )
    max_jump = max(pairwise_flat, key=lambda item: (item["hard_swap_count"], item["max_rank_aligned_position_jump"]))
    effective_values = [int(sample["effective_k"]) for sample in sample_reports]
    return {
        "schema_version": SCHEMA_VERSION,
        "alpha_grid": alphas,
        "sample_order": ordered_ids,
        "sample_order_sha256": _sha256_json(ordered_ids),
        "selection_contract": {
            "requested_budget": budget,
            "configured_max_unselected_hole": max_unselected_hole,
            "structured_temperature": temperature,
            "decoder": "global_structured_topk",
            "policy": "continuous_policy_logits",
            "gt_used_for_selection": False,
        },
        "aggregate": {
            "sample_count": batch_size,
            "valid_len_le_budget_count": int(valid_len_le_budget),
            "valid_len_le_budget_ratio": float(valid_len_le_budget / batch_size),
            "effective_k": {
                "min": min(effective_values),
                "max": max(effective_values),
                "mean": _mean(effective_values),
                "per_sample": effective_values,
            },
            "max_single_step_hard_swaps": int(max_jump["hard_swap_count"]),
            "max_single_step_rank_aligned_jump": int(max_jump["max_rank_aligned_position_jump"]),
            "max_single_step_sample_id": str(max_jump["sample_id"]),
            "max_single_step_interval": {"alpha_from": max_jump["alpha_from"], "alpha_to": max_jump["alpha_to"]},
            "soft_changed_without_hard_swap_count": int(sum(item["soft_changed_without_hard_swap"] for item in pairwise_flat)),
            "adjacent_alpha": aggregate_by_interval,
        },
        "samples": sample_reports,
    }


def selector_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized = {
        (str(key)[len("module.") :] if str(key).startswith("module.") else str(key)): value
        for key, value in state_dict.items()
    }
    selected = {
        key[len("frame_selector.") :]: value
        for key, value in normalized.items()
        if key.startswith("frame_selector.")
    }
    if not selected:
        raise ValueError("checkpoint has no frame_selector.* parameters")
    return selected


def _checkpoint_state(checkpoint: Mapping[str, Any], requested_key: str) -> tuple[str, Mapping[str, Any]]:
    key = requested_key
    if key == "auto":
        key = "state_dict_ema" if "state_dict_ema" in checkpoint else "state_dict"
    state = checkpoint.get(key)
    if not isinstance(state, Mapping):
        raise ValueError(f"checkpoint is missing mapping {key}")
    return key, state


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _git_root(path: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=path,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError("config must live in a readable Git worktree")
    return Path(result.stdout.strip()).resolve()


def _git_commit(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or len(commit) != 40:
        raise RuntimeError("unable to resolve an exact Git commit")
    return commit


def _load_fixed_batch(path: Path) -> dict[str, Any]:
    payload = _torch_load(path)
    _require(isinstance(payload, Mapping), "fixed batch must be a mapping")
    batch = payload.get("batch", payload)
    _require(isinstance(batch, Mapping), "fixed batch payload must be a mapping")
    inputs = batch.get("inputs")
    masks = batch.get("masks")
    sample_ids = batch.get("sample_ids", batch.get("sample_order"))
    _require(torch.is_tensor(inputs) and inputs.ndim >= 3, "fixed batch requires tensor inputs with a batch and temporal axis")
    _require(
        inputs.is_floating_point() or inputs.dtype == torch.uint8,
        "fixed inputs must be floating point or real-loader uint8 RGB",
    )
    if inputs.is_floating_point():
        _require(bool(torch.isfinite(inputs).all().item()), "fixed floating inputs must be finite")
    _require(torch.is_tensor(masks) and masks.ndim == 2, "fixed batch requires [B,T] masks")
    _require(int(inputs.shape[0]) == int(masks.shape[0]), "fixed inputs and masks batch dimensions must match")
    _require(isinstance(sample_ids, Sequence) and not isinstance(sample_ids, (str, bytes)), "fixed batch requires explicit sample_ids/sample_order")
    ordered_ids = [str(value) for value in sample_ids]
    _require(len(ordered_ids) == int(inputs.shape[0]), "fixed sample order must align with batch size")
    _require(all(ordered_ids) and len(set(ordered_ids)) == len(ordered_ids), "fixed sample ids must be non-empty and unique")
    gt_segments = batch.get("gt_segments")
    if gt_segments is not None:
        _require(
            torch.is_tensor(gt_segments) or isinstance(gt_segments, Sequence),
            "gt_segments must be a per-sample sequence or tensor",
        )
        _require(len(gt_segments) == int(inputs.shape[0]), "gt_segments must align with batch size")
    return {
        "inputs": inputs.detach().clone(),
        "masks": masks.detach().bool().clone(),
        "sample_ids": ordered_ids,
        "gt_segments": gt_segments,
        "batch_metadata": dict(payload.get("metadata", {})) if isinstance(payload.get("metadata", {}), Mapping) else {},
    }


def _validate_json_tree(value: Any, path: str = "$") -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite JSON value at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_tree(item, f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"non-string JSON key at {path}")
            _validate_json_tree(item, f"{path}.{key}")
        return
    raise TypeError(f"unsupported JSON value at {path}: {type(value).__name__}")


def _write_json_fail_closed(path: Path, report: Mapping[str, Any]) -> None:
    _validate_json_tree(report)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing audit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale audit temporary file exists: {temporary}")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _records_source(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    required = (
        "config",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_state_key",
        "git_commit",
    )
    canonical: dict[str, Any] | None = None
    for index, row in enumerate(rows):
        source = row.get("source")
        _require(isinstance(source, Mapping), f"record {index} is missing source provenance")
        current = {key: source.get(key) for key in required}
        _require(all(current[key] not in (None, "") for key in required), f"record {index} has incomplete source provenance")
        _require(bool(source.get("selector_only_inference")), f"record {index} was not produced by selector-only inference")
        _require(not bool(source.get("detector_backbone_executed")), f"record {index} executed the detector backbone")
        _require(not bool(source.get("uses_gt_for_selection")), f"record {index} used GT for selection")
        if canonical is None:
            canonical = current
        else:
            _require(current == canonical, "records mix checkpoints, configs, commits or state keys")
    _require(canonical is not None, "records source provenance is empty")
    return canonical


def run_records_audit(
    *,
    records_jsonl: str | Path,
    output_json: str | Path,
    alpha_grid: str | Sequence[float],
    budget: int,
    max_unselected_hole: int,
    temperature: float = 1.0,
    evaluate_gt: bool = False,
    device: str = "cpu",
) -> dict[str, Any]:
    """Audit hash-bound exported scorer logits without retaining raw RGB frames."""

    records_path = Path(records_jsonl).expanduser().resolve(strict=True)
    output_path = Path(output_json).expanduser().resolve()
    rows: list[Mapping[str, Any]] = []
    with records_path.open("r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, Mapping), f"record line {line_index} must be a mapping")
            _require(
                str(value.get("schema_version")) == "duca_selection_quality_record_v2",
                f"record line {line_index} has an unsupported schema",
            )
            rows.append(value)
    _require(bool(rows), "records JSONL is empty")
    source = _records_source(rows)

    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    _require(all(sample_ids), "record sample ids must be non-empty")
    _require(len(set(sample_ids)) == len(sample_ids), "record sample ids must be unique")
    valid_lengths = [int(row.get("valid_len", 0)) for row in rows]
    _require(all(length > 0 for length in valid_lengths), "record valid lengths must be positive")
    temporal_len = max(valid_lengths)
    scores = torch.zeros((len(rows), temporal_len), dtype=torch.float32)
    masks = torch.zeros((len(rows), temporal_len), dtype=torch.bool)
    gt_segments: list[Any] = []
    for index, (row, valid_len) in enumerate(zip(rows, valid_lengths)):
        values = row.get("transition_policy_scores")
        _require(
            isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and len(values) >= valid_len,
            f"{sample_ids[index]}: transition_policy_scores are missing or short",
        )
        score_row = torch.tensor([float(value) for value in values[:valid_len]], dtype=torch.float32)
        _require(bool(torch.isfinite(score_row).all().item()), f"{sample_ids[index]}: scorer logits are non-finite")
        scores[index, :valid_len] = score_row
        masks[index, :valid_len] = True
        gt_segments.append(row.get("gt_segments", []))

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
        torch.cuda.set_device(torch_device)
    trajectory = audit_tensor_trajectory(
        scores.to(torch_device),
        masks.to(torch_device),
        budget=int(budget),
        max_unselected_hole=int(max_unselected_hole),
        alpha_grid=alpha_grid,
        temperature=float(temperature),
        sample_ids=sample_ids,
        gt_segments=gt_segments if evaluate_gt else None,
    )
    script_path = Path(__file__).resolve()
    report = {
        **trajectory,
        "status": "complete",
        "provenance": {
            "records_jsonl": str(records_path),
            "records_sha256": _sha256_file(records_path),
            "record_count": len(rows),
            "record_source": source,
            "audit_script_path": str(script_path),
            "audit_script_sha256": _sha256_file(script_path),
            "alpha_grid_sha256": _sha256_json(parse_alpha_grid(alpha_grid)),
            "exported_score_precision": "decimal_values_as_serialized_in_records_jsonl",
        },
        "read_only_contract": {
            "selector_constructed": False,
            "adatad_detector_constructed": False,
            "adatad_heavy_backbone_executed": False,
            "optimizer_constructed": False,
            "gradients_enabled": False,
            "gt_evaluation_enabled": bool(evaluate_gt),
            "gt_used_for_selection": False,
        },
    }
    _write_json_fail_closed(output_path, report)
    return report


def run_frozen_audit(
    *,
    config: str | Path,
    checkpoint: str | Path,
    fixed_batch: str | Path,
    output_json: str | Path,
    alpha_grid: str | Sequence[float],
    device: str = "cpu",
    checkpoint_state_key: str = "auto",
    evaluate_gt: bool = False,
) -> dict[str, Any]:
    from mmengine.config import Config
    from opentad.models.builder import build_selector

    config_path = Path(config).expanduser().resolve(strict=True)
    checkpoint_path = Path(checkpoint).expanduser().resolve(strict=True)
    batch_path = Path(fixed_batch).expanduser().resolve(strict=True)
    output_path = Path(output_json).expanduser().resolve()
    repo_root = _git_root(config_path.parent)
    _require(config_path.is_relative_to(repo_root), "config must be inside the audited Git worktree")
    cfg = Config.fromfile(str(config_path))
    _require("model" in cfg and "frame_selector" in cfg.model, "config must define model.frame_selector")
    selector_cfg = cfg.model.frame_selector
    _require(str(selector_cfg.get("type")) == "DucaOnlineFrameSelector", "audit supports DucaOnlineFrameSelector only")
    _require(str(selector_cfg.get("selector_variant")) == "transition_only", "audit requires selector_variant=transition_only")
    _require(str(selector_cfg.get("acquisition_policy")) == "global_structured_topk", "audit requires global_structured_topk")
    _require(str(selector_cfg.get("budget_mode", "fixed")) == "fixed", "audit requires a fixed budget")
    budget = int(selector_cfg.get("budget"))
    max_hole_value = selector_cfg.get("max_unselected_hole")
    max_hole = None if max_hole_value is None else int(max_hole_value)
    temperature = float(selector_cfg.get("structured_temperature", 1.0))
    alphas = parse_alpha_grid(alpha_grid)

    fixed = _load_fixed_batch(batch_path)
    if evaluate_gt:
        _require(fixed["gt_segments"] is not None, "--evaluate-gt requires gt_segments in the frozen batch")
    inputs = fixed["inputs"]
    masks = fixed["masks"]
    dense_window_size = selector_cfg.get("dense_window_size")
    if dense_window_size is not None:
        _require(int(masks.shape[1]) == int(dense_window_size), "fixed batch temporal length must match dense_window_size")
    input_hash_before = _sha256_tensor(inputs)
    mask_hash_before = _sha256_tensor(masks)

    selector = build_selector(selector_cfg)
    _require(selector.__class__.__name__ == "DucaOnlineFrameSelector", "unexpected selector implementation")
    checkpoint_payload = _torch_load(checkpoint_path)
    _require(isinstance(checkpoint_payload, Mapping), "checkpoint must be a mapping")
    state_key, full_state = _checkpoint_state(checkpoint_payload, checkpoint_state_key)
    incompatible = selector.load_state_dict(selector_state_dict(full_state), strict=True)
    _require(not incompatible.missing_keys and not incompatible.unexpected_keys, "selector checkpoint is not strict-compatible")
    selector.requires_grad_(False)
    selector.eval()
    _require(not any(parameter.requires_grad for parameter in selector.parameters()), "selector parameters were not fully frozen")
    selector_state_before = _sha256_state_dict(selector.state_dict())

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA device requested but unavailable")
        torch.cuda.set_device(torch_device)
    selector = selector.to(torch_device)
    inputs_device = inputs.to(torch_device)
    masks_device = masks.to(torch_device)
    passed_input_hash_before = _sha256_tensor(inputs_device)
    passed_mask_hash_before = _sha256_tensor(masks_device)
    safe_metas = [{"video_name": sample_id, "video_id": sample_id} for sample_id in fixed["sample_ids"]]
    with torch.inference_mode():
        selector_output = selector.forward_test(inputs=inputs_device, masks=masks_device, metas=safe_metas)
    scores = selector_output.get("selector_outputs")
    _require(isinstance(scores, Mapping), "selector output is missing selector_outputs")
    center_scores = scores.get("center_scores")
    _require(torch.is_tensor(center_scores) and center_scores.shape == masks_device.shape, "selector output is missing aligned center_scores")
    transition_scores = scores.get("transition_policy_scores")
    if transition_scores is not None:
        _require(torch.is_tensor(transition_scores) and transition_scores.shape == center_scores.shape, "transition_policy_scores must align with center_scores")
        _require(
            torch.equal(center_scores.detach()[masks_device], transition_scores.detach()[masks_device]),
            "center_scores and transition_policy_scores diverged on the valid prefix",
        )
    output_valid = scores.get("valid_mask")
    _require(torch.is_tensor(output_valid) and torch.equal(output_valid.bool(), masks_device.bool()), "selector changed the fixed valid mask")
    frozen_logits = center_scores.detach().float().cpu()
    del selector_output, scores, center_scores, transition_scores

    selector_state_after = _sha256_state_dict(selector.state_dict())
    _require(selector_state_after == selector_state_before, "selector state changed during read-only inference")
    _require(_sha256_tensor(inputs_device) == passed_input_hash_before, "selector mutated the passed input tensor")
    _require(_sha256_tensor(masks_device) == passed_mask_hash_before, "selector mutated the passed mask tensor")
    _require(_sha256_tensor(inputs) == input_hash_before, "fixed inputs changed during inference")
    _require(_sha256_tensor(masks) == mask_hash_before, "fixed masks changed during inference")
    trajectory = audit_tensor_trajectory(
        frozen_logits,
        masks,
        budget=budget,
        max_unselected_hole=max_hole,
        alpha_grid=alphas,
        temperature=temperature,
        sample_ids=fixed["sample_ids"],
        gt_segments=fixed["gt_segments"] if evaluate_gt else None,
    )
    script_path = Path(__file__).resolve()
    report = {
        **trajectory,
        "status": "complete",
        "provenance": {
            "repository_root": str(repo_root),
            "git_commit": _git_commit(repo_root),
            "config_path": str(config_path),
            "config_sha256": _sha256_file(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_state_key": state_key,
            "checkpoint_epoch": checkpoint_payload.get("epoch"),
            "fixed_batch_path": str(batch_path),
            "fixed_batch_sha256": _sha256_file(batch_path),
            "fixed_batch_metadata": fixed["batch_metadata"],
            "audit_script_path": str(script_path),
            "audit_script_sha256": _sha256_file(script_path),
            "alpha_grid_sha256": _sha256_json(alphas),
        },
        "read_only_contract": {
            "selector_only_constructed": True,
            "adatad_detector_constructed": False,
            "adatad_heavy_backbone_executed": False,
            "optimizer_constructed": False,
            "gradients_enabled": False,
            "selector_state_sha256_before": selector_state_before,
            "selector_state_sha256_after": selector_state_after,
            "selector_state_unchanged": True,
            "fixed_inputs_unchanged": True,
            "fixed_masks_unchanged": True,
            "gt_evaluation_enabled": bool(evaluate_gt),
            "gt_passed_to_selector": False,
            "gt_used_for_selection": False,
        },
    }
    _write_json_fail_closed(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only DUCA uniform-to-learned homotopy trajectory audit.")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--fixed-batch", help="Frozen torch mapping with inputs, masks and explicit sample_ids.")
    parser.add_argument("--records-jsonl", help="Hash-bound selector-quality records containing frozen policy logits.")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--alpha-grid", required=True, help="Strictly increasing comma-separated values in [0,1].")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint-state-key", choices=["auto", "state_dict", "state_dict_ema"], default="auto")
    parser.add_argument("--evaluate-gt", action="store_true", help="Evaluate frozen GT only after all selections are decoded.")
    parser.add_argument("--budget", type=int, default=384)
    parser.add_argument("--max-unselected-hole", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args(argv)
    if args.records_jsonl:
        _require(not any((args.config, args.checkpoint, args.fixed_batch)), "records mode cannot mix config/checkpoint/fixed-batch inputs")
        report = run_records_audit(
            records_jsonl=args.records_jsonl,
            output_json=args.output_json,
            alpha_grid=args.alpha_grid,
            budget=args.budget,
            max_unselected_hole=args.max_unselected_hole,
            temperature=args.temperature,
            evaluate_gt=args.evaluate_gt,
            device=args.device,
        )
    else:
        _require(all((args.config, args.checkpoint, args.fixed_batch)), "frozen-batch mode requires config, checkpoint and fixed-batch")
        report = run_frozen_audit(
            config=args.config,
            checkpoint=args.checkpoint,
            fixed_batch=args.fixed_batch,
            output_json=args.output_json,
            alpha_grid=args.alpha_grid,
            device=args.device,
            checkpoint_state_key=args.checkpoint_state_key,
            evaluate_gt=args.evaluate_gt,
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "schema_version": report["schema_version"],
                "sample_count": report["aggregate"]["sample_count"],
                "max_single_step_hard_swaps": report["aggregate"]["max_single_step_hard_swaps"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
