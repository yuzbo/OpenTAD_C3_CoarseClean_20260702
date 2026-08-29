from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .structured_selection import exact_uniform_positions, global_structured_topk


NATIVE_TUBELET_POLICIES = {
    "native_tubelet_uniform",
    "native_tubelet_coreset",
    "native_tubelet_dynamic_uniform",
}

_DYNAMIC_CLIP_BUDGETS = (16, 20, 24)
_TUBELETS_PER_CLIP = 8


def build_native_tubelet_candidates(valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Group the dense frame axis into the VideoMAE-native two-frame units."""

    if valid_mask.ndim != 2:
        raise ValueError("native tubelet candidates require a [B,T] valid mask")
    batch, temporal_len = valid_mask.shape
    tubelet_count = temporal_len // 2
    if tubelet_count <= 0:
        raise ValueError("native tubelet selection requires at least two frames")
    frame_pairs = torch.arange(
        tubelet_count * 2, device=valid_mask.device, dtype=torch.long
    ).reshape(tubelet_count, 2)
    valid_mask = valid_mask.to(dtype=torch.bool)
    paired = valid_mask[:, : tubelet_count * 2].reshape(batch, tubelet_count, 2)
    tubelet_valid = paired.all(dim=-1)
    # Supported THUMOS windows are a valid prefix followed only by padding.
    seen_padding = (~tubelet_valid).cumsum(dim=1) > 0
    if bool((tubelet_valid & seen_padding).any().item()):
        raise ValueError("native tubelet validity must be a contiguous prefix")
    return frame_pairs, tubelet_valid


def aggregate_frame_signals_to_tubelets(
    actionness: torch.Tensor,
    boundary: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Aggregate frozen frame-level scout evidence onto native tubelets."""

    if actionness.shape != boundary.shape or actionness.shape != valid_mask.shape:
        raise ValueError("frame actionness, boundary, and mask must share [B,T]")
    if hidden.ndim != 3 or hidden.shape[:2] != actionness.shape:
        raise ValueError("scout hidden features must be [B,T,D]")
    _, tubelet_valid = build_native_tubelet_candidates(valid_mask)
    batch, tubelet_count = tubelet_valid.shape
    action_pair = actionness[:, : tubelet_count * 2].reshape(batch, tubelet_count, 2)
    boundary_pair = boundary[:, : tubelet_count * 2].reshape(batch, tubelet_count, 2)
    hidden_pair = hidden[:, : tubelet_count * 2].reshape(
        batch, tubelet_count, 2, hidden.shape[-1]
    )
    tubelet_hidden = hidden_pair.mean(dim=2).masked_fill(~tubelet_valid[:, :, None], 0.0)
    return {
        "actionness": action_pair.mean(dim=-1).masked_fill(~tubelet_valid, 0.0),
        "boundary": boundary_pair.amax(dim=-1).masked_fill(~tubelet_valid, 0.0),
        "hidden": tubelet_hidden,
        "valid_mask": tubelet_valid,
    }


def _stable_percentile_rank(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Return a tie-preserving empirical percentile rank on each valid row."""

    ranks = values.new_zeros(values.shape)
    for batch_idx in range(values.shape[0]):
        count = int(valid[batch_idx].long().sum().item())
        if count <= 1:
            continue
        row = values[batch_idx, :count]
        less = (row[:, None] > row[None, :]).sum(dim=1).to(row.dtype)
        equal = (row[:, None] == row[None, :]).sum(dim=1).to(row.dtype)
        ranks[batch_idx, :count] = (less + 0.5 * (equal - 1.0)) / float(count - 1)
    return ranks


def _temporal_novelty(hidden: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    normalized = F.normalize(hidden.float(), dim=-1, eps=1.0e-6)
    novelty = hidden.new_zeros(valid.shape, dtype=torch.float32)
    for batch_idx in range(hidden.shape[0]):
        count = int(valid[batch_idx].long().sum().item())
        if count <= 1:
            continue
        row = normalized[batch_idx, :count]
        pair_similarity = (row[:-1] * row[1:]).sum(dim=-1)
        novelty[batch_idx, 0] = 1.0 - pair_similarity[0]
        novelty[batch_idx, count - 1] = 1.0 - pair_similarity[-1]
        if count > 2:
            novelty[batch_idx, 1 : count - 1] = 1.0 - torch.maximum(
                pair_similarity[:-1], pair_similarity[1:]
            )
    return novelty.clamp_min(0.0).to(dtype=hidden.dtype)


def task_state_tubelet_scores(
    actionness: torch.Tensor,
    boundary: torch.Tensor,
    hidden: torch.Tensor,
    valid_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the frozen actionness/boundary/novelty coreset score."""

    novelty = _temporal_novelty(hidden, valid_mask)
    action_rank = _stable_percentile_rank(actionness, valid_mask)
    boundary_rank = _stable_percentile_rank(boundary, valid_mask)
    novelty_rank = _stable_percentile_rank(novelty, valid_mask)
    score = 0.35 * action_rank + 0.45 * boundary_rank + 0.20 * novelty_rank
    score = score.masked_fill(~valid_mask, 0.0)
    return {
        "score": score,
        "actionness_rank": action_rank,
        "boundary_rank": boundary_rank,
        "novelty_rank": novelty_rank,
        "novelty": novelty,
    }


def assign_dynamic_native_tubelet_clip_budgets(
    window_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Assign 16/20/24 real clip budgets by within-video demand rank.

    The input is a split-local, frozen-scout table.  No detector or label value
    is accepted by this function.  Ties are resolved by the physical window
    start, as fixed by the Pro decision.
    """

    forbidden = {
        "gt_segments",
        "gt_labels",
        "teacher",
        "predictions",
        "metrics",
        "raw_prediction_cache",
    }
    grouped: dict[str, list[tuple[int, Mapping[str, Any]]]] = {}
    for index, row in enumerate(window_rows):
        if not isinstance(row, Mapping):
            raise ValueError("dynamic window evidence rows must be mappings")
        if forbidden.intersection(row):
            raise ValueError("dynamic window evidence contains a forbidden decision input")
        video_name = str(row.get("video_name", ""))
        if not video_name:
            raise ValueError("dynamic window evidence requires video_name")
        grouped.setdefault(video_name, []).append((index, row))

    output: list[dict[str, Any] | None] = [None] * len(window_rows)
    for video_name, indexed_rows in grouped.items():
        indexed_rows = sorted(
            indexed_rows,
            key=lambda item: (int(item[1]["window_start_frame"]), item[0]),
        )
        component_values = []
        for key in ("mean_actionness", "p90_boundary", "p90_novelty"):
            values = torch.tensor(
                [float(row[key]) for _, row in indexed_rows], dtype=torch.float64
            )
            if not bool(torch.isfinite(values).all().item()):
                raise ValueError(f"non-finite dynamic window evidence: {key}")
            valid = torch.ones((1, values.numel()), dtype=torch.bool)
            component_values.append(_stable_percentile_rank(values[None], valid)[0])
        demand = (
            0.35 * component_values[0]
            + 0.45 * component_values[1]
            + 0.20 * component_values[2]
        )
        order = sorted(
            range(len(indexed_rows)),
            key=lambda pos: (
                float(demand[pos].item()),
                int(indexed_rows[pos][1]["window_start_frame"]),
            ),
        )
        half = len(order) // 2
        budgets = [20] * len(order)
        for pos in order[:half]:
            budgets[pos] = 16
        for pos in order[len(order) - half :]:
            budgets[pos] = 24
        if sum(budgets) != 20 * len(budgets):
            raise RuntimeError("dynamic native-tubelet budgets must average 20 clips per video")
        for local_pos, (original_index, row) in enumerate(indexed_rows):
            output[original_index] = {
                "video_name": video_name,
                "window_start_frame": int(row["window_start_frame"]),
                "mean_actionness": float(row["mean_actionness"]),
                "p90_boundary": float(row["p90_boundary"]),
                "p90_novelty": float(row["p90_novelty"]),
                "demand_score": float(demand[local_pos].item()),
                "clip_budget": int(budgets[local_pos]),
            }
    if any(row is None for row in output):
        raise RuntimeError("dynamic native-tubelet budget assignment lost a window")
    return [dict(row) for row in output if row is not None]


def select_native_tubelet_coreset(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    policy: str,
    selected_tubelets: int | Sequence[int] | torch.Tensor = 192,
    max_unselected_hole: int = 7,
    temperature: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Select a fixed number of native tubelets with deterministic padding."""

    if policy not in NATIVE_TUBELET_POLICIES:
        raise ValueError(f"unsupported native tubelet policy: {policy}")
    if scores.shape != valid_mask.shape:
        raise ValueError("tubelet scores and valid mask must share [B,U]")
    batch, tubelet_count = scores.shape
    if torch.is_tensor(selected_tubelets):
        requested = [int(value) for value in selected_tubelets.detach().cpu().tolist()]
    elif isinstance(selected_tubelets, Sequence) and not isinstance(selected_tubelets, (str, bytes)):
        requested = [int(value) for value in selected_tubelets]
    else:
        requested = [int(selected_tubelets)] * batch
    if len(requested) != batch or any(value <= 0 for value in requested):
        raise ValueError("selected_tubelets must provide one positive budget per batch row")
    max_selected = max(requested)
    positions = torch.full(
        (batch, max_selected), -1, device=scores.device, dtype=torch.long
    )
    selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    slot_mask = torch.zeros(
        (batch, max_selected), device=scores.device, dtype=torch.bool
    )
    for batch_idx in range(batch):
        valid_count = int(valid_mask[batch_idx].long().sum().item())
        if valid_count <= 0:
            raise ValueError("native tubelet selection requires at least one valid tubelet")
        if policy == "native_tubelet_dynamic_uniform":
            requested_count = requested[batch_idx]
            if requested_count % _TUBELETS_PER_CLIP != 0:
                raise ValueError(
                    "dynamic native-tubelet budgets must contain whole 16-frame clips"
                )
            if valid_count < requested_count:
                raise ValueError(
                    "dynamic native-tubelet budget exceeds the valid physical tubelet grid"
                )
        k = min(requested[batch_idx], valid_count)
        if policy in {"native_tubelet_uniform", "native_tubelet_dynamic_uniform"} or k == valid_count:
            row_positions = exact_uniform_positions(valid_count, k, device=scores.device)
        else:
            required = torch.zeros((1, valid_count), device=scores.device, dtype=torch.bool)
            required[0, 0] = True
            required[0, valid_count - 1] = True
            # The score is detached by contract. The tiny earlier-index bias is
            # used only when the frozen composite scores are exactly tied.
            tie = torch.arange(valid_count, device=scores.device, dtype=scores.dtype)
            row_scores = scores[batch_idx : batch_idx + 1, :valid_count] - tie[None] * 1.0e-7
            output = global_structured_topk(
                row_scores,
                k=k,
                max_unselected_hole=max_unselected_hole,
                required_mask=required,
                temperature=temperature,
                training=False,
            )
            row_positions = output.selected_positions[0]
        positions[batch_idx, :k] = row_positions
        slot_mask[batch_idx, :k] = True
        selected_mask[batch_idx, row_positions] = True
    return {
        "selected_positions": positions,
        "selected_mask": selected_mask,
        "slot_mask": slot_mask,
    }


def assign_discarded_tubelets_to_anchors(
    selected_positions: torch.Tensor,
    valid_count: int,
) -> torch.Tensor:
    """Assign every valid tubelet to its nearest selected anchor, ties earlier."""

    selected = selected_positions.to(dtype=torch.long)
    if selected.ndim != 1 or selected.numel() <= 0:
        raise ValueError("selected_positions must be a non-empty one-dimensional tensor")
    if bool((selected[1:] <= selected[:-1]).any().item()):
        raise ValueError("selected_positions must be strictly increasing")
    if int(selected[0].item()) < 0 or int(selected[-1].item()) >= int(valid_count):
        raise ValueError("selected_positions must lie on the valid tubelet grid")
    coordinates = torch.arange(valid_count, device=selected.device, dtype=torch.long)
    distances = (coordinates[:, None] - selected[None, :]).abs()
    # torch.argmin returns the first slot, which is the earlier physical anchor.
    return distances.argmin(dim=1)


def merge_discarded_scout_context(
    tubelet_hidden: torch.Tensor,
    tubelet_scores: torch.Tensor,
    selected_positions: torch.Tensor,
    *,
    valid_count: int,
    temperature: float = 0.7,
) -> torch.Tensor:
    """Return m_j-h_j using only frozen low-resolution scout features."""

    if temperature <= 0.0:
        raise ValueError("context recycling temperature must be positive")
    selected = selected_positions.to(device=tubelet_hidden.device, dtype=torch.long)
    assignments = assign_discarded_tubelets_to_anchors(selected, valid_count)
    residuals = []
    for slot, anchor in enumerate(selected):
        members = torch.nonzero(assignments == slot, as_tuple=False).flatten()
        weights = torch.softmax(tubelet_scores[members] / temperature, dim=0)
        merged = (weights[:, None] * tubelet_hidden[members]).sum(dim=0)
        residuals.append(merged - tubelet_hidden[anchor])
    return torch.stack(residuals, dim=0)


def _gather_frame_slots(
    inputs: torch.Tensor,
    frame_positions: torch.Tensor,
    frame_slot_mask: torch.Tensor,
) -> torch.Tensor:
    index = frame_positions.clamp_min(0).to(device=inputs.device, dtype=torch.long)
    if inputs.ndim == 5:
        gathered = torch.gather(
            inputs,
            2,
            index[:, None, :, None, None].expand(
                -1, inputs.shape[1], -1, inputs.shape[3], inputs.shape[4]
            ),
        )
        return gathered * frame_slot_mask[:, None, :, None, None].to(gathered.dtype)
    if inputs.ndim == 6:
        gathered = torch.gather(
            inputs,
            3,
            index[:, None, None, :, None, None].expand(
                -1,
                inputs.shape[1],
                inputs.shape[2],
                -1,
                inputs.shape[4],
                inputs.shape[5],
            ),
        )
        return gathered * frame_slot_mask[:, None, None, :, None, None].to(gathered.dtype)
    raise ValueError("native tubelet RGB gather expects [B,C,T,H,W] or [B,N,C,T,H,W]")


def gather_native_tubelet_rgb(
    inputs: torch.Tensor,
    selected_positions: torch.Tensor,
    slot_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pair_offsets = torch.tensor([0, 1], device=inputs.device, dtype=torch.long)
    frame_positions = selected_positions.clamp_min(0)[:, :, None] * 2 + pair_offsets
    frame_slot_mask = slot_mask[:, :, None].expand_as(frame_positions)
    frame_positions = frame_positions.reshape(selected_positions.shape[0], -1)
    frame_slot_mask = frame_slot_mask.reshape(selected_positions.shape[0], -1)
    return _gather_frame_slots(inputs, frame_positions, frame_slot_mask), frame_positions


def _native_tubelet_metas(
    metas: Sequence[Mapping[str, Any]] | None,
    *,
    selection: Mapping[str, torch.Tensor],
    evidence: Mapping[str, torch.Tensor],
    scout_output: Mapping[str, Any],
    policy: str,
) -> list[dict[str, Any]]:
    batch = int(selection["selected_positions"].shape[0])
    output = [{} for _ in range(batch)] if metas is None else [dict(row) for row in metas]
    if len(output) != batch:
        raise ValueError("native tubelet metadata must match batch size")
    for batch_idx, meta in enumerate(output):
        active = int(selection["slot_mask"][batch_idx].long().sum().item())
        valid_count = int(evidence["valid_mask"][batch_idx].long().sum().item())
        selected = selection["selected_positions"][batch_idx, :active]
        selected_list = [int(value) for value in selected.detach().cpu().tolist()]
        meta.update(
            {
                "duca_native_tubelet_policy": policy,
                "duca_native_tubelet_indices": selected_list,
                "duca_native_tubelet_centers": [2.0 * value + 0.5 for value in selected_list],
                "duca_native_tubelet_valid_len": valid_count,
                "duca_native_tubelet_selected_count": active,
                "duca_native_tubelet_actual_clips": active // _TUBELETS_PER_CLIP,
                "duca_native_tubelet_padded_to_global_max": False,
                "duca_native_tubelet_scout_hidden": evidence["hidden"][batch_idx, :valid_count].detach(),
                "duca_native_tubelet_scores": evidence["score"][batch_idx, :valid_count].detach(),
                "detector_prediction_inverse_map_required": False,
                "detector_output_coordinate_space": "physical_tubelet_grid",
                "irregular_native_axis": True,
                "irregular_selected_positions": selected_list,
                "irregular_selected_count": active,
                "irregular_dense_valid_len": valid_count,
                "irregular_selected_valid_len": active,
                "duca_online_actionness_source": scout_output.get("source_name", "frozen_c3_scout"),
            }
        )
    return output


def run_native_tubelet_selection(
    selector: Any,
    inputs: torch.Tensor,
    masks: torch.Tensor,
    metas: Sequence[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    """Run the frozen scout and gather the selected native tubelet RGB pairs."""

    policy = str(selector.acquisition_policy)
    if policy not in NATIVE_TUBELET_POLICIES:
        raise ValueError(f"selector is not configured for native tubelets: {policy}")
    if selector.raw_actionness_source is None:
        raise ValueError("native tubelet policies require the frozen C3 scout")
    masks = masks.to(device=inputs.device, dtype=torch.bool)
    selector.raw_actionness_source.eval()
    with torch.no_grad():
        scout = selector.raw_actionness_source(inputs, valid_mask=masks)
    actionness = scout["p_action"].detach()
    boundary = scout["transition_score"].detach()
    hidden = scout.get("coarse_hidden_features", scout.get("hidden_features"))
    if hidden is None:
        raise ValueError("native tubelet policies require frozen scout hidden features")
    aggregated = aggregate_frame_signals_to_tubelets(
        actionness,
        boundary,
        hidden.detach(),
        masks,
    )
    scoring = task_state_tubelet_scores(
        aggregated["actionness"],
        aggregated["boundary"],
        aggregated["hidden"],
        aggregated["valid_mask"],
    )
    evidence = {**aggregated, **scoring}
    if policy == "native_tubelet_dynamic_uniform":
        if metas is None:
            raise ValueError("dynamic native-tubelet selection requires window budget metadata")
        clip_budgets = [int(meta.get("duca_native_tubelet_budget_clips", 0)) for meta in metas]
        if any(value not in _DYNAMIC_CLIP_BUDGETS for value in clip_budgets):
            raise ValueError("dynamic native-tubelet clip budgets must be 16, 20, or 24")
        selected_tubelets: int | list[int] = [
            value * _TUBELETS_PER_CLIP for value in clip_budgets
        ]
    else:
        selected_tubelets = int(selector.native_tubelet_selected_count)
    selection = select_native_tubelet_coreset(
        scoring["score"],
        aggregated["valid_mask"],
        policy=policy,
        selected_tubelets=selected_tubelets,
        max_unselected_hole=int(selector.max_unselected_hole),
        temperature=float(selector.structured_temperature),
    )
    selected_inputs, selected_frame_positions = gather_native_tubelet_rgb(
        inputs,
        selection["selected_positions"],
        selection["slot_mask"],
    )
    updated_metas = _native_tubelet_metas(
        metas,
        selection=selection,
        evidence=evidence,
        scout_output=scout,
        policy=policy,
    )
    selector_outputs = {
        **selection,
        **evidence,
        "selected_frame_positions": selected_frame_positions,
        "selection_policy": policy,
        "selected_tubelet_count": selection["slot_mask"].long().sum(dim=1),
        "heavy_frame_slots": selection["slot_mask"].long().sum(dim=1) * 2,
        "actual_clip_count": selection["slot_mask"].long().sum(dim=1) // _TUBELETS_PER_CLIP,
        "scout_frozen": True,
    }
    selector.last_forward_summary = {
        "selection_policy": policy,
        "selected_tubelets": [
            int(value) for value in selector_outputs["selected_tubelet_count"].detach().cpu().tolist()
        ],
        "heavy_frame_slots": [
            int(value) for value in selector_outputs["heavy_frame_slots"].detach().cpu().tolist()
        ],
        "actual_clips": [
            int(value) for value in selector_outputs["actual_clip_count"].detach().cpu().tolist()
        ],
        "padded_to_global_max": False,
        "tubelet_size_frames": 2,
        "scout_frozen": True,
    }
    return {
        "inputs": selected_inputs,
        # This mask describes VideoMAE temporal tubelets, not RGB slots.
        "masks": selection["slot_mask"],
        "metas": updated_metas,
        "selector_outputs": selector_outputs,
    }


__all__ = [
    "NATIVE_TUBELET_POLICIES",
    "assign_dynamic_native_tubelet_clip_budgets",
    "aggregate_frame_signals_to_tubelets",
    "assign_discarded_tubelets_to_anchors",
    "build_native_tubelet_candidates",
    "gather_native_tubelet_rgb",
    "merge_discarded_scout_context",
    "run_native_tubelet_selection",
    "select_native_tubelet_coreset",
    "task_state_tubelet_scores",
]
