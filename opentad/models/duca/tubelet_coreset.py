from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn.functional as F

from .structured_selection import exact_uniform_positions, global_structured_topk


NATIVE_TUBELET_POLICIES = {
    "native_tubelet_uniform",
    "native_tubelet_coreset",
}


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


def select_native_tubelet_coreset(
    scores: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    policy: str,
    selected_tubelets: int = 192,
    max_unselected_hole: int = 7,
    temperature: float = 0.7,
) -> dict[str, torch.Tensor]:
    """Select a fixed number of native tubelets with deterministic padding."""

    if policy not in NATIVE_TUBELET_POLICIES:
        raise ValueError(f"unsupported native tubelet policy: {policy}")
    if scores.shape != valid_mask.shape:
        raise ValueError("tubelet scores and valid mask must share [B,U]")
    batch, tubelet_count = scores.shape
    positions = torch.full(
        (batch, selected_tubelets), -1, device=scores.device, dtype=torch.long
    )
    selected_mask = torch.zeros_like(valid_mask, dtype=torch.bool)
    slot_mask = torch.zeros(
        (batch, selected_tubelets), device=scores.device, dtype=torch.bool
    )
    for batch_idx in range(batch):
        valid_count = int(valid_mask[batch_idx].long().sum().item())
        if valid_count <= 0:
            raise ValueError("native tubelet selection requires at least one valid tubelet")
        k = min(int(selected_tubelets), valid_count)
        if policy == "native_tubelet_uniform" or k == valid_count:
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
    selection = select_native_tubelet_coreset(
        scoring["score"],
        aggregated["valid_mask"],
        policy=policy,
        selected_tubelets=int(selector.native_tubelet_selected_count),
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
        "heavy_frame_slots": int(selected_inputs.shape[3] if selected_inputs.ndim == 6 else selected_inputs.shape[2]),
        "scout_frozen": True,
    }
    selector.last_forward_summary = {
        "selection_policy": policy,
        "selected_tubelets": [
            int(value) for value in selector_outputs["selected_tubelet_count"].detach().cpu().tolist()
        ],
        "heavy_frame_slots": selector_outputs["heavy_frame_slots"],
        "tubelet_size_frames": 2,
        "scout_frozen": True,
    }
    return {
        "inputs": selected_inputs,
        # This mask describes the 192 VideoMAE temporal tubelets, not RGB slots.
        "masks": selection["slot_mask"],
        "metas": updated_metas,
        "selector_outputs": selector_outputs,
    }


__all__ = [
    "NATIVE_TUBELET_POLICIES",
    "aggregate_frame_signals_to_tubelets",
    "assign_discarded_tubelets_to_anchors",
    "build_native_tubelet_candidates",
    "gather_native_tubelet_rgb",
    "merge_discarded_scout_context",
    "run_native_tubelet_selection",
    "select_native_tubelet_coreset",
    "task_state_tubelet_scores",
]
