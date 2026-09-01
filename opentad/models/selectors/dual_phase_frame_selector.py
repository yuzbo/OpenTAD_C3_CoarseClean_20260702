from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..builder import SELECTORS
except Exception:
    SELECTORS = None
from ..duca.acquisition import dual_phase_orthogonal_budget_positions, DualPhaseBudgetSelection


class DualPhaseFrameSelector(nn.Module):
    """Dual-Phase Pre-Backbone Frame Selector.

    Decomposes acquisition budget K into:
    1. Global Scaffold (K_scaffold): uniform coverage for baseline semantic recall;
    2. Phase-Transition Bursts (K_burst): dense micro-clusters capturing action boundaries.

    Reduces input from raw 768 frames down to selected 384 frames before heavy VideoMAE backbone,
    providing boundary prior to B-AMoD and physical delta_t to CT-Conv1d.
    """

    def __init__(
        self,
        total_budget: int = 384,
        scaffold_budget: int = 128,
        burst_budget: int = 256,
        burst_radius: int = 2,
        scout_channels: int = 16,
    ):
        super().__init__()
        self.total_budget = int(total_budget)
        self.scaffold_budget = int(scaffold_budget)
        self.burst_budget = int(burst_budget)
        self.burst_radius = int(burst_radius)

        # VideoMAE normalization constants for scout branch
        self.register_buffer("scout_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1))
        self.register_buffer("scout_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1))

        # Lightweight 1D temporal scout to compute priority from downsampled frame features
        self.scout_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d((None, 4, 4)),  # spatial pool to [B, C, T, 4, 4]
            nn.Flatten(3),  # [B, C, T, 16]
        )
        self.scout_head = nn.Sequential(
            nn.Conv1d(3 * 16, scout_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(scout_channels, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def _compute_priority(self, inputs_5d: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute frame selection priority from video inputs [B, 3, T, H, W]."""
        B, C, T, H, W = inputs_5d.shape
        # Ensure float32 and VideoMAE normalization for the scout branch
        scout_in = inputs_5d.float()
        if scout_in.max() > 1.0:
            scout_in = (scout_in - self.scout_mean) / self.scout_std

        pooled = self.scout_proj(scout_in).permute(0, 1, 3, 2).reshape(B, C * 16, T)  # [B, 48, T]
        priority = self.scout_head(pooled).squeeze(1)  # [B, T]
        priority = priority * masks.float()
        return priority

    def _gather_frames(
        self,
        inputs: torch.Tensor,
        selected_positions: torch.Tensor,
        selected_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Gather selected frames along temporal dimension and zero-out invalid frames."""
        K = selected_positions.shape[1]
        if inputs.ndim == 6:
            B, N, C, T_in, H, W = inputs.shape
            clamped_pos = selected_positions.clamp(0, T_in - 1)
            idx = clamped_pos[:, None, None, :, None, None].expand(B, N, C, K, H, W)
            gathered = torch.gather(inputs, 3, idx)
            gathered = gathered * selected_masks[:, None, None, :, None, None].to(dtype=gathered.dtype)
        else:
            B, C, T_in, H, W = inputs.shape
            clamped_pos = selected_positions.clamp(0, T_in - 1)
            idx = clamped_pos[:, None, :, None, None].expand(B, C, K, H, W)
            gathered = torch.gather(inputs, 2, idx)
            gathered = gathered * selected_masks[:, None, :, None, None].to(dtype=gathered.dtype)
        return gathered

    def _gather_masks(self, masks: torch.Tensor, selected_positions: torch.Tensor) -> torch.Tensor:
        """Gather masks for selected positions."""
        B, T_in = masks.shape
        K = selected_positions.shape[1]
        clamped_pos = selected_positions.clamp(0, T_in - 1)
        gathered_masks = torch.gather(masks.bool(), 1, clamped_pos)
        # Position is valid if original pos >= 0 and mask was True
        valid_pos = selected_positions >= 0
        return gathered_masks & valid_pos

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: List[Dict],
        gt_segments: Optional[Any] = None,
        gt_labels: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Training forward pass."""
        return self._forward_impl(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, training=True)

    def forward_test(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: List[Dict],
    ) -> Dict[str, Any]:
        """Testing forward pass."""
        return self._forward_impl(inputs, masks, metas, training=False)

    def _forward_impl(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: List[Dict],
        gt_segments: Optional[Any] = None,
        gt_labels: Optional[Any] = None,
        training: bool = True,
    ) -> Dict[str, Any]:
        if inputs.ndim == 6:
            B, N, C, T_raw, H, W = inputs.shape
            assert N == 1, f"Expected N=1 for single-clip video inputs, got N={N}"
            inputs_5d = inputs[:, 0]
            orig_window_size = T_raw
        elif inputs.ndim == 5:
            B, C, T_raw, H, W = inputs.shape
            inputs_5d = inputs
            orig_window_size = T_raw
        else:
            raise ValueError(f"Expected 5D [B,C,T,H,W] or 6D [B,N,C,T,H,W] inputs, got {inputs.shape}")

        priority = self._compute_priority(inputs_5d, masks)

        selection: DualPhaseBudgetSelection = dual_phase_orthogonal_budget_positions(
            h65_priority=priority,
            valid_mask=masks,
            total_budget=self.total_budget,
            scaffold_budget=self.scaffold_budget,
            burst_budget=self.burst_budget,
            burst_radius=self.burst_radius,
        )

        selected_positions = selection.selected_positions  # [B, K]
        selected_masks = self._gather_masks(masks, selected_positions)
        selected_inputs = self._gather_frames(inputs, selected_positions, selected_masks)

        # Generate strictly monotonic temporal positions without -1 padding
        valid_pos_mask = selected_positions >= 0
        temporal_positions = selected_positions.float().clone()
        for b in range(B):
            valid_idx = torch.nonzero(valid_pos_mask[b], as_tuple=False).flatten()
            if len(valid_idx) == 0:
                temporal_positions[b] = torch.arange(self.total_budget, device=inputs.device, dtype=torch.float32)
            elif len(valid_idx) < self.total_budget:
                last_val = selected_positions[b, valid_idx[-1]].float()
                num_pad = self.total_budget - len(valid_idx)
                pad_vals = last_val + torch.arange(1, num_pad + 1, device=inputs.device, dtype=torch.float32)
                temporal_positions[b, len(valid_idx):] = pad_vals

        # Compute delta_t from consecutive monotonic positions
        diff = torch.zeros_like(temporal_positions, dtype=torch.float32)
        diff[:, :-1] = temporal_positions[:, 1:] - temporal_positions[:, :-1]
        diff[:, -1] = diff[:, -2] if self.total_budget > 1 else 1.0
        delta_t = diff.clamp_min(1.0)

        # Boundary prior score for B-AMoD: burst mask indicates boundary clusters
        boundary_prior = selection.burst_mask.float()

        # Update metas
        for i in range(len(metas)):
            metas[i]["selected_positions"] = selected_positions[i].detach()
            metas[i]["temporal_positions"] = temporal_positions[i].detach()
            metas[i]["delta_t"] = delta_t[i].detach()
            metas[i]["boundary_prior"] = boundary_prior[i].detach()
            metas[i]["original_window_size"] = orig_window_size
            metas[i]["selected_window_size"] = self.total_budget

        outputs = {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": metas,
            "selected_positions": selected_positions,
            "temporal_positions": temporal_positions,
            "boundary_prior": boundary_prior,
            "delta_t": delta_t,
        }
        if gt_segments is not None:
            outputs["gt_segments"] = gt_segments
        if gt_labels is not None:
            outputs["gt_labels"] = gt_labels
        return outputs


if SELECTORS is not None:
    SELECTORS.register_module()(DualPhaseFrameSelector)
