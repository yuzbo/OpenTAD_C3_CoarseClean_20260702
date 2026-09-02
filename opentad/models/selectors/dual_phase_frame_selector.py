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
    """Dual-Phase Pre-Backbone Frame Selector with Deterministic Motion Prior and Synchronized Tubelet Grid.

    Decomposes acquisition budget K into:
    1. Global Scaffold (K_scaffold): uniform coverage for baseline semantic recall;
    2. Phase-Transition Bursts (K_burst): dense micro-clusters capturing action boundaries.

    Uses a deterministic parameter-free frame-difference variation energy prior,
    eliminating uninitialized random network noise. Computes exact Tubelet physical midpoints
    and synchronizes temporal positions with 3D VideoMAE feature downsampling and 1D interpolation.
    Injects physical-grid ActionFormer metadata into metas for strict coordinate closure.
    """

    def __init__(
        self,
        total_budget: int = 384,
        scaffold_budget: int = 128,
        burst_budget: int = 256,
        burst_radius: int = 2,
        force_uniform: bool = False,
    ):
        super().__init__()
        self.total_budget = int(total_budget)
        self.scaffold_budget = int(scaffold_budget)
        self.burst_budget = int(burst_budget)
        self.burst_radius = int(burst_radius)
        self.force_uniform = bool(force_uniform)
        assert self.total_budget % 2 == 0, "total_budget must be even for tubelet pairing"

    def _compute_priority(self, inputs_5d: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """Compute frame selection priority via deterministic adjacent-frame variation energy.

        Args:
            inputs_5d: Video tensor [B, C, T, H, W] or float.
            masks: Boolean valid frame mask [B, T].

        Returns:
            priority: Normalized motion energy priority [B, T] in [0, 1].
        """
        B, C, T, H, W = inputs_5d.shape
        # Spatial average pool to 16x16 for efficient motion energy estimation
        downsampled = F.adaptive_avg_pool3d(inputs_5d.float(), (T, 16, 16))  # [B, C, T, 16, 16]

        priority = torch.zeros(B, T, device=inputs_5d.device, dtype=torch.float32)
        if T > 1:
            # Adjacent frame absolute difference averaged over channels and spatial grid
            diff = torch.abs(downsampled[:, :, 1:] - downsampled[:, :, :-1]).mean(dim=(1, 3, 4))  # [B, T-1]
            priority[:, :-1] = diff
            priority[:, -1] = diff[:, -1]

        # Per-sample min-max normalization to [0, 1]
        p_min = priority.min(dim=1, keepdim=True)[0]
        p_max = priority.max(dim=1, keepdim=True)[0]
        priority = (priority - p_min) / (p_max - p_min).clamp_min(1e-6)
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

        if self.force_uniform:
            rows = []
            scaffold_rows = []
            burst_rows = []
            actual_counts = []
            for b in range(B):
                valid_positions = torch.nonzero(masks[b], as_tuple=False).flatten()
                count = min(int(valid_positions.numel()), self.total_budget)
                padded = torch.full(
                    (self.total_budget,), -1, device=inputs.device, dtype=torch.long
                )
                if count:
                    offsets = torch.floor(
                        torch.arange(count, device=inputs.device, dtype=torch.float32)
                        * (float(valid_positions.numel()) / float(count))
                    ).long()
                    padded[:count] = valid_positions[offsets]
                rows.append(padded)
                scaffold_rows.append(
                    torch.arange(self.total_budget, device=inputs.device) < count
                )
                burst_rows.append(torch.zeros(self.total_budget, device=inputs.device, dtype=torch.bool))
                actual_counts.append(count)
            selected = torch.stack(rows, dim=0)
            selection = DualPhaseBudgetSelection(
                selected_positions=selected,
                scaffold_mask=torch.stack(scaffold_rows, dim=0),
                burst_mask=torch.stack(burst_rows, dim=0),
                actual_count=torch.tensor(actual_counts, device=inputs.device, dtype=torch.long),
                k_scaffold=min(self.total_budget, max(actual_counts, default=0)),
                k_burst=0,
                total_k=self.total_budget,
            )
        else:
            selection = dual_phase_orthogonal_budget_positions(
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
        raw_temporal_positions = selected_positions.float().clone()
        for b in range(B):
            valid_idx = torch.nonzero(valid_pos_mask[b], as_tuple=False).flatten()
            if len(valid_idx) == 0:
                raw_temporal_positions[b] = torch.arange(self.total_budget, device=inputs.device, dtype=torch.float32)
            elif len(valid_idx) < self.total_budget:
                last_val = selected_positions[b, valid_idx[-1]].float()
                num_pad = self.total_budget - len(valid_idx)
                pad_vals = last_val + torch.arange(1, num_pad + 1, device=inputs.device, dtype=torch.float32)
                raw_temporal_positions[b, len(valid_idx):] = pad_vals

        # Compute Tubelet midpoints and synchronized feature-grid temporal coordinates
        # VideoMAE conv3d (tubelet=2, stride=2) maps K frames -> K/2 tubelet tokens,
        # then post_processing_pipeline 1D interpolates K/2 tokens back to K tokens.
        # Synchronizing temporal_positions ensures exact 1-to-1 parity between features and timestamps.
        tubelet_len = self.total_budget // 2
        pos_even = raw_temporal_positions[:, 0::2]  # [B, tubelet_len]
        pos_odd = raw_temporal_positions[:, 1::2]   # [B, tubelet_len]
        tubelet_midpoints = 0.5 * (pos_even + pos_odd)  # [B, tubelet_len]

        # 1. Raw frame-pair physical time intervals inside each tubelet for CT-Tubelet speed normalization
        tubelet_delta_t = (pos_odd - pos_even).clamp_min(1.0)  # [B, tubelet_len]

        # 2. 1D linear interpolation matching VideoMAE post_processing_pipeline Interpolate(size=K, align_corners=False)
        synced_temporal_positions = F.interpolate(
            tubelet_midpoints.unsqueeze(1),
            size=self.total_budget,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # [B, K]

        # Enforce strict monotonic increase to prevent zero or negative intervals
        for b in range(B):
            for k in range(1, self.total_budget):
                if synced_temporal_positions[b, k] <= synced_temporal_positions[b, k - 1]:
                    synced_temporal_positions[b, k] = synced_temporal_positions[b, k - 1] + 1e-4

        # 3. Compute physical delta_t per token on detector feature grid for CT-Conv1d
        diff = torch.zeros_like(synced_temporal_positions, dtype=torch.float32)
        diff[:, :-1] = synced_temporal_positions[:, 1:] - synced_temporal_positions[:, :-1]
        diff[:, -1] = diff[:, -2] if self.total_budget > 1 else 1.0
        detector_delta_t = diff.clamp_min(1e-4)

        # Keep the frame-level prior for auditability, but route B-AMoD with
        # one score per VideoMAE tubelet.  A frame-level vector is not a valid
        # tubelet prior: expanding 384 frame scores over 192 tubelets repeats
        # the wrong temporal support.  Max-pooling each pair preserves any
        # boundary hit inside the tubelet and closes the token/grid contract.
        boundary_prior_frames = selection.burst_mask.float()
        boundary_prior = boundary_prior_frames
        boundary_prior_tubelet = boundary_prior_frames.reshape(B, tubelet_len, 2).amax(dim=-1)

        # Update metas with complete physical-grid ActionFormer contract
        for i in range(len(metas)):
            metas[i]["selected_positions"] = selected_positions[i].detach()
            metas[i]["temporal_positions"] = synced_temporal_positions[i].detach()
            metas[i]["irregular_selected_positions"] = synced_temporal_positions[i].detach()
            metas[i]["selected_dense_indices"] = synced_temporal_positions[i].detach()
            metas[i]["selected_valid_len"] = int(selected_masks[i].sum().item())
            metas[i]["irregular_selected_valid_len"] = float(masks[i].sum().item())
            metas[i]["irregular_dense_valid_len"] = float(masks[i].sum().item())
            metas[i]["irregular_native_axis"] = True
            metas[i]["tubelet_delta_t"] = tubelet_delta_t[i].detach()
            metas[i]["delta_t"] = detector_delta_t[i].detach()
            metas[i]["boundary_prior"] = boundary_prior[i].detach()
            metas[i]["boundary_prior_frames"] = boundary_prior_frames[i].detach()
            metas[i]["boundary_prior_tubelet"] = boundary_prior_tubelet[i].detach()
            metas[i]["original_window_size"] = orig_window_size
            metas[i]["selected_window_size"] = self.total_budget

        outputs = {
            "inputs": selected_inputs,
            "masks": selected_masks,
            "metas": metas,
            "selected_positions": selected_positions,
            "temporal_positions": synced_temporal_positions,
            "boundary_prior": boundary_prior,
            "boundary_prior_frames": boundary_prior_frames,
            "boundary_prior_tubelet": boundary_prior_tubelet,
            "tubelet_delta_t": tubelet_delta_t,
            "delta_t": detector_delta_t,
        }
        if gt_segments is not None:
            outputs["gt_segments"] = gt_segments
        if gt_labels is not None:
            outputs["gt_labels"] = gt_labels
        return outputs


if SELECTORS is not None:
    SELECTORS.register_module()(DualPhaseFrameSelector)
