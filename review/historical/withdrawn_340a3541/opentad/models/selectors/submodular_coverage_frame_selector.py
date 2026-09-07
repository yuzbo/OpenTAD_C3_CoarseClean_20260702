import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS


@SELECTORS.register_module()
class SubmodularCoverageFrameSelector(nn.Module):
    """Boundary-Sensitive Coverage-Aware Frame Selector (DUCA-Coverage-v1).

    This module replaces hard-split heuristic selection with a mathematically grounded
    submodular coverage optimization framework.

    Core Formulation:
      max_{S subset V, |S|=K} F(S) = sum_{t in S} Q(t) + beta * C(S)
      where:
        Q(t) = E(t) + alpha * |E(t+1) - E(t)|   (Quality: Motion Energy + Boundary Transition)
        C(S) = sum_{t in V} (1 - exp(-min_{s in S} d(t, s)^2 / (2 * sigma^2)))  (Saturated Submodular Coverage)

    Properties:
      1. Saturated exponential decay kernel prevents redundant frame clustering near dense action peaks.
      2. High boundary gradient alpha prioritizes sharp start/end transitions.
      3. Fully vectorized incremental greedy selection on GPU: O(K * T) without Host-Device pipeline stalls.
      4. Disentangles raw frame-pair intervals for CT-Tubelet speed normalization from synchronized detector coordinates.

    Args:
        total_budget (int): Total target frames K to select (default: 384).
        alpha_boundary (float): Weight for action transition boundary gradients (default: 1.5).
        beta_coverage (float): Weight for temporal coverage gain (default: 0.8).
        kernel_sigma (float): Bandwidth of temporal saturation Gaussian kernel (default: 0.05).
        downsample_res (int): Spatial resolution for fast motion energy calculation (default: 16).
    """

    def __init__(
        self,
        total_budget: int = 384,
        alpha_boundary: float = 1.5,
        beta_coverage: float = 0.8,
        kernel_sigma: float = 0.05,
        downsample_res: int = 16,
    ):
        super().__init__()
        self.total_budget = int(total_budget)
        self.alpha_boundary = float(alpha_boundary)
        self.beta_coverage = float(beta_coverage)
        self.kernel_sigma = float(kernel_sigma)
        self.downsample_res = int(downsample_res)

    def _compute_motion_energy(self, x: torch.Tensor, valid_len: int) -> torch.Tensor:
        """Compute low-cost frame difference variation energy on downsampled spatial resolution."""
        if x.dim() == 6:
            # [B, S, C, T, H, W] -> take first segment S=0
            x = x[:, 0]
        # x is [B, C, T, H, W]
        x = x.float()

        B, C, T, H, W = x.shape
        if H != self.downsample_res or W != self.downsample_res:
            x_down = F.interpolate(
                x.reshape(B * C * T, 1, H, W),
                size=(self.downsample_res, self.downsample_res),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, C, T, self.downsample_res, self.downsample_res)
        else:
            x_down = x

        # Adjacent frame difference: |I_{t+1} - I_t|
        diff = torch.abs(x_down[:, :, 1:, :, :] - x_down[:, :, :-1, :, :])  # [B, C, T-1, H, W]
        diff_energy = diff.mean(dim=(1, 3, 4))  # [B, T-1]

        # Pad last frame to maintain length T
        energy = F.pad(diff_energy, (0, 1), mode="replicate")  # [B, T]

        # Normalize per video
        energy_max = energy.amax(dim=-1, keepdim=True).clamp_min(1e-6)
        energy_norm = energy / energy_max
        return energy_norm

    def _submodular_greedy_select(
        self,
        quality_score: torch.Tensor,
        valid_masks: torch.Tensor,
        target_k: int,
    ) -> torch.Tensor:
        """Perform vectorized incremental submodular greedy keyframe selection.

        Args:
            quality_score: Tensor of shape [B, T] (quality score = motion energy + boundary gradient).
            valid_masks: BoolTensor of shape [B, T] indicating valid frames.
            target_k: Number of frames to select.

        Returns:
            selected_indices: LongTensor of shape [B, target_k] (sorted temporally).
        """
        B, T = quality_score.shape
        device = quality_score.device

        # Normalized temporal coordinates in [0, 1]
        t_coords = torch.linspace(0.0, 1.0, T, device=device).unsqueeze(0).expand(B, T)

        selected = torch.zeros((B, target_k), dtype=torch.long, device=device)
        cur_min_dist = torch.ones((B, T), device=device, dtype=torch.float32)

        # Base quality with padding mask applied (-10000.0 for invalid frames)
        masked_quality = quality_score.clone()
        masked_quality = masked_quality.masked_fill(~valid_masks, -10000.0)

        sigma_sq_2 = 2.0 * (self.kernel_sigma**2)

        for k in range(target_k):
            if k == 0:
                next_idx = torch.argmax(masked_quality, dim=-1)  # [B]
            else:
                # Saturated submodular coverage gain: 1 - exp(-dist^2 / (2*sigma^2))
                coverage_gain = 1.0 - torch.exp(-(cur_min_dist**2) / sigma_sq_2)
                obj_score = masked_quality + self.beta_coverage * coverage_gain
                # Mask out already selected keyframes
                obj_score.scatter_(1, selected[:, :k], -10000.0)
                next_idx = torch.argmax(obj_score, dim=-1)  # [B]

            selected[:, k] = next_idx

            # Incremental distance field update: min(cur_min_dist, |t - t_{next}|)
            new_t = t_coords.gather(1, next_idx.unsqueeze(1))  # [B, 1]
            new_dist = torch.abs(t_coords - new_t)  # [B, T]
            cur_min_dist = torch.minimum(cur_min_dist, new_dist)

        # Handle short videos where valid_count < target_k: avoid duplicate indices
        for b in range(B):
            v_cnt = int(valid_masks[b].sum().item())
            if v_cnt < target_k:
                unique_sel = torch.unique(selected[b, :k+1])
                pad_len = target_k - len(unique_sel)
                if pad_len > 0:
                    pad_idx = torch.arange(v_cnt, v_cnt + pad_len, device=device, dtype=torch.long) % T
                    selected[b] = torch.cat((unique_sel, pad_idx), dim=0)[:target_k]

        # Sort indices temporally to guarantee monotonic temporal ordering
        selected_sorted, _ = torch.sort(selected, dim=-1)
        return selected_sorted

    def forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: Optional[List[Dict]] = None,
        gt_segments: Optional[List[torch.Tensor]] = None,
        gt_labels: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Dict:
        """Select 384 keyframes via boundary-sensitive submodular coverage and synchronize geometry."""
        if inputs.dim() == 6:
            B, S, C, T, H, W = inputs.shape
            is_6d = True
        elif inputs.dim() == 5:
            B, C, T, H, W = inputs.shape
            S = 1
            is_6d = False
        else:
            raise ValueError(f"Inputs must be 5D or 6D tensor, got shape {inputs.shape}")

        device = inputs.device
        target_k = min(self.total_budget, T)

        # 1. Compute frame difference variation energy
        valid_masks = masks.to(device=device, dtype=torch.bool)
        motion_energy = self._compute_motion_energy(inputs, valid_len=T)  # [B, T]

        # 2. Boundary gradient: |E(t+1) - E(t)|
        boundary_grad = torch.abs(motion_energy[:, 1:] - motion_energy[:, :-1])
        boundary_grad = F.pad(boundary_grad, (0, 1), mode="replicate")  # [B, T]

        # 3. Combined quality score
        quality_score = motion_energy + self.alpha_boundary * boundary_grad  # [B, T]

        # 4. Incremental Submodular Coverage Greedy Selection
        selected_indices = self._submodular_greedy_select(
            quality_score=quality_score,
            valid_masks=valid_masks,
            target_k=target_k,
        )  # [B, target_k]

        # 5. Extract raw pixels for selected keyframes
        if is_6d:
            selected_frames = torch.stack(
                [
                    torch.index_select(inputs[b, 0], dim=1, index=selected_indices[b])
                    for b in range(B)
                ],
                dim=0,
            )  # [B, C, target_k, H, W]
            selected_frames = selected_frames.unsqueeze(1)  # [B, 1, C, target_k, H, W]
        else:
            selected_frames = torch.stack(
                [
                    torch.index_select(inputs[b], dim=1, index=selected_indices[b])
                    for b in range(B)
                ],
                dim=0,
            )  # [B, C, target_k, H, W]

        # 6. Physical time coordinates & Tubelet midpoint synchronization
        # VideoMAE compresses target_k (384) frames with stride=2 into 192 tubelets, then interpolates back to 384.
        tubelet_len = target_k // 2
        t_even = selected_indices[:, 0::2].float()  # [B, tubelet_len]
        t_odd = selected_indices[:, 1::2].float()   # [B, tubelet_len]

        # Raw frame-pair physical time intervals inside each tubelet for CT-Tubelet speed normalization
        tubelet_delta_t = (t_odd - t_even).clamp_min(1.0)  # [B, tubelet_len]
        tubelet_midpoints = 0.5 * (t_even + t_odd)         # [B, tubelet_len]

        # 1D linear interpolation matching VideoMAE feature Interpolate(size=target_k, align_corners=False)
        temporal_positions = F.interpolate(
            tubelet_midpoints.unsqueeze(1),
            size=target_k,
            mode="linear",
            align_corners=False,
        ).squeeze(1)  # [B, target_k]

        # 7. Sampling step delta_t for detector feature grid CT-Conv1d
        dt_forward = torch.diff(temporal_positions, dim=-1, prepend=temporal_positions[:, :1])
        dt_backward = torch.diff(temporal_positions, dim=-1, append=temporal_positions[:, -1:])
        detector_delta_t = 0.5 * (dt_forward.abs() + dt_backward.abs()).clamp_min(1e-4)  # [B, target_k]

        # 8. Synchronized boundary prior for B-AMoD: gather quality score onto selected frame grid [B, target_k]
        boundary_prior = torch.gather(quality_score, dim=1, index=selected_indices)

        selected_masks = torch.ones((B, target_k), device=device, dtype=torch.bool)
        for b in range(B):
            valid_count = int(valid_masks[b].sum().item())
            if valid_count < T:
                valid_sel = (selected_indices[b] < valid_count).sum().item()
                selected_masks[b, valid_sel:] = False

        # 9. Inject Physical-Grid metadata into metas for zero-truncation GT matching & seconds conversion
        if metas is not None:
            for b, meta in enumerate(metas):
                if isinstance(meta, dict):
                    meta["irregular_selected_positions"] = temporal_positions[b].detach().cpu()
                    meta["selected_dense_indices"] = selected_indices[b].detach().cpu()
                    meta["selected_valid_len"] = float(selected_masks[b].sum().item())
                    meta["irregular_selected_valid_len"] = float(T)
                    meta["irregular_dense_valid_len"] = float(T)
                    meta["irregular_native_axis"] = True
                    meta["tubelet_delta_t"] = tubelet_delta_t[b].detach().cpu()
                    meta["delta_t"] = detector_delta_t[b].detach().cpu()
                    meta["boundary_prior"] = boundary_prior[b].detach().cpu()
                    meta["temporal_positions"] = temporal_positions[b].detach().cpu()

        return {
            "inputs": selected_frames,
            "masks": selected_masks,
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "tubelet_delta_t": tubelet_delta_t,
            "delta_t": detector_delta_t,
            "temporal_positions": temporal_positions,
            "boundary_prior": boundary_prior,
            "selected_indices": selected_indices,
        }

    def forward_train(self, inputs, masks, metas=None, gt_segments=None, gt_labels=None, **kwargs):
        return self.forward(inputs, masks, metas, gt_segments, gt_labels, **kwargs)

    def forward_test(self, inputs, masks, metas=None, **kwargs):
        return self.forward(inputs, masks, metas, **kwargs)
