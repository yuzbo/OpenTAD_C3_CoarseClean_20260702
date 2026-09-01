import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import SELECTORS


@SELECTORS.register_module()
class SubmodularCoverageFrameSelector(nn.Module):
    """Boundary-Sensitive Submodular Coverage Keyframe Selector (DUCA-Coverage).

    Replaces rigid hard-split budget allocation with a monotone submodular coverage
    optimization solver. Dynamically balances:
      1. Motion Energy Quality: E(t) = 1/(C*H*W) sum |I_{t+1} - I_t|
      2. Boundary Gradient: G(t) = |E(t+1) - E(t)| (detects action onset/offset transitions)
      3. Saturated Temporal Coverage Gain: C(t) = 1 - exp(-min_{j in S} (t - t_j)^2 / (2*sigma^2))
         (prevents temporal coverage holes while avoiding over-allocation to empty backgrounds)

    Implemented via a pure GPU-vectorized incremental greedy algorithm (O(T) per step)
    without CPU-GPU synchronization.

    Args:
        total_budget (int): Total number of keyframes K to select (e.g. 384).
        alpha_boundary (float): Weight for action onset/offset boundary gradient.
        beta_coverage (float): Weight for temporal submodular coverage gain.
        kernel_sigma (float): Bandwidth parameter for saturated coverage kernel (in normalized [0, 1] units).
        downsample_res (int): Spatial downsampling resolution for scout pass (default 16).
    """

    def __init__(
        self,
        total_budget: int = 384,
        alpha_boundary: float = 1.2,
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

        if self.total_budget <= 0:
            raise ValueError(f"total_budget must be positive, got {total_budget}")
        if self.kernel_sigma <= 0:
            raise ValueError(f"kernel_sigma must be positive, got {kernel_sigma}")

    def _compute_motion_energy(self, inputs: torch.Tensor, valid_len: int) -> torch.Tensor:
        """Compute adjacent-frame pixel variation energy on low-res video tensor.

        Args:
            inputs: Tensor of shape [B, C, T, H, W] or [B, 1, C, T, H, W] or [B, num_seg, C, T, H, W].
            valid_len: Number of unpadded frames.

        Returns:
            energy: Tensor of shape [B, T] with normalized frame-difference variation energy.
        """
        if inputs.dim() == 6:
            # [B, num_segs, C, T, H, W] -> take first segment
            x = inputs[:, 0]  # [B, C, T, H, W]
        elif inputs.dim() == 5:
            x = inputs  # [B, C, T, H, W]
        else:
            raise ValueError(f"Unsupported input dimension {inputs.dim()} for video frames")

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
        # Global minimum distance to the current selected set (initialized to max distance 1.0)
        cur_min_dist = torch.ones((B, T), device=device, dtype=torch.float32)

        # Base quality with padding mask applied (-1e9 for invalid frames)
        masked_quality = quality_score.clone()
        masked_quality = masked_quality.masked_fill(~valid_masks, -1e9)

        sigma_sq_2 = 2.0 * (self.kernel_sigma**2)

        for k in range(target_k):
            if k == 0:
                next_idx = torch.argmax(masked_quality, dim=-1)  # [B]
            else:
                # Saturated submodular coverage gain: 1 - exp(-dist^2 / (2*sigma^2))
                coverage_gain = 1.0 - torch.exp(-(cur_min_dist**2) / sigma_sq_2)
                obj_score = masked_quality + self.beta_coverage * coverage_gain
                # Mask out already selected keyframes
                obj_score.scatter_(1, selected[:, :k], -1e9)
                next_idx = torch.argmax(obj_score, dim=-1)  # [B]

            selected[:, k] = next_idx

            # Incremental distance field update: min(cur_min_dist, |t - t_{next}|)
            new_t = t_coords.gather(1, next_idx.unsqueeze(1))  # [B, 1]
            new_dist = torch.abs(t_coords - new_t)  # [B, T]
            cur_min_dist = torch.minimum(cur_min_dist, new_dist)

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

        # 6. Physical time coordinates: Tubelet midpoints and 1D linear interpolation
        # VideoMAE compresses target_k (384) frames with stride=2 into 192 tubelets, then interpolates back to 384.
        # Compute 192 tubelet midpoints: tau_tubelet[j] = 0.5 * (t[2j] + t[2j+1])
        t_even = selected_indices[:, 0::2].float()  # [B, target_k // 2]
        t_odd = selected_indices[:, 1::2].float()   # [B, target_k // 2]
        tubelet_midpoints = 0.5 * (t_even + t_odd)  # [B, target_k // 2]

        # Interpolate tubelet midpoints to match feature dimension (target_k = 384)
        temporal_positions = F.interpolate(
            tubelet_midpoints.unsqueeze(1),
            size=target_k,
            mode="linear",
            align_corners=True,
        ).squeeze(1)  # [B, target_k]

        # 7. Sampling step delta_t for Continuous-Time Conv and CT-Tubelet speed normalization
        dt_forward = torch.diff(temporal_positions, dim=-1, prepend=temporal_positions[:, :1])
        dt_backward = torch.diff(temporal_positions, dim=-1, append=temporal_positions[:, -1:])
        delta_t = 0.5 * (dt_forward.abs() + dt_backward.abs()).clamp_min(1.0)  # [B, target_k]

        selected_masks = torch.ones((B, target_k), device=device, dtype=torch.bool)
        for b in range(B):
            valid_count = int(valid_masks[b].sum().item())
            if valid_count < T:
                valid_sel = (selected_indices[b] < valid_count).sum().item()
                selected_masks[b, valid_sel:] = False

        # 8. Inject Physical-Grid metadata into metas for zero-truncation GT matching & seconds conversion
        if metas is not None:
            for b, meta in enumerate(metas):
                if isinstance(meta, dict):
                    meta["irregular_selected_positions"] = temporal_positions[b].detach().cpu()
                    meta["selected_dense_indices"] = selected_indices[b].detach().cpu()
                    meta["selected_valid_len"] = float(selected_masks[b].sum().item())
                    meta["irregular_selected_valid_len"] = float(T)
                    meta["irregular_dense_valid_len"] = float(T)
                    meta["irregular_native_axis"] = True
                    meta["delta_t"] = delta_t[b].detach().cpu()
                    meta["temporal_positions"] = temporal_positions[b].detach().cpu()

        return {
            "inputs": selected_frames,
            "masks": selected_masks,
            "metas": metas,
            "gt_segments": gt_segments,
            "gt_labels": gt_labels,
            "delta_t": delta_t,
            "temporal_positions": temporal_positions,
            "boundary_prior": quality_score,
            "selected_indices": selected_indices,
        }

    def forward_train(self, inputs, masks, metas=None, gt_segments=None, gt_labels=None, **kwargs):
        return self.forward(inputs, masks, metas, gt_segments, gt_labels, **kwargs)

    def forward_test(self, inputs, masks, metas=None, **kwargs):
        return self.forward(inputs, masks, metas, **kwargs)
