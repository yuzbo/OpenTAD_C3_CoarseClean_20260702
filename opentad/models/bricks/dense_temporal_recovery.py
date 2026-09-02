"""Support-aware dense temporal recovery to uniform 384 detection grid."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from opentad.models.utils.numerics import assert_finite_tensor


class DenseTemporalRecovery(BaseModule):
    """Reconstructs features from irregular support tokens to a uniform 384 detection grid.

    Output grid: 384 uniform positions covering [0, 767]:
        grid_j = j * 767 / 383 for j in [0, 383].

    Stage 1: Support-aware triangular kernel scatter
        w_{i,j} = m_i * max(0, 1 - |grid_j - tau_i| / half_width_i)
        F_hat_j = (sum_i w_{i,j} F_i) / (sum_i w_{i,j} + eps)

    Stage 2: Zero-initialized depthwise-pointwise residual refinement
        Depthwise Conv1d(k=3) -> GELU -> Pointwise Conv1d -> zero-initialized gate.
        At initialization, output is bit-exact equivalent to Stage 1.
    """

    def __init__(
        self,
        embed_dims: int = 384,
        target_grid_size: int = 384,
        original_window_size: int = 768,
        kernel_size: int = 3,
        enabled: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.target_grid_size = target_grid_size
        self.original_window_size = original_window_size
        self.enabled = bool(enabled)

        # Precompute target grid coordinates in [0, original_window_size - 1]
        grid = torch.linspace(0, original_window_size - 1, target_grid_size, dtype=torch.float32)
        self.register_buffer("target_grid", grid)

        # Lightweight refinement conv
        self.dwconv = nn.Conv1d(
            embed_dims,
            embed_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=embed_dims,
        )
        self.act = nn.GELU()
        self.pwconv = nn.Conv1d(embed_dims, embed_dims, kernel_size=1)
        self.residual_gate = nn.Parameter(torch.zeros(1))

        # Initialize refinement to zero so initial output is strictly non-parametric interpolation
        nn.init.zeros_(self.pwconv.weight)
        nn.init.zeros_(self.pwconv.bias)

    def scatter_triangular(
        self,
        feats: torch.Tensor,
        centers: torch.Tensor,
        intervals: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Non-parametric support-aware triangular interpolation.

        Args:
            feats: [B, C, N] token features.
            centers: [B, N] original frame centers in [0, 767].
            intervals: [B, N, 2] (start, end) timestamps.
            masses: [B, N] support masses or None.
            valid_mask: [B, N] boolean valid token mask or None.
        Returns:
            interp: [B, C, 384] interpolated feature grid.
        """
        B, C, N = feats.shape
        M = self.target_grid_size
        for name, value in (
            ("recovery.feats", feats),
            ("recovery.centers", centers),
            ("recovery.intervals", intervals),
        ):
            assert_finite_tensor(value, name)
        if masses is not None:
            assert_finite_tensor(masses, "recovery.masses")
            if bool((masses < 0).any().item()):
                raise ValueError("recovery.masses must be non-negative")
        if valid_mask is not None:
            assert_finite_tensor(valid_mask, "recovery.valid_mask")
        grid = self.target_grid.view(1, 1, M)  # [1, 1, M]

        centers_3d = centers.unsqueeze(-1)  # [B, N, 1]
        # Half width: max(0.5 * (end - start), 1.0)
        half_width = (0.5 * (intervals[..., 1] - intervals[..., 0])).clamp_min(1.0).unsqueeze(-1)  # [B, N, 1]

        # Distance: |grid_j - center_i|: [B, N, M]
        dist = torch.abs(grid - centers_3d)

        # Triangular weights: max(0, 1 - dist / half_width)
        weights = F.relu(1.0 - dist / half_width)  # [B, N, M]

        if masses is not None:
            weights = weights * masses.unsqueeze(-1)  # [B, N, M]

        if valid_mask is not None:
            weights = weights * valid_mask.unsqueeze(-1).to(dtype=weights.dtype)

        # Weight sum per grid point: [B, 1, M]
        weight_sum = weights.sum(dim=1, keepdim=True)  # [B, 1, M]
        assert_finite_tensor(weights, "recovery.weights")
        assert_finite_tensor(weight_sum, "recovery.weight_sum")

        # Weighted feature sum: [B, C, M] = [B, C, N] @ [B, N, M]
        weighted_feats = torch.bmm(feats, weights)  # [B, C, M]

        # Safe normalization
        safe_mask = weight_sum > 1e-6
        interp = weighted_feats / weight_sum.clamp_min(1e-6)
        assert_finite_tensor(interp, "recovery.interpolated")

        # Fallback to nearest neighbor for any unhit grid points
        if not bool(safe_mask.all().item()):
            # Use the nearest valid support token.  An all-masked sample has no
            # evidence and therefore receives an explicit zero reconstruction.
            nearest_feats = torch.zeros_like(interp)
            for batch_idx in range(B):
                if valid_mask is None:
                    valid_idx = torch.arange(N, device=feats.device)
                else:
                    valid_idx = torch.nonzero(valid_mask[batch_idx].bool(), as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue
                nearest_local = dist[batch_idx, valid_idx].argmin(dim=0)
                nearest_idx = valid_idx[nearest_local]
                nearest_feats[batch_idx] = feats[batch_idx, :, nearest_idx]
            interp = torch.where(safe_mask.expand(-1, C, -1), interp, nearest_feats)
        assert_finite_tensor(interp, "recovery.interpolated_fallback")

        return interp

    def forward(
        self,
        feats: torch.Tensor,
        centers: torch.Tensor,
        intervals: torch.Tensor,
        masses: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
        dense_valid_len: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass reconstructing features to [B, C, 384].

        Args:
            feats: [B, C, N]
            centers: [B, N]
            intervals: [B, N, 2]
            masses: [B, N] or None
            valid_mask: [B, N] or None
        """
        if not self.enabled:
            # If disabled (e.g. NO_RECOVERY arm), return feats directly
            return feats

        # Stage 1: Non-parametric triangular scatter
        interp = self.scatter_triangular(feats, centers, intervals, masses, valid_mask)  # [B, C, 384]

        # Stage 2: Residual refinement
        res = self.pwconv(self.act(self.dwconv(interp)))
        out = interp + self.residual_gate * res
        if dense_valid_len is not None:
            valid_len = dense_valid_len.to(device=out.device).reshape(-1)
            if valid_len.numel() != out.shape[0]:
                raise ValueError("dense_valid_len must contain one value per sample")
            valid_grid = torch.ceil(
                valid_len.float() * float(self.target_grid_size) / float(self.original_window_size)
            ).clamp(min=0, max=self.target_grid_size).long()
            grid_mask = torch.arange(self.target_grid_size, device=out.device).view(1, -1) < valid_grid[:, None]
            out = out * grid_mask.to(dtype=out.dtype).unsqueeze(1)
        assert_finite_tensor(out, "recovery.output")
        return out
