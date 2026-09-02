"""Bounded tubelet interval adapter and continuous timestamp conditioner for DUCA evidence recovery."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule
from opentad.models.utils.numerics import assert_finite_tensor


class BoundedTubeletIntervalAdapter(BaseModule):
    """Bounded tubelet interval adapter for VideoMAE 3D patch embedding.

    Performs exact algebraic decomposition of the temporal kernel over 2-frame tubelets:
        X_mean = (X_0 + X_1) / 2
        X_diff = (X_1 - X_0) / 2
        W_mean = W_0 + W_1
        W_diff = W_1 - W_0
        Y = W_mean * X_mean + g(z) * W_diff * X_diff + bias

    When g(z) = 1.0, Y = W_0 * X_0 + W_1 * X_1 + bias identically.
    Condition function:
        g(z) = 1.0 + 0.5 * tanh(MLP(z)) in [0.5, 1.5]
    where MLP last layer is zero-initialized.
    """

    def __init__(
        self,
        embed_dims: int = 384,
        in_channels: int = 3,
        patch_size: int = 16,
        tubelet_size: int = 2,
        mlp_hidden_dim: int = 64,
        reference_delta_t: float = 1.0,
        enabled: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.reference_delta_t = float(reference_delta_t)
        self.enabled = bool(enabled)

        # Condition MLP: input z has 3 dims: [log(dt/dt_ref), local_density, normalized_support_width]
        self.condition_mlp = nn.Sequential(
            nn.Linear(3, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1),
        )
        # Zero-initialize the last linear layer so g(z) == 1.0 at initialization
        nn.init.zeros_(self.condition_mlp[-1].weight)
        nn.init.zeros_(self.condition_mlp[-1].bias)

    def compute_g(self, z: torch.Tensor) -> torch.Tensor:
        """Compute condition factor g(z) in range [0.5, 1.5].

        Args:
            z: [B, T_tubelet, 3] or broadcastable tensor of condition features.
        Returns:
            g: [B, T_tubelet, 1, 1, 1] scaling factor for W_diff * X_diff.
        """
        assert_finite_tensor(z, "interval.z_condition")
        if not self.enabled:
            return torch.ones(
                (*z.shape[:-1], 1, 1, 1),
                dtype=z.dtype,
                device=z.device,
            )
        mlp_out = self.condition_mlp(z.float())  # [..., 1]
        g = 1.0 + 0.5 * torch.tanh(mlp_out)
        assert_finite_tensor(g, "interval.condition_factor")
        return g.unsqueeze(-1).unsqueeze(-1).to(dtype=z.dtype)

    def forward_tubelet(
        self,
        x: torch.Tensor,
        weight_3d: torch.Tensor,
        bias_3d: Optional[torch.Tensor],
        stride_spatial: int,
        padding_spatial: int,
        z_condition: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply bounded tubelet convolution.

        Args:
            x: [B, C_in, T_frames, H, W] where T_frames is even.
            weight_3d: [C_out, C_in, 2, kH, kW]
            bias_3d: [C_out] or None
            stride_spatial: spatial stride (e.g. 16)
            padding_spatial: spatial padding (e.g. 0)
            z_condition: [B, T_tubelet, 3] condition features or None.
        """
        B, C_in, T_frames, H, W = x.shape

        if not self.enabled or z_condition is None:
            # Standard conv3d
            return F.conv3d(
                x,
                weight_3d,
                bias=bias_3d,
                stride=(self.tubelet_size, stride_spatial, stride_spatial),
                padding=(0, padding_spatial, padding_spatial),
            )

        if self.tubelet_size != 2 or weight_3d.shape[2] != 2:
            raise ValueError(
                "bounded interval adapter exact decomposition currently supports only tubelet_size=2 "
                f"and temporal kernel size 2, got tubelet_size={self.tubelet_size}, kernel={weight_3d.shape[2]}"
            )
        if T_frames % self.tubelet_size != 0:
            raise ValueError(
                f"input temporal length {T_frames} must be divisible by tubelet_size={self.tubelet_size}"
            )

        T_tubelet = T_frames // self.tubelet_size

        # Decompose input frames: X0 and X1
        x_reshaped = x.view(B, C_in, T_tubelet, 2, H, W)
        X0 = x_reshaped[:, :, :, 0]  # [B, C_in, T_tubelet, H, W]
        X1 = x_reshaped[:, :, :, 1]  # [B, C_in, T_tubelet, H, W]

        X_mean = 0.5 * (X0 + X1)
        X_diff = 0.5 * (X1 - X0)

        # Decompose weights W0 and W1 along temporal dim (dim 2 of weight)
        W0 = weight_3d[:, :, 0]  # [C_out, C_in, kH, kW]
        W1 = weight_3d[:, :, 1]  # [C_out, C_in, kH, kW]

        W_mean = W0 + W1  # [C_out, C_in, kH, kW]
        W_diff = W1 - W0  # [C_out, C_in, kH, kW]

        # Convolve X_mean with W_mean in 2D per tubelet
        # Flatten (B * T_tubelet) into batch dim for efficient 2D conv
        BT = B * T_tubelet
        X_mean_2d = X_mean.permute(0, 2, 1, 3, 4).reshape(BT, C_in, H, W)
        X_diff_2d = X_diff.permute(0, 2, 1, 3, 4).reshape(BT, C_in, H, W)

        Y_mean = F.conv2d(
            X_mean_2d,
            W_mean,
            bias=bias_3d,
            stride=(stride_spatial, stride_spatial),
            padding=(padding_spatial, padding_spatial),
        )  # [BT, C_out, H_out, W_out]

        Y_diff = F.conv2d(
            X_diff_2d,
            W_diff,
            bias=None,
            stride=(stride_spatial, stride_spatial),
            padding=(padding_spatial, padding_spatial),
        )  # [BT, C_out, H_out, W_out]

        _, C_out, H_out, W_out = Y_mean.shape
        Y_mean = Y_mean.view(B, T_tubelet, C_out, H_out, W_out)
        Y_diff = Y_diff.view(B, T_tubelet, C_out, H_out, W_out)

        # Compute condition g(z): [B, T_tubelet, 1, 1, 1]
        g = self.compute_g(z_condition)  # [B, T_tubelet, 1, 1, 1]

        Y = Y_mean + g * Y_diff  # [B, T_tubelet, C_out, H_out, W_out]
        assert_finite_tensor(Y, "interval.output")
        # Transpose back to [B, C_out, T_tubelet, H_out, W_out]
        Y = Y.permute(0, 2, 1, 3, 4).contiguous()
        return Y


class ContinuousTimestampConditioner(BaseModule):
    """Zero-initialized Continuous Timestamp Conditioner for VideoMAE self-attention.

    Generates relative temporal attention bias from original continuous timestamps:
        delta_tau = tau_i - tau_j in [-1, 1]
        fourier_feat = [sin(2*pi*f*delta_tau), cos(2*pi*f*delta_tau)]
        bias = MLP(fourier_feat) with zero-initialized output layer.
    """

    def __init__(
        self,
        num_heads: int = 6,
        num_fourier_bands: int = 8,
        mlp_hidden_dim: int = 64,
        max_period: float = 1.0,
        enabled: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.num_fourier_bands = num_fourier_bands
        self.enabled = bool(enabled)

        # Fourier frequencies: geometric progression
        freqs = 2.0 ** torch.arange(num_fourier_bands, dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)

        in_dim = num_fourier_bands * 2
        self.bias_mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, num_heads),
        )
        # Zero-initialize the final projection so initial bias is strictly zero
        nn.init.zeros_(self.bias_mlp[-1].weight)
        nn.init.zeros_(self.bias_mlp[-1].bias)

    def forward(
        self,
        tubelet_timestamps: torch.Tensor,
        spatial_tokens_per_tubelet: int = 100,
    ) -> Optional[torch.Tensor]:
        """Compute relative attention bias tensor.

        Args:
            tubelet_timestamps: [B, T_tubelets] normalized timestamps in [0, 1].
            spatial_tokens_per_tubelet: number of spatial tokens per tubelet (H*W, e.g. 100).
        Returns:
            attn_bias: [B, num_heads, N, N] where N = T_tubelets * spatial_tokens_per_tubelet,
                       or None if disabled.
        """
        if not self.enabled or tubelet_timestamps is None:
            return None

        B, T = tubelet_timestamps.shape

        # Keep the conditioning path in float32 under AMP.  H65 replay can
        # contain highly non-uniform timestamps; allowing the learned bias to
        # grow in fp16 can overflow before the detector's finite-loss check.
        with torch.autocast(device_type=tubelet_timestamps.device.type, enabled=False):
            timestamps_f32 = tubelet_timestamps.float()
            delta_tau = timestamps_f32.unsqueeze(-1) - timestamps_f32.unsqueeze(-2)
            # Fourier expansion: [B, T, T, num_fourier_bands * 2]
            angles = delta_tau.unsqueeze(-1) * self.freqs.float().view(1, 1, 1, -1)
            fourier_feat = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

            # Bound the learned additive attention bias to a numerically safe
            # range while preserving the exact zero-initialized behavior.
            tubelet_bias = 4.0 * torch.tanh(self.bias_mlp(fourier_feat))
            tubelet_bias = tubelet_bias.permute(0, 3, 1, 2)

        # Expand across spatial tokens: each tubelet expands to S tokens
        S = spatial_tokens_per_tubelet
        # [B, num_heads, T, 1, T, 1] -> [B, num_heads, T, S, T, S] -> [B, num_heads, T*S, T*S]
        attn_bias = (
            tubelet_bias.unsqueeze(3)
            .unsqueeze(5)
            .expand(-1, -1, -1, S, -1, S)
            .reshape(B, self.num_heads, T * S, T * S)
            .contiguous()
        )
        return attn_bias
