import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BRICKS


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    return torch.cat((-x2, x1), dim=-1)


@BRICKS.register_module()
class TimeAlignedRoPE(nn.Module):
    """Continuous Time-Aligned Rotary Position Embedding (TARoPE).

    Inspired by Qwen2.5-VL and VideoRoPE, this module generalizes standard 1D RoPE
    from discrete equidistant integer indices m in {0, 1, ..., T-1} to continuous physical
    timestamps tau in R^{B x T}.

    Mathematical Foundation:
      Given rotary inverse frequencies theta_j = base^{-2j/d} (j = 0, ..., d/2 - 1):
      For continuous timestamp tau_i at token i:
        phi_j(tau_i) = tau_i * theta_j
      The query/key transformation is:
        R(tau_i) x_i = x_i * cos(phi(tau_i)) + rotate_half(x_i) * sin(phi(tau_i))

      Inner-product shift-invariance property:
        < R(tau_q) q, R(tau_k) k > = f(q, k, tau_q - tau_k)
      The attention score depends exclusively on the physical elapsed time Delta tau = tau_q - tau_k,
      natively handling non-uniform temporal spacing and varying video frame rates.

    Args:
        dim (int): Feature dimension per attention head (must be even).
        base (float): Base for the geometric progression of frequencies (default: 10000.0).
        time_scale (float): Scaling factor to normalize timestamp magnitudes (default: 1.0).
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        time_scale: float = 1.0,
    ):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim must be even for rotary embedding, got {dim}")
        self.dim = int(dim)
        self.base = float(base)
        self.time_scale = float(time_scale)

        # Inverse frequencies: theta_j = base^(-2j/dim) for j in [0, dim/2 - 1]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward_frequencies(
        self,
        temporal_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cos and sin frequency tensors for continuous timestamps.

        Args:
            temporal_positions: Tensor of shape [B, T] or [T] containing continuous timestamps.

        Returns:
            cos: Tensor of shape [B, 1, T, dim] or [1, 1, T, dim]
            sin: Tensor of shape [B, 1, T, dim] or [1, 1, T, dim]
        """
        if temporal_positions.dim() == 1:
            # [T] -> [1, T]
            timestamps = temporal_positions.unsqueeze(0)
        else:
            timestamps = temporal_positions

        B, T = timestamps.shape
        device = timestamps.device
        dtype = timestamps.dtype

        # Ensure inv_freq is on the correct device and dtype
        inv_freq = self.inv_freq.to(device=device, dtype=torch.float32)

        # Scaled timestamps: [B, T, 1]
        t = (timestamps.float() * self.time_scale).unsqueeze(-1)  # [B, T, 1]
        # Outer product: phi = t * inv_freq -> [B, T, dim // 2]
        freqs = torch.matmul(t, inv_freq.unsqueeze(0))  # [B, T, dim // 2]

        # Duplicate along last dimension to match [B, T, dim]
        emb = torch.cat((freqs, freqs), dim=-1)  # [B, T, dim]

        # Reshape to [B, 1, T, dim] for broadcasting across attention heads
        cos = emb.cos().unsqueeze(1)  # [B, 1, T, dim]
        sin = emb.sin().unsqueeze(1)  # [B, 1, T, dim]
        return cos, sin

    def apply_rope(
        self,
        x: torch.Tensor,
        temporal_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Apply continuous time-aligned RoPE to input tensor x.

        Args:
            x: Tensor of shape [B, num_heads, T, dim] or [B, T, dim] or [B, T, num_heads, dim].
            temporal_positions: Tensor of shape [B, T] containing physical timestamps.

        Returns:
            x_rot: Tensor of the same shape and dtype as x.
        """
        cos, sin = self.forward_frequencies(temporal_positions)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)

        if x.dim() == 4:
            if x.shape[1] != cos.shape[1] and x.shape[2] == cos.shape[2]:
                # Shape is [B, num_heads, T, dim]
                return (x * cos) + (_rotate_half(x) * sin)
            elif x.shape[1] == cos.shape[2] and x.shape[3] == cos.shape[3]:
                # Shape is [B, T, num_heads, dim] -> cos needs [B, T, 1, dim]
                cos_t = cos.permute(0, 2, 1, 3)
                sin_t = sin.permute(0, 2, 1, 3)
                return (x * cos_t) + (_rotate_half(x) * sin_t)
            else:
                return (x * cos) + (_rotate_half(x) * sin)
        elif x.dim() == 3:
            # Shape is [B, T, dim]
            cos_3d = cos.squeeze(1)  # [B, T, dim]
            sin_3d = sin.squeeze(1)  # [B, T, dim]
            return (x * cos_3d) + (_rotate_half(x) * sin_3d)
        else:
            raise ValueError(f"Expected x to have 3 or 4 dimensions, got {x.dim()}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        temporal_positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Time-Aligned RoPE jointly to query and key tensors."""
        return self.apply_rope(q, temporal_positions), self.apply_rope(k, temporal_positions)


@BRICKS.register_module()
class TimeSpacingPositionalEncoding(nn.Module):
    """Time-Spacing and Continuous Timestamp Fourier Positional Encoding (TSPE).

    Generates multi-scale Fourier positional representations incorporating both:
      1. Absolute continuous physical timestamp tau_i
      2. Local irregular sampling step delta_t[i] = tau_{i+1} - tau_{i-1}

    Args:
        out_channels (int): Output feature dimension for positional encoding.
        num_frequencies (int): Number of multi-scale sinusoidal frequency bands (default: 32).
        base (float): Frequency scaling factor (default: 10000.0).
    """

    def __init__(
        self,
        out_channels: int,
        num_frequencies: int = 32,
        base: float = 10000.0,
    ):
        super().__init__()
        self.out_channels = int(out_channels)
        self.num_frequencies = int(num_frequencies)

        # Bands: omega_k = base^(-k / num_freqs)
        freq_bands = 1.0 / (
            base ** (torch.arange(0, num_frequencies).float() / num_frequencies)
        )
        self.register_buffer("freq_bands", freq_bands, persistent=False)

        # 2 components (tau and delta_t) * 2 (sin and cos) * num_frequencies = 4 * num_frequencies
        in_dim = 4 * num_frequencies
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_channels),
            nn.LayerNorm(out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
        )

    def forward(
        self,
        temporal_positions: torch.Tensor,
        delta_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute continuous temporal-spacing positional embeddings.

        Args:
            temporal_positions: Tensor of shape [B, T] with continuous timestamps.
            delta_t: Optional tensor of shape [B, T] with local sampling intervals.

        Returns:
            pe: Tensor of shape [B, T, out_channels] or [B, out_channels, T]
        """
        B, T = temporal_positions.shape
        device = temporal_positions.device

        if delta_t is None:
            # Auto-calculate symmetric difference delta_t
            dt_fwd = torch.diff(temporal_positions, dim=-1, prepend=temporal_positions[:, :1])
            dt_bwd = torch.diff(temporal_positions, dim=-1, append=temporal_positions[:, -1:])
            dt = 0.5 * (dt_fwd.abs() + dt_bwd.abs()).clamp_min(1e-4)
        else:
            dt = delta_t.clamp_min(1e-4)

        freqs = self.freq_bands.to(device=device, dtype=torch.float32)  # [num_freq]

        # Phase projections: [B, T, num_freq]
        phi_tau = temporal_positions.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
        phi_dt = dt.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)

        # Sinusoidal features: [B, T, 4 * num_freq]
        features = torch.cat(
            (
                phi_tau.sin(),
                phi_tau.cos(),
                phi_dt.sin(),
                phi_dt.cos(),
            ),
            dim=-1,
        )

        pe = self.proj(features)  # [B, T, out_channels]
        return pe
