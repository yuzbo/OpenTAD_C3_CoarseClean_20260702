"""FoveaScout: low-resolution dense temporal observation front-end.

The scout sees the dense video once, before the heavy detector.  It combines a
lightweight 2D stem (32x32 RGB observations) with a masked 1D temporal stack,
so it remains cheap enough for dense training while producing a
``[B, T, D]`` temporal memory for the Query-Bridge.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FoveaScout(nn.Module):
    """2D stem + masked dilated 1D temporal encoder."""

    def __init__(
        self,
        in_dim: int = 3 * 32 * 32,
        hidden_dim: int = 96,
        temporal_layers: int = 4,
        kernel_size: int = 5,
        dilations: Tuple[int, ...] = (1, 2, 4, 8),
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.temporal_layers = int(temporal_layers)
        if len(dilations) != self.temporal_layers:
            raise ValueError("scout dilations must have one entry per temporal layer")

        # 2D stem: 3x32x32 -> 32x4x4 -> hidden_dim.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.spatial_proj = nn.Linear(32 * 4 * 4, self.hidden_dim)

        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=int(kernel_size),
                    padding=int(dilation * (kernel_size - 1) // 2),
                    dilation=int(dilation),
                )
                for dilation in dilations
            ]
        )
        self.temporal_norms = nn.ModuleList(
            [nn.LayerNorm(self.hidden_dim) for _ in range(self.temporal_layers)]
        )
        self.dropout = nn.Dropout(float(dropout))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.stem:
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.spatial_proj.weight)
        nn.init.zeros_(self.spatial_proj.bias)
        for conv in self.temporal_convs:
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="relu")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)
        for norm in self.temporal_norms:
            nn.init.ones_(norm.weight)
            nn.init.zeros_(norm.bias)

    def forward(self, observations: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Encode dense 32x32 observations.

        Args:
            observations: ``[B, 3, T, 32, 32]`` dense low-resolution frames.
            valid: ``[B, T]`` bool prefix mask (True = valid).

        Returns:
            ``[B, T, D]`` scout memory.  Invalid trailing positions are zeroed.
        """
        if observations.ndim != 5:
            raise ValueError("FoveaScout expects [B,3,T,32,32] observations")
        batch, _, length, _, _ = observations.shape
        if valid.ndim != 2 or valid.shape[0] != batch or valid.shape[1] != length:
            raise ValueError("FoveaScout valid mask must be [B,T]")
        if not valid.dtype == torch.bool:
            valid = valid.bool()

        flat = observations.permute(0, 2, 1, 3, 4).reshape(batch * length, 3, observations.shape[-2], observations.shape[-1])
        stemmed = self.stem(flat)  # [B*T, 32, 4, 4]
        stemmed = stemmed.flatten(1)
        z = self.spatial_proj(stemmed).reshape(batch, length, self.hidden_dim)  # [B,T,D]

        mask = valid.unsqueeze(2)  # [B,T,1]
        z = z * mask.to(z.dtype)
        for conv, norm in zip(self.temporal_convs, self.temporal_norms):
            x = z.transpose(1, 2)  # [B,D,T]
            residual = x
            x = conv(x)
            x = x.transpose(1, 2)  # [B,T,D]
            x = norm(x)
            x = F.gelu(x)
            x = self.dropout(x)
            z = z + x
            z = z * mask.to(z.dtype)
        return z.contiguous()
