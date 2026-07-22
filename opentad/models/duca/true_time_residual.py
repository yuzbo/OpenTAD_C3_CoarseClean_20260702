from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn


class TrueTimeFeatureResidual(nn.Module):
    """Inject selected-frame timing as a zero-initialized feature residual."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 64,
        descriptor_mode: str = "actual",
    ) -> None:
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.descriptor_mode = str(descriptor_mode)
        if self.feature_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("true-time residual dimensions must be positive")
        if self.descriptor_mode not in {"actual", "reversed", "constant"}:
            raise ValueError("descriptor_mode must be actual, reversed, or constant")
        self.projector = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.projector[-1].weight)
        nn.init.zeros_(self.projector[-1].bias)
        self.last_summary: dict[str, Any] = {}

    @staticmethod
    def _row_descriptors(
        positions: Sequence[int],
        dense_valid_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        pos = torch.as_tensor(list(positions), device=device, dtype=torch.float32)
        if pos.ndim != 1 or pos.numel() <= 0:
            raise ValueError("true-time residual requires at least one selected position")
        if torch.any(pos < 0) or torch.any(pos[1:] <= pos[:-1]):
            raise ValueError("selected true-time positions must be strictly increasing")
        valid_len = int(dense_valid_len)
        if valid_len <= int(pos[-1].item()):
            raise ValueError("truetime_dense_valid_len must exceed the last selected position")
        denom = float(max(valid_len - 1, 1))
        left = torch.empty_like(pos)
        right = torch.empty_like(pos)
        left[0] = pos[0] + 1.0
        left[1:] = pos[1:] - pos[:-1]
        right[-1] = float(valid_len) - pos[-1]
        right[:-1] = pos[1:] - pos[:-1]
        asymmetry = (right - left) / (right + left).clamp_min(1.0)
        return torch.stack((pos / denom, left / denom, right / denom, asymmetry), dim=-1).to(dtype=dtype)

    def _descriptors(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("true-time residual expects features [B,C,T]")
        if masks.shape != (features.shape[0], features.shape[2]):
            raise ValueError("true-time residual masks must be [B,T]")
        if len(metas) != int(features.shape[0]):
            raise ValueError("true-time residual metadata must match batch size")
        rows = features.new_zeros((features.shape[0], features.shape[2], 4))
        for batch_idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError("true-time residual metadata rows must be mappings")
            active = int(masks[batch_idx].long().sum().item())
            positions = meta.get("selected_axis_to_true_time_dense_index")
            if not isinstance(positions, Sequence) or isinstance(positions, (str, bytes)):
                raise ValueError("missing selected_axis_to_true_time_dense_index metadata")
            if len(positions) != active:
                raise ValueError("selected true-time positions must match active detector tokens")
            valid_len = int(meta.get("truetime_dense_valid_len", 0))
            row = self._row_descriptors(
                [int(value) for value in positions],
                valid_len,
                device=features.device,
                dtype=features.dtype,
            )
            if self.descriptor_mode == "reversed":
                row = torch.flip(row, dims=(0,))
            elif self.descriptor_mode == "constant":
                row = row.mean(dim=0, keepdim=True).expand_as(row)
            rows[batch_idx, :active] = row
        return rows

    def forward(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if int(features.shape[1]) != self.feature_dim:
            raise ValueError(
                f"true-time residual expected feature_dim={self.feature_dim}, got {features.shape[1]}"
            )
        valid = masks.to(device=features.device, dtype=torch.bool)
        descriptors = self._descriptors(features, valid, metas)
        residual = self.projector(descriptors).transpose(1, 2)
        residual = residual.masked_fill(~valid[:, None, :], 0.0)
        self.last_summary = {
            "descriptor_mode": self.descriptor_mode,
            "feature_dim": self.feature_dim,
            "active_tokens": [int(value) for value in valid.long().sum(dim=1).detach().cpu().tolist()],
            "residual_abs_max": float(residual.detach().abs().amax().cpu().item()),
        }
        return features + residual
