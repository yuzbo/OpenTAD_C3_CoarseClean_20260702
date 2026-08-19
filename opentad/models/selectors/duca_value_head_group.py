from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class DucaValueOutput:
    value: torch.Tensor
    provenance: dict[str, Any] = field(default_factory=dict)


class DucaValueHeadGroup(nn.Module):
    """Single signed utility residual V(t) with a query cross-attention portal."""

    def __init__(self, hidden_dim: int, num_queries: int = 4, num_heads: int = 4, dropout: float = 0.0, init_scale: float = 0.02):
        super().__init__()
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if int(num_queries) != 4:
            raise ValueError("DUCA-UVT freezes four class-agnostic task queries")
        if int(num_heads) <= 0 or int(hidden_dim) % int(num_heads) != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        self.hidden_dim = int(hidden_dim)
        self.queries = nn.Parameter(torch.randn(int(num_queries), self.hidden_dim) * float(init_scale))
        self.cross_attention = nn.MultiheadAttention(self.hidden_dim, int(num_heads), batch_first=True, dropout=float(dropout))
        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.activation = nn.GELU()
        self.value_head = nn.Linear(self.hidden_dim, 1)

    def forward(self, value_evidence: torch.Tensor, valid: torch.Tensor) -> DucaValueOutput:
        if not torch.is_tensor(value_evidence) or value_evidence.ndim != 3:
            raise ValueError(f"value_evidence must be [B,T,D], got {tuple(value_evidence.shape)}")
        batch, length, dim = value_evidence.shape
        if dim != self.hidden_dim:
            raise ValueError(f"value_evidence dimension {dim} != hidden_dim {self.hidden_dim}")
        if tuple(valid.shape) != (batch, length):
            raise ValueError("valid mask must match value_evidence")
        valid_bool = valid.bool()
        if not bool(valid_bool.any().item()):
            raise ValueError("value portal requires at least one valid position")
        queries = self.queries.unsqueeze(0).expand(batch, -1, -1)
        query_context, _ = self.cross_attention(queries, value_evidence, value_evidence, key_padding_mask=~valid_bool)
        global_value = self.activation(self.global_proj(query_context.mean(dim=1)))
        modulated = value_evidence + global_value[:, None, :]
        modulated = modulated.masked_fill(~valid_bool.unsqueeze(-1), 0.0)
        value = self.value_head(modulated).squeeze(-1).masked_fill(~valid_bool, 0.0)
        provenance = {
            "uses_dense_detector_teacher": False,
            "uses_self_ema_teacher": False,
            "uses_gt_geometry_target": False,
            "uses_detector_feedback_at_inference": False,
            "uses_raw_prediction_cache": False,
        }
        return DucaValueOutput(value=value, provenance=provenance)
