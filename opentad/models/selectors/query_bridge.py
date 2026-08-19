"""Query-Bridge: a small query bank that reads the FoveaScout memory.

The bridge produces two things:

* ``contribution``: a class-agnostic ``[B, M, T]`` attention heatmap used to
  supervise and select frames, and
* ``query_memory`` (``Q1``): the updated ``[B, M, D]`` query state.

``Q1`` is selector-internal.  It never enters the heavy detector and never
leaks into inference-time selection; it only supplies the query context used
by the three manual score heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
import torch.nn as nn


@dataclass
class QueryBridgeOutput:
    contribution: torch.Tensor
    query_memory: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class QueryBank(nn.Module):
    """Learned class-agnostic query tokens ``Q0``."""

    def __init__(self, hidden_dim: int, num_queries: int = 4) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.q0 = nn.Parameter(torch.randn(1, self.num_queries, self.hidden_dim) * 0.02)

    def forward(self, batch: int) -> torch.Tensor:
        return self.q0.expand(batch, self.num_queries, self.hidden_dim)


class LightQueryDecoder(nn.Module):
    """One lightweight transformer decoder layer (query self-attn + cross-attn + FFN)."""

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.10) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Dropout(float(dropout)),
        )
        self.norm1 = nn.LayerNorm(self.hidden_dim)
        self.norm2 = nn.LayerNorm(self.hidden_dim)
        self.norm3 = nn.LayerNorm(self.hidden_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, query: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Cross-attend ``query`` to masked temporal ``memory``.

        Args:
            query: ``[B, M, D]``.
            memory: ``[B, T, D]``.
            valid: ``[B, T]`` bool mask; True positions are kept.

        Returns:
            updated ``[B, M, D]`` queries.
        """
        key_padding_mask = ~valid.bool()
        q = query
        self_out, _ = self.self_attn(q, q, q)
        q = self.norm1(q + self_out)
        cross_out, _ = self.cross_attn(
            q,
            memory,
            memory,
            key_padding_mask=key_padding_mask,
        )
        q = self.norm2(q + cross_out)
        q = self.norm3(q + self.ffn(q))
        return q


class QueryBridgeWithDecoder(nn.Module):
    """QueryBank -> stacked LightQueryDecoder -> contribution heatmap."""

    def __init__(
        self,
        hidden_dim: int = 96,
        num_queries: int = 4,
        num_decoder_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.query_bank = QueryBank(self.hidden_dim, self.num_queries)
        self.decoders = nn.ModuleList(
            [
                LightQueryDecoder(self.hidden_dim, num_heads=int(num_heads), dropout=float(dropout))
                for _ in range(int(num_decoder_layers))
            ]
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.query_bank.q0, mean=0.0, std=0.02)

    def forward(self, z: torch.Tensor, valid: torch.Tensor) -> QueryBridgeOutput:
        """Build the contribution heatmap and internal query memory.

        Args:
            z: ``[B, T, D]`` scout memory.
            valid: ``[B, T]`` bool prefix mask.

        Returns:
            :class:`QueryBridgeOutput` with ``contribution`` ``[B, M, T]``
            (softmax over valid time positions for each query) and
            ``query_memory`` ``[B, M, D]``.
        """
        if z.ndim != 3:
            raise ValueError("QueryBridge expects [B,T,D] scout memory")
        batch, length, hidden = z.shape
        if valid.ndim != 2 or valid.shape[0] != batch or valid.shape[1] != length:
            raise ValueError("QueryBridge valid mask must be [B,T]")
        valid_bool = valid.bool()
        q = self.query_bank(batch)
        for decoder in self.decoders:
            q = decoder(q, z, valid_bool)
        logits = torch.einsum("bmd,btd->bmt", q, z) / math.sqrt(float(hidden))
        contribution = logits.masked_fill(~valid_bool.unsqueeze(1), -torch.inf)
        contribution = torch.softmax(contribution, dim=-1)
        contribution = contribution.masked_fill(~valid_bool.unsqueeze(1), 0.0)
        return QueryBridgeOutput(
            contribution=contribution,
            query_memory=q.contiguous(),
            metadata={
                "query_memory_is_selector_internal": True,
                "query_memory_enters_heavy_detector": False,
                "contribution_axis": "query,temporal",
                "num_queries": self.num_queries,
            },
        )
