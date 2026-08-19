"""Three manual score branches + coarse proposal head for FoveaSampler.

The three branches are kept separate by design: saliency, boundary evidence and
uncertainty-gated query context.  They are fused as::

    s_t = saliency_t + boundary_edge_t + uncertainty_context_t

``boundary_edge`` is the softplus of ``max(start_logit, end_logit)`` and
``uncertainty_context = uncertainty * ||query_context||``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _valid_2d(tensor: torch.Tensor, valid: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.ndim != 2:
        raise ValueError(f"{name} must be [B,T]")
    return tensor


class FoveaHeads(nn.Module):
    """Saliency / boundary / uncertainty manual score heads."""

    def __init__(self, hidden_dim: int = 96, num_queries: int = 4) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_queries = int(num_queries)
        self.saliency_head = nn.Linear(self.hidden_dim, 1)
        self.boundary_head = nn.Linear(self.hidden_dim, 2)
        self.uncertainty_head = nn.Linear(self.hidden_dim, 1)
        self.context_gate = nn.Linear(self.hidden_dim + self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for linear in (self.saliency_head, self.boundary_head, self.uncertainty_head, self.context_gate):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        z: torch.Tensor,
        contribution: torch.Tensor,
        query_memory: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Fuse the three manual score branches.

        Args:
            z: ``[B,T,D]`` scout memory.
            contribution: ``[B,M,T]`` query-frame contribution heatmap.
            query_memory: ``[B,M,D]`` selector-internal ``Q1``.
            valid: ``[B,T]`` bool prefix mask.

        Returns:
            dict with ``saliency``, ``boundary_start``, ``boundary_end``,
            ``boundary_edge``, ``uncertainty``, ``uncertainty_context``,
            ``query_context`` and the fused ``frame_score``.
        """
        if valid.ndim != 2 or tuple(z.shape[:2]) != tuple(valid.shape):
            raise ValueError("FoveaHeads expects [B,T] valid")
        if contribution.ndim != 3 or contribution.shape[0] != z.shape[0] or contribution.shape[-1] != z.shape[1]:
            raise ValueError("FoveaHeads expects [B,M,T] contribution")
        if query_memory.ndim != 3 or query_memory.shape[0] != z.shape[0] or query_memory.shape[1] != contribution.shape[1]:
            raise ValueError("FoveaHeads query_memory must be [B,M,D]")

        saliency = self.saliency_head(z).squeeze(-1)  # [B,T]
        boundary_logits = self.boundary_head(z)  # [B,T,2]
        boundary_start = boundary_logits[..., 0]
        boundary_end = boundary_logits[..., 1]
        boundary_edge = F.softplus(torch.maximum(boundary_start, boundary_end))

        uncertainty_logits = self.uncertainty_head(z).squeeze(-1)
        uncertainty = torch.sigmoid(uncertainty_logits)

        query_context = torch.einsum("bmt,bmd->btd", contribution, query_memory)  # [B,T,D]
        context_gate_input = torch.cat([z, query_context], dim=-1)
        local_context = torch.tanh(self.context_gate(context_gate_input).squeeze(-1))
        local_context = local_context * query_context.norm(dim=-1).clamp_max(10.0)
        uncertainty_context = uncertainty * local_context

        frame_score = saliency + boundary_edge + uncertainty_context
        mask = valid.bool()
        frame_score = frame_score.masked_fill(~mask, -torch.inf)
        return {
            "saliency": saliency,
            "boundary_start": boundary_start,
            "boundary_end": boundary_end,
            "boundary_edge": boundary_edge,
            "uncertainty": uncertainty,
            "uncertainty_logits": uncertainty_logits,
            "uncertainty_context": uncertainty_context,
            "query_context": query_context,
            "frame_score": frame_score,
        }


class CoarseProposalHead(nn.Module):
    """Auxiliary coarse proposal head used only for ``L_coarse``."""

    def __init__(self, hidden_dim: int = 96) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.fuse = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.cls_head = nn.Linear(self.hidden_dim, 1)
        self.center_head = nn.Linear(self.hidden_dim, 1)
        self.width_head = nn.Linear(self.hidden_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for linear in (self.fuse, self.cls_head, self.center_head, self.width_head):
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)

    def forward(
        self,
        z: torch.Tensor,
        contribution: torch.Tensor,
        valid: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict coarse per-position actionness / center offset / width.

        These outputs only supervise the scout front-end and the bridge; they
        are never used by the sampler at inference.
        """
        if z.ndim != 3 or contribution.ndim != 3:
            raise ValueError("CoarseProposalHead expects z [B,T,D] and contribution [B,M,T]")
        query_context = torch.einsum("bmt,bmd->btd", contribution, z)
        # pool over query tokens, keep valid positions only
        valid_bool = valid.bool()
        query_pool = query_context.masked_fill(~valid_bool.unsqueeze(-1), 0.0)
        fused = torch.tanh(self.fuse(torch.cat([z, query_pool], dim=-1)))
        coarse_logits = self.cls_head(fused).squeeze(-1)
        coarse_center = torch.tanh(self.center_head(fused).squeeze(-1))
        coarse_width = F.softplus(self.width_head(fused).squeeze(-1))
        coarse_logits = coarse_logits.masked_fill(~valid_bool, -10.0)
        coarse_center = coarse_center.masked_fill(~valid_bool, 0.0)
        coarse_width = coarse_width.masked_fill(~valid_bool, 0.0)
        return {
            "coarse_logits": coarse_logits,
            "coarse_center": coarse_center,
            "coarse_width": coarse_width,
        }
