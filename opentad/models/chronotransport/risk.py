from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .actions import ChronoAction


class ScheduleQuantileRiskPredictor(nn.Module):
    """Schedule-conditioned deploy-visible upper-regret predictor.

    Signals have shape ``[B, C, G, F]``. Candidate actions may use shape
    ``[K, C, G]`` or ``[B, K, C, G]``. The model returns one non-negative
    aggregate risk per window and candidate.
    """

    def __init__(
        self,
        signal_dims: int,
        num_groups: int,
        hidden_dims: int = 64,
        quantile: float = 0.9,
        action_embed_dims: int = 8,
    ) -> None:
        super().__init__()
        self.signal_dims = int(signal_dims)
        self.num_groups = int(num_groups)
        self.hidden_dims = int(hidden_dims)
        self.quantile = float(quantile)
        if self.signal_dims <= 0 or self.num_groups <= 0 or self.hidden_dims <= 0:
            raise ValueError("risk predictor dimensions must be positive")
        if not 0.0 < self.quantile < 1.0:
            raise ValueError("quantile must lie in (0, 1)")

        self.action_embedding = nn.Embedding(len(ChronoAction), int(action_embed_dims))
        self.group_embedding = nn.Embedding(self.num_groups, int(action_embed_dims))
        input_dims = self.signal_dims + 2 * int(action_embed_dims) + 1
        self.cell_mlp = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.GELU(),
            nn.Linear(self.hidden_dims, 1),
        )
        self.register_buffer("calibration_offset", torch.tensor(0.0), persistent=True)
        self.register_buffer("_debug_action_risk", torch.empty(0), persistent=False)

    def set_debug_action_risk(self, *, recompute: float, transport: float, hold: float) -> None:
        values = torch.tensor([recompute, transport, hold], dtype=torch.float32)
        if not torch.isfinite(values).all() or bool((values < 0).any().item()):
            raise ValueError("debug action risks must be finite and non-negative")
        self._debug_action_risk = values.to(device=self.calibration_offset.device)

    def clear_debug_action_risk(self) -> None:
        self._debug_action_risk = torch.empty(0, device=self.calibration_offset.device)

    def set_calibration_offset(self, value: float | Tensor) -> None:
        value = torch.as_tensor(value, dtype=self.calibration_offset.dtype, device=self.calibration_offset.device)
        if value.numel() != 1 or not torch.isfinite(value).all() or float(value.item()) < 0.0:
            raise ValueError("calibration offset must be one finite non-negative scalar")
        self.calibration_offset.copy_(value.reshape(()))

    @staticmethod
    def conformal_offset(
        prediction: Tensor,
        target: Tensor,
        *,
        coverage: float = 0.9,
    ) -> Tensor:
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have identical shape")
        coverage = float(coverage)
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        residual = (target - prediction).flatten().clamp_min(0.0)
        if residual.numel() == 0:
            raise ValueError("calibration residual set must be non-empty")
        # Finite-sample split-conformal quantile.
        rank = min(residual.numel(), int(math.ceil((residual.numel() + 1) * coverage)))
        return residual.kthvalue(rank).values


    @staticmethod
    def candidate_age(actions: Tensor) -> Tensor:
        """Return time since last RECOMPUTE for every candidate cell.

        Accepted shapes are ``[K,C,G]`` and ``[B,K,C,G]``; the result appends
        one feature dimension. Age is a deploy-visible deterministic function
        of the proposed schedule, not a teacher signal.
        """

        if actions.ndim not in (3, 4):
            raise ValueError("candidate actions must have shape [K,C,G] or [B,K,C,G]")
        actions = actions.to(dtype=torch.long)
        age = torch.zeros_like(actions, dtype=torch.float32)
        running = torch.zeros_like(actions[..., 0, :], dtype=torch.float32)
        for chunk_index in range(int(actions.shape[-2])):
            current = actions[..., chunk_index, :]
            running = torch.where(
                current == int(ChronoAction.RECOMPUTE),
                torch.zeros_like(running),
                running + 1.0,
            )
            age[..., chunk_index, :] = running
        return age.unsqueeze(-1)

    def _normalize_actions(self, signals: Tensor, actions: Tensor) -> Tensor:
        batch_size, num_chunks, num_groups, _ = signals.shape
        if actions.ndim == 3:
            if tuple(actions.shape[1:]) != (num_chunks, num_groups):
                raise ValueError("candidate action shape does not match signals")
            actions = actions.unsqueeze(0).expand(batch_size, -1, -1, -1)
        elif actions.ndim == 4:
            if int(actions.shape[0]) == 1 and batch_size != 1:
                actions = actions.expand(batch_size, -1, -1, -1)
            if int(actions.shape[0]) != batch_size or tuple(actions.shape[2:]) != (num_chunks, num_groups):
                raise ValueError("batched candidate action shape does not match signals")
        else:
            raise ValueError("candidate actions must have shape [K,C,G] or [B,K,C,G]")
        return actions.to(device=signals.device, dtype=torch.long)

    def forward(self, signals: Tensor, actions: Tensor) -> Tensor:
        if signals.ndim != 4:
            raise ValueError("risk signals must have shape [B, C, G, F]")
        if int(signals.shape[2]) != self.num_groups or int(signals.shape[3]) != self.signal_dims:
            raise ValueError("risk signal shape does not match predictor configuration")
        actions = self._normalize_actions(signals, actions)
        batch_size, candidates, num_chunks, num_groups = actions.shape
        if bool(((actions < 0) | (actions >= len(ChronoAction))).any().item()):
            raise ValueError("candidate actions contain invalid ChronoAction values")

        if self._debug_action_risk.numel() == len(ChronoAction):
            per_cell = self._debug_action_risk.to(device=signals.device, dtype=signals.dtype)[actions]
        else:
            expanded_signals = signals.unsqueeze(1).expand(-1, candidates, -1, -1, -1)
            action_embed = self.action_embedding(actions)
            group_ids = torch.arange(num_groups, device=signals.device)
            group_embed = self.group_embedding(group_ids).view(1, 1, 1, num_groups, -1)
            group_embed = group_embed.expand(batch_size, candidates, num_chunks, -1, -1)
            age = self.candidate_age(actions).to(device=signals.device, dtype=signals.dtype)
            normalized_age = age / (1.0 + age)
            features = torch.cat((expanded_signals, action_embed, group_embed, normalized_age), dim=-1)
            per_cell = F.softplus(self.cell_mlp(features).squeeze(-1))

        aggregate = per_cell.sum(dim=(-1, -2))
        return aggregate + self.calibration_offset.to(dtype=aggregate.dtype)
