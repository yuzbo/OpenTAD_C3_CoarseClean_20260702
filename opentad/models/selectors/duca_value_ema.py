from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .duca_value_head_group import DucaValueHeadGroup


class DucaValueEMA(nn.Module):
    """Parameter-only EMA of the single DUCA value head."""

    def __init__(self, value_head: DucaValueHeadGroup, decay: float = 0.999):
        super().__init__()
        if not 0.0 < float(decay) < 1.0:
            raise ValueError("ema_decay must lie in (0,1)")
        self.decay = float(decay)
        self.ema_head = copy.deepcopy(value_head)
        for param in self.ema_head.parameters():
            param.requires_grad = False
        self.ema_head.eval()
        self._copy_online_to_ema(value_head)

    @torch.no_grad()
    def _copy_online_to_ema(self, online_head: DucaValueHeadGroup) -> None:
        for ema_param, online_param in zip(self.ema_head.parameters(), online_head.parameters()):
            ema_param.copy_(online_param)

    @torch.no_grad()
    def update(self, online_head: DucaValueHeadGroup) -> None:
        for ema_param, online_param in zip(self.ema_head.parameters(), online_head.parameters()):
            ema_param.mul_(self.decay).add_(online_param.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def detach_targets(self, value_evidence: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return self.ema_head(value_evidence, valid).value.detach()
