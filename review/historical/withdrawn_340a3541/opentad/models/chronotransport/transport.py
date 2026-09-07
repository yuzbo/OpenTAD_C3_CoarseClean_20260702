from __future__ import annotations

import torch
from torch import Tensor, nn


class TemporalTransportAdapter(nn.Module):
    """Low-rank feature transport conditioned on current innovation and age.

    The adapter starts as an exact HOLD-on-latest-cache operator because ``up_proj`` is zero
    initialized. This makes Stage-B optimization fail-safe and gives an explicit
    TRANSPORT-without-correction ablation.
    """

    def __init__(
        self,
        embed_dims: int,
        bottleneck_dims: int = 64,
        max_age: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.bottleneck_dims = int(bottleneck_dims)
        self.max_age = int(max_age)
        if self.embed_dims <= 0 or self.bottleneck_dims <= 0:
            raise ValueError("embed_dims and bottleneck_dims must be positive")
        if self.max_age <= 0:
            raise ValueError("max_age must be positive")

        self.down_proj = nn.Linear(self.embed_dims, self.bottleneck_dims)
        self.age_embedding = nn.Embedding(self.max_age + 1, self.bottleneck_dims)
        self.norm = nn.LayerNorm(self.bottleneck_dims)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(float(dropout))
        self.up_proj = nn.Linear(self.bottleneck_dims, self.embed_dims)
        self.gate = nn.Parameter(torch.tensor(1.0))

        nn.init.trunc_normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.down_proj.bias)
        nn.init.trunc_normal_(self.age_embedding.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, cached: Tensor, current: Tensor, age: Tensor | int) -> Tensor:
        if cached.shape != current.shape:
            raise ValueError("cached and current transport tensors must have identical shape")
        if cached.ndim != 3:
            raise ValueError("transport expects [B, N, C] tensors")
        if int(cached.shape[-1]) != self.embed_dims:
            raise ValueError("transport channel dimension does not match embed_dims")
        if not torch.isfinite(cached).all() or not torch.isfinite(current).all():
            raise ValueError("transport inputs must be finite")

        if not isinstance(age, Tensor):
            age = torch.full(
                (int(cached.shape[0]),),
                int(age),
                dtype=torch.long,
                device=cached.device,
            )
        age = age.to(device=cached.device, dtype=torch.long).reshape(-1)
        if int(age.numel()) == 1 and int(cached.shape[0]) != 1:
            age = age.expand(int(cached.shape[0]))
        if int(age.numel()) != int(cached.shape[0]):
            raise ValueError("age must have one value per transport sample")
        age = age.clamp_(0, self.max_age)

        innovation = current - cached
        hidden = self.down_proj(innovation)
        hidden = hidden + self.age_embedding(age).unsqueeze(1)
        hidden = self.dropout(self.act(self.norm(hidden)))
        correction = self.up_proj(hidden)
        # tanh keeps the learned residual bounded while preserving the zero-init
        # exact HOLD behavior.
        return cached + torch.tanh(self.gate) * correction
