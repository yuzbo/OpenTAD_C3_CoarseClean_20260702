from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DynamicBudgetDecision:
    """Differentiable sample-wise budget decision with a hard forward value."""

    budget_hard: torch.Tensor
    budget_soft: torch.Tensor
    expected_cost: torch.Tensor
    continue_logits: torch.Tensor
    continue_soft: torch.Tensor
    continue_hard: torch.Tensor
    prefix_soft: torch.Tensor
    prefix_hard: torch.Tensor
    marginal_utility: torch.Tensor
    lambda_dual: torch.Tensor
    budget_min: int
    budget_max: int
    budget_multiple: int
    target_budget: float
    policy_name: str = "prefix_marginal_utility_stop"
    dual_target_unit: str = "detector_valid_temporal_observations"

    @property
    def hard_requested_k(self) -> torch.Tensor:
        return self.budget_hard

    @property
    def st_budget_k(self) -> torch.Tensor:
        return self.budget_soft

    @property
    def soft_expected_k(self) -> torch.Tensor:
        return self.expected_cost

    def validate(self, batch_size: Optional[int] = None) -> "DynamicBudgetDecision":
        if self.budget_hard.ndim != 1:
            raise ValueError("budget_hard must be [B]")
        if self.budget_soft.shape != self.budget_hard.shape:
            raise ValueError("budget_soft must match budget_hard")
        if batch_size is not None and self.budget_hard.numel() != int(batch_size):
            raise ValueError("budget decision batch size mismatch")
        if int(self.budget_min) <= 0:
            raise ValueError("budget_min must be positive")
        if int(self.budget_max) < int(self.budget_min):
            raise ValueError("budget_max must be >= budget_min")
        if int(self.budget_multiple) <= 0:
            raise ValueError("budget_multiple must be positive")
        if (int(self.budget_max) - int(self.budget_min)) % int(self.budget_multiple) != 0:
            raise ValueError("budget_max - budget_min must be divisible by budget_multiple")
        hard = self.budget_hard.to(dtype=torch.long)
        if torch.any(hard < int(self.budget_min)) or torch.any(hard > int(self.budget_max)):
            raise ValueError("budget_hard must lie within [budget_min, budget_max]")
        if torch.any((hard - int(self.budget_min)) % int(self.budget_multiple) != 0):
            raise ValueError("budget_hard must align to budget_multiple")
        if not torch.isfinite(self.budget_soft).all():
            raise ValueError("budget_soft must be finite")
        if self.expected_cost.shape != self.budget_hard.shape or not torch.isfinite(self.expected_cost).all():
            raise ValueError("expected_cost must be a finite [B] true soft expectation")
        if torch.any(self.budget_soft < float(self.budget_min) - 1e-4):
            raise ValueError("budget_soft below budget_min")
        if torch.any(self.budget_soft > float(self.budget_max) + 1e-4):
            raise ValueError("budget_soft above budget_max")
        if self.prefix_hard.ndim != 2:
            raise ValueError("prefix_hard must be [B,J]")
        if self.prefix_hard.shape != self.prefix_soft.shape:
            raise ValueError("prefix_hard/prefix_soft shape mismatch")
        if self.prefix_hard.shape != self.continue_hard.shape:
            raise ValueError("prefix_hard/continue_hard shape mismatch")
        if self.prefix_hard.shape[1] > 1 and torch.any(self.prefix_hard[:, 1:] > self.prefix_hard[:, :-1]):
            raise ValueError("prefix_hard must be monotonic non-increasing")
        return self


class PrefixMarginalUtilityBudgetController(nn.Module):
    """Predict K(x) by stopping when marginal detector utility no longer pays for cost."""

    policy_name = "prefix_marginal_utility_stop"

    def __init__(
        self,
        hidden_dim: int,
        budget_min: int = 64,
        budget_max: int = 384,
        budget_multiple: int = 16,
        target_budget: Optional[float] = None,
        tau: float = 1.0,
        lambda_init: float = 1e-3,
        lambda_max: float = 10.0,
        dual_lr: float = 1e-2,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.budget_min = int(budget_min)
        self.budget_max = int(budget_max)
        self.budget_multiple = int(budget_multiple)
        self.target_budget = float(self.budget_max if target_budget is None else target_budget)
        self.tau = float(tau)
        self.lambda_max = float(lambda_max)
        self.dual_lr = float(dual_lr)
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.budget_min <= 0:
            raise ValueError("budget_min must be positive")
        if self.budget_max < self.budget_min:
            raise ValueError("budget_min must be <= budget_max")
        if self.budget_multiple <= 0:
            raise ValueError("budget_multiple must be positive")
        if (self.budget_max - self.budget_min) % self.budget_multiple != 0:
            raise ValueError("budget_multiple must divide budget_max - budget_min")
        if not (0.0 < self.target_budget <= float(self.budget_max)):
            raise ValueError("target_budget must lie in (0, budget_max]")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")

        self.num_extra_blocks = (self.budget_max - self.budget_min) // self.budget_multiple
        rank_count = max(1, self.num_extra_blocks)
        self.rank_embed = nn.Embedding(rank_count, self.hidden_dim)
        self.global_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.block_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.delta_head = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )
        self.register_buffer("lambda_dual", torch.tensor(float(lambda_init), dtype=torch.float32), persistent=True)

    def forward(
        self,
        selection_features: torch.Tensor,
        center_scores: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> DynamicBudgetDecision:
        if selection_features.ndim != 3:
            raise ValueError("selection_features must be [B,T,H]")
        if center_scores.ndim != 2:
            raise ValueError("center_scores must be [B,T]")
        if valid_mask.shape != center_scores.shape:
            raise ValueError("valid_mask must match center_scores")
        if selection_features.shape[:2] != center_scores.shape:
            raise ValueError("selection_features and center_scores must share [B,T]")
        if selection_features.shape[-1] != self.hidden_dim:
            raise ValueError(f"selection_features hidden dim must be {self.hidden_dim}")
        valid = valid_mask.bool()
        if torch.any(valid.long().sum(dim=1) <= 0):
            raise ValueError("each sample must contain at least one valid observation")

        dtype = selection_features.dtype
        device = selection_features.device
        batch_size, temporal_len, hidden_dim = selection_features.shape
        masked_scores = center_scores.masked_fill(~valid, torch.finfo(center_scores.dtype).min / 4.0)
        topk = min(max(1, self.budget_max), temporal_len)
        ranked_idx = torch.topk(masked_scores, k=topk, dim=1).indices
        ranked = torch.gather(
            selection_features,
            dim=1,
            index=ranked_idx[:, :, None].expand(-1, -1, hidden_dim),
        )
        ranked_valid = torch.gather(valid, dim=1, index=ranked_idx).to(dtype=dtype)
        ranked = ranked * ranked_valid[:, :, None]
        if topk < self.budget_max:
            pad = torch.zeros(batch_size, self.budget_max - topk, hidden_dim, device=device, dtype=dtype)
            ranked = torch.cat((ranked, pad), dim=1)
            ranked_valid = torch.cat(
                (ranked_valid, torch.zeros(batch_size, self.budget_max - topk, device=device, dtype=dtype)),
                dim=1,
            )

        valid_float = valid.to(dtype=dtype)
        global_feat = (selection_features * valid_float[:, :, None]).sum(dim=1) / valid_float.sum(
            dim=1, keepdim=True
        ).clamp_min(1.0)
        if self.num_extra_blocks == 0:
            empty = torch.zeros(batch_size, 0, device=device, dtype=dtype)
            budget = torch.full((batch_size,), self.budget_min, device=device, dtype=torch.long)
            decision = DynamicBudgetDecision(
                budget_hard=budget,
                budget_soft=budget.to(dtype=dtype),
                expected_cost=budget.to(dtype=dtype),
                continue_logits=empty,
                continue_soft=empty,
                continue_hard=empty,
                prefix_soft=empty,
                prefix_hard=empty,
                marginal_utility=empty,
                lambda_dual=self.lambda_dual.to(device=device, dtype=dtype),
                budget_min=self.budget_min,
                budget_max=self.budget_max,
                budget_multiple=self.budget_multiple,
                target_budget=self.target_budget,
            )
            return decision.validate(batch_size=batch_size)

        blocks = []
        for block_idx in range(self.num_extra_blocks):
            start = self.budget_min + block_idx * self.budget_multiple
            end = start + self.budget_multiple
            block = ranked[:, start:end, :]
            block_valid = ranked_valid[:, start:end]
            block_mean = block.sum(dim=1) / block_valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            blocks.append(block_mean)
        block_features = torch.stack(blocks, dim=1)
        rank_ids = torch.arange(self.num_extra_blocks, device=device)
        fused = self.global_proj(global_feat)[:, None, :] + self.block_proj(block_features) + self.rank_embed(rank_ids)
        marginal = F.softplus(self.delta_head(fused).squeeze(-1))
        cost = self.lambda_dual.to(device=device, dtype=dtype).clamp(0.0, self.lambda_max)
        continue_logits = (marginal - cost) / self.tau
        continue_soft_raw = torch.sigmoid(continue_logits)
        continue_hard_raw = (continue_soft_raw >= 0.5).to(dtype=dtype)
        prefix_soft_raw = torch.cumprod(continue_soft_raw, dim=1)
        prefix_hard = torch.cumprod(continue_hard_raw, dim=1)
        prefix_st = prefix_hard + prefix_soft_raw - prefix_soft_raw.detach()
        soft_expected_k = float(self.budget_min) + float(self.budget_multiple) * prefix_soft_raw.sum(dim=1)
        budget_soft = float(self.budget_min) + float(self.budget_multiple) * prefix_st.sum(dim=1)
        budget_hard = self.budget_min + self.budget_multiple * prefix_hard.sum(dim=1).to(dtype=torch.long)
        budget_hard = budget_hard.clamp(min=self.budget_min, max=self.budget_max)
        budget_soft = budget_soft.clamp(min=float(self.budget_min), max=float(self.budget_max))
        soft_expected_k = soft_expected_k.clamp(min=float(self.budget_min), max=float(self.budget_max))
        decision = DynamicBudgetDecision(
            budget_hard=budget_hard,
            budget_soft=budget_soft,
            expected_cost=soft_expected_k,
            continue_logits=continue_logits,
            continue_soft=continue_soft_raw,
            continue_hard=continue_hard_raw,
            prefix_soft=prefix_st,
            prefix_hard=prefix_hard,
            marginal_utility=marginal,
            lambda_dual=self.lambda_dual.to(device=device, dtype=dtype),
            budget_min=self.budget_min,
            budget_max=self.budget_max,
            budget_multiple=self.budget_multiple,
            target_budget=self.target_budget,
        )
        return decision.validate(batch_size=batch_size)

    @torch.no_grad()
    def update_dual(self, observed_mean_budget: torch.Tensor | float) -> torch.Tensor:
        value = torch.as_tensor(observed_mean_budget, device=self.lambda_dual.device, dtype=self.lambda_dual.dtype)
        normalized_residual = (value - float(self.target_budget)) / max(float(self.target_budget), 1.0)
        self.lambda_dual.add_(self.dual_lr * normalized_residual)
        self.lambda_dual.clamp_(0.0, self.lambda_max)
        return self.lambda_dual
