"""FoveatedSampler: exact-K foveated frame selection.

Inference is fully deterministic (global evidence top-k + boundary
neighbourhood quota + greedy MMR).  Training uses the Gumbel-TopK
straight-through surrogate so detector feedback can flow back into the score
branches.  Boundary neighbourhoods are boosted in the training logits so the
surrogate and the deterministic policy keep the same foveation prior.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GumbelTopK(nn.Module):
    """Perturbed top-K with hard one-hot transport and a soft surrogate."""

    def __init__(self, tau: float = 1.0, num_samples: int = 1) -> None:
        super().__init__()
        self.tau = float(tau)
        self.num_samples = int(num_samples)

    def forward(self, logits: torch.Tensor, valid: torch.Tensor, k: int) -> Dict[str, torch.Tensor]:
        if logits.ndim != 2 or valid.shape != logits.shape:
            raise ValueError("GumbelTopK expects [B,T] logits and valid")
        if k <= 0:
            raise ValueError("GumbelTopK requires k > 0")
        masked = logits.masked_fill(~valid.bool(), -torch.inf)
        tau = max(self.tau, 1.0e-3)
        soft_sample = torch.zeros_like(masked)
        for _ in range(self.num_samples):
            gumbel = -torch.log(-torch.log(torch.rand_like(masked).clamp_min(1.0e-8).clamp_max(1.0 - 1.0e-8)))
            y = (masked + gumbel) / tau
            soft_sample = soft_sample + torch.softmax(y, dim=-1)
        soft_sample = soft_sample / float(self.num_samples)
        soft_sample = soft_sample * float(k) / soft_sample.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        soft_sample = soft_sample.masked_fill(~valid.bool(), 0.0)
        hard_indices = torch.topk(masked + gumbel, k, dim=-1).indices
        onehot = torch.zeros_like(masked).scatter(1, hard_indices, 1.0)
        return {
            "indices": hard_indices.detach(),
            "onehot": onehot.detach(),
            "soft_sample": soft_sample,
        }


class FoveatedSampler(nn.Module):
    """Foveated exact-K frame sampler with deterministic MMR inference."""

    def __init__(
        self,
        target_k: int = 384,
        min_k: int = 256,
        max_k: int = 512,
        budget_step: int = 16,
        boundary_quota: int = 64,
        boundary_center_top_m: int = 8,
        boundary_radius: int = 2,
        boundary_pair_max_gap: int = 8,
        mmr_lambda: float = 0.10,
        dynamic_budget: bool = True,
        gumbel_tau: float = 1.0,
        gumbel_num_samples: int = 1,
    ) -> None:
        super().__init__()
        if target_k % budget_step != 0 or min_k % budget_step != 0 or max_k % budget_step != 0:
            raise ValueError("sampler budgets must be multiples of budget_step (16)")
        self.target_k = int(target_k)
        self.min_k = int(min_k)
        self.max_k = int(max_k)
        self.budget_step = int(budget_step)
        self.boundary_quota = int(boundary_quota)
        self.boundary_center_top_m = int(boundary_center_top_m)
        self.boundary_radius = int(boundary_radius)
        self.boundary_pair_max_gap = int(boundary_pair_max_gap)
        self.mmr_lambda = float(mmr_lambda)
        self.dynamic_budget = bool(dynamic_budget)
        self.gumbel = GumbelTopK(tau=float(gumbel_tau), num_samples=int(gumbel_num_samples))
        self._similarity_sigma = float(max(2.0, self.boundary_pair_max_gap))

    def resolve_budget(
        self,
        score: torch.Tensor,
        valid: torch.Tensor,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> int:
        if not self.dynamic_budget:
            return int(self.target_k)
        if uncertainty is not None and torch.is_tensor(uncertainty):
            if uncertainty.ndim == 2:
                evidence = uncertainty[0].masked_select(valid[0].bool())
            else:
                evidence = uncertainty.masked_select(valid.bool())
            mu = float(evidence.mean().item()) if evidence.numel() else 0.5
        else:
            evidence = torch.sigmoid(score[0].masked_select(valid[0].bool()))
            mu = float(evidence.mean().item()) if evidence.numel() else 0.5
        span = int((self.max_k - self.min_k) // 2)
        centered = float(max(-1.0, min(1.0, (mu - 0.5) * 2.0)))
        delta_steps = int(round(centered * float(span) / float(self.budget_step)))
        budget = int(self.target_k + delta_steps * self.budget_step)
        return int(max(self.min_k, min(self.max_k, budget)))

    @staticmethod
    def _floor_to_budget_step(length: int, step: int) -> int:
        if length <= 0:
            return 0
        return (length // step) * step

    def _boundary_centers(self, score: torch.Tensor, valid_len: int) -> List[int]:
        if valid_len < 3 or self.boundary_center_top_m <= 0:
            return []
        s = score[:valid_len]
        radius = max(1, int(self.boundary_radius))
        padded = F.pad(s, (radius, radius), value=-torch.inf)
        windows = padded.unfold(0, 2 * radius + 1, 1)
        local_max = s >= windows.max(dim=1).values
        candidates = local_max.nonzero(as_tuple=False).squeeze(-1)
        if candidates.numel() == 0:
            return []
        if candidates.dim() == 0:
            candidates = candidates.unsqueeze(0)
        top_scores = s[candidates]
        top = torch.topk(top_scores, min(self.boundary_center_top_m, int(candidates.numel()))).indices
        return candidates[top].tolist()

    def _boundary_protected(
        self,
        score: torch.Tensor,
        valid_len: int,
        budget: int,
    ) -> Tuple[List[int], List[int]]:
        centers = self._boundary_centers(score, valid_len)
        protected: List[int] = []
        seen = set()
        for center in centers:
            for j in range(
                max(0, center - int(self.boundary_radius)),
                min(valid_len, center + int(self.boundary_radius) + 1),
            ):
                if j not in seen:
                    seen.add(j)
                    protected.append(j)
        if len(centers) >= 2 and abs(centers[0] - centers[1]) <= self.boundary_pair_max_gap:
            lo, hi = min(centers[0], centers[1]), max(centers[0], centers[1])
            for j in range(lo, hi + 1):
                if j not in seen:
                    seen.add(j)
                    protected.append(j)
        protected = sorted(protected)
        if len(protected) > self.boundary_quota:
            ranked = sorted(protected, key=lambda j: -float(score[j].item()))
            protected = sorted(ranked[: self.boundary_quota])
        return centers, protected

    def _greedy_mmr(
        self,
        score: torch.Tensor,
        valid_len: int,
        budget: int,
        protected: List[int],
    ) -> List[int]:
        selected = list(protected)
        if len(selected) >= budget:
            return sorted(selected[:budget])
        pool_size = min(valid_len, max(budget * 4, budget + 512))
        top_pool = torch.topk(score[:valid_len], min(valid_len, pool_size)).indices.tolist()
        pool = sorted(set(top_pool) | set(selected))
        if len(pool) < budget:
            pool = sorted(set(range(valid_len)))
        pool_tensor = torch.as_tensor(pool, device=score.device, dtype=torch.long)
        pool_scores = score[pool_tensor]

        while len(selected) < budget and pool_tensor.numel():
            if not selected:
                best = int(torch.argmax(pool_scores).item())
                selected.append(int(pool_tensor[best].item()))
                keep = torch.ones(pool_tensor.numel(), dtype=torch.bool, device=pool_tensor.device)
                keep[best] = False
                pool_tensor = pool_tensor[keep]
                pool_scores = pool_scores[keep]
                continue
            sel_tensor = torch.as_tensor(selected, device=score.device, dtype=torch.long)
            dist = (pool_tensor.unsqueeze(1) - sel_tensor.unsqueeze(0)).abs().float()
            similarity = torch.exp(-(dist.pow(2)) / (2.0 * self._similarity_sigma ** 2))
            max_sim = similarity.max(dim=1).values
            gain = pool_scores - self.mmr_lambda * max_sim
            best = int(torch.argmax(gain).item())
            selected.append(int(pool_tensor[best].item()))
            keep = torch.ones(pool_tensor.numel(), dtype=torch.bool, device=pool_tensor.device)
            keep[best] = False
            pool_tensor = pool_tensor[keep]
            pool_scores = pool_scores[keep]
        if len(selected) < budget:
            remaining = sorted(set(range(valid_len)) - set(selected))
            selected.extend(remaining[: budget - len(selected)])
        return sorted(selected[:budget])

    def forward(
        self,
        frame_score: torch.Tensor,
        valid: torch.Tensor,
        contribution: Optional[torch.Tensor] = None,
        training: bool = True,
        global_start_offset: int = 0,
        uncertainty: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if frame_score.ndim != 2 or valid.shape != frame_score.shape:
            raise ValueError("FoveatedSampler expects [B,T] frame_score and valid")
        if frame_score.shape[0] != 1:
            raise ValueError("FoveatedSampler currently supports batch_size=1 (dynamic backbone constraint)")
        valid_bool = valid[0].bool()
        valid_len = int(valid_bool.long().sum().item())
        if valid_len <= 0:
            raise ValueError("FoveatedSampler requires at least one valid frame")

        budget = self.resolve_budget(frame_score[0], valid_bool.unsqueeze(0), uncertainty=uncertainty)
        step_capacity = self._floor_to_budget_step(valid_len, self.budget_step)
        if step_capacity <= 0:
            step_capacity = valid_len
            budget = valid_len
        else:
            budget = min(budget, step_capacity)
        if budget <= 0:
            raise ValueError("resolved foveated budget must be positive")

        centers, protected = self._boundary_protected(frame_score[0], valid_len, budget)
        if training:
            boosted = frame_score.clone()
            if protected:
                protected_tensor = torch.as_tensor(protected, device=frame_score.device, dtype=torch.long)
                boosted[0, protected_tensor] = frame_score[0, protected_tensor] + 10.0
            gumbel_out = self.gumbel(boosted, valid, budget)
            indices = gumbel_out["indices"]
            onehot = gumbel_out["onehot"]
            soft_sample = gumbel_out["soft_sample"]
            transport = onehot + (soft_sample - soft_sample.detach())
            probs = soft_sample
        else:
            positions = self._greedy_mmr(frame_score[0], valid_len, budget, protected)
            indices = torch.as_tensor([positions], device=frame_score.device, dtype=torch.long)
            onehot = torch.zeros_like(frame_score).scatter(1, indices, 1.0)
            transport = onehot
            probs = onehot

        positions = indices + int(global_start_offset)
        metadata: Dict[str, Any] = {
            "mode": "training_gumbel_topk" if training else "inference_greedy_mmr",
            "budget": int(budget),
            "target_k": int(self.target_k),
            "boundary_centers": int(len(centers)),
            "boundary_protected": int(len(protected)),
            "transport_is_straight_through": bool(training),
            "inference_is_deterministic": not bool(training),
            "selected_positions_are_sorted_unique": True,
        }
        return {
            "indices": indices,
            "positions": positions,
            "transport": transport,
            "probs": probs,
            "metadata": metadata,
        }
