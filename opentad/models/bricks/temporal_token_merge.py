"""Boundary-Protected Temporal Token Merging for VideoMAE clips."""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule


class BoundaryProtectedTemporalTokenMerge(BaseModule):
    """Boundary-Protected Temporal Token Merging across spatial patches and adjacent temporal tubelets.

    Schedule: reduces tubelets per 16-frame chunk: 8 -> 7 (after Block 3), 7 -> 6 (after Block 6), 6 -> 5 (after Block 9).
    Protection: Top-2 boundary tubelets in each chunk cannot be merged or absorbed.
    """

    def __init__(
        self,
        merge_blocks: Tuple[int, ...] = (2, 5, 8),  # 0-indexed: Block 3, Block 6, Block 9
        protected_boundary_tubelets: int = 2,
        enabled: bool = True,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.merge_blocks = set(merge_blocks)
        self.protected_boundary_tubelets = int(protected_boundary_tubelets)
        self.enabled = bool(enabled)

    def should_merge(self, block_index: int) -> bool:
        return self.enabled and (block_index in self.merge_blocks)

    def merge_step(
        self,
        x: torch.Tensor,
        spatial_tokens: int,
        support_mass: torch.Tensor,
        support_centers: torch.Tensor,
        support_intervals: torch.Tensor,
        boundary_scores: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one token merge step on x for each chunk independently.

        Args:
            x: [B_chunks, T * S, C]
            spatial_tokens: S (e.g. 100)
            support_mass: [B_chunks, T]
            support_centers: [B_chunks, T]
            support_intervals: [B_chunks, T, 2]
            boundary_scores: [B_chunks, T]
        Returns:
            x_merged: [B_chunks, (T - 1) * S, C]
            new_mass: [B_chunks, T - 1]
            new_centers: [B_chunks, T - 1]
            new_intervals: [B_chunks, T - 1, 2]
            new_boundary_scores: [B_chunks, T - 1]
        """
        B_chunks, N_tokens, C = x.shape
        S = spatial_tokens
        T = N_tokens // S
        device = x.device

        if boundary_scores is None:
            boundary_scores = torch.zeros((B_chunks, T), device=device, dtype=torch.float32)

        if not self.enabled or T <= 1:
            return x, support_mass, support_centers, support_intervals, boundary_scores

        # Reshape x to [B_chunks, T, S, C]
        x_reshaped = x.view(B_chunks, T, S, C)

        # Identify protected tubelets: top boundary scores per chunk (at most T - 2 to ensure mergeable pairs exist)
        protected_mask = torch.zeros((B_chunks, T), dtype=torch.bool, device=device)
        num_protected = min(self.protected_boundary_tubelets, max(0, T - 2))
        if num_protected > 0:
            _, topk_idx = torch.topk(boundary_scores, k=num_protected, dim=-1)
            protected_mask.scatter_(1, topk_idx, True)

        # Compute cosine similarity between adjacent temporal tubelets (t, t+1) averaged across spatial tokens
        t_curr = x_reshaped[:, :-1]  # [B_chunks, T-1, S, C]
        t_next = x_reshaped[:, 1:]   # [B_chunks, T-1, S, C]
        cos_sim = F.cosine_similarity(t_curr, t_next, dim=-1).mean(dim=-1)  # [B_chunks, T-1]

        # Ineligible pairs: if EITHER tubelet t OR tubelet t+1 is protected, the pair CANNOT merge
        pair_protected = protected_mask[:, :-1] | protected_mask[:, 1:]  # [B_chunks, T-1]

        # Perform merge per chunk
        new_x_list = []
        new_m_list = []
        new_c_list = []
        new_int_list = []
        new_b_list = []

        for b in range(B_chunks):
            unprotected_pairs = (~pair_protected[b]).nonzero(as_tuple=True)[0]
            if len(unprotected_pairs) == 0:
                # Exact compression still needs one merge. Choose the least exposed
                # boundary pair instead of dropping a protected tubelet.
                pair_boundary = torch.maximum(boundary_scores[b, :-1], boundary_scores[b, 1:])
                min_boundary = pair_boundary.min()
                candidate_pairs = (pair_boundary <= min_boundary).nonzero(as_tuple=True)[0]
            else:
                candidate_pairs = unprotected_pairs

            best_offset = int(torch.argmax(cos_sim[b, candidate_pairs]).item())
            p = int(candidate_pairs[best_offset].item())

            m1 = support_mass[b, p]
            m2 = support_mass[b, p + 1]
            m_sum = m1 + m2 + 1e-7

            # Merged feature: mass-weighted average over spatial patches
            feat_p = (m1 * x_reshaped[b, p] + m2 * x_reshaped[b, p + 1]) / m_sum  # [S, C]
            c_p = (m1 * support_centers[b, p] + m2 * support_centers[b, p + 1]) / m_sum
            int_p = torch.stack([
                torch.minimum(support_intervals[b, p, 0], support_intervals[b, p + 1, 0]),
                torch.maximum(support_intervals[b, p, 1], support_intervals[b, p + 1, 1]),
            ])
            b_p = torch.maximum(boundary_scores[b, p], boundary_scores[b, p + 1])

            # Reconstruct list of (T - 1) tubelets
            b_feats = []
            b_masses = []
            b_centers = []
            b_intervals = []
            b_scores = []

            for t_idx in range(T):
                if t_idx == p:
                    b_feats.append(feat_p)
                    b_masses.append(m_sum)
                    b_centers.append(c_p)
                    b_intervals.append(int_p)
                    b_scores.append(b_p)
                elif t_idx == p + 1:
                    continue  # absorbed
                else:
                    b_feats.append(x_reshaped[b, t_idx])
                    b_masses.append(support_mass[b, t_idx])
                    b_centers.append(support_centers[b, t_idx])
                    b_intervals.append(support_intervals[b, t_idx])
                    b_scores.append(boundary_scores[b, t_idx])

            new_x_list.append(torch.stack(b_feats, dim=0))  # [T-1, S, C]
            new_m_list.append(torch.stack(b_masses, dim=0))  # [T-1]
            new_c_list.append(torch.stack(b_centers, dim=0))  # [T-1]
            new_int_list.append(torch.stack(b_intervals, dim=0))  # [T-1, 2]
            new_b_list.append(torch.stack(b_scores, dim=0))  # [T-1]

        new_x_tensor = torch.stack(new_x_list, dim=0).view(B_chunks, (T - 1) * S, C)
        new_m_tensor = torch.stack(new_m_list, dim=0)
        new_c_tensor = torch.stack(new_c_list, dim=0)
        new_int_tensor = torch.stack(new_int_list, dim=0)
        new_b_tensor = torch.stack(new_b_list, dim=0)

        return new_x_tensor, new_m_tensor, new_c_tensor, new_int_tensor, new_b_tensor
