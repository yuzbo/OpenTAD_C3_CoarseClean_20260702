"""DUCA Evidence Recovery: Coverage-Constrained Semantic Acquisition and Boundary-Protected Recovery."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from opentad.models.builder import MODELS, SELECTORS
from opentad.models.bricks.bounded_interval_adapter import (
    BoundedTubeletIntervalAdapter,
    ContinuousTimestampConditioner,
)
from opentad.models.bricks.dense_temporal_recovery import DenseTemporalRecovery
from opentad.models.bricks.temporal_token_merge import BoundaryProtectedTemporalTokenMerge
from opentad.models.duca.structured_selection import global_structured_topk
from opentad.models.duca.transition_only import balanced_binary_actionness_loss
from opentad.models.utils.numerics import assert_finite_tensor
from opentad.models.utils.truetime_geometry import SELECTED_AXIS, TRUE_TIME_AXIS, TrueTimeMap


H65_POSITION_META_KEYS = (
    "bata_selected_dense_indices",
    "irregular_selected_positions",
    "selected_dense_indices",
)


def _max_unselected_hole(positions: torch.Tensor, temporal_len: int) -> int:
    if positions.numel() == 0:
        return int(temporal_len)
    sentinels = torch.cat(
        (
            positions.new_tensor([-1]),
            positions.to(dtype=torch.long),
            positions.new_tensor([int(temporal_len)]),
        )
    )
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _support_from_selected_positions(
    selected_positions: torch.Tensor,
    *,
    boundary_prob: Optional[torch.Tensor],
    window_size: int,
) -> Dict[str, torch.Tensor]:
    assert_finite_tensor(selected_positions, "selection.selected_positions")
    B, K = selected_positions.shape
    if K % 2 != 0:
        raise ValueError("DUCA Evidence Recovery requires an even frame budget for tubelet pairing")
    device = selected_positions.device
    num_tubelets = K // 2
    pos_even = selected_positions[:, 0::2].float()
    pos_odd = selected_positions[:, 1::2].float()
    centers = 0.5 * (pos_even + pos_odd)
    intervals = torch.stack([pos_even, pos_odd], dim=-1)
    mass = torch.ones((B, num_tubelets), dtype=torch.float32, device=device) * 2.0
    norm_timestamps = centers / max(float(window_size - 1), 1.0)
    dt = (pos_odd - pos_even).clamp_min(1.0)
    z_cond = torch.stack(
        [torch.log(dt), 2.0 / dt, dt / max(float(window_size), 1.0)],
        dim=-1,
    )
    assert_finite_tensor(centers, "selection.support_centers")
    assert_finite_tensor(intervals, "selection.support_intervals")
    assert_finite_tensor(z_cond, "selection.tubelet_z_condition")
    if boundary_prob is None:
        tubelet_b = torch.zeros((B, num_tubelets), dtype=torch.float32, device=device)
    else:
        max_idx = boundary_prob.shape[1] - 1
        b_even = torch.gather(boundary_prob, 1, selected_positions[:, 0::2].clamp(0, max_idx))
        b_odd = torch.gather(boundary_prob, 1, selected_positions[:, 1::2].clamp(0, max_idx))
        tubelet_b = torch.maximum(b_even, b_odd)
    assert_finite_tensor(tubelet_b, "selection.boundary_scores_per_tubelet")
    return {
        "tubelet_timestamps": norm_timestamps,
        "tubelet_z_condition": z_cond,
        "support_centers": centers,
        "support_intervals": intervals,
        "support_mass": mass,
        "boundary_scores_per_tubelet": tubelet_b,
    }


class ASFormerDenseSemanticScout(BaseModule):
    """Low-cost ASFormer dense semantic scout extracting multi-signal temporal evidence."""

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 64,
        num_layers: int = 4,
        kernel_size: int = 5,
        window_size: int = 768,
        context_window: int = 4,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.hidden_dim = hidden_dim
        self.window_size = window_size
        self.context_window = int(context_window)

        # Spatial stem for low-resolution 64x64 inputs: downsample spatially to 1x1
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, kernel_size=4, stride=4, padding=0),  # 64 -> 16
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=4, padding=0),  # 16 -> 4
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Temporal dilated residual layers (ASFormer-style dilated 1D convs)
        self.temporal_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(
                        hidden_dim,
                        hidden_dim,
                        kernel_size=kernel_size,
                        padding=(kernel_size // 2) * (2**i),
                        dilation=2**i,
                    ),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                )
                for i in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # Actionness & boundary heads
        self.action_head = nn.Linear(hidden_dim, 1)
        self.boundary_head = nn.Linear(hidden_dim, 1)

        # Utility MLP: [actionness, abs_delta_actionness, boundary_prob, boundary_uncert, feat_residual, novelty] -> utility
        self.utility_mlp = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Zero-initialize the last linear layer of utility MLP
        nn.init.zeros_(self.utility_mlp[-1].weight)
        nn.init.zeros_(self.utility_mlp[-1].bias)

    def forward(
        self,
        lowres_rgb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-signal temporal evidence."""
        if lowres_rgb.ndim == 5:
            if lowres_rgb.shape[1] == 3:  # [B, 3, T, H, W]
                B, C, T, H, W = lowres_rgb.shape
                frames = lowres_rgb.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
            else:  # [B, T, 3, H, W]
                B, T, C, H, W = lowres_rgb.shape
                frames = lowres_rgb.reshape(B * T, C, H, W)
        else:
            raise ValueError(f"lowres_rgb must be 5D tensor, got {lowres_rgb.shape}")

        if valid_mask is None:
            valid_mask = torch.ones((B, T), dtype=torch.bool, device=lowres_rgb.device)

        # 1. Spatial stem: [B*T, D, 1, 1] -> [B, D, T]
        stem_feats = self.spatial_stem(frames.float()).view(B, T, self.hidden_dim).permute(0, 2, 1)

        assert_finite_tensor(stem_feats, "scout.stem_features")

        # 2. Temporal dilated convolution
        x = stem_feats
        for layer in self.temporal_layers:
            x = x + layer(x)

        # LayerNorm along feature dim: [B, T, D]
        hidden = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)  # [B, D, T]
        assert_finite_tensor(hidden, "scout.hidden")
        hidden_t = hidden.permute(0, 2, 1)  # [B, T, D]

        # 3. Actionness & Boundary predictions
        action_logits = self.action_head(hidden_t).squeeze(-1)  # [B, T]
        boundary_logits = self.boundary_head(hidden_t).squeeze(-1)  # [B, T]
        assert_finite_tensor(action_logits, "scout.action_logits")
        assert_finite_tensor(boundary_logits, "scout.boundary_logits")

        actionness = torch.sigmoid(action_logits)  # [B, T]
        boundary_prob = torch.sigmoid(boundary_logits)  # [B, T]
        boundary_uncert = 4.0 * boundary_prob * (1.0 - boundary_prob)  # [B, T]

        # First difference of actionness: abs(a[t] - a[t-1])
        delta_action = torch.zeros_like(actionness)
        delta_action[:, 1:] = torch.abs(actionness[:, 1:] - actionness[:, :-1])

        # Feature residual: ||LN(h[t]) - LN(h[t-1])||_2 / sqrt(D)
        feat_res = torch.zeros_like(actionness)
        feat_res[:, 1:] = torch.norm(hidden_t[:, 1:] - hidden_t[:, :-1], p=2, dim=-1) / math.sqrt(self.hidden_dim)

        # Context novelty: 1 - cosine(h[t], mean(h[t-w : t+w]))
        w = self.context_window
        padded_h = F.pad(hidden_t.permute(0, 2, 1), (w, w), mode="replicate").permute(0, 2, 1)
        windows = padded_h.unfold(dimension=1, size=2 * w + 1, step=1)  # [B, T, D, 2w+1]
        context_mean = windows.mean(dim=-1)  # [B, T, D]
        context_novelty = 1.0 - F.cosine_similarity(hidden_t, context_mean, dim=-1).clamp(-1.0, 1.0)

        # 4. Utility computation via MLP
        feat_stack = torch.stack(
            [actionness, delta_action, boundary_prob, boundary_uncert, feat_res, context_novelty],
            dim=-1,
        )  # [B, T, 6]

        # Initial fallback: 0.5 * actionness + 0.5 * boundary_prob
        base_utility = 0.5 * actionness + 0.5 * boundary_prob
        mlp_residual = self.utility_mlp(feat_stack).squeeze(-1)
        utility = torch.sigmoid(base_utility + mlp_residual)
        assert_finite_tensor(utility, "scout.utility")

        return {
            "hidden": hidden,  # [B, D, T]
            "actionness": actionness,  # [B, T]
            "action_logits": action_logits,  # [B, T]
            "boundary_prob": boundary_prob,  # [B, T]
            "boundary_logits": boundary_logits,  # [B, T]
            "boundary_uncert": boundary_uncert,  # [B, T]
            "delta_action": delta_action,  # [B, T]
            "feature_residual": feat_res,  # [B, T]
            "context_novelty": context_novelty,  # [B, T]
            "utility": utility,  # [B, T]
        }


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute temperature-scaled KL divergence distillation loss with valid mask."""
    p_s = F.log_softmax(student_logits / temperature, dim=-1)
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(p_s, p_t, reduction="none").sum(dim=-1) * (temperature**2)
    if valid_mask is not None:
        mask_f = valid_mask.float()
        return (kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return kl.mean()


def compute_two_view_consistency_loss(
    view1_logits: torch.Tensor,
    view2_logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute two-view consistency loss using symmetric KL divergence with valid mask."""
    p1 = F.log_softmax(view1_logits, dim=-1)
    p2 = F.log_softmax(view2_logits, dim=-1)
    q1 = F.softmax(view1_logits, dim=-1)
    q2 = F.softmax(view2_logits, dim=-1)
    kl_12 = F.kl_div(p1, q2, reduction="none").sum(dim=-1)
    kl_21 = F.kl_div(p2, q1, reduction="none").sum(dim=-1)
    sym_kl = 0.5 * (kl_12 + kl_21)
    if valid_mask is not None:
        mask_f = valid_mask.float()
        return (sym_kl * mask_f).sum() / mask_f.sum().clamp_min(1.0)
    return sym_kl.mean()


def partition_semantic_segments(

    change_score: torch.Tensor,
    valid_len: int,
    min_seg_len: int = 8,
    max_seg_len: int = 64,
) -> List[Tuple[int, int]]:
    """Partition a sequence [0, valid_len) into semantic segments based on change scores."""
    if valid_len <= min_seg_len:
        return [(0, valid_len)]

    scores = change_score[:valid_len].detach().cpu().numpy()
    split_points = [0]
    current = 0

    while current < valid_len:
        next_min = min(current + min_seg_len, valid_len)
        next_max = min(current + max_seg_len, valid_len)
        if next_max == valid_len:
            split_points.append(valid_len)
            break

        window_scores = scores[next_min:next_max]
        if len(window_scores) > 0:
            best_offset = int(window_scores.argmax())
            split_pt = next_min + best_offset
        else:
            split_pt = next_min
        split_points.append(split_pt)
        current = split_pt

    segments = [(split_points[i], split_points[i + 1]) for i in range(len(split_points) - 1)]
    return segments


def largest_remainder_quota(
    segment_weights: List[float],
    total_budget: int,
    min_per_seg: int = 2,
) -> List[int]:
    """Allocate integer quotas using the Largest Remainder Method (Hare-Niemeyer)."""
    num_segs = len(segment_weights)
    if num_segs == 0:
        return []
    if any(not math.isfinite(float(weight)) for weight in segment_weights):
        raise FloatingPointError(
            f"non-finite segment weight before quota allocation: {segment_weights!r}"
        )
    if any(float(weight) < 0.0 for weight in segment_weights):
        raise ValueError(f"segment weights must be non-negative, got {segment_weights!r}")
    if total_budget <= num_segs * min_per_seg:
        base = total_budget // num_segs
        rem = total_budget % num_segs
        return [base + (1 if i < rem else 0) for i in range(num_segs)]
    remaining_budget = total_budget - num_segs * min_per_seg
    total_w = sum(segment_weights) + 1e-7
    if not math.isfinite(total_w) or total_w <= 0.0:
        raise FloatingPointError(f"invalid total segment weight: {total_w!r}")

    exact_quotas = [w / total_w * remaining_budget for w in segment_weights]
    integer_parts = [int(math.floor(q)) for q in exact_quotas]
    remainders = [q - math.floor(q) for q in exact_quotas]

    allocated = sum(integer_parts)
    leftover = remaining_budget - allocated

    order = sorted(range(num_segs), key=lambda i: remainders[i], reverse=True)
    for i in range(leftover):
        integer_parts[order[i]] += 1

    quotas = [min_per_seg + q for q in integer_parts]
    return quotas


class EvidenceRecoverySelector(BaseModule):
    """Coverage-constrained semantic acquisition selector with DP & Largest-Remainder Quotas."""

    def __init__(
        self,
        budget: int = 384,
        window_size: int = 768,
        use_coverage: bool = True,
        max_hole: int = 16,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.use_coverage = bool(use_coverage)
        self.max_hole = int(max_hole)

    def select(
        self,
        utility: torch.Tensor,
        boundary_prob: torch.Tensor,
        delta_action: torch.Tensor,
        feat_residual: torch.Tensor,
        context_novelty: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Perform exact-K selection with optional max-hole coverage."""
        assert_finite_tensor(utility, "selector.utility")
        assert_finite_tensor(boundary_prob, "selector.boundary_prob")
        assert_finite_tensor(delta_action, "selector.delta_action")
        assert_finite_tensor(feat_residual, "selector.feature_residual")
        assert_finite_tensor(context_novelty, "selector.context_novelty")
        B, T = utility.shape
        K = min(self.budget, T)
        device = utility.device

        change_score = delta_action + boundary_prob + feat_residual + context_novelty
        assert_finite_tensor(change_score, "selector.change_score")

        selected_positions_list = []
        selected_valid_counts = []
        dense_valid_lens = []
        observed_holes = []

        for b in range(B):
            valid_len = int(valid_mask[b].sum().item())
            if valid_len <= 0:
                valid_len = T
            eff_k = min(K, valid_len)

            if not self.use_coverage:
                # NO_COVERAGE ARM: pure exact semantic Top-K within valid prefix
                u_b = utility[b, :valid_len]
                _, topk_idx = torch.topk(u_b, k=eff_k, largest=True)
                sel_active = torch.sort(topk_idx)[0]
            else:
                # FULL / COVERAGE-CONSTRAINED: semantic quota prior plus exact-K/max-hole DP.
                segments = partition_semantic_segments(change_score[b], valid_len)
                seg_weights = [
                    float(utility[b, start:end].sum().item() + 1e-4)
                    for start, end in segments
                ]
                if any(not math.isfinite(weight) for weight in seg_weights):
                    raise FloatingPointError(
                        f"non-finite segment weights at batch {b}: {seg_weights!r}"
                    )
                quotas = largest_remainder_quota(seg_weights, total_budget=eff_k, min_per_seg=2)
                quota_prior = torch.zeros((valid_len,), dtype=utility.dtype, device=device)
                for (start, end), q in zip(segments, quotas):
                    q_act = min(max(int(q), 0), end - start)
                    if q_act <= 0:
                        continue
                    seg_u = utility[b, start:end]
                    _, idx = torch.topk(seg_u, k=q_act, largest=True)
                    quota_prior[start + idx] = 1.0

                row_logits = utility[b, :valid_len]
                if quota_prior.any():
                    bonus = row_logits.detach().float().std().clamp_min(1.0) * 1.0e-3
                    row_logits = row_logits + quota_prior.to(dtype=row_logits.dtype) * bonus.to(dtype=row_logits.dtype)
                structured = global_structured_topk(
                    row_logits.unsqueeze(0),
                    k=eff_k,
                    max_unselected_hole=self.max_hole,
                    training=False,
                )
                sel_active = structured.selected_positions[0]

            if eff_k < K:
                pad_value = max(valid_len - 1, 0)
                sel_pos = F.pad(sel_active, (0, K - eff_k), value=pad_value)
            else:
                sel_pos = sel_active

            selected_positions_list.append(sel_pos)
            selected_valid_counts.append(torch.tensor(eff_k, dtype=torch.long, device=device))
            dense_valid_lens.append(torch.tensor(valid_len, dtype=torch.long, device=device))
            observed_holes.append(torch.tensor(_max_unselected_hole(sel_active, valid_len), dtype=torch.long, device=device))

        selected_positions = torch.stack(selected_positions_list, dim=0)
        assert_finite_tensor(selected_positions, "selector.selected_positions")
        for b in range(B):
            active = int(selected_valid_counts[b].item())
            row = selected_positions[b, :active]
            if active and bool((row < 0).any().item() or (row >= valid_mask.shape[1]).any().item()):
                raise ValueError("selected positions must lie inside the valid temporal domain")
            if active > 1 and bool((row[1:] <= row[:-1]).any().item()):
                raise ValueError("selected positions must be strictly increasing")
        support = _support_from_selected_positions(
            selected_positions,
            boundary_prob=boundary_prob,
            window_size=self.window_size,
        )

        return {
            "selected_positions": selected_positions,  # [B, 384]
            "selected_valid_counts": torch.stack(selected_valid_counts, dim=0),
            "dense_valid_len": torch.stack(dense_valid_lens, dim=0),
            "observed_max_unselected_hole": torch.stack(observed_holes, dim=0),
            **support,
        }


class DucaEvidenceRecoveryModule(BaseModule):
    """Integrated DUCA Evidence Recovery module managing scout, selection, and losses."""

    def __init__(
        self,
        budget: int = 384,
        window_size: int = 768,
        use_coverage: bool = True,
        use_time_conditioning: bool = True,
        use_temporal_merge: bool = True,
        use_dense_recovery: bool = True,
        use_robust_training: bool = True,
        use_h65_selection: bool = False,
        max_hole: int = 16,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.use_coverage = bool(use_coverage)
        self.use_time_conditioning = bool(use_time_conditioning)
        self.use_temporal_merge = bool(use_temporal_merge)
        self.use_dense_recovery = bool(use_dense_recovery)
        self.use_robust_training = bool(use_robust_training)
        self.use_h65_selection = bool(use_h65_selection)
        self.max_hole = int(max_hole)

        if not self.use_h65_selection:
            self.scout = ASFormerDenseSemanticScout(window_size=window_size)
            self.selector = EvidenceRecoverySelector(
                budget=budget,
                window_size=window_size,
                use_coverage=use_coverage,
                max_hole=max_hole,
            )
        else:
            self.scout = None
            self.selector = None

        self.recovery = DenseTemporalRecovery(
            target_grid_size=budget,
            original_window_size=window_size,
            enabled=use_dense_recovery,
        )

    def acquire(
        self,
        lowres_rgb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        h65_positions: Optional[torch.Tensor] = None,
        diagnostic_context: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Perform semantic evidence acquisition or pure deterministic H65 replay."""
        B = lowres_rgb.shape[0]
        T = self.window_size if lowres_rgb.ndim < 4 else (lowres_rgb.shape[2] if lowres_rgb.shape[1] == 3 else lowres_rgb.shape[1])

        if self.use_h65_selection:
            if h65_positions is None:
                raise ValueError("DUCA H65 replay requires H65 selected positions in metas; uniform fallback is forbidden")
            # Validate the source tensor before any float->integer conversion.  A
            # NaN cast can otherwise become an opaque integer sentinel and hide the
            # first invalid operation from the diagnostic.
            assert_finite_tensor(
                h65_positions,
                "h65_positions",
                **dict(diagnostic_context or {}),
            )
            # Preserve the historical finite-path cast exactly; the diagnostic
            # only changes the non-finite failure from an opaque integer sentinel
            # into an actionable exception.
            sel_pos = h65_positions.to(device=lowres_rgb.device, dtype=torch.long)
            if sel_pos.ndim != 2 or sel_pos.shape[0] != B or sel_pos.shape[1] != self.budget:
                raise ValueError(
                    "h65_positions must be [B,budget], "
                    f"got {tuple(sel_pos.shape)} for B={B}, budget={self.budget}"
                )
            dense_valid_len = (
                valid_mask.long().sum(dim=-1).to(device=lowres_rgb.device)
                if valid_mask is not None
                else torch.full((B,), T, dtype=torch.long, device=lowres_rgb.device)
            )
            selected_valid_counts = (sel_pos < dense_valid_len[:, None]).sum(dim=1).clamp(max=self.budget)
            support = _support_from_selected_positions(
                sel_pos,
                boundary_prob=None,
                window_size=self.window_size,
            )
            sel_out = {
                "selected_positions": sel_pos,
                "selected_valid_counts": selected_valid_counts,
                "dense_valid_len": dense_valid_len,
                "observed_max_unselected_hole": torch.tensor(
                    [
                        _max_unselected_hole(
                            sel_pos[b, : max(1, int(selected_valid_counts[b].item()))],
                            max(1, int(dense_valid_len[b].item())),
                        )
                        for b in range(B)
                    ],
                    dtype=torch.long,
                    device=sel_pos.device,
                ),
                **support,
            }
            return {
                "scout": None,
                "selection": sel_out,
            }

        # Active semantic acquisition path
        assert_finite_tensor(lowres_rgb, "scout.lowres_rgb", **dict(diagnostic_context or {}))
        scout_out = self.scout(lowres_rgb, valid_mask=valid_mask)
        sel_out = self.selector.select(
            utility=scout_out["utility"],
            boundary_prob=scout_out["boundary_prob"],
            delta_action=scout_out["delta_action"],
            feat_residual=scout_out["feature_residual"],
            context_novelty=scout_out["context_novelty"],
            valid_mask=valid_mask if valid_mask is not None else torch.ones_like(scout_out["utility"], dtype=torch.bool),
        )

        return {
            "scout": scout_out,
            "selection": sel_out,
        }


@MODELS.register_module()
class DucaEvidenceRecoveryFrameSelector(BaseModule):

    """Pre-backbone frame selector implementing the DUCA Evidence Recovery strategy."""

    def __init__(
        self,
        budget: int = 384,
        window_size: int = 768,
        use_coverage: bool = True,
        use_time_conditioning: bool = True,
        use_temporal_merge: bool = True,
        use_dense_recovery: bool = True,
        use_robust_training: bool = True,
        use_h65_selection: bool = False,
        tubelet_size: int = 2,
        max_hole: int = 16,

        h65_position_keys: Optional[Tuple[str, ...]] = None,
        loss_weights: Optional[Dict[str, float]] = None,
        init_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.budget = int(budget)
        self.window_size = int(window_size)
        self.tubelet_size = int(tubelet_size)
        self.use_coverage = bool(use_coverage)
        self.use_time_conditioning = bool(use_time_conditioning)
        self.use_temporal_merge = bool(use_temporal_merge)
        self.use_dense_recovery = bool(use_dense_recovery)
        self.use_robust_training = bool(use_robust_training)
        self.use_h65_selection = bool(use_h65_selection)
        self.max_hole = int(max_hole)
        self.h65_position_keys = tuple(h65_position_keys or H65_POSITION_META_KEYS)


        # Standard detector compatibility attributes
        self.require_counterfactual_utility_teacher = False
        self.last_forward_summary = {}

        self.loss_weights = {
            "scout_action": 1.0,
            "scout_boundary": 1.0,
            "recovery_cycle": 0.1,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)

        self.module = DucaEvidenceRecoveryModule(
            budget=budget,
            window_size=window_size,
            use_coverage=use_coverage,
            use_time_conditioning=use_time_conditioning,
            use_temporal_merge=use_temporal_merge,
            use_dense_recovery=use_dense_recovery,
            use_robust_training=use_robust_training,
            use_h65_selection=use_h65_selection,
            max_hole=max_hole,
        )

    def _dense_valid_lens(self, masks: torch.Tensor) -> torch.Tensor:
        valid_lens = masks.long().sum(dim=-1).to(device=masks.device)
        return valid_lens.clamp_min(1)

    def _extract_h65_positions(
        self,
        metas: Optional[List[Dict[str, Any]]],
        dense_valid_lens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        if not self.use_h65_selection:
            raise RuntimeError("_extract_h65_positions should only be called for H65 replay arms")
        if metas is None or len(metas) != int(dense_valid_lens.numel()):
            raise ValueError("DUCA H65 replay requires one meta dict per batch sample")
        rows = []
        for idx, meta in enumerate(metas):
            if not isinstance(meta, Mapping):
                raise ValueError(f"metas[{idx}] must be a mapping for H65 replay")
            value = None
            source_key = None
            for key in self.h65_position_keys:
                if key in meta and meta[key] is not None:
                    value = meta[key]
                    source_key = key
                    break
            if value is None:
                raise ValueError(
                    "DUCA H65 replay requires selected-position metadata; "
                    f"missing all of {self.h65_position_keys!r} for metas[{idx}]"
                )
            row = torch.as_tensor(value, device=device)
            row = row.reshape(-1)
            active_count = int(row.numel())
            if active_count <= 0 or active_count > self.budget:
                raise ValueError(
                    f"H65 positions from {source_key} must contain 1..{self.budget} entries, "
                    f"got {active_count}"
                )
            if row.is_floating_point():
                rounded = row.round()
                if not bool(torch.allclose(row, rounded, atol=1e-4, rtol=0.0)):
                    raise ValueError(f"H65 positions from {source_key} must be integer frame indices")
                row = rounded
            row = row.to(dtype=torch.long)
            if bool((row < 0).any().item()) or bool((row >= self.window_size).any().item()):
                raise ValueError(f"H65 positions from {source_key} must lie inside [0,{self.window_size})")
            if row.numel() > 1 and bool((row[1:] <= row[:-1]).any().item()):
                raise ValueError(f"H65 positions from {source_key} must be strictly increasing")
            dense_valid_len = int(dense_valid_lens[idx].item())
            if bool((row >= dense_valid_len).any().item()):
                raise ValueError(f"H65 positions from {source_key} exceed dense valid length {dense_valid_len}")
            if active_count < self.budget:
                if dense_valid_len >= self.window_size:
                    raise ValueError(
                        f"H65 positions from {source_key} are short ({active_count}) for a full-valid window"
                    )
                pad_value = min(max(dense_valid_len, int(row[-1].item())), self.window_size - 1)
                row = F.pad(row, (0, self.budget - active_count), value=pad_value)
            rows.append(row)
        return torch.stack(rows, dim=0)

    def _make_gathered_mask(
        self,
        *,
        batch_size: int,
        selected_count: int,
        selected_valid_counts: torch.Tensor,
        masks: torch.Tensor,
    ) -> torch.Tensor:
        gathered_masks = torch.zeros((batch_size, selected_count), dtype=masks.dtype, device=masks.device)
        for b in range(batch_size):
            gathered_masks[b, : min(selected_count, int(selected_valid_counts[b].item()))] = True
        return gathered_masks

    def _can_use_physical_time(self, selected_positions: torch.Tensor, dense_valid_lens: torch.Tensor) -> bool:
        if not self.use_time_conditioning:
            return False
        if selected_positions.ndim != 2 or selected_positions.shape[1] != self.budget:
            return False
        if bool((dense_valid_lens < self.budget).any().item()):
            return False
        if bool((selected_positions[:, 1:] <= selected_positions[:, :-1]).any().item()):
            return False
        if bool((selected_positions >= dense_valid_lens[:, None]).any().item()):
            return False
        return True

    def _initial_detector_positions(self, sel_out: Dict[str, torch.Tensor]) -> torch.Tensor:
        return sel_out["support_centers"]

    def _clone_and_write_metas(
        self,
        metas: Optional[List[Dict[str, Any]]],
        *,
        sel_pos: torch.Tensor,
        dense_valid_lens: torch.Tensor,
        selected_valid_counts: torch.Tensor,
        detector_positions: Optional[torch.Tensor] = None,
        detector_valid_counts: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, Any]]:
        batch = int(sel_pos.shape[0])
        if metas is None:
            out = [{} for _ in range(batch)]
        else:
            if len(metas) != batch:
                raise ValueError("metas length must match batch size")
            out = [dict(meta) for meta in metas]

        acq_cpu = sel_pos.detach().cpu()
        dense_lens_cpu = dense_valid_lens.detach().cpu().long()
        selected_counts_cpu = selected_valid_counts.detach().cpu().long()
        if detector_positions is None:
            detector_positions = self._initial_detector_positions(
                {
                    "support_centers": _support_from_selected_positions(
                        sel_pos,
                        boundary_prob=None,
                        window_size=self.window_size,
                    )["support_centers"]
                }
            )
        detector_cpu = detector_positions.detach().cpu().float()
        if detector_valid_counts is None:
            detector_valid_counts = torch.full(
                (batch,),
                detector_positions.shape[1],
                dtype=torch.long,
                device=sel_pos.device,
            )
        detector_counts_cpu = detector_valid_counts.detach().cpu().long()

        for b, meta in enumerate(out):
            acq_count = min(int(selected_counts_cpu[b].item()), int(acq_cpu.shape[1]))
            acq_positions = [int(v) for v in acq_cpu[b, :acq_count].tolist()]
            dense_valid_len = int(dense_lens_cpu[b].item())
            meta["duca_acquisition_positions"] = acq_positions
            meta["duca_acquisition_budget"] = int(self.budget)
            meta["duca_acquisition_selected_count"] = acq_count
            meta["duca_acquisition_dense_valid_len"] = dense_valid_len
            meta["selected_dense_indices"] = acq_positions
            meta["selected_valid_len"] = acq_count
            meta["truetime_dense_len"] = int(self.window_size)
            meta["truetime_dense_valid_len"] = dense_valid_len
            meta["irregular_dense_valid_len"] = dense_valid_len

            if self.use_dense_recovery:
                meta["dense_recovery_scale"] = float(self.window_size - 1) / float(self.budget - 1)
                meta["irregular_selected_positions"] = acq_positions
                meta["irregular_selected_count"] = acq_count
                meta["irregular_selected_valid_len"] = acq_count
                meta["irregular_native_axis"] = True
                meta["detector_output_coordinate_space"] = TRUE_TIME_AXIS
                meta["detector_prediction_inverse_map_required"] = False
            else:
                det_count = min(int(detector_counts_cpu[b].item()), int(detector_cpu.shape[1]))
                det_positions = [float(v) for v in detector_cpu[b, :det_count].tolist()]
                meta["irregular_selected_positions"] = det_positions
                meta["irregular_selected_count"] = det_count
                meta["irregular_selected_valid_len"] = det_count
                meta["selected_axis_to_true_time_dense_index"] = det_positions
                meta["truetime_selected_positions"] = det_positions
                meta["irregular_native_axis"] = False
                meta["detector_output_coordinate_space"] = SELECTED_AXIS
                meta["detector_prediction_inverse_map_required"] = True
                meta["gt_remapped_to_selected_axis"] = False
                meta["gt_coordinate_space"] = TRUE_TIME_AXIS
        return out

    def _build_extra_backbone_kwargs(
        self,
        sel_out: Dict[str, torch.Tensor],
        *,
        sel_pos: torch.Tensor,
        dense_valid_lens: torch.Tensor,
    ) -> Dict[str, Any]:
        kwargs = {
            "tubelet_z_condition": sel_out["tubelet_z_condition"] if self.use_time_conditioning else None,
            "tubelet_timestamps": sel_out["tubelet_timestamps"] if self.use_time_conditioning else None,
            "boundary_scores": sel_out["boundary_scores_per_tubelet"] if self.use_temporal_merge else None,
            "support_mass": sel_out["support_mass"],
            "support_centers": sel_out["support_centers"],
            "support_intervals": sel_out["support_intervals"],
        }
        if self._can_use_physical_time(sel_pos, dense_valid_lens):
            kwargs["irregular_selected_positions"] = sel_pos
            kwargs["irregular_dense_valid_len"] = dense_valid_lens
        return kwargs

    def _remap_segments_to_axis(
        self,
        gt_segments: List[torch.Tensor],
        metas: List[Dict[str, Any]],
        *,
        target_positions: torch.Tensor,
        valid_counts: torch.Tensor,
    ) -> List[torch.Tensor]:
        remapped = []
        target_cpu = target_positions.detach().cpu()
        for b, (segs, meta) in enumerate(zip(gt_segments, metas)):
            if segs is None or len(segs) == 0:
                remapped.append(segs)
                continue
            count = min(int(valid_counts[b].item()), int(target_cpu.shape[1]))
            if count <= 0:
                remapped.append(segs.new_zeros((0, 2)))
                continue
            positions = target_cpu[b, :count].to(dtype=torch.float32)
            mapper = TrueTimeMap(
                positions,
                dense_len=int(meta.get("truetime_dense_len", self.window_size)),
                valid_len=int(meta.get("truetime_dense_valid_len", self.window_size)),
            )
            mapped = mapper.remap_segments(
                segs,
                source_coordinate_space=TRUE_TIME_AXIS,
                target_coordinate_space=SELECTED_AXIS,
            ).to(device=segs.device, dtype=segs.dtype)
            remapped.append(mapped)
        return remapped

    def _update_no_recovery_axis_after_backbone(
        self,
        *,
        selector_outputs: Dict[str, Any],
        masks: torch.Tensor,
        backbone_support_metadata: Optional[Dict[str, Any]],
    ) -> None:
        if self.use_dense_recovery:
            return
        metas = selector_outputs.get("metas")
        if metas is None:
            return
        centers = None if backbone_support_metadata is None else backbone_support_metadata.get("support_centers")
        if centers is None:
            centers = selector_outputs.get("support_centers")
        if centers is None:
            raise RuntimeError("NO_RECOVERY requires support centers for selected-axis detector mapping")
        if centers.shape[0] != len(metas) or centers.shape[1] != masks.shape[-1]:
            raise RuntimeError(
                "NO_RECOVERY support centers must align with detector feature axis; "
                f"centers={tuple(centers.shape)}, masks={tuple(masks.shape)}"
            )
        valid_counts = masks.long().sum(dim=-1).to(device=centers.device)
        updated_metas = self._clone_and_write_metas(
            metas,
            sel_pos=selector_outputs["selected_positions"],
            dense_valid_lens=selector_outputs["dense_valid_len"].to(device=centers.device),
            selected_valid_counts=selector_outputs["selected_valid_counts"].to(device=centers.device),
            detector_positions=centers,
            detector_valid_counts=valid_counts,
        )
        selector_outputs["metas"] = updated_metas
        if "gt_segments" in selector_outputs:
            source_segments = selector_outputs.get("gt_segments_original", selector_outputs["gt_segments"])
            selector_outputs["gt_segments"] = self._remap_segments_to_axis(
                source_segments,
                updated_metas,
                target_positions=centers,
                valid_counts=valid_counts,
            )
            for meta, original, mapped in zip(updated_metas, source_segments, selector_outputs["gt_segments"]):
                meta["gt_segments_original_time"] = (
                    original.detach().cpu().tolist() if torch.is_tensor(original) else original
                )
                meta["gt_segments_selected_axis"] = mapped.detach().cpu().tolist()
                meta["gt_remapped_to_selected_axis"] = True
                meta["gt_coordinate_space"] = SELECTED_AXIS
                meta["gt_original_coordinate_space"] = TRUE_TIME_AXIS

    def _prepare_lowres_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Resize high-res video inputs to [B, 3, T, 64, 64] for fast scout processing."""
        if inputs.ndim == 6:
            frames = inputs[:, 0]  # [B, 3, T, H, W]
        else:
            frames = inputs  # [B, 3, T, H, W]
        B, C, T, H, W = frames.shape
        if (H, W) == (64, 64):
            return frames
        reshaped = frames.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        lowres = F.interpolate(reshaped.float(), size=(64, 64), mode="bilinear", align_corners=False)
        return lowres.view(B, T, C, 64, 64).permute(0, 2, 1, 3, 4)

    def _gather_frames(self, inputs: torch.Tensor, selected_positions: torch.Tensor) -> torch.Tensor:
        """Gather selected frames along temporal dimension."""
        time_dim = 3 if inputs.ndim == 6 else 2
        pos = selected_positions.to(device=inputs.device, dtype=torch.long).clamp(0, inputs.shape[time_dim] - 1)
        expand_shape = list(inputs.shape)
        expand_shape[time_dim] = pos.shape[1]
        gather_idx = pos.view(
            pos.shape[0],
            *([1] * (time_dim - 1)),
            pos.shape[1],
            *([1] * (inputs.ndim - time_dim - 1)),
        ).expand(expand_shape)
        return torch.gather(inputs, dim=time_dim, index=gather_idx)

    def forward_train(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: List[Dict[str, Any]],
        gt_segments: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        lowres = self._prepare_lowres_inputs(inputs)
        dense_valid_lens = self._dense_valid_lens(masks)
        h65_positions = (
            self._extract_h65_positions(metas, dense_valid_lens, device=inputs.device)
            if self.use_h65_selection
            else None
        )
        diagnostic_context = {"batch_size": int(inputs.shape[0])}
        if metas:
            diagnostic_context["video_name"] = metas[0].get("video_name", metas[0].get("video_id", ""))
        acq = self.module.acquire(
            lowres,
            valid_mask=masks,
            h65_positions=h65_positions,
            diagnostic_context=diagnostic_context,
        )
        scout_out = acq["scout"]
        sel_out = acq["selection"]

        sel_pos = sel_out["selected_positions"]  # [B, 384]
        gathered_inputs = self._gather_frames(inputs, sel_pos)
        gathered_masks = self._make_gathered_mask(
            batch_size=inputs.shape[0],
            selected_count=sel_pos.shape[1],
            selected_valid_counts=sel_out["selected_valid_counts"],
            masks=masks,
        )

        # Compute selector losses (keys must end with _loss)
        losses = {}
        if self.use_robust_training and scout_out is not None:
            B, T = masks.shape
            action_targets = torch.zeros((B, T), dtype=torch.float32, device=inputs.device)
            boundary_targets = torch.zeros((B, T), dtype=torch.float32, device=inputs.device)
            for b, segs in enumerate(gt_segments):
                if segs is not None and len(segs) > 0:
                    for seg in segs:
                        s = int(torch.clamp(seg[0], 0, T - 1).item())
                        e = int(torch.clamp(seg[1], 0, T - 1).item())
                        action_targets[b, s : e + 1] = 1.0
                        boundary_targets[b, max(0, s - 1) : min(T, s + 2)] = 1.0
                        boundary_targets[b, max(0, e - 1) : min(T, e + 2)] = 1.0

            loss_act, _ = balanced_binary_actionness_loss(scout_out["action_logits"], action_targets, masks)
            
            # Masked boundary loss
            loss_bnd_raw = F.binary_cross_entropy_with_logits(scout_out["boundary_logits"], boundary_targets, reduction="none")
            loss_bnd = (loss_bnd_raw * masks.float()).sum() / masks.float().sum().clamp_min(1.0)

            losses["scout_action_loss"] = self.loss_weights["scout_action"] * loss_act
            losses["scout_boundary_loss"] = self.loss_weights["scout_boundary"] * loss_bnd

        if self.use_dense_recovery:
            scale = float(self.budget - 1) / float(self.window_size - 1)
            remapped_gt_segments = [
                segs * scale if segs is not None and len(segs) > 0 else segs
                for segs in gt_segments
            ]
            detector_positions = None
            detector_valid_counts = None
        else:
            remapped_gt_segments = list(gt_segments)
            detector_positions = sel_out["support_centers"]
            detector_valid_counts = sel_out["selected_valid_counts"] // max(1, self.tubelet_size)

        out_metas = self._clone_and_write_metas(
            metas,
            sel_pos=sel_pos,
            dense_valid_lens=dense_valid_lens,
            selected_valid_counts=sel_out["selected_valid_counts"],
            detector_positions=detector_positions,
            detector_valid_counts=detector_valid_counts,
        )
        if not self.use_dense_recovery:
            remapped_gt_segments = self._remap_segments_to_axis(
                remapped_gt_segments,
                out_metas,
                target_positions=sel_out["support_centers"],
                valid_counts=detector_valid_counts,
            )
            for meta, original, mapped in zip(out_metas, gt_segments, remapped_gt_segments):
                meta["gt_segments_original_time"] = (
                    original.detach().cpu().tolist() if torch.is_tensor(original) else original
                )
                meta["gt_segments_selected_axis"] = mapped.detach().cpu().tolist()
                meta["gt_remapped_to_selected_axis"] = True
                meta["gt_coordinate_space"] = SELECTED_AXIS
                meta["gt_original_coordinate_space"] = TRUE_TIME_AXIS

        extra_backbone_kwargs = self._build_extra_backbone_kwargs(
            sel_out,
            sel_pos=sel_pos,
            dense_valid_lens=dense_valid_lens,
        )

        return {
            "inputs": gathered_inputs,
            "masks": gathered_masks,
            "metas": out_metas,
            "gt_segments": remapped_gt_segments,
            "gt_labels": gt_labels,
            "gt_segments_original": gt_segments,
            "losses": losses,
            "extra_backbone_kwargs": extra_backbone_kwargs,
            "selected_positions": sel_pos,
            "selected_valid_counts": sel_out["selected_valid_counts"],
            "dense_valid_len": dense_valid_lens,
            "observed_max_unselected_hole": sel_out.get("observed_max_unselected_hole"),
            "support_centers": sel_out["support_centers"],
            "support_intervals": sel_out["support_intervals"],
            "support_mass": sel_out["support_mass"],
        }

    def forward_test(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
        metas: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        lowres = self._prepare_lowres_inputs(inputs)
        dense_valid_lens = self._dense_valid_lens(masks)
        h65_positions = (
            self._extract_h65_positions(metas, dense_valid_lens, device=inputs.device)
            if self.use_h65_selection
            else None
        )
        diagnostic_context = {"batch_size": int(inputs.shape[0])}
        if metas:
            diagnostic_context["video_name"] = metas[0].get("video_name", metas[0].get("video_id", ""))
        acq = self.module.acquire(
            lowres,
            valid_mask=masks,
            h65_positions=h65_positions,
            diagnostic_context=diagnostic_context,
        )
        sel_out = acq["selection"]

        sel_pos = sel_out["selected_positions"]
        gathered_inputs = self._gather_frames(inputs, sel_pos)
        gathered_masks = self._make_gathered_mask(
            batch_size=inputs.shape[0],
            selected_count=sel_pos.shape[1],
            selected_valid_counts=sel_out["selected_valid_counts"],
            masks=masks,
        )

        if self.use_dense_recovery:
            detector_positions = None
            detector_valid_counts = None
        else:
            detector_positions = sel_out["support_centers"]
            detector_valid_counts = sel_out["selected_valid_counts"] // max(1, self.tubelet_size)
        out_metas = self._clone_and_write_metas(
            metas,
            sel_pos=sel_pos,
            dense_valid_lens=dense_valid_lens,
            selected_valid_counts=sel_out["selected_valid_counts"],
            detector_positions=detector_positions,
            detector_valid_counts=detector_valid_counts,
        )

        extra_backbone_kwargs = self._build_extra_backbone_kwargs(
            sel_out,
            sel_pos=sel_pos,
            dense_valid_lens=dense_valid_lens,
        )

        return {
            "inputs": gathered_inputs,
            "masks": gathered_masks,
            "metas": out_metas,
            "extra_backbone_kwargs": extra_backbone_kwargs,
            "selected_positions": sel_pos,
            "selected_valid_counts": sel_out["selected_valid_counts"],
            "dense_valid_len": dense_valid_lens,
            "observed_max_unselected_hole": sel_out.get("observed_max_unselected_hole"),
            "support_centers": sel_out["support_centers"],
            "support_intervals": sel_out["support_intervals"],
            "support_mass": sel_out["support_mass"],
        }

    def recover_features(
        self,
        feats: torch.Tensor,
        masks: torch.Tensor,
        selector_outputs: Dict[str, Any],
        backbone_support_metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Recover gathered and merged features back to uniform detection grid."""
        if not self.use_dense_recovery:
            # If recovery is disabled, adjust masks length to match feats length
            N_tokens = feats.shape[-1]
            if masks.shape[-1] != N_tokens:
                selected_valid_counts = selector_outputs.get("selected_valid_counts")
                if selected_valid_counts is None:
                    ratio = float(N_tokens) / float(masks.shape[-1])
                    valid_lens = (masks.long().sum(dim=-1).float() * ratio).round().long()
                else:
                    valid_lens = (
                        selected_valid_counts.to(device=masks.device).float()
                        * float(N_tokens)
                        / float(masks.shape[-1])
                    ).floor().long()
                new_masks = torch.zeros((feats.shape[0], N_tokens), dtype=masks.dtype, device=masks.device)
                for b in range(feats.shape[0]):
                    new_masks[b, :min(N_tokens, valid_lens[b].item())] = True
                masks = new_masks
            self._update_no_recovery_axis_after_backbone(
                selector_outputs=selector_outputs,
                masks=masks,
                backbone_support_metadata=backbone_support_metadata,
            )
            return feats, masks

        # If backbone performed Token Merging, use updated support coordinates
        if backbone_support_metadata is not None:
            centers = backbone_support_metadata.get("support_centers")
            intervals = backbone_support_metadata.get("support_intervals")
            masses = backbone_support_metadata.get("support_mass")
        else:
            centers = selector_outputs.get("support_centers")
            intervals = selector_outputs.get("support_intervals")
            masses = selector_outputs.get("support_mass")

        N_tokens = feats.shape[-1]
        if centers is None or intervals is None or masses is None:
            raise RuntimeError("dense recovery requires support centers, intervals and masses")
        if centers.shape[0] != feats.shape[0] or centers.shape[1] != N_tokens:
            raise RuntimeError(
                "dense recovery support coordinates must align with backbone feature axis; "
                f"centers={tuple(centers.shape)}, feats={tuple(feats.shape)}"
            )
        if masks.shape[-1] == N_tokens:
            support_mask = masks.to(device=feats.device, dtype=torch.bool)
        else:
            support_mask = torch.ones((feats.shape[0], N_tokens), dtype=torch.bool, device=feats.device)

        dense_valid_lens = selector_outputs.get("dense_valid_len")
        if dense_valid_lens is None:
            ratio = float(self.budget) / float(masks.shape[-1]) if masks.shape[-1] != self.budget else 1.0
            valid_lens = (masks.long().sum(dim=-1).float() * ratio).round().long()
        else:
            valid_lens = (
                dense_valid_lens.to(device=masks.device).float()
                * float(self.budget)
                / float(self.window_size)
            ).ceil().long()

        recovered = self.module.recovery(
            feats=feats,
            centers=centers,
            intervals=intervals,
            masses=masses,
            valid_mask=support_mask,
            dense_valid_len=(
                selector_outputs.get("dense_valid_len")
                if selector_outputs.get("dense_valid_len") is not None
                else masks.long().sum(dim=-1)
            ),
        )

        # Recovered mask has length self.budget matching target grid
        new_masks = torch.zeros((feats.shape[0], self.budget), dtype=masks.dtype, device=masks.device)
        for b in range(feats.shape[0]):
            new_masks[b, :min(self.budget, valid_lens[b].item())] = True
        # Recovery must never expose interpolated values beyond each sample's
        # valid dense prefix.  Keep the mask and feature tensor in lockstep so
        # downstream projection/FPN/head cannot consume padded tail evidence.
        recovered = recovered * new_masks.to(device=recovered.device, dtype=recovered.dtype).unsqueeze(1)
        assert_finite_tensor(recovered, "recovery.masked_output")
        return recovered, new_masks
