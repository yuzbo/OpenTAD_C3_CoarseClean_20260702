import math

import torch
import torch.nn as nn

from ..builder import PROJECTIONS
from ..utils.phystime_geometry import (
    build_physical_query_pyramid,
    geometry_from_metas,
    support_overlap_mass,
)


class PhysicalQueryEmbedding(nn.Module):
    def __init__(self, out_channels, num_fourier_bands=4):
        super().__init__()
        frequencies = 2.0 ** torch.arange(int(num_fourier_bands), dtype=torch.float32)
        self.register_buffer("frequencies", frequencies, persistent=False)
        in_channels = 5 + 2 * int(num_fourier_bands)
        self.net = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels),
        )

    def forward(self, centers_sec, widths_sec, duration_sec):
        duration = duration_sec[:, None].clamp_min(torch.finfo(centers_sec.dtype).eps)
        normalized_center = centers_sec / duration
        normalized_width = widths_sec / duration
        phase = normalized_center[..., None] * self.frequencies.to(dtype=centers_sec.dtype) * (2.0 * math.pi)
        features = torch.cat(
            (
                centers_sec[..., None],
                normalized_center[..., None],
                torch.log1p(duration).expand_as(centers_sec)[..., None],
                widths_sec[..., None],
                normalized_width[..., None],
                torch.sin(phase),
                torch.cos(phase),
            ),
            dim=-1,
        )
        return self.net(features)


class SupportIntegratedMeasureAttention(nn.Module):
    """Cross-attention whose base measure is physical support overlap."""

    def __init__(
        self,
        in_channels,
        out_channels,
        attention_channels=128,
        content_logits=True,
        relative_time_logits=True,
        observation_measure="support_overlap",
        point_radius_cells=4.0,
        dropout=0.0,
        keep_uncovered_queries=False,
        use_null_evidence=True,
        support_context_scale=1.0,
        min_assignment_coverage=0.0,
        eps=1.0e-8,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.attention_channels = int(attention_channels)
        self.content_logits = bool(content_logits)
        self.relative_time_logits = bool(relative_time_logits)
        self.observation_measure = str(observation_measure)
        self.point_radius_cells = float(point_radius_cells)
        self.keep_uncovered_queries = bool(keep_uncovered_queries)
        self.use_null_evidence = bool(use_null_evidence)
        self.support_context_scale = float(support_context_scale)
        self.min_assignment_coverage = float(min_assignment_coverage)
        self.eps = float(eps)
        if self.observation_measure not in {"support_overlap", "point_gaussian"}:
            raise ValueError(f"unsupported observation_measure: {self.observation_measure}")
        if self.point_radius_cells <= 0:
            raise ValueError("point_radius_cells must be positive")
        if self.support_context_scale <= 0:
            raise ValueError("support_context_scale must be positive")
        if self.min_assignment_coverage < 0:
            raise ValueError("min_assignment_coverage must be non-negative")

        self.value_proj = nn.Linear(self.in_channels, self.out_channels)
        self.output_proj = nn.Linear(self.out_channels, self.out_channels)
        self.query_value_embedding = PhysicalQueryEmbedding(self.out_channels)
        self.coverage_embedding = nn.Sequential(
            nn.Linear(2, self.out_channels),
            nn.GELU(),
            nn.Linear(self.out_channels, self.out_channels),
        )
        nn.init.zeros_(self.query_value_embedding.net[-1].weight)
        nn.init.zeros_(self.query_value_embedding.net[-1].bias)
        nn.init.zeros_(self.coverage_embedding[-1].weight)
        nn.init.zeros_(self.coverage_embedding[-1].bias)
        if self.keep_uncovered_queries and self.use_null_evidence:
            self.null_evidence = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.register_parameter("null_evidence", None)
        self.dropout = nn.Dropout(float(dropout))
        if self.content_logits:
            self.query_embedding = PhysicalQueryEmbedding(self.attention_channels)
            self.key_proj = nn.Linear(self.in_channels, self.attention_channels)
        if self.relative_time_logits:
            self.relative_time_mlp = nn.Sequential(
                nn.Linear(4, self.attention_channels),
                nn.GELU(),
                nn.Linear(self.attention_channels, 1),
            )

    def forward(self, observations, observation_geometry, query_geometry):
        if observations.ndim != 3:
            raise ValueError("observations must have shape [B, K, C]")
        timestamps = observation_geometry["timestamps_sec"]
        ownership = observation_geometry["ownership_intervals_sec"]
        observation_mask = observation_geometry["valid_mask"]
        duration = observation_geometry["duration_sec"]
        query_intervals = query_geometry["intervals_sec"]
        query_mask = query_geometry["valid_mask"]
        query_centers = query_geometry["centers_sec"]
        query_widths = query_geometry["widths_sec"]
        if observations.shape[:2] != timestamps.shape or observations.shape[-1] != self.in_channels:
            raise ValueError("observation features and physical geometry have incompatible shapes")

        context_widths = (query_widths * self.support_context_scale).clamp_min(self.eps)
        context_intervals = torch.stack(
            (
                query_centers - 0.5 * context_widths,
                query_centers + 0.5 * context_widths,
            ),
            dim=-1,
        )
        if self.observation_measure == "support_overlap":
            mass = support_overlap_mass(ownership, context_intervals, observation_mask)
        else:
            safe_width = context_widths[:, :, None].clamp_min(self.eps)
            normalized_distance = (timestamps[:, None, :] - query_centers[:, :, None]).abs() / safe_width
            mass = torch.exp(-0.5 * normalized_distance.square())
            mass = mass * (normalized_distance <= self.point_radius_cells).to(mass.dtype)
            mass = mass * observation_mask[:, None, :].to(mass.dtype)
        # Query-pyramid padding uses zero geometry.  The epsilon-clamped context
        # interval can otherwise overlap support at t=0 in support-overlap mode.
        mass = mass * query_mask[:, :, None].to(mass.dtype)
        logits = observations.new_zeros(mass.shape)
        if self.content_logits:
            queries = self.query_embedding(query_centers, query_widths, duration)
            keys = self.key_proj(observations)
            logits = logits + torch.einsum("bqd,bkd->bqk", queries, keys) / math.sqrt(self.attention_channels)
        if self.relative_time_logits:
            safe_width = query_widths[:, :, None].clamp_min(self.eps)
            signed_offset = (timestamps[:, None, :] - query_centers[:, :, None]) / safe_width
            if self.observation_measure == "support_overlap":
                support_width = ownership[..., 1] - ownership[..., 0]
            else:
                support_width = torch.zeros_like(timestamps)
            relative_features = torch.stack(
                (
                    signed_offset,
                    signed_offset.abs(),
                    support_width[:, None, :] / safe_width,
                    torch.log1p(signed_offset.abs()),
                ),
                dim=-1,
            )
            logits = logits + self.relative_time_mlp(relative_features).squeeze(-1)

        covered = mass > 0
        masked_logits = logits.masked_fill(~covered, float("-inf"))
        row_max = masked_logits.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        unnormalized = torch.exp(masked_logits - row_max) * mass
        denominator = unnormalized.sum(dim=-1, keepdim=True)
        weights = unnormalized / denominator.clamp_min(self.eps)
        weights = self.dropout(weights)

        values = self.value_proj(observations)
        output = self.output_proj(torch.einsum("bqk,bkc->bqc", weights, values))
        if self.observation_measure == "support_overlap":
            coverage = mass.sum(dim=-1)
        else:
            coverage = (mass.sum(dim=-1) > self.eps).to(mass.dtype) * query_widths
        covered_mask = query_mask & (coverage > self.eps)
        output_mask = query_mask if self.keep_uncovered_queries else covered_mask
        if self.null_evidence is not None:
            null = self.null_evidence.to(dtype=output.dtype, device=output.device)
            output = torch.where(covered_mask[..., None], output, null.view(1, 1, -1))
        coverage_ratio = (coverage / query_widths.clamp_min(self.eps)).clamp(0.0, 1.0)
        query_value = self.query_value_embedding(query_centers, query_widths, duration).to(dtype=output.dtype)
        coverage_feature = self.coverage_embedding(
            torch.stack((coverage_ratio, torch.log1p(coverage_ratio)), dim=-1).to(dtype=output.dtype)
        )
        output = output + query_value + coverage_feature
        output = output * output_mask[..., None].to(output.dtype)
        assignment_mask = query_mask & (coverage_ratio >= self.min_assignment_coverage)
        diagnostics = {
            "attention_weights": weights,
            "overlap_mass": mass,
            "coverage_sec": coverage,
            "covered_mask": covered_mask,
            "coverage_ratio": coverage_ratio,
            "domain_valid_mask": query_mask,
            "evidence_mask": covered_mask,
            "assignment_mask": assignment_mask,
        }
        return output, output_mask, diagnostics


@PROJECTIONS.register_module()
class PhysTimeMeasureProjection(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        base_spacing_sec,
        num_levels,
        attention_channels=128,
        observation_measure="support_overlap",
        point_radius_cells=4.0,
        dropout=0.0,
        keep_uncovered_queries=False,
        use_null_evidence=True,
        support_context_scale=1.0,
        min_assignment_coverage=0.0,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.base_spacing_sec = float(base_spacing_sec)
        self.num_levels = int(num_levels)
        self.keep_uncovered_queries = bool(keep_uncovered_queries)
        self.use_null_evidence = bool(use_null_evidence)
        self.support_context_scale = float(support_context_scale)
        self.min_assignment_coverage = float(min_assignment_coverage)
        self.level_attentions = nn.ModuleList(
            [
                SupportIntegratedMeasureAttention(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    attention_channels=attention_channels,
                    observation_measure=observation_measure,
                    point_radius_cells=point_radius_cells,
                    dropout=dropout,
                    keep_uncovered_queries=self.keep_uncovered_queries,
                    use_null_evidence=self.use_null_evidence,
                    support_context_scale=self.support_context_scale,
                    min_assignment_coverage=self.min_assignment_coverage,
                )
                for _ in range(self.num_levels)
            ]
        )

    def forward(self, x, masks, metas):
        if x.ndim != 3:
            raise ValueError("PhysTimeMeasureProjection expects features with shape [B, C, K]")
        if x.shape[1] != self.in_channels or masks.shape != (x.shape[0], x.shape[2]):
            raise ValueError("PhysTime projection input feature and mask shapes are inconsistent")
        # Absolute seconds can exceed 1,000 on THUMOS. FP16 cannot preserve
        # sub-frame support widths at that magnitude and content logits can
        # overflow, so the complete physical measure path stays in FP32.
        with torch.cuda.amp.autocast(enabled=False):
            geometry = geometry_from_metas(
                metas, masks, dtype=torch.float32, device=x.device
            )
            query_pyramid = build_physical_query_pyramid(
                geometry["duration_sec"],
                geometry["domain_start_sec"],
                geometry["domain_end_sec"],
                base_spacing_sec=self.base_spacing_sec,
                num_levels=self.num_levels,
            )

            observations = x.float().transpose(1, 2)
            features = []
            level_masks = []
            level_geometry = []
            for attention, query_geometry in zip(self.level_attentions, query_pyramid):
                level_feature, level_mask, diagnostics = attention(
                    observations, geometry, query_geometry
                )
                features.append(level_feature.transpose(1, 2))
                level_masks.append(level_mask)
                level_info = dict(query_geometry)
                level_info["domain_valid_mask"] = query_geometry["valid_mask"]
                level_info["evidence_mask"] = diagnostics["evidence_mask"]
                level_info["assignment_mask"] = diagnostics["assignment_mask"]
                level_info["valid_mask"] = level_mask
                level_info["coverage_sec"] = diagnostics["coverage_sec"]
                level_info["coverage_ratio"] = diagnostics["coverage_ratio"]
                level_geometry.append(level_info)
        return tuple(features), tuple(level_masks), tuple(level_geometry)
