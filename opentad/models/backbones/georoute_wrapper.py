"""Native-token GeoRoute backbone for a single-heavy-forward AdaTAD path."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .backbone_wrapper import BackboneWrapper
from .georoute_routing import (
    DYNAMIC_BRANCH_CALIBRATION_MODES,
    DYNAMIC_ROUTE_MODES,
    POLICY_ESTIMATORS,
    ROUTE_MODES,
    SCORE_FUNCTION_TEMPORAL_REDUCTIONS,
    STRUCTURED_ROUTE_MODES,
    build_refresh_mask,
    calibrate_dynamic_residual_modifier,
    decode_continuous_geometry,
    interpolate_temporal_knots,
    native_cell_extent_floor,
    native_patch_centers,
    roi_logits_from_geometry,
    roi_modifier_from_geometry,
    score_function_policy_loss,
    select_continuous_strict_rectangle,
    select_dynamic_global_exact_budget,
    select_exact_k,
    select_fixed_quota_structured_exact_k,
    select_qbase_global_exact_k,
    select_rectangle_constrained_qbase_8x8,
    select_rectangle_core_outside_qbase_7x7,
    select_strict_rectangle_8x8,
)
from .native_crop_wrapper import deterministic_linear_2x


GEOROUTE_BACKBONE_SCHEMA = "georoute_native_packed_backbone_v7"

OFFICIAL_QBASE_SUPPORTS = frozenset(
    {
        "strict_rect8x8_q48",
        "strict_rect8x8_shuf48",
        "q48_global",
        "strict_rect7x7_core49_q15",
        "strict_rect7x7_core49_shuf15",
        "q64_global",
    }
)
OFFICIAL_R3_SUPPORTS = frozenset(
    {
        "continuous_rect_dynamic",
        "continuous_rect_dynamic_area_shift97",
    }
)
OFFICIAL_MULTIBRANCH_SUPPORTS = OFFICIAL_QBASE_SUPPORTS | OFFICIAL_R3_SUPPORTS


class GeoRouteScout(nn.Module):
    """Low-cost global observer that predicts geometry and residual saliency."""

    def __init__(
        self,
        channels: int = 48,
        *,
        dynamic_utility: bool = False,
    ) -> None:
        super().__init__()
        if int(channels) <= 0:
            raise ValueError("scout channels must be positive")
        self.stem = nn.Sequential(
            nn.Conv3d(3, channels, kernel_size=(2, 8, 8), stride=(2, 8, 8), bias=False),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.GELU(),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=channels,
                bias=False,
            ),
            nn.GroupNorm(num_groups=min(8, channels), num_channels=channels),
            nn.GELU(),
        )
        self.geometry_head = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(channels, 4, kernel_size=1),
        )
        self.residual_head = nn.Conv3d(channels, 1, kernel_size=1)
        self.base_utility_head = (
            nn.Conv3d(channels, 1, kernel_size=1)
            if bool(dynamic_utility)
            else None
        )

    @staticmethod
    def _resize_native_field(
        field: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        if field.ndim != 4:
            raise ValueError("scout native field must be [B,T,H,W]")
        batch, tubelets, scout_h, scout_w = map(int, field.shape)
        target_h, target_w = map(int, source_grid_hw)
        return F.interpolate(
            field.reshape(batch * tubelets, 1, scout_h, scout_w),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, tubelets, target_h * target_w)

    def _encode(self, scout: torch.Tensor) -> torch.Tensor:
        if scout.ndim != 5 or scout.shape[1] != 3:
            raise ValueError("scout must be [B,3,T,H,W]")
        return self.stem(scout)

    def forward(self, scout: torch.Tensor, *, source_grid_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        features = self._encode(scout)
        geometry_logits = self.geometry_head(features.mean(dim=(-1, -2))).transpose(1, 2)
        residual = self._resize_native_field(
            self.residual_head(features).squeeze(1),
            source_grid_hw=source_grid_hw,
        )
        return geometry_logits, residual

    def forward_dynamic(
        self,
        scout: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return stop-gradient policy fields and live auxiliary scout features."""

        if self.base_utility_head is None:
            raise RuntimeError("dynamic scout utility head is not configured")
        features = self._encode(scout)
        policy_features = features.detach()
        geometry_logits = self.geometry_head(
            policy_features.mean(dim=(-1, -2))
        ).transpose(1, 2)
        q_base = self._resize_native_field(
            self.base_utility_head(policy_features).squeeze(1),
            source_grid_hw=source_grid_hw,
        )
        residual = self._resize_native_field(
            self.residual_head(policy_features).squeeze(1),
            source_grid_hw=source_grid_hw,
        )
        return geometry_logits, q_base, residual, features


class GeoRouteSparseTemporalAdapter(nn.Module):
    """Aggregate selected spatial tokens with explicit route geometry.

    The adapter receives the exact native-patch coordinates selected by the
    hard route.  It can therefore distinguish a token's source-frame location
    and its offset inside the continuous ROI without materializing a resized
    crop or a second VideoMAE pass.  Disabling this input is a registered
    ablation rather than an implicit fallback.
    """

    def __init__(self, channels: int = 384) -> None:
        super().__init__()
        self.geometry_projection = nn.Linear(4, channels)
        self.coordinate_projection = nn.Sequential(
            nn.Linear(4, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.norm = nn.LayerNorm(channels)
        self.temporal = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.output = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(
        self,
        selected_features: torch.Tensor,
        selected_scores: torch.Tensor,
        geometry: torch.Tensor,
        selected_coordinates: torch.Tensor,
        *,
        use_absolute_coordinates: bool,
        use_roi_relative_coordinates: bool,
        use_geometry_projection: bool,
        pooling_mode: str,
    ) -> torch.Tensor:
        if selected_features.ndim != 4:
            raise ValueError("selected features must be [B,T,K,C]")
        if selected_scores.shape != selected_features.shape[:3]:
            raise ValueError("selected score geometry must match selected features")
        if geometry.shape != (*selected_features.shape[:2], 4):
            raise ValueError("continuous geometry must be [B,T,4]")
        if selected_coordinates.shape != (*selected_features.shape[:3], 2):
            raise ValueError("selected coordinates must be [B,T,K,2]")
        if not bool(torch.isfinite(selected_coordinates).all().item()):
            raise ValueError("selected native coordinates must be finite")
        if use_absolute_coordinates or use_roi_relative_coordinates:
            absolute = selected_coordinates if use_absolute_coordinates else torch.zeros_like(selected_coordinates)
            relative = (selected_coordinates - geometry[:, :, None, :2]) / geometry[:, :, None, 2:].clamp_min(1e-6)
            if not use_roi_relative_coordinates:
                relative = torch.zeros_like(relative)
            coordinate_features = torch.cat((absolute, relative), dim=-1)
            selected_features = selected_features + self.coordinate_projection(coordinate_features)
        if pooling_mode == "uniform_selected":
            weights = torch.full_like(
                selected_scores,
                1.0 / float(selected_scores.shape[-1]),
            ).unsqueeze(-1)
        elif pooling_mode == "route_score_ablation":
            weights = torch.softmax(selected_scores, dim=-1).unsqueeze(-1)
        else:
            raise ValueError(f"unsupported GeoRoute pooling mode {pooling_mode!r}")
        pooled = (weights * selected_features).sum(dim=2)
        if use_geometry_projection:
            pooled = pooled + self.geometry_projection(geometry)
        pooled = self.norm(pooled)
        temporal = self.output(self.temporal(pooled.transpose(1, 2))).transpose(1, 2)
        return (pooled + temporal).transpose(1, 2)

    def forward_ragged(
        self,
        selected_features: torch.Tensor,
        selected_scores: torch.Tensor,
        geometry: torch.Tensor,
        selected_coordinates: torch.Tensor,
        tubelet_indices: torch.Tensor,
        *,
        use_absolute_coordinates: bool,
        use_roi_relative_coordinates: bool,
        use_geometry_projection: bool,
        pooling_mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Aggregate a true ragged union with an explicit masked-zero carrier."""

        if selected_features.ndim != 3:
            raise ValueError("ragged selected features must be [B,S,C]")
        if selected_scores.shape != selected_features.shape[:2]:
            raise ValueError("ragged selected scores must match [B,S]")
        if tubelet_indices.shape != selected_features.shape[:2]:
            raise ValueError("ragged tubelet indices must match [B,S]")
        if tubelet_indices.dtype != torch.long:
            raise TypeError("ragged tubelet indices must be torch.long")
        if selected_coordinates.shape != (*selected_features.shape[:2], 2):
            raise ValueError("ragged selected coordinates must be [B,S,2]")
        if geometry.ndim != 3 or geometry.shape[0] != selected_features.shape[0] or geometry.shape[-1] != 4:
            raise ValueError("ragged continuous geometry must be [B,T,4]")
        if pooling_mode != "uniform_selected":
            raise ValueError(
                "dynamic ragged SCNR requires uniform_selected pooling"
            )
        batch_size, selected_count, channels = map(int, selected_features.shape)
        tubelets = int(geometry.shape[1])
        if selected_count <= 0:
            raise ValueError("ragged aggregation requires a positive window budget")
        if bool(
            ((tubelet_indices < 0) | (tubelet_indices >= tubelets)).any().item()
        ):
            raise ValueError("ragged tubelet index falls outside geometry time")
        if not bool(torch.isfinite(selected_coordinates).all().item()):
            raise ValueError("ragged selected coordinates must be finite")

        if use_absolute_coordinates or use_roi_relative_coordinates:
            selected_geometry = geometry.gather(
                1,
                tubelet_indices.unsqueeze(-1).expand(
                    batch_size,
                    selected_count,
                    4,
                ),
            )
            absolute = (
                selected_coordinates
                if use_absolute_coordinates
                else torch.zeros_like(selected_coordinates)
            )
            relative = (
                selected_coordinates - selected_geometry[..., :2]
            ) / selected_geometry[..., 2:].clamp_min(1e-6)
            if not use_roi_relative_coordinates:
                relative = torch.zeros_like(relative)
            selected_features = selected_features + self.coordinate_projection(
                torch.cat((absolute, relative), dim=-1)
            )

        pooled = selected_features.new_zeros(
            batch_size,
            tubelets,
            channels,
        )
        pooled.scatter_add_(
            1,
            tubelet_indices.unsqueeze(-1).expand(
                batch_size,
                selected_count,
                channels,
            ),
            selected_features,
        )
        counts = selected_features.new_zeros(batch_size, tubelets)
        counts.scatter_add_(
            1,
            tubelet_indices,
            torch.ones_like(tubelet_indices, dtype=selected_features.dtype),
        )
        heavy_valid_mask = counts > 0
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        if use_geometry_projection:
            pooled = pooled + self.geometry_projection(geometry)
        valid = heavy_valid_mask.unsqueeze(-1).to(dtype=pooled.dtype)
        pooled = self.norm(pooled) * valid
        temporal = self.output(
            self.temporal(pooled.transpose(1, 2))
        ).transpose(1, 2)
        # Re-mask after every bias-bearing temporal path.  Empty tubelets are an
        # exact zero carrier rather than an unmarked learned pseudo-observation.
        output = (pooled + temporal) * valid
        empty = (~heavy_valid_mask).unsqueeze(-1).expand_as(output)
        if not bool(output.masked_select(empty).eq(0).all().item()):
            raise RuntimeError("masked-zero carrier became content bearing")
        return output.transpose(1, 2), heavy_valid_mask


def _normalize_uint8_video(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.uint8:
        raise TypeError("GeoRoute source/scout inputs must remain uint8 before native patch gather")
    return (value.to(torch.float32) - mean) / std


class GeoRoutePostBackboneAggregationWrapper(BackboneWrapper):
    """Official dense VideoMAE forward with a matched post-backbone adapter.

    Arm B aggregates all 100 dense spatial tokens per tubelet.  Arm C applies
    the existing ROI-only exact-K selector to that same completed dense feature
    lattice before the identical sparse adapter.  Neither arm changes the heavy
    VideoMAE execution surface, so this path is accuracy/causal evidence only
    and cannot support an efficiency claim.
    """

    WRAPPER_TYPE = "georoute_postbackbone_sparse_aggregation_v1"

    def __init__(self, cfg) -> None:
        custom_cfg = cfg.custom
        self.selection_mode = str(
            getattr(custom_cfg, "georoute_postbackbone_selection", "all")
        )
        self.window_size = int(
            getattr(custom_cfg, "georoute_postbackbone_window_size", 768)
        )
        self.chunk_num = int(
            getattr(custom_cfg, "georoute_postbackbone_chunk_num", 48)
        )
        self.tubelet_size = int(
            getattr(custom_cfg, "georoute_postbackbone_tubelet_size", 2)
        )
        self.source_grid_hw = tuple(
            int(value)
            for value in getattr(
                custom_cfg,
                "georoute_postbackbone_source_grid_hw",
                (10, 10),
            )
        )
        self.roi_tokens = int(
            getattr(custom_cfg, "georoute_postbackbone_roi_tokens", 64)
        )
        self.scout_size = int(
            getattr(custom_cfg, "georoute_postbackbone_scout_size", 96)
        )
        self.roi_temperature = float(
            getattr(custom_cfg, "georoute_postbackbone_roi_temperature", 0.25)
        )
        self.policy_temperature = float(
            getattr(custom_cfg, "georoute_postbackbone_policy_temperature", 0.5)
        )
        self.min_roi_extent_cells = int(
            getattr(
                custom_cfg,
                "georoute_postbackbone_min_roi_extent_cells",
                1,
            )
        )
        self.max_roi_extent = float(
            getattr(custom_cfg, "georoute_postbackbone_max_roi_extent", 1.0)
        )
        self.pooling_mode = str(
            getattr(
                custom_cfg,
                "georoute_postbackbone_pooling_mode",
                "uniform_selected",
            )
        )
        side_channels = {
            "absolute_coordinates": bool(
                getattr(
                    custom_cfg,
                    "georoute_postbackbone_absolute_coordinates_enabled",
                    False,
                )
            ),
            "roi_relative_coordinates": bool(
                getattr(
                    custom_cfg,
                    "georoute_postbackbone_roi_relative_coordinates_enabled",
                    False,
                )
            ),
            "geometry_projection": bool(
                getattr(
                    custom_cfg,
                    "georoute_postbackbone_geometry_projection_enabled",
                    False,
                )
            ),
        }
        if self.selection_mode not in {"all", "roi"}:
            raise ValueError("post-backbone selection must be 'all' or 'roi'")
        if self.window_size != 768 or self.chunk_num != 48 or self.tubelet_size != 2:
            raise ValueError(
                "official post-backbone B/C requires the frozen 768/48/tubelet-2 lattice"
            )
        if self.source_grid_hw != (10, 10):
            raise ValueError("official post-backbone B/C requires the 10x10 feature grid")
        if self.selection_mode == "roi" and self.roi_tokens != 64:
            raise ValueError("official post-backbone C requires exact ROI K=64")
        if self.scout_size != 96:
            raise ValueError("official post-backbone C requires the existing 96x96 scout")
        if self.pooling_mode != "uniform_selected":
            raise ValueError("official post-backbone B/C requires uniform sparse aggregation")
        if any(side_channels.values()):
            raise ValueError("official post-backbone B/C forbids adapter side channels")
        if not (0.0 < self.roi_temperature and 0.0 < self.policy_temperature):
            raise ValueError("official post-backbone ROI temperatures must be positive")
        if self.min_roi_extent_cells != 1 or self.max_roi_extent != 1.0:
            raise ValueError("official post-backbone C requires the frozen native-cell extent bounds")

        super().__init__(cfg)
        channels = int(self.model.backbone.embed_dims)
        # Construct this module before C's conditional selector so B and C use
        # the same seeded sparse-adapter initialization.
        self.sparse_adapter = GeoRouteSparseTemporalAdapter(channels=channels)
        for module in (
            self.sparse_adapter.geometry_projection,
            self.sparse_adapter.coordinate_projection,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

        if self.selection_mode == "roi":
            self.scout = GeoRouteScout(channels=48, dynamic_utility=False)
            for parameter in self.scout.residual_head.parameters():
                parameter.requires_grad = False
            self.register_buffer(
                "source_mean",
                torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1),
            )
            self.register_buffer(
                "source_std",
                torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1),
            )
        self._pending_postbackbone_route: dict[str, torch.Tensor] | None = None

    def _compute_roi_route(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 6
            or tuple(frames.shape[1:3]) != (1, 3)
            or frames.dtype != torch.uint8
        ):
            raise ValueError(
                "official post-backbone C requires augmented uint8 [B,1,3,T,H,W] input"
            )
        source = frames[:, 0]
        if int(source.shape[2]) != self.window_size or tuple(source.shape[-2:]) != (
            160,
            160,
        ):
            raise ValueError(
                "official post-backbone C requires the augmented 768x160x160 input"
            )
        batch_size, _channels, frame_count, height, width = map(int, source.shape)
        with torch.autocast(device_type=source.device.type, enabled=False):
            normalized = _normalize_uint8_video(
                source,
                self.source_mean,
                self.source_std,
            )
            scout_input = F.interpolate(
                normalized.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * frame_count,
                    3,
                    height,
                    width,
                ),
                size=(self.scout_size, self.scout_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(
                batch_size,
                frame_count,
                3,
                self.scout_size,
                self.scout_size,
            ).permute(0, 2, 1, 3, 4).contiguous()
            scout_features = self.scout._encode(scout_input)
            geometry_logits = self.scout.geometry_head(
                scout_features.mean(dim=(-1, -2))
            ).transpose(1, 2)
            geometry = decode_continuous_geometry(
                interpolate_temporal_knots(geometry_logits, stride=1),
                min_extent=native_cell_extent_floor(
                    self.source_grid_hw[0],
                    self.source_grid_hw[1],
                    cells_per_axis=self.min_roi_extent_cells,
                ),
                max_extent=self.max_roi_extent,
            )
            roi_logits = roi_logits_from_geometry(
                geometry,
                grid_height=self.source_grid_hw[0],
                grid_width=self.source_grid_hw[1],
                temperature=self.roi_temperature,
            )
            valid_mask = torch.ones_like(roi_logits, dtype=torch.bool)
            route = select_exact_k(
                roi_logits=roi_logits,
                residual_logits=torch.zeros_like(roi_logits),
                mode="roi",
                tokens_per_tubelet=self.roi_tokens,
                context_tokens=0,
                roi_fraction=1.0,
                training=self.training,
                estimator="straight_through" if self.training else "none",
                temperature=self.policy_temperature,
                valid_mask=valid_mask,
            )
        return {
            "geometry": geometry,
            "indices": route["indices"],
            "st_gate": route["st_gate"],
        }

    def forward(self, frames, masks=None):
        if self._pending_postbackbone_route is not None:
            raise RuntimeError("post-backbone ROI route was not consumed exactly once")
        if self.selection_mode == "roi":
            self._pending_postbackbone_route = self._compute_roi_route(frames)
        try:
            # The complete official preprocessing and dense VideoMAE forward are
            # delegated unchanged.  Selection is consumed only by the overridden
            # post-backbone aggregation hook below.
            return super().forward(frames, masks)
        finally:
            self._pending_postbackbone_route = None

    def unflatten_and_pool_features(self, features, batches, num_segs):
        if features.ndim != 5:
            raise ValueError("official VideoMAE feature map must be [B,C,T,H,W]")
        if int(num_segs) != 1 or int(batches) % self.chunk_num:
            raise ValueError("official post-backbone B/C requires one segment and 48 chunks")
        if int(features.shape[0]) != int(batches) * int(num_segs):
            raise ValueError("official heavy feature batch does not match its chunk lineage")
        channels = int(self.model.backbone.embed_dims)
        local_tubelets = int(features.shape[2])
        if (
            int(features.shape[1]) != channels
            or local_tubelets != 8
            or tuple(features.shape[-2:]) != self.source_grid_hw
        ):
            raise ValueError(
                "official post-backbone B/C requires dense [*,384,8,10,10] features"
            )
        window_batch = int(batches) // self.chunk_num
        tubelets = self.chunk_num * local_tubelets
        dense_features = features.reshape(
            window_batch,
            self.chunk_num,
            channels,
            local_tubelets,
            self.source_grid_hw[0],
            self.source_grid_hw[1],
        ).permute(0, 1, 3, 4, 5, 2).reshape(
            window_batch,
            tubelets,
            self.source_grid_hw[0] * self.source_grid_hw[1],
            channels,
        )

        if self.selection_mode == "all":
            selected_features = dense_features
            geometry = dense_features.new_ones((window_batch, tubelets, 4))
            geometry[..., :2] = 0.5
        else:
            route = self._pending_postbackbone_route
            if route is None or route["indices"].shape != (
                window_batch,
                tubelets,
                self.roi_tokens,
            ):
                raise RuntimeError("post-backbone ROI route does not match dense heavy output")
            indices = route["indices"]
            selected_features = dense_features.gather(
                2,
                indices.unsqueeze(-1).expand(
                    window_batch,
                    tubelets,
                    self.roi_tokens,
                    channels,
                ),
            )
            selected_features = selected_features * route["st_gate"].to(
                dtype=selected_features.dtype
            ).unsqueeze(-1)
            geometry = route["geometry"].to(dtype=selected_features.dtype)

        selected_scores = selected_features.new_zeros(selected_features.shape[:3])
        selected_coordinates = selected_features.new_zeros(
            (*selected_features.shape[:3], 2)
        )
        aggregated = self.sparse_adapter(
            selected_features,
            selected_scores,
            geometry,
            selected_coordinates,
            use_absolute_coordinates=False,
            use_roi_relative_coordinates=False,
            use_geometry_projection=False,
            pooling_mode="uniform_selected",
        )
        return F.interpolate(
            aggregated,
            size=self.window_size,
            mode="linear",
            align_corners=False,
        )


def _temporal_class_occupancy_targets(
    gt_segments: Sequence[torch.Tensor],
    gt_labels: Sequence[torch.Tensor],
    *,
    batch_size: int,
    num_classes: int,
    output_length: int,
    detector_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build fit-only coarse TAD occupancy for the train-only scout head."""

    if len(gt_segments) != batch_size or len(gt_labels) != batch_size:
        raise ValueError("dynamic scout GT inputs do not match the feature batch")
    if min(num_classes, output_length, detector_length) <= 0:
        raise ValueError("dynamic scout occupancy geometry must be positive")
    centers = (
        torch.arange(output_length, device=device, dtype=dtype) + 0.5
    ) * (float(detector_length) / float(output_length))
    target = torch.zeros(
        (batch_size, num_classes, output_length),
        device=device,
        dtype=dtype,
    )
    for batch_index, (segments_raw, labels_raw) in enumerate(
        zip(gt_segments, gt_labels)
    ):
        segments = torch.as_tensor(
            segments_raw,
            device=device,
            dtype=dtype,
        ).reshape(-1, 2)
        labels = torch.as_tensor(
            labels_raw,
            device=device,
            dtype=torch.long,
        ).reshape(-1)
        if segments.shape[0] != labels.shape[0]:
            raise ValueError("dynamic scout GT segments and labels differ in length")
        if not bool(torch.isfinite(segments).all().item()):
            raise ValueError("dynamic scout GT segments must be finite")
        if bool(((labels < 0) | (labels >= num_classes)).any().item()):
            raise ValueError("dynamic scout GT label is outside the class range")
        for segment, label in zip(segments, labels):
            start, end = segment.unbind()
            if end <= start:
                continue
            target[batch_index, label, (centers >= start) & (centers < end)] = 1.0
    return target


def dynamic_proxy_weight_at_step(
    successful_update: int,
    *,
    initial_weight: float,
    anneal_start: int,
    anneal_end: int,
) -> float:
    """Hold, linearly anneal, then remove the backward-only soft proxy."""

    if int(successful_update) < 0:
        raise ValueError("successful update must be non-negative")
    if not (0.0 <= float(initial_weight)):
        raise ValueError("dynamic proxy initial weight must be non-negative")
    if not (0 <= int(anneal_start) < int(anneal_end)):
        raise ValueError("dynamic proxy anneal steps must satisfy 0 <= start < end")
    step = int(successful_update)
    if step < int(anneal_start):
        return float(initial_weight)
    if step >= int(anneal_end):
        return 0.0
    fraction = (step - int(anneal_start)) / float(
        int(anneal_end) - int(anneal_start)
    )
    return float(initial_weight) * (1.0 - fraction)


def extract_native_tubelets(
    source: torch.Tensor,
    *,
    patch_size: int,
    tubelet_size: int,
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int], torch.Tensor]:
    """Crop to complete native patches and expose an explicit validity mask."""

    if source.ndim != 5 or source.shape[1] != 3:
        raise ValueError("source must be [B,3,T,H,W]")
    batch, channels, frames, height, width = map(int, source.shape)
    if channels != 3 or frames % int(tubelet_size):
        raise ValueError("source must contain RGB frames divisible by VideoMAE tubelet size")
    valid_height = (height // int(patch_size)) * int(patch_size)
    valid_width = (width // int(patch_size)) * int(patch_size)
    if valid_height <= 0 or valid_width <= 0:
        raise ValueError("source is smaller than one complete native spatial patch")
    ignored_bottom = height - valid_height
    ignored_right = width - valid_width
    source = source[..., :valid_height, :valid_width].contiguous()
    grid_height = valid_height // int(patch_size)
    grid_width = valid_width // int(patch_size)
    tubelets = frames // int(tubelet_size)
    native = (
        source.reshape(
            batch,
            channels,
            tubelets,
            tubelet_size,
            grid_height,
            patch_size,
            grid_width,
            patch_size,
        )
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(
            batch,
            tubelets,
            grid_height * grid_width,
            channels,
            tubelet_size,
            patch_size,
            patch_size,
        )
        .contiguous()
    )
    valid_patch_mask = torch.ones(
        (batch, tubelets, grid_height * grid_width),
        device=source.device,
        dtype=torch.bool,
    )
    return (
        native,
        (grid_height, grid_width),
        (ignored_bottom, ignored_right),
        valid_patch_mask,
    )


class GeoRouteBackboneWrapper(BackboneWrapper):
    """Route native tubelets through one heavy VideoMAE execution.

    The whole-frame scout is intentionally lightweight.  The pretrained
    VideoMAE is invoked once only on native patch tubelets selected by a
    continuous geometry prior plus optional free residual tokens.  No local
    crop is resized and no second heavy global/local backbone pass exists.
    """

    def __init__(self, cfg):
        custom_cfg = cfg.custom
        self.source_key = str(getattr(custom_cfg, "georoute_source_key", "source"))
        self.scout_key = str(getattr(custom_cfg, "georoute_scout_key", "scout"))
        self.window_size = int(getattr(custom_cfg, "georoute_window_size", 768))
        self.scout_size = int(getattr(custom_cfg, "georoute_scout_size", 96))
        self.patch_size = int(getattr(custom_cfg, "georoute_patch_size", 16))
        self.tubelet_size = int(getattr(custom_cfg, "georoute_tubelet_size", 2))
        self.tokens_per_tubelet = int(getattr(custom_cfg, "georoute_tokens_per_tubelet", 48))
        self.context_tokens = int(getattr(custom_cfg, "georoute_context_tokens", 4))
        self.roi_fraction = float(getattr(custom_cfg, "georoute_roi_fraction", 0.50))
        self.route_mode = str(getattr(custom_cfg, "georoute_route_mode", "hybrid"))
        raw_official_support = getattr(
            custom_cfg,
            "georoute_official_support",
            None,
        )
        self.official_support = (
            None if raw_official_support is None else str(raw_official_support)
        )
        self.refresh_carry_mode = str(
            getattr(custom_cfg, "zoomtoken_refresh_carry_mode", "full64")
        )
        self.refresh_query_tokens = int(
            getattr(custom_cfg, "zoomtoken_query_tokens", 64)
        )
        self.refresh_kv_tokens = int(
            getattr(custom_cfg, "zoomtoken_kv_tokens", 64)
        )
        self.refresh_mlp_tokens = int(
            getattr(custom_cfg, "zoomtoken_mlp_tokens", 64)
        )
        self.temporal_carry_enabled = bool(
            getattr(custom_cfg, "zoomtoken_temporal_carry", False)
        )
        self.temporal_carry_detached = bool(
            getattr(custom_cfg, "zoomtoken_carry_detach", False)
        )
        self.temporal_carry_per_block = bool(
            getattr(custom_cfg, "zoomtoken_carry_mix_per_block", False)
        )
        self.requires_route_window_ordinals = self.official_support in {
            "strict_rect8x8_shuf48",
            "strict_rect7x7_core49_shuf15",
        }
        self.official_roi_tokens = int(
            getattr(custom_cfg, "georoute_official_roi_tokens", 64)
        )
        self.policy_estimator = str(getattr(custom_cfg, "georoute_policy_estimator", "straight_through"))
        self.policy_temperature = float(getattr(custom_cfg, "georoute_policy_temperature", 0.5))
        self.r3_soft_membership_temperature = float(
            getattr(
                custom_cfg,
                "georoute_r3_soft_membership_temperature",
                0.025,
            )
        )
        self.r3_area_shift_tubelets = int(
            getattr(custom_cfg, "georoute_r3_area_shift_tubelets", 0)
        )
        self.window_token_budget = int(
            getattr(
                custom_cfg,
                "georoute_window_token_budget",
                self.tokens_per_tubelet * (self.window_size // self.tubelet_size),
            )
        )
        self.zero_carrier_mode = str(
            getattr(custom_cfg, "georoute_zero_carrier_mode", "masked_zero")
        )
        self.branch_calibration_mode = str(
            getattr(custom_cfg, "georoute_branch_calibration_mode", "none")
        )
        self.dynamic_roi_modifier_enabled = bool(
            getattr(custom_cfg, "georoute_dynamic_roi_modifier_enabled", True)
        )
        self.dynamic_residual_modifier_enabled = bool(
            getattr(
                custom_cfg,
                "georoute_dynamic_residual_modifier_enabled",
                True,
            )
        )
        self.dynamic_aux_num_classes = int(
            getattr(custom_cfg, "georoute_dynamic_aux_num_classes", 20)
        )
        self.dynamic_aux_detector_length = int(
            getattr(
                custom_cfg,
                "georoute_dynamic_aux_detector_length",
                self.window_size,
            )
        )
        self.dynamic_aux_weight = float(
            getattr(custom_cfg, "georoute_dynamic_aux_weight", 0.25)
        )
        self.dynamic_proxy_initial_weight = float(
            getattr(custom_cfg, "georoute_dynamic_proxy_initial_weight", 0.50)
        )
        self.dynamic_proxy_anneal_start = int(
            getattr(custom_cfg, "georoute_dynamic_proxy_anneal_start", 1600)
        )
        self.dynamic_proxy_anneal_end = int(
            getattr(custom_cfg, "georoute_dynamic_proxy_anneal_end", 3200)
        )
        self.random_seed = int(getattr(custom_cfg, "georoute_random_seed", 3407))
        self.route_study_seed = int(
            getattr(custom_cfg, "georoute_route_study_seed", self.random_seed)
        )
        self.structured_context_tokens = int(
            getattr(
                custom_cfg,
                "georoute_structured_context_tokens",
                self.context_tokens,
            )
        )
        self.structured_roi_tokens = int(
            getattr(custom_cfg, "georoute_structured_roi_tokens", 0)
        )
        self.structured_residual_tokens = int(
            getattr(custom_cfg, "georoute_structured_residual_tokens", 0)
        )
        self.geometry_temporal_shift_tubelets = int(
            getattr(custom_cfg, "georoute_geometry_temporal_shift_tubelets", 0)
        )
        self.roi_temperature = float(getattr(custom_cfg, "georoute_roi_temperature", 0.25))
        self.geometry_stride_tubelets = int(getattr(custom_cfg, "georoute_geometry_stride_tubelets", 1))
        self.absolute_position_enabled = bool(getattr(custom_cfg, "georoute_absolute_position_enabled", True))
        self.absolute_coordinates_enabled = bool(getattr(custom_cfg, "georoute_absolute_coordinates_enabled", True))
        self.roi_relative_coordinates_enabled = bool(
            getattr(
                custom_cfg,
                "georoute_roi_relative_coordinates_enabled",
                self.absolute_coordinates_enabled,
            )
        )
        self.geometry_projection_enabled = bool(getattr(custom_cfg, "georoute_geometry_projection_enabled", True))
        self.diagnostic_telemetry_enabled = bool(getattr(custom_cfg, "georoute_diagnostic_telemetry_enabled", False))
        self.role_calibration_telemetry_enabled = bool(
            getattr(
                custom_cfg,
                "georoute_role_calibration_telemetry_enabled",
                False,
            )
        )
        self.amp_diagnostic_enabled = bool(
            getattr(custom_cfg, "georoute_amp_diagnostic_enabled", False)
        )
        self.gradient_decomposition_enabled = bool(
            getattr(
                custom_cfg,
                "georoute_gradient_decomposition_enabled",
                False,
            )
        )
        if self.gradient_decomposition_enabled and not self.amp_diagnostic_enabled:
            raise ValueError("gradient decomposition requires AMP diagnostics")
        self.pooling_mode = str(getattr(custom_cfg, "georoute_pooling_mode", "uniform_selected"))
        self.adapter_mode = str(
            getattr(
                custom_cfg,
                "georoute_adapter_mode",
                "coordinate_lineage_packed",
            )
        )
        # A causal control can expose exactly the same learned geometry to the
        # aggregation adapter while keeping a fixed token lattice.  It tests
        # whether a gain comes from spatial *selection*, rather than merely
        # adding an extra geometry-conditioned feature pathway.
        self.geometry_side_channel = bool(getattr(custom_cfg, "georoute_geometry_side_channel", False))
        self.roi_extent_floor_mode = str(
            getattr(
                custom_cfg,
                "georoute_roi_extent_floor_mode",
                "static_normalized",
            )
        )
        raw_roi_extent_floor_cells = getattr(
            custom_cfg,
            "georoute_roi_extent_floor_cells",
            1,
        )
        if (
            isinstance(raw_roi_extent_floor_cells, bool)
            or int(raw_roi_extent_floor_cells) != raw_roi_extent_floor_cells
        ):
            raise ValueError(
                "georoute_roi_extent_floor_cells must be one positive integer"
            )
        self.roi_extent_floor_cells = int(raw_roi_extent_floor_cells)
        self.min_roi_extent = float(getattr(custom_cfg, "georoute_min_roi_extent", 0.20))
        self.max_roi_extent = float(getattr(custom_cfg, "georoute_max_roi_extent", 1.00))
        self.geometry_smoothness_weight = float(getattr(custom_cfg, "georoute_geometry_smoothness_weight", 0.0))
        self.area_prior_weight = float(getattr(custom_cfg, "georoute_area_prior_weight", 0.0))
        self.area_prior = float(getattr(custom_cfg, "georoute_area_prior", 0.30))
        self.score_function_weight = float(getattr(custom_cfg, "georoute_score_function_weight", 1.0))
        self.score_function_baseline_momentum = float(getattr(custom_cfg, "georoute_score_function_baseline_momentum", 0.95))
        self.score_function_temporal_reduction = str(
            getattr(
                custom_cfg,
                "georoute_score_function_temporal_reduction",
                "sum",
            )
        )
        # This switch is intentionally P0-only.  It runs a dense numerical
        # reference before the real packed call and is forbidden in ordinary
        # development/paper cells so it can never be mistaken for model cost.
        self.p0_dense_reference_check = bool(getattr(custom_cfg, "georoute_p0_dense_reference_check", False))
        self.output_length = int(getattr(custom_cfg, "georoute_output_length", self.window_size))
        self.max_batch_size = int(getattr(custom_cfg, "georoute_max_batch_size", 1))
        if self.official_support not in {
            None,
            "all_native",
            "roi_k64",
            "strict_rect8x8",
        } | OFFICIAL_MULTIBRANCH_SUPPORTS:
            raise ValueError(
                "unsupported frozen official pre-backbone support"
            )
        refresh_contracts = {
            "full64": (64, 64, 64, False),
            "drop32": (32, 32, 32, False),
            "mod32_kv": (32, 64, 32, False),
            "rc32_kv": (32, 64, 32, True),
            "dsr6_kv": (32, 64, 32, False),
            "apm32_ctx64": (32, 64, 32, False),
            "cur32_ctx64": (32, 64, 32, False),
            "apm_c32_full64": (64, 64, 64, False),
        }
        if self.refresh_carry_mode not in refresh_contracts:
            raise ValueError("unsupported ZoomToken refresh-carry mode")
        expected_query, expected_kv, expected_mlp, expected_carry = (
            refresh_contracts[self.refresh_carry_mode]
        )
        if (
            self.refresh_query_tokens,
            self.refresh_kv_tokens,
            self.refresh_mlp_tokens,
        ) != (expected_query, expected_kv, expected_mlp):
            raise ValueError("ZoomToken refresh token contract does not match its arm")
        if self.temporal_carry_enabled != expected_carry:
            raise ValueError("only RC32-KV may enable temporal carry")
        if expected_carry:
            if not self.temporal_carry_detached or not self.temporal_carry_per_block:
                raise ValueError("RC32-KV requires detached per-block temporal carry")
        elif self.temporal_carry_detached or self.temporal_carry_per_block:
            raise ValueError("non-RC arms cannot configure temporal carry")
        if self.refresh_carry_mode != "full64" and self.official_support != "strict_rect8x8":
            raise ValueError("refresh-carry arms require the frozen strict R1 K64 support")
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unsupported GeoRoute route mode {self.route_mode!r}")
        if self.policy_estimator not in POLICY_ESTIMATORS:
            raise ValueError(f"unsupported GeoRoute estimator {self.policy_estimator!r}")
        if self.branch_calibration_mode not in DYNAMIC_BRANCH_CALIBRATION_MODES:
            raise ValueError(
                "georoute_branch_calibration_mode must be none or "
                "residual_window_center"
            )
        if (
            self.route_mode not in DYNAMIC_ROUTE_MODES
            and self.branch_calibration_mode != "none"
        ):
            raise ValueError(
                "dynamic branch calibration is valid only for dynamic SCNR"
            )
        if self.pooling_mode not in {"uniform_selected", "route_score_ablation"}:
            raise ValueError("georoute_pooling_mode must be uniform_selected or route_score_ablation")
        if self.adapter_mode != "coordinate_lineage_packed":
            raise ValueError("GeoRoute correctness protocol requires " "georoute_adapter_mode='coordinate_lineage_packed'")
        if self.window_size <= 0 or self.window_size % self.tubelet_size:
            raise ValueError("GeoRoute window size must be divisible by its tubelet size")
        if self.window_size % 16:
            raise ValueError("GeoRoute currently requires 16-frame VideoMAE clips")
        if self.output_length != 2 * (self.window_size // self.tubelet_size):
            raise ValueError("GeoRoute detector contract requires exact 384-to-768 temporal restoration")
        if self.scout_size <= 0 or self.scout_size % 16:
            raise ValueError("GeoRoute scout size must be a positive multiple of 16")
        if not (0.0 <= self.score_function_baseline_momentum < 1.0):
            raise ValueError("score-function baseline momentum must lie in [0,1)")
        if (
            self.score_function_temporal_reduction
            not in SCORE_FUNCTION_TEMPORAL_REDUCTIONS
        ):
            raise ValueError(
                "georoute_score_function_temporal_reduction must be sum or mean"
            )
        if self.geometry_stride_tubelets <= 0:
            raise ValueError("GeoRoute geometry stride must be positive")
        if self.roi_extent_floor_mode not in {
            "static_normalized",
            "native_cells",
        }:
            raise ValueError(
                "georoute_roi_extent_floor_mode must be static_normalized or "
                "native_cells"
            )
        if self.roi_extent_floor_cells <= 0:
            raise ValueError(
                "georoute_roi_extent_floor_cells must be one positive integer"
            )
        if not (0.0 < self.min_roi_extent <= self.max_roi_extent <= 1.0):
            raise ValueError(
                "static GeoRoute extents must satisfy 0 < min <= max <= 1"
            )
        if self.p0_dense_reference_check and self.route_mode != "dense":
            raise ValueError("GeoRoute dense numerical reference is valid only for route_mode='dense'")
        if self.geometry_side_channel and self.route_mode not in {"uniform", "random"}:
            raise ValueError("GeoRoute geometry-side-channel control is valid only for fixed uniform/random routes")
        if self.route_mode in STRUCTURED_ROUTE_MODES:
            structured_total = (
                self.structured_context_tokens
                + self.structured_roi_tokens
                + self.structured_residual_tokens
            )
            if any(
                value < 0
                for value in (
                    self.structured_context_tokens,
                    self.structured_roi_tokens,
                    self.structured_residual_tokens,
                )
            ) or structured_total != self.tokens_per_tubelet:
                raise ValueError(
                    "structured GeoRoute quotas must be non-negative and sum to exact K"
                )
            if self.route_mode == "structured_context_residual" and self.structured_roi_tokens != 0:
                raise ValueError("structured_context_residual requires zero ROI quota")
            if self.route_mode == "structured_context_roi" and self.structured_residual_tokens != 0:
                raise ValueError("structured_context_roi requires zero residual quota")
            if self.route_mode in {
                "structured_hybrid",
                "structured_hybrid_geometry_shift",
            } and (
                self.structured_roi_tokens <= 0
                or self.structured_residual_tokens <= 0
            ):
                raise ValueError("structured hybrid requires both learned roles")
            if (
                not self.absolute_position_enabled
                or self.absolute_coordinates_enabled
                or self.roi_relative_coordinates_enabled
                or self.geometry_projection_enabled
                or self.geometry_side_channel
                or self.pooling_mode != "uniform_selected"
                or self.geometry_smoothness_weight != 0.0
                or self.area_prior_weight != 0.0
            ):
                raise ValueError(
                    "structured causal-pilot modes require support-only representation isolation"
                )
            if self.policy_estimator == "score_function" and (
                self.score_function_temporal_reduction != "mean"
                or self.max_batch_size != 1
            ):
                raise ValueError(
                    "structured PL requires temporal mean and local batch capacity one"
                )
            tubelets = self.window_size // self.tubelet_size
            if self.route_mode == "structured_hybrid_geometry_shift":
                if not (
                    0 < self.geometry_temporal_shift_tubelets < tubelets
                ):
                    raise ValueError(
                        "geometry-shift control requires a nonzero in-range tubelet shift"
                    )
            elif self.geometry_temporal_shift_tubelets != 0:
                raise ValueError(
                    "geometry temporal shift is reserved for its named control"
                )
        if self.route_mode in DYNAMIC_ROUTE_MODES:
            if self.policy_estimator != "straight_through":
                raise ValueError(
                    "dynamic SCNR main route requires straight_through estimator"
                )
            if self.window_token_budget <= 0:
                raise ValueError("dynamic SCNR window token budget must be positive")
            if self.zero_carrier_mode != "masked_zero":
                raise ValueError(
                    "dynamic SCNR main route requires masked_zero carrier"
                )
            if self.dynamic_aux_num_classes <= 0:
                raise ValueError("dynamic SCNR auxiliary class count must be positive")
            if self.dynamic_aux_detector_length != self.window_size:
                raise ValueError(
                    "dynamic SCNR auxiliary detector length must match the window"
                )
            if self.dynamic_aux_weight <= 0.0:
                raise ValueError("dynamic SCNR auxiliary weight must be positive")
            dynamic_proxy_weight_at_step(
                0,
                initial_weight=self.dynamic_proxy_initial_weight,
                anneal_start=self.dynamic_proxy_anneal_start,
                anneal_end=self.dynamic_proxy_anneal_end,
            )
            if (
                not self.absolute_position_enabled
                or self.absolute_coordinates_enabled
                or self.roi_relative_coordinates_enabled
                or self.geometry_projection_enabled
                or self.geometry_side_channel
                or self.pooling_mode != "uniform_selected"
                or self.geometry_smoothness_weight != 0.0
                or self.area_prior_weight != 0.0
                or self.p0_dense_reference_check
                or self.gradient_decomposition_enabled
            ):
                raise ValueError(
                    "dynamic SCNR requires support-only, zero-regularization ragged isolation"
                )
            if self.roi_extent_floor_mode != "native_cells":
                raise ValueError(
                    "dynamic SCNR requires a runtime native-cell ROI floor"
                )
        if self.role_calibration_telemetry_enabled and (
            self.route_mode not in DYNAMIC_ROUTE_MODES
            or not self.diagnostic_telemetry_enabled
        ):
            raise ValueError(
                "role calibration telemetry requires dynamic SCNR diagnostic replay"
            )
        if self.official_support is not None:
            if (
                self.window_size != 768
                or self.patch_size != 16
                or self.tubelet_size != 2
                or self.scout_size != 96
                or self.output_length != 768
                or self.max_batch_size != 1
            ):
                raise ValueError(
                    "official pre-backbone B/C/R1 requires 768 frames, native "
                    "2x16x16 tubelets, a 96x96 scout and local batch one"
                )
            if (
                self.route_mode != "roi"
                or self.official_roi_tokens != 64
                or self.policy_estimator != "straight_through"
                or not self.absolute_position_enabled
                or self.absolute_coordinates_enabled
                or self.roi_relative_coordinates_enabled
                or self.geometry_projection_enabled
                or self.geometry_side_channel
                or self.pooling_mode != "uniform_selected"
                or self.geometry_smoothness_weight != 0.0
                or self.area_prior_weight != 0.0
                or self.p0_dense_reference_check
            ):
                raise ValueError(
                    "official pre-backbone B/C/R1 requires fixed ROI-only "
                    "support with no residual, side channel or proxy path"
                )
            if self.official_support == "strict_rect8x8" and (
                self.policy_temperature != 0.5
                or self.geometry_stride_tubelets != 1
            ):
                raise ValueError(
                    "strict rectangle R1 requires temperature 0.5 and "
                    "per-tubelet geometry"
                )
            if self.official_support in OFFICIAL_QBASE_SUPPORTS and (
                self.policy_temperature != 0.5
                or self.geometry_stride_tubelets != 1
            ):
                raise ValueError(
                    "R2/R4/Q controls require temperature 0.5 and per-tubelet routing"
                )
            if self.official_support in OFFICIAL_R3_SUPPORTS:
                expected_shift = (
                    97
                    if self.official_support
                    == "continuous_rect_dynamic_area_shift97"
                    else 0
                )
                if (
                    self.geometry_stride_tubelets != 1
                    or self.r3_soft_membership_temperature != 0.025
                    or self.r3_area_shift_tubelets != expected_shift
                ):
                    raise ValueError(
                        "R3 requires per-tubelet geometry, temperature 0.025 and "
                        "only its named AREA-SHIFT97 control"
                    )
        super().__init__(cfg)
        if int(self.model.backbone.patch_size) != self.patch_size:
            raise ValueError("GeoRoute patch size must match the loaded VideoMAE")
        if (
            bool(getattr(self.model.backbone, "with_cp", False))
            and self.official_support is None
        ):
            raise ValueError("GeoRoute requires VideoMAE with_cp=False for one-forward accounting")
        self._freeze_shared_backbone_except_adapters()
        if self.refresh_carry_mode == "rc32_kv":
            self.zoomtoken_refresh_carry_alpha = nn.Parameter(
                torch.zeros(len(self.model.backbone.blocks), dtype=torch.float32)
            )
        else:
            self.register_parameter("zoomtoken_refresh_carry_alpha", None)
        self.scout = GeoRouteScout(
            channels=48,
            dynamic_utility=(
                self.route_mode in DYNAMIC_ROUTE_MODES
                or self.official_support in OFFICIAL_QBASE_SUPPORTS
            ),
        )
        if self.official_support in OFFICIAL_R3_SUPPORTS:
            final_geometry = self.scout.geometry_head[-1]
            nn.init.zeros_(final_geometry.weight)
            nn.init.zeros_(final_geometry.bias)
            initial_extent_logit = torch.logit(torch.tensor((0.8 - 0.02) / 0.98))
            with torch.no_grad():
                final_geometry.bias[2:].fill_(float(initial_extent_logit.item()))
        self._configure_scout_trainability()
        self.sparse_adapter = GeoRouteSparseTemporalAdapter(channels=int(self.model.backbone.embed_dims))
        self._configure_sparse_adapter_trainability()
        self.dynamic_aux_head = (
            nn.Conv1d(48, self.dynamic_aux_num_classes, kernel_size=1)
            if self.route_mode in DYNAMIC_ROUTE_MODES
            else None
        )
        self.register_buffer("source_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1))
        self.register_buffer("source_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1))
        self.register_buffer("score_function_baseline", torch.zeros(()))
        self.register_buffer("score_function_baseline_initialized", torch.zeros((), dtype=torch.bool))
        self._successful_update_index: int | None = None
        self._pending_regularization: dict[str, torch.Tensor] | None = None
        self._pending_score_function: dict[str, torch.Tensor] | None = None
        self._pending_dynamic_auxiliary: dict[str, Any] | None = None
        self._pending_r3_epoch_g: torch.Tensor | None = None
        self._pending_r3_update_index: int | None = None
        self._gradient_decomposition_payload: dict[str, Any] | None = None
        self.latest_georoute_audit: dict[str, Any] | None = None
        self.latest_heavy_valid_mask: torch.Tensor | None = None
        if self.official_support in OFFICIAL_R3_SUPPORTS:
            self.register_buffer("r3_dual_lambda", torch.zeros(()))
            self.register_buffer("r3_epoch_g_sum", torch.zeros(()))
            self.register_buffer(
                "r3_epoch_successful_updates", torch.zeros((), dtype=torch.long)
            )
            self.register_buffer(
                "r3_last_completed_update", torch.full((), -1, dtype=torch.long)
            )
            self.register_buffer(
                "r3_last_completed_epoch", torch.full((), -1, dtype=torch.long)
            )

    def _freeze_shared_backbone_except_adapters(self) -> None:
        trainable_adapter_parameters = 0
        for name, parameter in self.model.backbone.named_parameters():
            is_adapter = ".adapter." in f".{name}."
            parameter.requires_grad = is_adapter
            if is_adapter:
                trainable_adapter_parameters += parameter.numel()
        if trainable_adapter_parameters <= 0:
            raise RuntimeError("GeoRoute requires trainable VideoMAE adapters")
        self.trainable_adapter_parameters = trainable_adapter_parameters

    def _configure_scout_trainability(self) -> None:
        if self.route_mode in DYNAMIC_ROUTE_MODES:
            # The scout stem is trained only by the train-only auxiliary head.
            # ``GeoRouteScout.forward_dynamic`` detaches the stem features before
            # every policy head, so detector/proxy route gradients cannot alter
            # the observer representation through an implicit path.
            for parameter in self.scout.parameters():
                parameter.requires_grad = True
            return
        if self.official_support in OFFICIAL_MULTIBRANCH_SUPPORTS:
            needs_qbase = self.official_support in OFFICIAL_QBASE_SUPPORTS
            needs_geometry = self.official_support not in {
                "q48_global",
                "q64_global",
            }
            for parameter in self.scout.parameters():
                parameter.requires_grad = True
            if not needs_geometry:
                for parameter in self.scout.geometry_head.parameters():
                    parameter.requires_grad = False
            for parameter in self.scout.residual_head.parameters():
                parameter.requires_grad = False
            if self.scout.base_utility_head is not None and not needs_qbase:
                for parameter in self.scout.base_utility_head.parameters():
                    parameter.requires_grad = False
            return
        needs_geometry = (
            self.route_mode in {"roi", "hybrid"}
            or self.geometry_side_channel
            or (
                self.route_mode in STRUCTURED_ROUTE_MODES
                and self.structured_roi_tokens > 0
            )
        )
        needs_residual = self.route_mode in {"free", "hybrid"} or (
            self.route_mode in STRUCTURED_ROUTE_MODES
            and self.structured_residual_tokens > 0
        )
        needs_stem = needs_geometry or needs_residual
        for parameter in self.scout.parameters():
            parameter.requires_grad = needs_stem
        if not needs_geometry:
            for parameter in self.scout.geometry_head.parameters():
                parameter.requires_grad = False
        if not needs_residual:
            for parameter in self.scout.residual_head.parameters():
                parameter.requires_grad = False

    def _configure_sparse_adapter_trainability(self) -> None:
        if (
            self.route_mode not in DYNAMIC_ROUTE_MODES
            and self.official_support is None
        ):
            return
        # Support-only isolation disables both representation side channels.
        # Freeze them explicitly so DDP/optimizer audits do not see intentionally
        # unused trainable parameters in the dynamic main cell.
        for module in (
            self.sparse_adapter.geometry_projection,
            self.sparse_adapter.coordinate_projection,
        ):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def set_norm_layer(self) -> None:
        """Freeze only pretrained VideoMAE normalization, not route modules."""

        if not self.norm_eval:
            return
        for module in self.model.backbone.modules():
            if isinstance(module, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                module.eval()

    def set_successful_update_index(self, index: int) -> None:
        if int(index) < 0:
            raise ValueError("successful update index must be non-negative")
        self._successful_update_index = int(index)

    def commit_successful_update(self, index: int) -> None:
        """Commit R3 budget statistics only after a real optimizer update."""

        if self.official_support not in OFFICIAL_R3_SUPPORTS:
            return
        if (
            self._pending_r3_epoch_g is None
            or self._pending_r3_update_index != int(index)
        ):
            raise RuntimeError("R3 successful update lacks its matching budget item")
        if int(self.r3_last_completed_update.item()) >= int(index):
            raise RuntimeError("R3 successful update identity is not monotonic")
        self.r3_epoch_g_sum.add_(self._pending_r3_epoch_g)
        self.r3_epoch_successful_updates.add_(1)
        self.r3_last_completed_update.fill_(int(index))
        self._pending_r3_epoch_g = None
        self._pending_r3_update_index = None

    def finish_training_epoch(
        self,
        epoch: int,
        next_successful_update_index: int,
    ) -> None:
        """Update the frozen R3 dual once from successful-update epoch mean."""

        if self.official_support not in OFFICIAL_R3_SUPPORTS:
            return
        count = int(self.r3_epoch_successful_updates.item())
        if count <= 0:
            raise RuntimeError("R3 cannot finish an epoch without successful updates")
        if int(self.r3_last_completed_update.item()) + 1 != int(
            next_successful_update_index
        ):
            raise RuntimeError("R3 epoch completion disagrees with update identity")
        epoch_g = self.r3_epoch_g_sum / float(count)
        self.r3_dual_lambda.copy_((self.r3_dual_lambda + epoch_g).clamp(-4.0, 4.0))
        self.r3_epoch_g_sum.zero_()
        self.r3_epoch_successful_updates.zero_()
        self.r3_last_completed_epoch.fill_(int(epoch))

    def export_r3_recovery_state(self) -> dict[str, Any] | None:
        if self.official_support not in OFFICIAL_R3_SUPPORTS:
            return None
        if self._pending_r3_epoch_g is not None:
            raise RuntimeError("R3 recovery capture requires an epoch boundary")
        return {
            "schema_version": "zoomtoken_r3_dual_state_v001",
            "dual_lambda": float(self.r3_dual_lambda.item()),
            "epoch_g_sum": float(self.r3_epoch_g_sum.item()),
            "epoch_successful_updates": int(
                self.r3_epoch_successful_updates.item()
            ),
            "last_completed_update": int(self.r3_last_completed_update.item()),
            "last_completed_epoch": int(self.r3_last_completed_epoch.item()),
        }

    def restore_r3_recovery_state(self, state: Mapping[str, Any] | None) -> None:
        if self.official_support not in OFFICIAL_R3_SUPPORTS:
            if state is not None:
                raise ValueError("non-R3 arm cannot restore R3 dual state")
            return
        expected_keys = {
            "schema_version",
            "dual_lambda",
            "epoch_g_sum",
            "epoch_successful_updates",
            "last_completed_update",
            "last_completed_epoch",
        }
        if not isinstance(state, Mapping) or set(state) != expected_keys:
            raise ValueError("R3 recovery lacks the complete frozen dual state")
        if state["schema_version"] != "zoomtoken_r3_dual_state_v001":
            raise ValueError("R3 recovery dual-state schema mismatch")
        values = (state["dual_lambda"], state["epoch_g_sum"])
        if any(not isinstance(value, (int, float)) for value in values):
            raise ValueError("R3 recovery dual scalars are invalid")
        count = state["epoch_successful_updates"]
        update = state["last_completed_update"]
        epoch = state["last_completed_epoch"]
        if (
            not isinstance(count, int)
            or count < 0
            or not isinstance(update, int)
            or update < -1
            or not isinstance(epoch, int)
            or epoch < -1
        ):
            raise ValueError("R3 recovery update/epoch identity is invalid")
        if not (-4.0 <= float(state["dual_lambda"]) <= 4.0):
            raise ValueError("R3 recovery lambda leaves the frozen clip range")
        self.r3_dual_lambda.fill_(float(state["dual_lambda"]))
        self.r3_epoch_g_sum.fill_(float(state["epoch_g_sum"]))
        self.r3_epoch_successful_updates.fill_(count)
        self.r3_last_completed_update.fill_(update)
        self.r3_last_completed_epoch.fill_(epoch)
        self._pending_r3_epoch_g = None
        self._pending_r3_update_index = None

    def _validate_inputs(self, frames: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(frames, Mapping) or set(frames) != {
            self.source_key,
            self.scout_key,
        }:
            raise ValueError("GeoRoute inputs are fail-closed; expected exactly " f"{sorted((self.source_key, self.scout_key))}")
        source = frames[self.source_key]
        scout = frames[self.scout_key]
        if not isinstance(source, torch.Tensor) or source.ndim != 6 or source.shape[1:3] != (1, 3):
            raise ValueError("GeoRoute source must be uint8 [B,1,3,T,H,W]")
        if not isinstance(scout, torch.Tensor) or scout.ndim != 6 or scout.shape[1:3] != (1, 3):
            raise ValueError("GeoRoute scout must be uint8 [B,1,3,T,H,W]")
        if source.dtype != torch.uint8 or scout.dtype != torch.uint8:
            raise TypeError("GeoRoute source and scout must remain uint8 before the route")
        if source.shape[0] != scout.shape[0] or source.shape[3] != scout.shape[3]:
            raise ValueError("GeoRoute source/scout batch and temporal axes must match")
        if source.shape[3] != self.window_size or scout.shape[3] != self.window_size:
            raise ValueError("GeoRoute source/scout must preserve the full configured temporal axis")
        if tuple(scout.shape[-2:]) != (self.scout_size, self.scout_size):
            raise ValueError("GeoRoute scout has an unexpected spatial resolution")
        if source.shape[0] > self.max_batch_size:
            raise ValueError("GeoRoute native grid batches require batch_size <= configured max_batch_size")
        if (
            self.route_mode in STRUCTURED_ROUTE_MODES
            and self.policy_estimator == "score_function"
            and source.shape[0] != 1
        ):
            raise ValueError("structured PL detector risk requires local batch size exactly one")
        return source[:, 0].contiguous(), scout[:, 0].contiguous()

    def _validate_official_fixed_support_input(
        self,
        frames: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not isinstance(frames, torch.Tensor)
            or frames.ndim != 6
            or tuple(frames.shape[1:3]) != (1, 3)
            or frames.dtype != torch.uint8
        ):
            raise ValueError(
                "official pre-backbone B/C/R1 requires augmented uint8 "
                "[B,1,3,T,H,W] input"
            )
        source = frames[:, 0]
        if (
            int(source.shape[0]) > self.max_batch_size
            or int(source.shape[2]) != self.window_size
            or tuple(source.shape[-2:]) != (160, 160)
        ):
            raise ValueError(
                "official pre-backbone B/C/R1 requires local batch one and the "
                "augmented 768x160x160 input"
            )
        return source.contiguous()

    def _official_fixed_support_route(
        self,
        source: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
        valid_patch_mask: torch.Tensor,
        window_ordinals: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if self.official_support is None:
            raise RuntimeError("official fixed-support route is not configured")
        if source_grid_hw != (10, 10):
            raise ValueError("official pre-backbone B/C/R1 requires a 10x10 native grid")
        batch_size, _channels, frame_count, height, width = map(int, source.shape)
        with torch.autocast(device_type=source.device.type, enabled=False):
            normalized = _normalize_uint8_video(
                source,
                self.source_mean,
                self.source_std,
            )
            scout_input = F.interpolate(
                normalized.permute(0, 2, 1, 3, 4).reshape(
                    batch_size * frame_count,
                    3,
                    height,
                    width,
                ),
                size=(self.scout_size, self.scout_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(
                batch_size,
                frame_count,
                3,
                self.scout_size,
                self.scout_size,
            ).permute(0, 2, 1, 3, 4).contiguous()
            scout_features = self.scout._encode(scout_input)
            geometry_logits = self.scout.geometry_head(
                scout_features.mean(dim=(-1, -2))
            ).transpose(1, 2)
            geometry_logits = interpolate_temporal_knots(
                geometry_logits,
                stride=self.geometry_stride_tubelets,
            )
            q_base = None
            if self.official_support in OFFICIAL_QBASE_SUPPORTS:
                if self.scout.base_utility_head is None:
                    raise RuntimeError("official q_base support lacks its utility head")
                q_base = self.scout._resize_native_field(
                    self.scout.base_utility_head(scout_features).squeeze(1),
                    source_grid_hw=source_grid_hw,
                )
            if self.official_support == "strict_rect8x8":
                route = select_strict_rectangle_8x8(
                    geometry_logits,
                    training=self.training,
                    temperature=self.policy_temperature,
                    valid_mask=valid_patch_mask,
                )
                geometry = route["geometry"]
            elif self.official_support in {
                "strict_rect8x8_q48",
                "strict_rect8x8_shuf48",
            }:
                route = select_rectangle_constrained_qbase_8x8(
                    geometry_logits,
                    q_base,
                    training=self.training,
                    temperature=self.policy_temperature,
                    valid_mask=valid_patch_mask,
                    shuffle_seed=(
                        self.random_seed
                        if self.official_support == "strict_rect8x8_shuf48"
                        else None
                    ),
                    window_ordinals=window_ordinals,
                )
                geometry = route["geometry"]
            elif self.official_support in {
                "strict_rect7x7_core49_q15",
                "strict_rect7x7_core49_shuf15",
            }:
                route = select_rectangle_core_outside_qbase_7x7(
                    geometry_logits,
                    q_base,
                    training=self.training,
                    temperature=self.policy_temperature,
                    valid_mask=valid_patch_mask,
                    shuffle_seed=(
                        self.random_seed
                        if self.official_support
                        == "strict_rect7x7_core49_shuf15"
                        else None
                    ),
                    window_ordinals=window_ordinals,
                )
                geometry = route["geometry"]
            elif self.official_support in {"q48_global", "q64_global"}:
                target_k = 48 if self.official_support == "q48_global" else 64
                route = select_qbase_global_exact_k(
                    q_base,
                    target_k=target_k,
                    training=self.training,
                    temperature=self.policy_temperature,
                    valid_mask=valid_patch_mask,
                    mode=self.official_support,
                )
                geometry = torch.tensor(
                    [0.5, 0.5, 1.0, 1.0],
                    device=q_base.device,
                    dtype=q_base.dtype,
                ).view(1, 1, 4).expand(*q_base.shape[:2], 4)
            elif self.official_support in OFFICIAL_R3_SUPPORTS:
                route = select_continuous_strict_rectangle(
                    geometry_logits,
                    training=self.training,
                    valid_mask=valid_patch_mask,
                    soft_temperature=self.r3_soft_membership_temperature,
                    area_shift_tubelets=self.r3_area_shift_tubelets,
                )
                geometry = route["geometry"]
            else:
                geometry = decode_continuous_geometry(
                    geometry_logits,
                    min_extent=self._minimum_roi_extent_wh(source_grid_hw),
                    max_extent=self.max_roi_extent,
                )
                roi_logits = roi_logits_from_geometry(
                    geometry,
                    grid_height=source_grid_hw[0],
                    grid_width=source_grid_hw[1],
                    temperature=self.roi_temperature,
                )
                selected_per_tubelet = (
                    int(roi_logits.shape[-1])
                    if self.official_support == "all_native"
                    else self.official_roi_tokens
                )
                route = select_exact_k(
                    roi_logits=roi_logits,
                    residual_logits=torch.zeros_like(roi_logits),
                    mode="roi",
                    tokens_per_tubelet=selected_per_tubelet,
                    context_tokens=0,
                    roi_fraction=1.0,
                    training=self.training,
                    estimator="straight_through" if self.training else "none",
                    temperature=self.policy_temperature,
                    valid_mask=valid_patch_mask,
                )
        if self.official_support in OFFICIAL_R3_SUPPORTS:
            return {
                **route,
                "geometry": geometry,
            }
        spatial_indices, sort_order = route["indices"].sort(dim=-1)
        st_gate = route["st_gate"].gather(-1, sort_order)
        if self.official_support == "all_native":
            expected = torch.arange(
                int(roi_logits.shape[-1]),
                device=spatial_indices.device,
                dtype=torch.long,
            ).view(1, 1, -1).expand_as(spatial_indices)
            if not torch.equal(spatial_indices, expected):
                raise RuntimeError("official arm B did not preserve all native support")
        elif (
            self.official_support == "roi_k64"
            and int(spatial_indices.shape[-1]) != self.official_roi_tokens
        ):
            raise RuntimeError("official arm C did not preserve fixed ROI K=64")
        elif (
            self.official_support == "strict_rect8x8"
            and int(spatial_indices.shape[-1]) != 64
        ):
            raise RuntimeError("strict rectangle R1 did not preserve exact K64")
        expected_new_k = {
            "strict_rect8x8_q48": 48,
            "strict_rect8x8_shuf48": 48,
            "q48_global": 48,
            "strict_rect7x7_core49_q15": 64,
            "strict_rect7x7_core49_shuf15": 64,
            "q64_global": 64,
        }.get(self.official_support)
        if expected_new_k is not None and int(spatial_indices.shape[-1]) != expected_new_k:
            raise RuntimeError("official R2/R4/Q support violates its exact token count")
        receipt: dict[str, Any] = {
            "geometry": geometry,
            "spatial_indices": spatial_indices,
            "st_gate": st_gate,
        }
        if self.official_support == "strict_rect8x8":
            receipt.update(
                {
                    "routing_schema": route["schema_version"],
                    "candidate_top_left_row_col": route[
                        "candidate_top_left_row_col"
                    ],
                    "block_top_left_row_col": route["block_top_left_row_col"],
                    "block_size_hw": route["block_size_hw"],
                    "candidate_count": route["candidate_count"],
                    "hole_count": route["hole_count"],
                }
            )
        elif self.official_support in OFFICIAL_QBASE_SUPPORTS:
            receipt.update(
                {
                    "routing_schema": route["schema_version"],
                    "route_mode": route["mode"],
                    "target_k": route["target_k"],
                    "candidate_count": route["candidate_count"],
                    "padded_token_count": route["padded_token_count"],
                }
            )
            for key in (
                "candidate_top_left_row_col",
                "block_top_left_row_col",
                "block_size_hw",
                "hole_count",
                "core_count",
                "outside_candidate_count",
                "outside_count",
                "shuffle_enabled",
            ):
                if key in route:
                    receipt[key] = route[key]
        return receipt

    def _r3_augmented_lagrangian_loss(
        self,
        route: Mapping[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        local_soft = route["soft_count_sum"].to(torch.float32)
        local_denominator = route["valid_tubelet_count"].to(
            device=local_soft.device,
            dtype=torch.float32,
        ) * 64.0
        totals = torch.stack((local_soft.detach(), local_denominator.detach()))
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(totals)
            world_size = torch.distributed.get_world_size()
        global_denominator = totals[1].clamp_min(1.0)
        global_g = totals[0] / global_denominator - 1.0
        live_g = global_g + (
            float(world_size)
            * (local_soft - local_soft.detach())
            / global_denominator
        )
        loss = self.r3_dual_lambda.detach() * live_g + 0.5 * live_g.square()
        if not bool(torch.isfinite(loss).item()):
            raise FloatingPointError("R3 augmented-Lagrangian budget is non-finite")
        return loss, global_g

    def _gather_selected_native_tubelets(
        self,
        native: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        if native.shape[:2] != indices.shape[:2]:
            raise ValueError("native tubelets and GeoRoute indices must share [B,T]")
        gather_index = indices[..., None, None, None, None].expand(
            *indices.shape,
            *native.shape[3:],
        )
        return native.gather(2, gather_index)

    def _build_strict_rectangle_refresh_mask(
        self,
        native: torch.Tensor,
        spatial_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Build the causal exact-K32 refresh mask inside strict R1 K64 support."""

        if native.ndim != 7 or spatial_indices.ndim != 3:
            raise ValueError("refresh routing requires native [B,T,N,...] and support [B,T,K]")
        batch_size, tubelets, spatial_tokens = map(int, native.shape[:3])
        if tuple(spatial_indices.shape[:2]) != (batch_size, tubelets) or int(
            spatial_indices.shape[-1]
        ) != 64:
            raise ValueError("refresh routing requires strict K64 support per tubelet")

        support_native = self._gather_selected_native_tubelets(native, spatial_indices)
        motion = torch.zeros(
            (batch_size, tubelets, 64),
            device=native.device,
            dtype=torch.float32,
        )
        cache_valid = torch.zeros_like(motion, dtype=torch.bool)
        if tubelets > 1:
            previous_indices = spatial_indices[:, :-1]
            current_indices = spatial_indices[:, 1:]
            previous_same_spatial = self._gather_selected_native_tubelets(
                native[:, :-1],
                current_indices,
            )
            motion[:, 1:] = (
                support_native[:, 1:].to(torch.float32)
                - previous_same_spatial.to(torch.float32)
            ).abs().mean(dim=(-1, -2, -3, -4))
            cache_valid[:, 1:] = (
                current_indices.unsqueeze(-1) == previous_indices.unsqueeze(-2)
            ).any(dim=-1)

        age_lattice = torch.zeros(
            (batch_size, spatial_tokens),
            device=native.device,
            dtype=torch.long,
        )
        masks = []
        for tubelet_index in range(tubelets):
            support_index = spatial_indices[:, tubelet_index]
            support_age = age_lattice.gather(1, support_index)
            selected = build_refresh_mask(
                motion[:, tubelet_index : tubelet_index + 1],
                cache_valid[:, tubelet_index : tubelet_index + 1],
                support_age.unsqueeze(1),
            )[:, 0]
            masks.append(selected)
            age_lattice = (age_lattice + 1).clamp_max(2)
            refreshed_index = support_index[selected].reshape(batch_size, 32)
            age_lattice.scatter_(1, refreshed_index, 0)
        return torch.stack(masks, dim=1)

    @staticmethod
    def _selected_native_coordinates(
        indices: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        if indices.ndim != 3:
            raise ValueError("GeoRoute indices must be [B,T,K]")
        centres = native_patch_centers(
            source_grid_hw[0],
            source_grid_hw[1],
            device=indices.device,
            dtype=torch.float32,
        )
        gather_index = indices[..., None].expand(*indices.shape, 2)
        return centres.view(1, 1, -1, 2).expand(indices.shape[0], indices.shape[1], -1, 2).gather(2, gather_index)

    @staticmethod
    def _gather_selected_native_physical(
        native: torch.Tensor,
        physical_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Gather one sorted global physical-token union without padding."""

        if native.ndim != 7:
            raise ValueError("native tubelets must be [B,T,N,3,2,P,P]")
        if physical_indices.ndim != 2 or physical_indices.shape[0] != native.shape[0]:
            raise ValueError("physical indices must be [B,S]")
        if physical_indices.dtype != torch.long:
            raise TypeError("physical indices must be torch.long")
        batch_size, tubelets, spatial_tokens = map(int, native.shape[:3])
        capacity = tubelets * spatial_tokens
        if bool(
            ((physical_indices < 0) | (physical_indices >= capacity)).any().item()
        ):
            raise ValueError("physical selection falls outside the native lattice")
        if physical_indices.shape[1] > 1 and not bool(
            (physical_indices[:, 1:] > physical_indices[:, :-1]).all().item()
        ):
            raise ValueError("physical selection must be strictly increasing and unique")
        flattened = native.reshape(batch_size, capacity, *native.shape[3:])
        gather_index = physical_indices.reshape(
            batch_size,
            physical_indices.shape[1],
            *([1] * (native.ndim - 3)),
        ).expand(
            batch_size,
            physical_indices.shape[1],
            *native.shape[3:],
        )
        return flattened.gather(1, gather_index)

    @staticmethod
    def _selected_physical_coordinates(
        spatial_indices: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        if spatial_indices.ndim != 2 or spatial_indices.dtype != torch.long:
            raise ValueError("ragged spatial indices must be long [B,S]")
        centres = native_patch_centers(
            source_grid_hw[0],
            source_grid_hw[1],
            device=spatial_indices.device,
            dtype=torch.float32,
        )
        item_count = int(centres.shape[0])
        if bool(
            ((spatial_indices < 0) | (spatial_indices >= item_count)).any().item()
        ):
            raise ValueError("ragged spatial index falls outside the native grid")
        return centres.index_select(0, spatial_indices.reshape(-1)).reshape(
            *spatial_indices.shape,
            2,
        )

    def _regularization(self, geometry: torch.Tensor) -> torch.Tensor:
        smoothness = geometry.new_zeros(())
        if geometry.shape[1] > 1:
            smoothness = (geometry[:, 1:] - geometry[:, :-1]).square().mean()
        area = (geometry[..., 2] * geometry[..., 3]).mean()
        area_loss = (area - self.area_prior).square()
        return self.geometry_smoothness_weight * smoothness + self.area_prior_weight * area_loss

    def _minimum_roi_extent_wh(
        self,
        source_grid_hw: tuple[int, int],
    ) -> tuple[float, float]:
        """Resolve the configured ROI floor in normalized ``(width, height)``."""

        if self.roi_extent_floor_mode == "static_normalized":
            minimum = (self.min_roi_extent, self.min_roi_extent)
        else:
            minimum = native_cell_extent_floor(
                source_grid_hw[0],
                source_grid_hw[1],
                cells_per_axis=self.roi_extent_floor_cells,
            )
        if any(value > self.max_roi_extent for value in minimum):
            raise ValueError(
                "runtime native-cell ROI floor exceeds georoute_max_roi_extent"
            )
        return minimum

    @staticmethod
    def _fixed_full_frame_geometry(
        *,
        batch_size: int,
        tubelets: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        geometry = torch.ones(
            (batch_size, tubelets, 4),
            device=device,
            dtype=dtype,
        )
        geometry[..., :2] = 0.5
        return geometry

    @staticmethod
    def _geometry_shift_control(
        geometry: torch.Tensor,
        *,
        shift_tubelets: int,
    ) -> torch.Tensor:
        """Cyclically misalign a geometry trajectory from its video content."""

        if geometry.ndim != 3 or geometry.shape[-1] != 4:
            raise ValueError("geometry shift control requires [B,T,4]")
        tubelets = int(geometry.shape[1])
        if not (0 < int(shift_tubelets) < tubelets):
            raise ValueError("geometry shift must be nonzero and smaller than T")
        # Unit/P0 controls prove the multiset identity.  The production path does
        # not sort or synchronize here, so the negative control adds only roll.
        # Frozen permutation: output tubelet t consumes geometry
        # ``(t + shift) mod T``.
        return torch.roll(geometry, shifts=-int(shift_tubelets), dims=1)

    def _compute_route_fields(
        self,
        scout: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(scout.shape[0])
        tubelets = self.window_size // self.tubelet_size
        item_count = int(source_grid_hw[0]) * int(source_grid_hw[1])
        needs_geometry = (
            self.route_mode in {"roi", "hybrid"}
            or self.geometry_side_channel
            or (
                self.route_mode in STRUCTURED_ROUTE_MODES
                and self.structured_roi_tokens > 0
            )
        )
        needs_residual = self.route_mode in {"free", "hybrid"} or (
            self.route_mode in STRUCTURED_ROUTE_MODES
            and self.structured_residual_tokens > 0
        )
        needs_scout = needs_geometry or needs_residual

        if needs_scout:
            # REINFORCE multiplies the joint route log-probability by a
            # detector-derived advantage.  Even when the scalar loss is FP32,
            # propagating its scaled gradient through an autocast FP16 scout
            # can overflow before GradScaler can unscale it.  The scout is the
            # low-cost route path, so keep its complete forward/backward graph
            # in FP32 without changing the registered estimator or objective.
            with torch.autocast(device_type=scout.device.type, enabled=False):
                normalized_scout = _normalize_uint8_video(
                    scout,
                    self.source_mean,
                    self.source_std,
                )
                geometry_logits, residual_logits = self.scout(
                    normalized_scout,
                    source_grid_hw=source_grid_hw,
                )
            if geometry_logits.dtype != torch.float32 or residual_logits.dtype != torch.float32:
                raise FloatingPointError("GeoRoute scout must remain FP32 outside autocast")
        else:
            geometry_logits = torch.zeros(
                (batch_size, tubelets, 4),
                device=scout.device,
                dtype=torch.float32,
            )
            residual_logits = torch.zeros(
                (batch_size, tubelets, item_count),
                device=scout.device,
                dtype=torch.float32,
            )

        if needs_geometry:
            minimum_extent_wh = self._minimum_roi_extent_wh(source_grid_hw)
            geometry = decode_continuous_geometry(
                interpolate_temporal_knots(
                    geometry_logits,
                    stride=self.geometry_stride_tubelets,
                ),
                min_extent=minimum_extent_wh,
                max_extent=self.max_roi_extent,
            )
            regularization = self._regularization(geometry)
        else:
            geometry = self._fixed_full_frame_geometry(
                batch_size=batch_size,
                tubelets=tubelets,
                device=scout.device,
                dtype=residual_logits.dtype,
            )
            regularization = geometry.new_zeros(())

        if not needs_residual:
            residual_logits = torch.zeros(
                (batch_size, tubelets, item_count),
                device=scout.device,
                dtype=geometry.dtype,
            )
        return geometry, residual_logits, regularization

    def _compute_dynamic_route_fields(
        self,
        scout: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Compute Scheme-A policy fields while isolating the scout stem."""

        if self.route_mode not in DYNAMIC_ROUTE_MODES:
            raise RuntimeError("dynamic route fields require dynamic_scnr mode")
        with torch.autocast(device_type=scout.device.type, enabled=False):
            normalized_scout = _normalize_uint8_video(
                scout,
                self.source_mean,
                self.source_std,
            )
            (
                geometry_logits,
                q_base,
                residual_logits,
                scout_features,
            ) = self.scout.forward_dynamic(
                normalized_scout,
                source_grid_hw=source_grid_hw,
            )
            geometry = decode_continuous_geometry(
                interpolate_temporal_knots(
                    geometry_logits,
                    stride=self.geometry_stride_tubelets,
                ),
                min_extent=self._minimum_roi_extent_wh(source_grid_hw),
                max_extent=self.max_roi_extent,
            )
            regularization = self._regularization(geometry)
        tensors = (
            geometry,
            q_base,
            residual_logits,
            scout_features,
            regularization,
        )
        if any(value.dtype != torch.float32 for value in tensors):
            raise FloatingPointError("dynamic SCNR scout and policy must remain FP32")
        if not all(bool(torch.isfinite(value).all().item()) for value in tensors):
            raise FloatingPointError("dynamic SCNR scout or policy produced nonfinite values")
        if regularization.numel() != 1 or float(regularization.detach().item()) != 0.0:
            raise RuntimeError("dynamic SCNR main route forbids geometry regularization")
        return tensors

    @staticmethod
    def _dynamic_soft_proxy_features(
        scout_features: torch.Tensor,
        soft_probability: torch.Tensor,
        *,
        source_grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        """Pool detached scout evidence through the differentiable global route."""

        if scout_features.ndim != 5:
            raise ValueError("dynamic scout features must be [B,C,T,H,W]")
        if soft_probability.ndim != 3:
            raise ValueError("dynamic soft route must be [B,T,N]")
        batch_size, channels, tubelets, scout_h, scout_w = map(
            int,
            scout_features.shape,
        )
        grid_height, grid_width = map(int, source_grid_hw)
        if soft_probability.shape != (
            batch_size,
            tubelets,
            grid_height * grid_width,
        ):
            raise ValueError("dynamic soft route and scout feature lattice differ")
        # This detach is part of the method contract: the proxy teaches only
        # policy heads and the shared auxiliary classifier, never the observer
        # stem and never the heavy VideoMAE.
        detached = scout_features.detach().permute(0, 2, 1, 3, 4).reshape(
            batch_size * tubelets,
            channels,
            scout_h,
            scout_w,
        )
        native = F.interpolate(
            detached,
            size=(grid_height, grid_width),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size,
            tubelets,
            channels,
            grid_height * grid_width,
        )
        probability = soft_probability.unsqueeze(2)
        mass = probability.sum(dim=-1).clamp_min(
            torch.finfo(soft_probability.dtype).tiny
        )
        pooled = (native * probability).sum(dim=-1) / mass
        return pooled.permute(0, 2, 1).contiguous()

    @staticmethod
    def _dynamic_calibration_distribution(values: torch.Tensor) -> dict[str, Any]:
        """Return a compact, mergeable diagnostic distribution."""

        values = values.detach().flatten().to(device="cpu", dtype=torch.float32)
        if values.numel() <= 0 or not bool(torch.isfinite(values).all().item()):
            raise ValueError("dynamic role calibration values must be finite and non-empty")
        quantiles = torch.quantile(
            values,
            torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]),
        )
        return {
            "count": int(values.numel()),
            "min": float(values.min().item()),
            "p05": float(quantiles[0].item()),
            "p25": float(quantiles[1].item()),
            "p50": float(quantiles[2].item()),
            "p75": float(quantiles[3].item()),
            "p95": float(quantiles[4].item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "std_population": float(values.std(unbiased=False).item()),
            "negative_count": int((values < 0.0).sum().item()),
            "zero_count": int((values == 0.0).sum().item()),
            "positive_count": int((values > 0.0).sum().item()),
        }

    @classmethod
    def _dynamic_policy_calibration_telemetry(
        cls,
        *,
        route: Mapping[str, Any],
        q_base: torch.Tensor,
        delta_roi: torch.Tensor,
        delta_residual: torch.Tensor,
        valid_patch_mask: torch.Tensor,
    ) -> dict[str, Any]:
        """Diagnose role identifiability without imposing role quotas."""

        if (
            q_base.ndim != 3
            or q_base.shape[0] != 1
            or delta_roi.shape != q_base.shape
            or delta_residual.shape != q_base.shape
            or valid_patch_mask.shape != q_base.shape
            or valid_patch_mask.dtype != torch.bool
        ):
            raise ValueError(
                "dynamic role calibration requires aligned one-sample policy fields"
            )
        fields = (q_base, delta_roi, delta_residual)
        if not all(bool(torch.isfinite(value).all().item()) for value in fields):
            raise FloatingPointError("dynamic role calibration observed nonfinite policy fields")

        valid = valid_patch_mask.detach().to(device="cpu", dtype=torch.bool)
        selected = route["selected_mask"].detach().to(device="cpu", dtype=torch.bool)
        if (
            selected.shape != valid.shape
            or not bool((selected <= valid).all().item())
            or int(selected.sum().item()) != int(route["window_budget"])
        ):
            raise RuntimeError("dynamic role calibration received an invalid hard route")
        unselected = valid & ~selected
        if not bool(valid.any().item()) or not bool(selected.any().item()) or not bool(
            unselected.any().item()
        ):
            raise RuntimeError("dynamic role calibration requires selected and unselected support")

        q_base_cpu, delta_roi_cpu, delta_residual_cpu = (
            value.detach().to(device="cpu", dtype=torch.float32) for value in fields
        )
        modifiers = torch.stack(
            (torch.zeros_like(q_base_cpu), delta_roi_cpu, delta_residual_cpu),
            dim=-1,
        )
        winner_values, winner_roles = modifiers.max(dim=-1)
        winner_margin = torch.topk(modifiers, k=2, dim=-1).values
        winner_margin = winner_margin[..., 0] - winner_margin[..., 1]
        role_names = ("context", "roi", "residual")

        def _role_counts(mask: torch.Tensor) -> dict[str, int]:
            counts = torch.bincount(
                winner_roles.masked_select(mask),
                minlength=len(role_names),
            )
            return {
                name: int(counts[index].item())
                for index, name in enumerate(role_names)
            }

        valid_role_counts = _role_counts(valid)
        selected_role_counts = _role_counts(selected)
        unselected_role_counts = _role_counts(unselected)
        expected_selected = {
            name: int(route["role_counts"][name]) for name in role_names
        }
        if selected_role_counts != expected_selected:
            raise RuntimeError(
                "dynamic role calibration disagrees with selected role receipt"
            )
        valid_count = int(valid.sum().item())
        selected_count = int(selected.sum().item())
        valid_role_fractions = {
            name: count / float(valid_count)
            for name, count in valid_role_counts.items()
        }
        selected_role_fractions = {
            name: count / float(selected_count)
            for name, count in selected_role_counts.items()
        }
        unselected_count = int(unselected.sum().item())
        unselected_role_fractions = {
            name: count / float(unselected_count)
            for name, count in unselected_role_counts.items()
        }
        dominant_role = max(
            role_names,
            key=lambda name: (selected_role_counts[name], -role_names.index(name)),
        )

        def _scoped_distribution(value: torch.Tensor) -> dict[str, Any]:
            return {
                "valid": cls._dynamic_calibration_distribution(
                    value.masked_select(valid)
                ),
                "selected": cls._dynamic_calibration_distribution(
                    value.masked_select(selected)
                ),
                "unselected": cls._dynamic_calibration_distribution(
                    value.masked_select(unselected)
                ),
            }

        return {
            "schema_version": "scnr_dynamic_role_calibration_window_v1",
            "measurement_scope": "accuracy_replay_only_excluded_from_timed_cost",
            "diagnostic_only": True,
            "changes_route_or_execution": False,
            "role_target_fractions_used": False,
            "fixed_role_quota_used": False,
            "q_base_shared_across_roles": True,
            "context_modifier_definition": "exact_zero_baseline_no_learned_q_ctx",
            "valid_candidate_count": valid_count,
            "selected_candidate_count": selected_count,
            "unselected_candidate_count": unselected_count,
            "role_order": list(role_names),
            "valid_role_counts": valid_role_counts,
            "valid_role_fractions": valid_role_fractions,
            "selected_role_counts": selected_role_counts,
            "selected_role_fractions": selected_role_fractions,
            "unselected_role_counts": unselected_role_counts,
            "unselected_role_fractions": unselected_role_fractions,
            "selected_missing_roles": [
                name for name in role_names if selected_role_counts[name] == 0
            ],
            "selected_dominant_role": dominant_role,
            "selected_dominant_role_fraction": selected_role_fractions[
                dominant_role
            ],
            "selected_over_valid_role_fraction_ratio": {
                name: (
                    selected_role_fractions[name] / valid_role_fractions[name]
                    if valid_role_fractions[name] > 0.0
                    else None
                )
                for name in role_names
            },
            "fields": {
                "q_base": _scoped_distribution(q_base_cpu),
                "delta_roi": _scoped_distribution(delta_roi_cpu),
                "delta_residual": _scoped_distribution(delta_residual_cpu),
                "residual_minus_roi": _scoped_distribution(
                    delta_residual_cpu - delta_roi_cpu
                ),
                "roi_minus_context": _scoped_distribution(delta_roi_cpu),
                "residual_minus_context": _scoped_distribution(
                    delta_residual_cpu
                ),
                "winning_modifier": _scoped_distribution(winner_values),
                "winner_top1_minus_top2_margin": _scoped_distribution(
                    winner_margin
                ),
            },
            "gt_for_route_used": False,
            "teacher_used": False,
            "oracle_used": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }

    @classmethod
    def _dynamic_diagnostic_route_telemetry(
        cls,
        *,
        route: Mapping[str, Any],
        geometry: torch.Tensor,
        source_grid_hw: tuple[int, int],
        minimum_extent_wh: tuple[float, float],
        maximum_extent_wh: tuple[float, float],
        packed: Mapping[str, Any],
        policy_calibration: Mapping[str, Any] | None = None,
        branch_calibration: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Serialize result-blind dynamic geometry and execution diagnostics."""

        geometry = geometry.detach().to(device="cpu", dtype=torch.float32)
        k_t = route["k_per_tubelet"].detach().to(device="cpu", dtype=torch.long)
        tubelet_indices = route["tubelet_indices"].detach().to(device="cpu", dtype=torch.long)
        role_ids = route["selected_role_ids"].detach().to(device="cpu", dtype=torch.long)
        physical_indices = route["physical_indices"].detach().to(device="cpu", dtype=torch.long)
        if (
            geometry.ndim != 3
            or geometry.shape[0] != 1
            or geometry.shape[-1] != 4
            or k_t.shape != geometry.shape[:2]
            or tubelet_indices.shape != role_ids.shape
            or tubelet_indices.shape != physical_indices.shape
            or tubelet_indices.shape[0] != 1
        ):
            raise ValueError("dynamic diagnostic telemetry requires one aligned sample")
        if not bool(torch.isfinite(geometry).all().item()):
            raise FloatingPointError("dynamic diagnostic geometry is nonfinite")

        tubelet_count = int(geometry.shape[1])
        window_budget = int(physical_indices.shape[1])
        item_count = int(source_grid_hw[0]) * int(source_grid_hw[1])
        if (
            window_budget <= 0
            or int(k_t.sum().item()) != window_budget
            or bool((tubelet_indices < 0).any().item())
            or bool((tubelet_indices >= tubelet_count).any().item())
            or bool((role_ids < 0).any().item())
            or bool((role_ids > 2).any().item())
            or bool((physical_indices < 0).any().item())
            or bool((physical_indices >= tubelet_count * item_count).any().item())
        ):
            raise ValueError("dynamic diagnostic route leaves its physical lattice")
        if not torch.equal(
            torch.div(physical_indices, item_count, rounding_mode="floor"),
            tubelet_indices,
        ):
            raise RuntimeError("dynamic diagnostic physical indices disagree with tubelet lineage")
        if int(torch.unique(physical_indices).numel()) != window_budget:
            raise RuntimeError("dynamic diagnostic physical route is not one-copy exact-B")

        role_names = ("context", "roi", "residual")
        linear_role = tubelet_indices[0] * len(role_names) + role_ids[0]
        roles_per_tubelet = torch.bincount(
            linear_role,
            minlength=tubelet_count * len(role_names),
        ).reshape(tubelet_count, len(role_names))
        if not torch.equal(roles_per_tubelet.sum(dim=-1), k_t[0]):
            raise RuntimeError("dynamic role diagnostics do not partition K_t")
        aggregate_roles = {
            name: int(roles_per_tubelet[:, role_id].sum().item())
            for role_id, name in enumerate(role_names)
        }
        if aggregate_roles != {
            name: int(route["role_counts"][name]) for name in role_names
        }:
            raise RuntimeError("dynamic role diagnostics disagree with route receipt")

        minimum = torch.tensor(minimum_extent_wh, dtype=torch.float32)
        maximum = torch.tensor(maximum_extent_wh, dtype=torch.float32)
        if (
            minimum.shape != (2,)
            or maximum.shape != (2,)
            or not bool((minimum > 0.0).all().item())
            or not bool((maximum <= 1.0).all().item())
            or not bool((minimum < maximum).all().item())
        ):
            raise ValueError("dynamic diagnostic extent bounds are invalid")
        centers = geometry[0, :, :2]
        extents = geometry[0, :, 2:]
        normalized_extents = (extents - minimum) / (maximum - minimum)
        tolerance = 1e-5
        if bool((normalized_extents < -tolerance).any().item()) or bool(
            (normalized_extents > 1.0 + tolerance).any().item()
        ):
            raise RuntimeError("dynamic diagnostic geometry leaves decoded bounds")
        if bool((centers < extents / 2.0 - tolerance).any().item()) or bool(
            (centers > 1.0 - extents / 2.0 + tolerance).any().item()
        ):
            raise RuntimeError("dynamic diagnostic ROI leaves the normalized frame")
        normalized_extents = normalized_extents.clamp(0.0, 1.0)

        packed_contract = {
            "schema_version": "videomae_native_ragged_v1",
            "execution_mode": "true_clip_ragged_no_padding",
            "batch_size": 1,
            "total_tubelets": tubelet_count,
            "source_grid_hw": list(map(int, source_grid_hw)),
            "spatial_tokens_per_tubelet": item_count,
            "window_token_budget": window_budget,
            "requested_physical_tokens_per_window": window_budget,
            "unique_physical_tokens_per_window": window_budget,
            "padded_heavy_tokens_per_window": 0,
            "executed_patch_tokens_per_window": window_budget,
            "heavy_backbone_forward_count": 1,
            "dense_adapter_forward_count": 0,
            "adapter_execution": "coordinate_lineage_true_ragged",
        }
        if any(packed.get(key) != value for key, value in packed_contract.items()):
            raise RuntimeError("dynamic diagnostic ragged ledger is invalid")
        clip_rows = packed.get("clip_token_counts")
        pair_rows = packed.get("attention_pairs_per_window")
        if (
            not isinstance(clip_rows, list)
            or len(clip_rows) != 1
            or not isinstance(pair_rows, list)
            or len(pair_rows) != 1
            or sum(map(int, clip_rows[0])) != window_budget
            or sum(int(value) ** 2 for value in clip_rows[0]) != int(pair_rows[0])
        ):
            raise RuntimeError("dynamic diagnostic ragged cost ledger is invalid")

        def _distribution(values: torch.Tensor) -> dict[str, float]:
            values = values.flatten().to(torch.float32)
            quantiles = torch.quantile(
                values,
                torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95]),
            )
            return {
                "min": float(values.min().item()),
                "p05": float(quantiles[0].item()),
                "p25": float(quantiles[1].item()),
                "p50": float(quantiles[2].item()),
                "p75": float(quantiles[3].item()),
                "p95": float(quantiles[4].item()),
                "max": float(values.max().item()),
                "mean": float(values.mean().item()),
            }

        k_values = k_t[0]
        k_unique, k_counts = torch.unique(k_values, sorted=True, return_counts=True)
        widths = extents[:, 0]
        heights = extents[:, 1]
        saturation_fraction = 0.01
        center_steps = centers[1:] - centers[:-1]
        extent_steps = extents[1:] - extents[:-1]
        payload = {
            "schema_version": "georoute_dynamic_diagnostic_window_telemetry_v1",
            "measurement_scope": "accuracy_replay_only_excluded_from_timed_cost",
            "batch_size": 1,
            "tubelet_count": tubelet_count,
            "item_count": item_count,
            "source_grid_hw": list(map(int, source_grid_hw)),
            "window_token_budget": window_budget,
            "selected_physical_index_sha256": cls._tensor_sha256(physical_indices),
            "k_t": {
                "values": [int(value) for value in k_values.tolist()],
                "min": int(k_values.min().item()),
                "max": int(k_values.max().item()),
                "zero_count": int((k_values == 0).sum().item()),
                "histogram": {
                    str(int(key)): int(count)
                    for key, count in zip(k_unique.tolist(), k_counts.tolist())
                },
            },
            "roles": {
                "order": list(role_names),
                "aggregate_counts": aggregate_roles,
                "aggregate_fractions": {
                    name: aggregate_roles[name] / float(window_budget)
                    for name in role_names
                },
                "per_tubelet_counts": [
                    [int(value) for value in row]
                    for row in roles_per_tubelet.tolist()
                ],
            },
            "geometry": {
                "parameter_order": ["cx", "cy", "w", "h"],
                "values": [
                    [float(value) for value in row] for row in geometry[0].tolist()
                ],
                "minimum_extent_wh": [float(value) for value in minimum.tolist()],
                "maximum_extent_wh": [float(value) for value in maximum.tolist()],
                "width": _distribution(widths),
                "height": _distribution(heights),
                "area": _distribution(widths * heights),
                "floor_saturation_definition": "normalized_distance_from_floor_le_0.01",
                "ceiling_saturation_definition": "normalized_distance_from_floor_ge_0.99",
                "width_floor_saturation_rate": float(
                    (normalized_extents[:, 0] <= saturation_fraction).float().mean().item()
                ),
                "height_floor_saturation_rate": float(
                    (normalized_extents[:, 1] <= saturation_fraction).float().mean().item()
                ),
                "width_ceiling_saturation_rate": float(
                    (normalized_extents[:, 0] >= 1.0 - saturation_fraction).float().mean().item()
                ),
                "height_ceiling_saturation_rate": float(
                    (normalized_extents[:, 1] >= 1.0 - saturation_fraction).float().mean().item()
                ),
                "center_step_l2_mean": (
                    float(torch.linalg.vector_norm(center_steps, dim=-1).mean().item())
                    if center_steps.numel()
                    else 0.0
                ),
                "extent_step_l2_mean": (
                    float(torch.linalg.vector_norm(extent_steps, dim=-1).mean().item())
                    if extent_steps.numel()
                    else 0.0
                ),
            },
            "ragged_execution": {
                "clip_token_counts": [int(value) for value in clip_rows[0]],
                "attention_pairs": int(pair_rows[0]),
                "requested_physical_tokens": window_budget,
                "unique_physical_tokens": window_budget,
                "padded_heavy_tokens": 0,
                "executed_patch_tokens": window_budget,
                "ragged_attention_bucket_call_count": int(
                    packed.get("ragged_attention_bucket_call_count", -1)
                ),
                "ragged_mlp_bucket_call_count": int(
                    packed.get("ragged_mlp_bucket_call_count", -1)
                ),
            },
            "role_assignment_changes_execution": False,
            "gt_for_route_used": False,
            "teacher_used": False,
            "oracle_used": False,
            "official_test_opened": False,
            "paper_claim_allowed": False,
        }
        if policy_calibration is not None:
            if (
                policy_calibration.get("schema_version")
                != "scnr_dynamic_role_calibration_window_v1"
                or policy_calibration.get("diagnostic_only") is not True
                or policy_calibration.get("changes_route_or_execution") is not False
            ):
                raise ValueError("dynamic role calibration telemetry is invalid")
            payload["policy_calibration"] = dict(policy_calibration)
        if branch_calibration is not None:
            expected_mode = branch_calibration.get("mode")
            if (
                branch_calibration.get("schema_version")
                != "scnr_dynamic_branch_calibration_window_v1"
                or expected_mode not in DYNAMIC_BRANCH_CALIBRATION_MODES
                or branch_calibration.get("target") != "delta_residual"
                or not isinstance(
                    branch_calibration.get("valid_candidate_count"), int
                )
                or isinstance(branch_calibration.get("valid_candidate_count"), bool)
                or not 0
                < int(branch_calibration["valid_candidate_count"])
                <= tubelet_count * item_count
                or branch_calibration.get("changes_q_base") is not False
                or branch_calibration.get("changes_delta_roi") is not False
                or branch_calibration.get("changes_context_zero_modifier") is not False
                or branch_calibration.get("changes_budget_or_role_quota") is not False
                or branch_calibration.get("mean_detached") is not False
            ):
                raise ValueError("dynamic branch calibration telemetry is invalid")
            payload["branch_calibration"] = dict(branch_calibration)
        return payload

    @staticmethod
    def _route_score_statistics(
        scores: torch.Tensor,
        *,
        selected_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> dict[str, float | None]:
        selected = scores.detach().masked_select(selected_mask)
        unselected_mask = valid_mask & ~selected_mask
        unselected = scores.detach().masked_select(unselected_mask)
        result: dict[str, float | None] = {
            "selected_mean": (float(selected.float().mean().item()) if selected.numel() else None),
            "unselected_mean": (float(unselected.float().mean().item()) if unselected.numel() else None),
            "hard_margin_mean": None,
        }
        if bool(unselected_mask.any().item()):
            selected_floor = scores.detach().masked_fill(~selected_mask, float("inf")).amin(dim=-1)
            unselected_ceiling = scores.detach().masked_fill(~unselected_mask, float("-inf")).amax(dim=-1)
            result["hard_margin_mean"] = float((selected_floor - unselected_ceiling).float().mean().item())
        return result

    @staticmethod
    def _detached_tensor_statistics(value: torch.Tensor) -> dict[str, Any]:
        detached = value.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = int(detached.numel())
        finite_values = detached.float().masked_select(finite_mask)
        return {
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "shape": list(detached.shape),
            "finite": finite_count == total_count,
            "finite_count": finite_count,
            "nonfinite_count": total_count - finite_count,
            "finite_min": (
                float(finite_values.min().item()) if finite_count else None
            ),
            "finite_max": (
                float(finite_values.max().item()) if finite_count else None
            ),
            "finite_mean": (
                float(finite_values.mean().item()) if finite_count else None
            ),
            "scalar_value": (
                float(finite_values.item())
                if total_count == 1 and finite_count == 1
                else None
            ),
        }

    @staticmethod
    def _tensor_sha256(value: torch.Tensor) -> str:
        canonical = (
            value.detach()
            .to(device="cpu", dtype=torch.float32)
            .contiguous()
        )
        return hashlib.sha256(canonical.numpy().tobytes()).hexdigest()

    @staticmethod
    def _observed_path_pl_entropy(
        logits: torch.Tensor,
        *,
        ordered_indices: torch.Tensor,
        available_mask: torch.Tensor,
        temperature: float,
    ) -> dict[str, Any]:
        """Summarize conditional categorical entropy along an observed PL path."""

        if ordered_indices.shape[-1] == 0:
            return {
                "applicable": False,
                "ordered_slot_count": 0,
                "conditional_entropy_mean": None,
                "conditional_entropy_min": None,
                "conditional_entropy_max": None,
                "observed_ordered_log_probability_mean": None,
            }
        available = available_mask.detach().clone()
        scaled = logits.detach().float() / float(temperature)
        entropies: list[torch.Tensor] = []
        observed_log_probability = torch.zeros(
            logits.shape[:2],
            device=logits.device,
            dtype=torch.float32,
        )
        for slot in range(ordered_indices.shape[-1]):
            choice = ordered_indices[..., slot : slot + 1]
            if bool((~available.gather(-1, choice)).any().item()):
                raise RuntimeError("telemetry observed an unavailable PL choice")
            log_probability = F.log_softmax(
                scaled.masked_fill(~available, float("-inf")),
                dim=-1,
            )
            probability = log_probability.exp()
            entropy = -torch.where(
                available,
                probability * log_probability,
                torch.zeros_like(probability),
            ).sum(dim=-1)
            entropies.append(entropy)
            observed_log_probability = observed_log_probability + log_probability.gather(
                -1,
                choice,
            ).squeeze(-1)
            available = available.scatter(-1, choice, False)
        values = torch.stack(entropies, dim=-1)
        return {
            "applicable": True,
            "ordered_slot_count": int(ordered_indices.shape[-1]),
            "measurement": "mean_observed_path_conditional_categorical_entropy",
            "conditional_entropy_mean": float(values.mean().item()),
            "conditional_entropy_min": float(values.min().item()),
            "conditional_entropy_max": float(values.max().item()),
            "observed_ordered_log_probability_mean": float(
                observed_log_probability.mean().item()
            ),
        }

    @classmethod
    def _diagnostic_route_telemetry(
        cls,
        *,
        route: Mapping[str, Any],
        roi_logits: torch.Tensor,
        residual_logits: torch.Tensor,
        valid_patch_mask: torch.Tensor,
        selected_coordinates: torch.Tensor,
        geometry: torch.Tensor,
        original_geometry: torch.Tensor,
        source_grid_hw: tuple[int, int],
        policy_temperature: float,
    ) -> dict[str, Any]:
        selected_mask = route["selected_mask"].detach()
        selected_coordinates = selected_coordinates.detach().float()
        geometry = geometry.detach().float()
        selected_count = int(selected_mask.sum().item())
        if selected_count <= 0:
            raise RuntimeError("GeoRoute telemetry observed an empty hard route")

        if selected_mask.shape[1] > 1:
            previous = selected_mask[:, :-1]
            following = selected_mask[:, 1:]
            intersection = (previous & following).sum(dim=-1).float()
            union = (previous | following).sum(dim=-1).float()
            adjacent_jaccard = intersection / union.clamp_min(1.0)
            lineage_retention = intersection / float(route["target_k"])
            center_step = (geometry[:, 1:, :2] - geometry[:, :-1, :2]).square().sum(dim=-1).sqrt()
            extent_step = (geometry[:, 1:, 2:] - geometry[:, :-1, 2:]).square().sum(dim=-1).sqrt()
        else:
            intersection = geometry.new_zeros((geometry.shape[0], 0))
            union = geometry.new_zeros((geometry.shape[0], 0))
            adjacent_jaccard = geometry.new_zeros((geometry.shape[0], 0))
            lineage_retention = geometry.new_zeros((geometry.shape[0], 0))
            center_step = geometry.new_zeros((geometry.shape[0], 0))
            extent_step = geometry.new_zeros((geometry.shape[0], 0))

        x = selected_coordinates[..., 0]
        y = selected_coordinates[..., 1]
        quadrants = torch.stack(
            (
                ((x < 0.5) & (y < 0.5)).sum(),
                ((x >= 0.5) & (y < 0.5)).sum(),
                ((x < 0.5) & (y >= 0.5)).sum(),
                ((x >= 0.5) & (y >= 0.5)).sum(),
            )
        ).float()
        surrogate = route["soft_membership"].detach().float()
        valid_surrogate = surrogate.masked_select(valid_patch_mask)
        selected_surrogate = surrogate.masked_select(selected_mask)
        unselected_surrogate = surrogate.masked_select(valid_patch_mask & ~selected_mask)
        hard = selected_mask.to(dtype=surrogate.dtype)
        hard_soft_l1 = (surrogate - hard).abs().masked_select(valid_patch_mask).mean()
        indices = route["indices"].detach().to("cpu").contiguous()
        role_indices = route.get("role_indices", {})

        def _mean_or_zero(value: torch.Tensor) -> float:
            return float(value.mean().item()) if value.numel() else 0.0

        def _role_summary(role: str) -> dict[str, Any]:
            ordered_role = role_indices.get(role)
            if ordered_role is None or ordered_role.shape[-1] == 0:
                return {
                    "applicable": False,
                    "tokens_per_tubelet": 0,
                }
            ordered_role = ordered_role.detach()
            role_coordinates = cls._selected_native_coordinates(
                ordered_role,
                source_grid_hw=source_grid_hw,
            ).detach().float()
            role_mask = torch.zeros_like(selected_mask).scatter(
                -1,
                ordered_role,
                True,
            )
            role_x = role_coordinates[..., 0]
            role_y = role_coordinates[..., 1]
            if role_mask.shape[1] > 1:
                role_previous = role_mask[:, :-1]
                role_following = role_mask[:, 1:]
                role_intersection = (role_previous & role_following).sum(dim=-1).float()
                role_union = (role_previous | role_following).sum(dim=-1).float()
                role_jaccard = role_intersection / role_union.clamp_min(1.0)
                role_lineage = role_intersection / float(ordered_role.shape[-1])
            else:
                role_intersection = geometry.new_zeros((geometry.shape[0], 0))
                role_jaccard = geometry.new_zeros((geometry.shape[0], 0))
                role_lineage = geometry.new_zeros((geometry.shape[0], 0))
            return {
                "applicable": True,
                "tokens_per_tubelet": int(ordered_role.shape[-1]),
                "ordered_index_sha256": cls._tensor_sha256(
                    ordered_role.to(dtype=torch.float32)
                ),
                "x_span_mean": float(
                    (role_x.amax(dim=-1) - role_x.amin(dim=-1)).mean().item()
                ),
                "y_span_mean": float(
                    (role_y.amax(dim=-1) - role_y.amin(dim=-1)).mean().item()
                ),
                "adjacent_pair_count": int(role_intersection.numel()),
                "adjacent_jaccard_mean": _mean_or_zero(role_jaccard),
                "lineage_survival_mean": _mean_or_zero(role_lineage),
            }

        role_telemetry = {
            role: _role_summary(role)
            for role in ("context", "roi", "residual")
        }
        branch_entropy: dict[str, Any] = {}
        if role_indices:
            context_mask = torch.zeros_like(selected_mask).scatter(
                -1,
                role_indices["context"],
                True,
            )
            roi_mask = torch.zeros_like(selected_mask).scatter(
                -1,
                role_indices["roi"],
                True,
            )
            after_context = valid_patch_mask & ~context_mask
            branch_entropy["roi"] = cls._observed_path_pl_entropy(
                roi_logits,
                ordered_indices=role_indices["roi"],
                available_mask=after_context,
                temperature=policy_temperature,
            )
            branch_entropy["residual"] = cls._observed_path_pl_entropy(
                residual_logits,
                ordered_indices=role_indices["residual"],
                available_mask=after_context & ~roi_mask,
                temperature=policy_temperature,
            )
        route_rng = dict(route.get("route_rng", {}))
        route_rng_sha256 = (
            hashlib.sha256(
                json.dumps(
                    route_rng,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if route_rng
            else None
        )

        return {
            "schema_version": "georoute_diagnostic_window_telemetry_v2",
            "batch_size": int(selected_mask.shape[0]),
            "tubelet_count": int(selected_mask.shape[1]),
            "item_count": int(selected_mask.shape[2]),
            "target_k": int(route["target_k"]),
            "selected_index_sha256": hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
            "adjacent": {
                "pair_count": int(intersection.numel()),
                "intersection_mean": _mean_or_zero(intersection),
                "union_mean": _mean_or_zero(union),
                "jaccard_mean": _mean_or_zero(adjacent_jaccard),
                "lineage_retention_mean": _mean_or_zero(lineage_retention),
            },
            "coordinates": {
                "x_mean": float(x.mean().item()),
                "y_mean": float(y.mean().item()),
                "x_span_mean": float((x.amax(dim=-1) - x.amin(dim=-1)).mean().item()),
                "y_span_mean": float((y.amax(dim=-1) - y.amin(dim=-1)).mean().item()),
                "quadrant_fraction": [float(value) for value in (quadrants / float(selected_count)).tolist()],
            },
            "geometry": {
                "area_mean": float((geometry[..., 2] * geometry[..., 3]).mean().item()),
                "center_step_l2_mean": _mean_or_zero(center_step),
                "extent_step_l2_mean": _mean_or_zero(extent_step),
                "original_trajectory_sha256": cls._tensor_sha256(
                    original_geometry
                ),
                "routing_trajectory_sha256": cls._tensor_sha256(geometry),
                "trajectory_changed": not torch.equal(
                    original_geometry.detach(),
                    geometry.detach(),
                ),
            },
            "scores": {
                "roi": cls._route_score_statistics(
                    roi_logits,
                    selected_mask=selected_mask,
                    valid_mask=valid_patch_mask,
                ),
                "residual": cls._route_score_statistics(
                    residual_logits,
                    selected_mask=selected_mask,
                    valid_mask=valid_patch_mask,
                ),
            },
            "surrogate": {
                "valid_mean": _mean_or_zero(valid_surrogate),
                "selected_mean": _mean_or_zero(selected_surrogate),
                "unselected_mean": _mean_or_zero(unselected_surrogate),
                "hard_soft_l1_mean": float(hard_soft_l1.item()),
            },
            "role_counts": dict(route["role_counts"]),
            "roles": role_telemetry,
            "branch_entropy": branch_entropy,
            "branch_gradient": {
                role: {
                    "applicable": bool(
                        role_indices
                        and role_indices[role].shape[-1] > 0
                        and logits.requires_grad
                    ),
                    "observed": False,
                    "measurement": "autograd_hook_before_optimizer_unscale",
                }
                for role, logits in (
                    ("roi", roi_logits),
                    ("residual", residual_logits),
                )
            },
            "route_rng": route_rng,
            "route_rng_sha256": route_rng_sha256,
        }

    def _forward_dynamic_scnr(
        self,
        *,
        source: torch.Tensor,
        scout: torch.Tensor,
        native: torch.Tensor,
        source_grid_hw: tuple[int, int],
        ignored_remainder: tuple[int, int],
        valid_patch_mask: torch.Tensor,
        masks: torch.Tensor | None,
    ) -> torch.Tensor:
        """Execute the approved dynamic exact-B Stage-1 route end to end."""

        if self.route_mode not in DYNAMIC_ROUTE_MODES:
            raise RuntimeError("dynamic SCNR forward called for a legacy route")
        if self.training:
            if self._successful_update_index is None:
                raise RuntimeError(
                    "dynamic SCNR training requires the successful-update hook"
                )
            if (
                self._pending_regularization is not None
                or self._pending_score_function is not None
                or self._pending_dynamic_auxiliary is not None
            ):
                raise RuntimeError(
                    "dynamic SCNR pending training losses were not consumed exactly once"
                )

        (
            geometry,
            q_base,
            delta_residual,
            scout_features,
            geometry_regularization,
        ) = self._compute_dynamic_route_fields(
            scout,
            source_grid_hw=source_grid_hw,
        )
        delta_roi = roi_modifier_from_geometry(
            geometry,
            grid_height=source_grid_hw[0],
            grid_width=source_grid_hw[1],
            temperature=self.roi_temperature,
        )
        if not self.dynamic_roi_modifier_enabled:
            delta_roi = torch.zeros_like(delta_roi)
        delta_residual_raw = delta_residual
        if not self.dynamic_residual_modifier_enabled:
            delta_residual_raw = torch.zeros_like(delta_residual_raw)
        delta_residual, residual_valid_mean_before = (
            calibrate_dynamic_residual_modifier(
                delta_residual_raw,
                valid_mask=valid_patch_mask,
                mode=self.branch_calibration_mode,
            )
        )
        route = select_dynamic_global_exact_budget(
            q_base=q_base,
            delta_roi=delta_roi,
            delta_residual=delta_residual,
            window_budget=self.window_token_budget,
            training=self.training,
            estimator=self.policy_estimator if self.training else "none",
            temperature=self.policy_temperature,
            valid_mask=valid_patch_mask,
        )
        physical_indices = route["physical_indices"]
        selected_native = self._gather_selected_native_physical(
            native,
            physical_indices,
        )
        selected_native = _normalize_uint8_video(
            selected_native.reshape(
                -1,
                3,
                self.tubelet_size,
                self.patch_size,
                self.patch_size,
            ),
            self.source_mean,
            self.source_std,
        ).reshape_as(selected_native)

        ragged_invocations_before = int(
            self.model.backbone.native_ragged_forward_invocations
        )
        selected_features = self.model.backbone.forward_native_ragged(
            selected_native,
            physical_indices,
            total_tubelets=int(native.shape[1]),
            source_grid_hw=source_grid_hw,
            use_absolute_position=self.absolute_position_enabled,
        )
        ragged_invocations_after = int(
            self.model.backbone.native_ragged_forward_invocations
        )
        ragged_invocation_delta = (
            ragged_invocations_after - ragged_invocations_before
        )
        if ragged_invocation_delta != 1:
            raise RuntimeError(
                "dynamic SCNR did not execute exactly one ragged VideoMAE forward"
            )
        if selected_features.shape[:2] != physical_indices.shape:
            raise RuntimeError("dynamic heavy output differs from exact selected union")
        selected_features = selected_features * route["st_gate"].to(
            selected_features.dtype
        ).unsqueeze(-1)
        selected_scores = torch.zeros_like(
            route["selected_aggregation_logits"],
            dtype=selected_features.dtype,
        )
        selected_coordinates = self._selected_physical_coordinates(
            route["spatial_indices"],
            source_grid_hw=source_grid_hw,
        ).to(dtype=selected_features.dtype)
        intermediate, heavy_valid_mask = self.sparse_adapter.forward_ragged(
            selected_features,
            selected_scores,
            geometry,
            selected_coordinates,
            route["tubelet_indices"],
            use_absolute_coordinates=self.absolute_coordinates_enabled,
            use_roi_relative_coordinates=self.roi_relative_coordinates_enabled,
            use_geometry_projection=self.geometry_projection_enabled,
            pooling_mode=self.pooling_mode,
        )
        expected_heavy_valid = route["k_per_tubelet"] > 0
        if not torch.equal(heavy_valid_mask, expected_heavy_valid):
            raise RuntimeError("masked-zero carrier disagrees with dynamic K_t")
        empty_values = intermediate.transpose(1, 2).masked_select(
            (~heavy_valid_mask).unsqueeze(-1).expand(
                *heavy_valid_mask.shape,
                intermediate.shape[1],
            )
        )
        if not bool(empty_values.eq(0).all().item()):
            raise RuntimeError("empty dynamic tubelet carries nonzero heavy content")
        self.latest_heavy_valid_mask = heavy_valid_mask.detach()

        output = deterministic_linear_2x(intermediate)
        if output.shape != (
            source.shape[0],
            int(self.model.backbone.embed_dims),
            self.output_length,
        ):
            raise RuntimeError(
                "dynamic SCNR violated the AdaTAD backbone feature contract"
            )
        if masks is not None:
            if masks.shape != (output.shape[0], output.shape[-1]):
                raise ValueError("dynamic SCNR masks must match the detector time axis")
            output = output * masks.to(output.device).unsqueeze(1).detach().to(
                output.dtype
            )

        soft_probability = route["soft_probability"]
        if self.training:
            if soft_probability is None or self.dynamic_aux_head is None:
                raise RuntimeError("dynamic SCNR training lost its soft proxy path")
            with torch.autocast(device_type=scout.device.type, enabled=False):
                auxiliary_logits = self.dynamic_aux_head(
                    scout_features.mean(dim=(-1, -2))
                )
                proxy_features = self._dynamic_soft_proxy_features(
                    scout_features,
                    soft_probability,
                    source_grid_hw=source_grid_hw,
                )
                proxy_logits = self.dynamic_aux_head(proxy_features)
            if auxiliary_logits.dtype != torch.float32 or proxy_logits.dtype != torch.float32:
                raise FloatingPointError("dynamic auxiliary paths must remain FP32")
            self._pending_regularization = {"geometry": geometry_regularization}
            self._pending_score_function = None
            self._pending_dynamic_auxiliary = {
                "auxiliary_logits": auxiliary_logits,
                "proxy_logits": proxy_logits,
                "successful_update": int(self._successful_update_index),
            }
        else:
            self._pending_regularization = None
            self._pending_score_function = None
            self._pending_dynamic_auxiliary = None

        packed = dict(self.model.backbone.latest_native_packed_summary or {})
        executed_per_window = int(
            packed.get("executed_patch_tokens_per_window", -1)
        )
        padded_per_window = int(packed.get("padded_heavy_tokens_per_window", -1))
        if executed_per_window != self.window_token_budget or padded_per_window != 0:
            raise RuntimeError(
                "dynamic ragged ledger differs from exact-B zero-padding contract"
            )
        k_per_tubelet = route["k_per_tubelet"].detach()
        role_counts_per_window = route["role_counts_per_window"].detach()
        soft_budget_sum = (
            None
            if soft_probability is None
            else soft_probability.detach().reshape(source.shape[0], -1).sum(dim=-1)
        )
        diagnostic_telemetry = None
        if self.diagnostic_telemetry_enabled:
            residual_valid_count = valid_patch_mask.reshape(
                valid_patch_mask.shape[0], -1
            ).sum(dim=-1)
            residual_valid_mean_after = (
                delta_residual.masked_fill(~valid_patch_mask, 0.0)
                .reshape(delta_residual.shape[0], -1)
                .sum(dim=-1)
                / residual_valid_count.to(dtype=delta_residual.dtype)
            )
            branch_calibration = {
                "schema_version": "scnr_dynamic_branch_calibration_window_v1",
                "mode": self.branch_calibration_mode,
                "target": "delta_residual",
                "scope": (
                    "complete_window_all_valid_candidates"
                    if self.branch_calibration_mode
                    == "residual_window_center"
                    else "disabled"
                ),
                "valid_candidate_count": int(residual_valid_count[0].item()),
                "residual_valid_mean_before": float(
                    residual_valid_mean_before[0].detach().item()
                ),
                "residual_valid_mean_after": float(
                    residual_valid_mean_after[0].detach().item()
                ),
                "changes_q_base": False,
                "changes_delta_roi": False,
                "changes_context_zero_modifier": False,
                "changes_budget_or_role_quota": False,
                "mean_detached": False,
            }
            policy_calibration = None
            if self.role_calibration_telemetry_enabled:
                policy_calibration = self._dynamic_policy_calibration_telemetry(
                    route=route,
                    q_base=q_base,
                    delta_roi=delta_roi,
                    delta_residual=delta_residual,
                    valid_patch_mask=valid_patch_mask,
                )
            diagnostic_telemetry = self._dynamic_diagnostic_route_telemetry(
                route=route,
                geometry=geometry,
                source_grid_hw=source_grid_hw,
                minimum_extent_wh=self._minimum_roi_extent_wh(source_grid_hw),
                maximum_extent_wh=(
                    self.max_roi_extent,
                    self.max_roi_extent,
                ),
                packed=packed,
                policy_calibration=policy_calibration,
                branch_calibration=branch_calibration,
            )
        self.latest_georoute_audit = {
            "schema_version": GEOROUTE_BACKBONE_SCHEMA,
            "routing_schema": route["schema_version"],
            "route_mode": self.route_mode,
            "policy_estimator": self.policy_estimator,
            "estimator_claim": (
                "biased_straight_through_plus_backward_only_global_soft_proxy"
                if self.training
                else "hard_exact_global_top_b_inference"
            ),
            "successful_update": (
                int(self._successful_update_index) if self.training else None
            ),
            "scout_autocast_enabled": False,
            "scout_compute_dtype": str(q_base.dtype),
            "scout_policy_stop_gradient": True,
            "auxiliary_updates_scout_stem": bool(self.training),
            "proxy_updates_scout_stem": False,
            "proxy_updates_heavy_backbone": False,
            "proxy_inference_enabled": False,
            "proxy_soft_budget_sum": (
                None if soft_budget_sum is None else soft_budget_sum.cpu().tolist()
            ),
            "policy_temperature": self.policy_temperature,
            "roi_temperature": self.roi_temperature,
            "roi_modifier_geometry": (
                "signed_ellipse_with_semiaxes_half_decoded_full_extent"
            ),
            "branch_calibration": {
                "mode": self.branch_calibration_mode,
                "scope": (
                    "complete_window_all_valid_candidates"
                    if self.branch_calibration_mode
                    == "residual_window_center"
                    else "disabled"
                ),
                "changes_q_base": False,
                "changes_delta_roi": False,
                "changes_context_zero_modifier": False,
                "changes_budget_or_role_quota": False,
                "mean_detached": False,
                "residual_valid_mean_before": (
                    residual_valid_mean_before.detach().cpu().tolist()
                ),
                "raw_delta_residual": self._detached_tensor_statistics(
                    delta_residual_raw
                ),
                "effective_delta_residual": self._detached_tensor_statistics(
                    delta_residual
                ),
            },
            "dynamic_roi_modifier_enabled": self.dynamic_roi_modifier_enabled,
            "dynamic_residual_modifier_enabled": (
                self.dynamic_residual_modifier_enabled
            ),
            "window_token_budget": self.window_token_budget,
            "window_budget_is_global": True,
            "independent_count_head": False,
            "fixed_context_quota": False,
            "fixed_per_tubelet_k": False,
            "k_t_allows_zero": True,
            "k_per_tubelet": k_per_tubelet.cpu().tolist(),
            "k_t_min": int(k_per_tubelet.min().item()),
            "k_t_max": int(k_per_tubelet.max().item()),
            "k_t_zero_count": int((k_per_tubelet == 0).sum().item()),
            "role_counts": dict(route["role_counts"]),
            "role_counts_per_window": role_counts_per_window.cpu().tolist(),
            "role_assignment_changes_execution": False,
            "hard_utility": self._detached_tensor_statistics(route["hard_utility"]),
            "soft_utility": self._detached_tensor_statistics(route["soft_utility"]),
            "q_base": self._detached_tensor_statistics(q_base),
            "delta_roi": self._detached_tensor_statistics(delta_roi),
            "delta_residual": self._detached_tensor_statistics(delta_residual),
            "physical_indices_sha256": self._tensor_sha256(physical_indices),
            "heavy_valid_mask_sha256": self._tensor_sha256(heavy_valid_mask),
            "heavy_valid_mask_matches_k_t": True,
            "zero_carrier_mode": self.zero_carrier_mode,
            "requested_physical_tokens_per_window": self.window_token_budget,
            "unique_physical_tokens_per_window": int(physical_indices.shape[1]),
            "padded_heavy_tokens_per_window": padded_per_window,
            "executed_patch_tokens_per_window": executed_per_window,
            "shared_backbone_instances": 1,
            "heavy_backbone_forward_count": ragged_invocation_delta,
            "native_ragged_invocation_counter_before": ragged_invocations_before,
            "native_ragged_invocation_counter_after": ragged_invocations_after,
            "source_input_shape": list(source.shape),
            "source_grid_hw": list(source_grid_hw),
            "source_padding_bottom_right": [0, 0],
            "source_ignored_remainder_bottom_right": list(ignored_remainder),
            "native_tubelet_shape": list(native.shape),
            "selected_native_tubelet_shape": list(selected_native.shape),
            "intermediate_shape": list(intermediate.shape),
            "output_shape": list(output.shape),
            "geometry_shape": list(geometry.shape),
            "geometry_extent_floor_mode": self.roi_extent_floor_mode,
            "geometry_extent_floor_cells": self.roi_extent_floor_cells,
            "geometry_min_extent_wh": list(
                self._minimum_roi_extent_wh(source_grid_hw)
            ),
            "geometry_max_extent_wh": [
                self.max_roi_extent,
                self.max_roi_extent,
            ],
            "geometry_smoothness_weight": self.geometry_smoothness_weight,
            "area_prior_weight": self.area_prior_weight,
            "full_frame_size_penalty_enabled": False,
            "absolute_position_enabled": self.absolute_position_enabled,
            "absolute_coordinates_enabled": self.absolute_coordinates_enabled,
            "roi_relative_coordinates_enabled": self.roi_relative_coordinates_enabled,
            "geometry_projection_enabled": self.geometry_projection_enabled,
            "diagnostic_telemetry_enabled": self.diagnostic_telemetry_enabled,
            "pooling_mode": self.pooling_mode,
            "packed": packed,
            "uses_grid_sample": False,
            "uses_resized_local_crop": False,
            "uses_gt_for_route": False,
            "uses_gt_for_auxiliary_fit_only": bool(self.training),
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        if diagnostic_telemetry is not None:
            self.latest_georoute_audit[
                "diagnostic_telemetry"
            ] = diagnostic_telemetry
        if self.role_calibration_telemetry_enabled:
            self.latest_georoute_audit[
                "role_calibration_telemetry_enabled"
            ] = True
        return output.to(torch.float32)

    def _forward_official_fixed_support(
        self,
        frames: torch.Tensor,
        masks: torch.Tensor | None,
        window_ordinals: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the matched official route with selection before VideoMAE."""

        del masks
        if self.training and (
            self._pending_regularization is not None
            or self._pending_score_function is not None
            or self._pending_dynamic_auxiliary is not None
        ):
            raise RuntimeError(
                "official pre-backbone pending training losses were not "
                "consumed exactly once"
            )
        source = self._validate_official_fixed_support_input(frames)
        self.set_norm_layer()
        (
            native,
            source_grid_hw,
            ignored_remainder,
            valid_patch_mask,
        ) = extract_native_tubelets(
            source,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        route = self._official_fixed_support_route(
            source,
            source_grid_hw=source_grid_hw,
            valid_patch_mask=valid_patch_mask,
            window_ordinals=window_ordinals,
        )
        batch_size, tubelets, spatial_tokens = map(int, native.shape[:3])
        refresh_support_mask = None
        backbone_refresh_mask = None
        selected_gate = route["st_gate"]
        if self.official_support in OFFICIAL_R3_SUPPORTS:
            physical_indices = route["physical_indices"]
            selected_per_tubelet = None
        else:
            spatial_indices = route["spatial_indices"]
            if self.refresh_carry_mode in {
                "drop32",
                "mod32_kv",
                "rc32_kv",
                "dsr6_kv",
            }:
                refresh_support_mask = self._build_strict_rectangle_refresh_mask(
                    native,
                    spatial_indices,
                )
                if self.refresh_carry_mode == "drop32":
                    spatial_indices = spatial_indices[refresh_support_mask].reshape(
                        batch_size,
                        tubelets,
                        32,
                    )
                    selected_gate = selected_gate[refresh_support_mask].reshape(
                        batch_size,
                        tubelets,
                        32,
                    )
                else:
                    backbone_refresh_mask = refresh_support_mask.reshape(
                        batch_size,
                        tubelets * 64,
                    )
            selected_per_tubelet = int(spatial_indices.shape[-1])
            tubelet_offsets = (
                torch.arange(
                    tubelets,
                    device=spatial_indices.device,
                    dtype=torch.long,
                )
                * spatial_tokens
            ).view(1, tubelets, 1)
            physical_indices = (
                spatial_indices + tubelet_offsets
            ).reshape(batch_size, tubelets * selected_per_tubelet)

        # Materialize the selected native 2x16x16 tubelets before entering any
        # VideoMAE block.  Every arm uses this exact gather and ragged heavy path;
        # only its physical support (100, ROI K64 or strict-rectangle K64) differs.
        selected_native = self._gather_selected_native_physical(
            native,
            physical_indices,
        )
        selected_native = _normalize_uint8_video(
            selected_native.reshape(
                -1,
                3,
                self.tubelet_size,
                self.patch_size,
                self.patch_size,
            ),
            self.source_mean,
            self.source_std,
        ).reshape_as(selected_native)
        ragged_invocations_before = int(
            self.model.backbone.native_ragged_forward_invocations
        )
        selected_features = self.model.backbone.forward_native_ragged(
            selected_native,
            physical_indices,
            total_tubelets=tubelets,
            source_grid_hw=source_grid_hw,
            use_absolute_position=self.absolute_position_enabled,
            refresh_mask=backbone_refresh_mask,
            refresh_mode=self.refresh_carry_mode,
            refresh_alpha=self.zoomtoken_refresh_carry_alpha,
        )
        ragged_invocation_delta = int(
            self.model.backbone.native_ragged_forward_invocations
        ) - ragged_invocations_before
        if ragged_invocation_delta != 1:
            raise RuntimeError(
                "official pre-backbone B/C/R1 did not execute exactly one ragged "
                "VideoMAE forward"
            )
        if tuple(selected_features.shape[:2]) != tuple(physical_indices.shape):
            raise RuntimeError(
                "official pre-backbone heavy output differs from selected support"
            )
        selected_features = selected_features * selected_gate.reshape(
            batch_size,
            -1,
        ).to(dtype=selected_features.dtype).unsqueeze(-1)
        tubelet_indices = torch.div(
            physical_indices,
            spatial_tokens,
            rounding_mode="floor",
        )
        selected_spatial_indices = physical_indices.remainder(spatial_tokens)
        selected_coordinates = self._selected_physical_coordinates(
            selected_spatial_indices,
            source_grid_hw=source_grid_hw,
        ).to(dtype=selected_features.dtype)
        intermediate, heavy_valid_mask = self.sparse_adapter.forward_ragged(
            selected_features,
            torch.zeros_like(physical_indices, dtype=selected_features.dtype),
            route["geometry"],
            selected_coordinates,
            tubelet_indices,
            use_absolute_coordinates=False,
            use_roi_relative_coordinates=False,
            use_geometry_projection=False,
            pooling_mode="uniform_selected",
        )
        if self.official_support in OFFICIAL_R3_SUPPORTS:
            expected_heavy_valid = route["k_per_tubelet"] > 0
            if not torch.equal(heavy_valid_mask, expected_heavy_valid):
                raise RuntimeError("R3 masked-zero carrier disagrees with natural K_t")
        elif not bool(heavy_valid_mask.all().item()):
            raise RuntimeError(
                "fixed per-tubelet support unexpectedly produced an empty tubelet"
            )
        self.latest_heavy_valid_mask = heavy_valid_mask.detach()
        output = deterministic_linear_2x(intermediate)
        expected_output_shape = (
            batch_size,
            int(self.model.backbone.embed_dims),
            self.output_length,
        )
        if tuple(output.shape) != expected_output_shape:
            raise RuntimeError(
                "official pre-backbone B/C/R1 violated the AdaTAD feature contract"
            )
        packed = dict(self.model.backbone.latest_native_packed_summary or {})
        strict_rectangle_audit = None
        if self.official_support == "strict_rect8x8":
            executed_tokens_per_tubelet = (
                32 if self.refresh_carry_mode == "drop32" else 64
            )
            expected_physical_tokens = tubelets * executed_tokens_per_tubelet
            packed_contract = {
                "schema_version": "videomae_native_ragged_v1",
                "execution_mode": "true_clip_ragged_no_padding",
                "window_token_budget": expected_physical_tokens,
                "requested_physical_tokens_per_window": expected_physical_tokens,
                "unique_physical_tokens_per_window": expected_physical_tokens,
                "padded_heavy_tokens_per_window": 0,
                "executed_patch_tokens_per_window": expected_physical_tokens,
                "heavy_backbone_forward_count": 1,
                "dense_adapter_forward_count": 0,
                "refresh_execution_mode": self.refresh_carry_mode,
            }
            if self.refresh_carry_mode in {
                "mod32_kv",
                "rc32_kv",
                "dsr6_kv",
                "apm32_ctx64",
                "cur32_ctx64",
                "apm_c32_full64",
            }:
                packed_contract["kv_context_tokens_per_window"] = tubelets * 64
                if self.refresh_carry_mode == "apm_c32_full64":
                    packed_contract["refresh_query_tokens_per_window"] = tubelets * 64
                elif self.refresh_carry_mode not in {
                    "apm32_ctx64",
                    "cur32_ctx64",
                }:
                    packed_contract["refresh_query_tokens_per_window"] = tubelets * 32
            if self.refresh_carry_mode == "dsr6_kv":
                packed_contract.update(
                    {
                        "full_update_block_count": 6,
                        "refresh_update_block_count": 6,
                    }
                )
            if any(packed.get(key) != value for key, value in packed_contract.items()):
                raise RuntimeError(
                    "strict rectangle R1 ragged ledger violates exact-K64 zero-padding"
                )
            strict_rectangle_audit = {
                "routing_schema": route["routing_schema"],
                "support_topology": "one_complete_hole_free_8x8_block",
                "source_grid_hw": [10, 10],
                "candidate_count": int(route["candidate_count"]),
                "candidate_top_left_row_col": route[
                    "candidate_top_left_row_col"
                ].detach().cpu().tolist(),
                "block_top_left_row_col": route[
                    "block_top_left_row_col"
                ].detach().cpu().tolist(),
                "block_size_hw": list(route["block_size_hw"]),
                "fixed_width_height": [0.8, 0.8],
                "hole_count": int(route["hole_count"]),
                "tokens_per_tubelet": 64,
                "executed_patch_tokens_per_tubelet": executed_tokens_per_tubelet,
                "refresh_query_tokens_per_tubelet": (
                    "K32_normal_K64_fallback"
                    if self.refresh_carry_mode
                    in {"apm32_ctx64", "cur32_ctx64"}
                    else (
                        64
                        if self.refresh_carry_mode
                        in {"full64", "apm_c32_full64"}
                        else 32
                    )
                ),
                "kv_context_tokens_per_tubelet": (
                    64
                    if self.refresh_carry_mode
                    in {
                        "mod32_kv",
                        "rc32_kv",
                        "dsr6_kv",
                        "apm32_ctx64",
                        "cur32_ctx64",
                        "apm_c32_full64",
                    }
                    else executed_tokens_per_tubelet
                ),
                "refresh_carry_mode": self.refresh_carry_mode,
                "requested_physical_tokens_per_window": expected_physical_tokens,
                "unique_physical_tokens_per_window": expected_physical_tokens,
                "executed_patch_tokens_per_window": expected_physical_tokens,
                "padded_heavy_tokens_per_window": 0,
                "dummy_tokens_used": False,
                "hard_forward_membership_exact_one": True,
                "categorical_temperature": 0.5,
                "raw_native_gather_before_patch_embedding": True,
            }
            if self.refresh_carry_mode == "dsr6_kv":
                strict_rectangle_audit.update(
                    {
                        "full_update_block_count": 6,
                        "refresh_update_block_count": 6,
                    }
                )
            if self.refresh_carry_mode in {
                "apm32_ctx64",
                "cur32_ctx64",
                "apm_c32_full64",
            }:
                temporal_alignment = packed.get("temporal_alignment")
                if not isinstance(temporal_alignment, Mapping):
                    raise RuntimeError(
                        "APM32/CUR32 execution omitted its temporal alignment ledger"
                    )
                expected_temporal = {
                    "schema_version": "zoomtoken_apm32_ctx64_alignment_v1",
                    "carrier_mode": self.refresh_carry_mode,
                    "memory_tensor": "pre_position_patch_embedding",
                    "memory_lifetime_tubelets": 1,
                    "clip_reset_tubelets": 8,
                    "similarity_threshold": 0.80,
                    "search_radius": 2,
                    "new_trainable_parameters": 0,
                    "previous_memory_detached": True,
                    "current_position_restored": True,
                    "future_tubelet_access": False,
                }
                if any(
                    temporal_alignment.get(key) != value
                    for key, value in expected_temporal.items()
                ):
                    raise RuntimeError(
                        "APM32/CUR32 temporal alignment ledger violates the frozen contract"
                    )
                total_alignment_tubelets = int(
                    temporal_alignment.get("total_tubelets", -1)
                )
                fallback_tubelets = int(
                    temporal_alignment.get("fallback_tubelets", -1)
                )
                normal_tubelets = int(temporal_alignment.get("normal_tubelets", -1))
                if (
                    total_alignment_tubelets != batch_size * tubelets
                    or fallback_tubelets < 0
                    or normal_tubelets < 0
                    or fallback_tubelets + normal_tubelets
                    != total_alignment_tubelets
                ):
                    raise RuntimeError(
                        "APM32/CUR32 fallback ledger does not cover every tubelet"
                    )
                strict_rectangle_audit["temporal_alignment"] = dict(
                    temporal_alignment
                )
                strict_rectangle_audit[
                    "refresh_query_tokens_per_window"
                ] = int(packed["refresh_query_tokens_per_window"])
                if self.refresh_carry_mode == "apm_c32_full64":
                    carrier_by_batch = packed.get(
                        "memory_carrier_tokens_per_window_by_batch"
                    )
                    if not isinstance(carrier_by_batch, list) or len(
                        carrier_by_batch
                    ) != batch_size:
                        raise RuntimeError(
                            "APM-C32/FULL64 omitted its memory-carrier ledger"
                        )
                    strict_rectangle_audit.update(
                        {
                            "memory_carrier_tokens_per_window_by_batch": list(
                                carrier_by_batch
                            ),
                            "deep_update_tokens_per_tubelet": 64,
                            "kv_tokens_per_tubelet": 64,
                            "adapter_tokens_per_tubelet": 64,
                            "fallback_deep_update_tokens_per_tubelet": 64,
                        }
                    )
        multibranch_audit = None
        r3_budget_loss = None
        r3_global_g = None
        if self.official_support in OFFICIAL_MULTIBRANCH_SUPPORTS:
            expected_physical_tokens = int(physical_indices.shape[1])
            packed_contract = {
                "schema_version": "videomae_native_ragged_v1",
                "execution_mode": "true_clip_ragged_no_padding",
                "window_token_budget": expected_physical_tokens,
                "requested_physical_tokens_per_window": expected_physical_tokens,
                "unique_physical_tokens_per_window": expected_physical_tokens,
                "padded_heavy_tokens_per_window": 0,
                "executed_patch_tokens_per_window": expected_physical_tokens,
                "heavy_backbone_forward_count": 1,
                "dense_adapter_forward_count": 0,
            }
            if any(packed.get(key) != value for key, value in packed_contract.items()):
                raise RuntimeError("R2/R3/R4 ragged ledger violates zero-padding")
            multibranch_audit = {
                "routing_schema": route["schema_version"]
                if "schema_version" in route
                else route["routing_schema"],
                "route_mode": route.get("mode", route.get("route_mode")),
                "raw_native_gather_before_patch_embedding": True,
                "one_forward_native_ragged": True,
                "padded_heavy_tokens_per_window": 0,
                "dummy_tokens_used": False,
                "physical_tokens_per_window": expected_physical_tokens,
                "hard_forward_membership_exact_one": bool(
                    route["st_gate"].detach().eq(1).all().item()
                ),
            }
            if self.official_support in OFFICIAL_QBASE_SUPPORTS:
                multibranch_audit.update(
                    {
                        "target_k": int(route["target_k"]),
                        "candidate_count": int(route["candidate_count"]),
                        "shuffle_enabled": bool(route.get("shuffle_enabled", False)),
                        "q_base_roi_modifier_enabled": False,
                        "q_base_residual_modifier_enabled": False,
                        "q_base_geometry_side_channel_enabled": False,
                    }
                )
                if "core_count" in route:
                    multibranch_audit.update(
                        {
                            "support_topology": "complete_7x7_core49_plus_outside15",
                            "core_count": int(route["core_count"]),
                            "outside_count": int(route["outside_count"]),
                        }
                    )
                elif self.official_support.startswith("strict_rect8x8"):
                    multibranch_audit["support_topology"] = (
                        "complete_8x8_candidate_top48"
                    )
                else:
                    multibranch_audit["support_topology"] = "global_q_base"
            else:
                r3_budget_loss, r3_global_g = self._r3_augmented_lagrangian_loss(
                    route
                )
                k_per_tubelet = route["k_per_tubelet"]
                multibranch_audit.update(
                    {
                        "support_topology": "continuous_strict_hard_rectangle_all_members",
                        "k_t_min": int(k_per_tubelet.min().item()),
                        "k_t_max": int(k_per_tubelet.max().item()),
                        "k_t_sum": int(k_per_tubelet.sum().item()),
                        "masked_zero_tubelets": int((k_per_tubelet == 0).sum().item()),
                        "area_shift_tubelets": int(route["area_shift_tubelets"]),
                        "budget_g": float(r3_global_g.item()),
                        "dual_lambda": float(self.r3_dual_lambda.item()),
                        "extra_anti_collapse_loss_enabled": False,
                    }
                )
        self.latest_georoute_audit = {
            "schema_version": (
                "georoute_official_prebackbone_r1_v1"
                if self.official_support == "strict_rect8x8"
                else (
                    "georoute_official_prebackbone_r234_v1"
                    if self.official_support in OFFICIAL_MULTIBRANCH_SUPPORTS
                    else "georoute_official_prebackbone_bc_v1"
                )
            ),
            "official_support": self.official_support,
            "selection_application": "pre_heavy_videomae",
            "native_materialization_before_heavy": True,
            "selected_tokens_per_tubelet": selected_per_tubelet,
            "refresh_carry_mode": self.refresh_carry_mode,
            "refresh_query_tokens_per_tubelet": (
                "K32_normal_K64_fallback"
                if self.refresh_carry_mode
                in {"apm32_ctx64", "cur32_ctx64"}
                else (
                    selected_per_tubelet
                    if self.refresh_carry_mode
                    in {"full64", "apm_c32_full64"}
                    else 32
                )
            ),
            "refresh_support_tokens_per_tubelet": (
                64 if self.refresh_carry_mode != "full64" else selected_per_tubelet
            ),
            "temporal_carry_enabled": self.temporal_carry_enabled,
            "temporal_carry_detached": self.temporal_carry_detached,
            "physical_tokens_per_window": int(physical_indices.shape[1]),
            "heavy_backbone_forward_count": ragged_invocation_delta,
            "heavy_execution": "true_clip_ragged_no_padding",
            "ignored_source_remainder_hw": list(ignored_remainder),
            "residual_enabled": False,
            "adapter_side_channel_enabled": False,
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_raw_prediction": False,
            "geometry_regularization_enabled": False,
            "packed": packed,
        }
        if strict_rectangle_audit is not None:
            self.latest_georoute_audit["strict_rectangle"] = strict_rectangle_audit
        if multibranch_audit is not None:
            self.latest_georoute_audit["route_ledger"] = multibranch_audit
        if self.training:
            # ActionFormer capability-dispatches the shared GeoRoute consumer on
            # every training forward.  This arm has no scientific regularizer,
            # but it must still publish one defined, exactly-once zero contract.
            self._pending_regularization = {"geometry": output.new_zeros(())}
            if self.official_support in OFFICIAL_R3_SUPPORTS:
                if self._successful_update_index is None:
                    raise RuntimeError("R3 requires the successful-update hook")
                if (
                    self._pending_r3_epoch_g is not None
                    and self._pending_r3_update_index
                    != int(self._successful_update_index)
                ):
                    raise RuntimeError("R3 budget item crossed update identities")
                self._pending_regularization["r3_budget"] = r3_budget_loss
                self._pending_r3_epoch_g = r3_global_g.detach()
                self._pending_r3_update_index = int(self._successful_update_index)
            self._pending_score_function = None
            self._pending_dynamic_auxiliary = None
        else:
            self._pending_regularization = None
            self._pending_score_function = None
            self._pending_dynamic_auxiliary = None
        return output.to(torch.float32)

    def forward_with_window_ordinals(
        self,
        frames: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        if not self.requires_route_window_ordinals:
            raise RuntimeError("window ordinals are reserved for shuffle controls")
        if not isinstance(metas, Sequence) or len(metas) != int(frames.shape[0]):
            raise ValueError("route metadata must contain one item per local sample")
        forbidden = {
            "gt_segments",
            "gt_labels",
            "prediction",
            "teacher",
            "oracle",
            "raw_prediction",
        }
        ordinals = []
        for meta in metas:
            if not isinstance(meta, Mapping) or forbidden.intersection(meta):
                raise ValueError("route metadata exposes forbidden result information")
            ordinal = meta.get("window_ordinal", None)
            if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
                raise ValueError("shuffle control requires a non-negative window ordinal")
            ordinals.append(ordinal)
        return self._forward_official_fixed_support(
            frames,
            None,
            window_ordinals=torch.tensor(
                ordinals,
                device=frames.device,
                dtype=torch.long,
            ),
        )

    def forward(self, frames, masks=None):
        if self.official_support is not None:
            return self._forward_official_fixed_support(frames, masks)
        source, scout = self._validate_inputs(frames)
        self.set_norm_layer()
        (
            native,
            source_grid_hw,
            ignored_remainder,
            valid_patch_mask,
        ) = extract_native_tubelets(
            source,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        if self.route_mode in DYNAMIC_ROUTE_MODES:
            return self._forward_dynamic_scnr(
                source=source,
                scout=scout,
                native=native,
                source_grid_hw=source_grid_hw,
                ignored_remainder=ignored_remainder,
                valid_patch_mask=valid_patch_mask,
                masks=masks,
            )
        learned_geometry_enabled = bool(
            self.route_mode in {"roi", "hybrid"}
            or self.geometry_side_channel
            or (
                self.route_mode in STRUCTURED_ROUTE_MODES
                and self.structured_roi_tokens > 0
            )
        )
        geometry_min_extent_wh = (
            self._minimum_roi_extent_wh(source_grid_hw)
            if learned_geometry_enabled
            else None
        )
        geometry, residual_logits, geometry_regularization = self._compute_route_fields(
            scout,
            source_grid_hw=source_grid_hw,
        )
        routing_geometry = geometry
        if self.route_mode == "structured_hybrid_geometry_shift":
            routing_geometry = self._geometry_shift_control(
                geometry,
                shift_tubelets=self.geometry_temporal_shift_tubelets,
            )
        roi_logits = roi_logits_from_geometry(
            routing_geometry,
            grid_height=source_grid_hw[0],
            grid_width=source_grid_hw[1],
            temperature=self.roi_temperature,
        )
        if self.route_mode in STRUCTURED_ROUTE_MODES:
            distributed_rank = (
                int(torch.distributed.get_rank())
                if torch.distributed.is_available()
                and torch.distributed.is_initialized()
                else 0
            )
            route = select_fixed_quota_structured_exact_k(
                roi_logits=roi_logits,
                residual_logits=residual_logits,
                mode=self.route_mode,
                context_tokens=self.structured_context_tokens,
                roi_tokens=self.structured_roi_tokens,
                residual_tokens=self.structured_residual_tokens,
                training=self.training,
                estimator=self.policy_estimator,
                temperature=self.policy_temperature,
                valid_mask=valid_patch_mask,
                study_seed=self.route_study_seed,
                successful_update_index=self._successful_update_index,
                distributed_rank=distributed_rank,
            )
        else:
            route = select_exact_k(
                roi_logits=roi_logits,
                residual_logits=residual_logits,
                mode=self.route_mode,
                tokens_per_tubelet=self.tokens_per_tubelet,
                context_tokens=self.context_tokens,
                roi_fraction=self.roi_fraction,
                training=self.training,
                estimator=self.policy_estimator,
                temperature=self.policy_temperature,
                valid_mask=valid_patch_mask,
                random_seed=self.random_seed,
            )
        if self.gradient_decomposition_enabled:
            if self._gradient_decomposition_payload is not None:
                raise RuntimeError(
                    "gradient decomposition payload was not consumed exactly once"
                )
            self._gradient_decomposition_payload = {
                "policy_estimator": self.policy_estimator,
                "logits": residual_logits,
                "ordered_indices": route["ordered_indices"].detach(),
                "ordered_log_prob": (
                    None
                    if route["ordered_log_prob"] is None
                    else route["ordered_log_prob"].detach()
                ),
                "valid_mask": valid_patch_mask.detach(),
                "temperature": float(self.policy_temperature),
                "weight": float(self.score_function_weight),
                "baseline_momentum": float(
                    self.score_function_baseline_momentum
                ),
                "temporal_reduction": self.score_function_temporal_reduction,
                "target_k": int(route["target_k"]),
                "item_count": int(route["item_count"]),
            }
        if not bool(valid_patch_mask.gather(-1, route["indices"]).all().item()):
            raise RuntimeError("GeoRoute selected an invalid native patch")
        selected_native = self._gather_selected_native_tubelets(native, route["indices"])
        selected_native = _normalize_uint8_video(
            selected_native.reshape(-1, 3, 2, self.patch_size, self.patch_size),
            self.source_mean,
            self.source_std,
        ).reshape_as(selected_native)
        packed_invocations_before = int(self.model.backbone.native_packed_forward_invocations)
        dense_reference_audit = None
        if self.p0_dense_reference_check:
            if source.device.type != "cuda":
                raise RuntimeError("GeoRoute P0 dense reference requires CUDA RNG isolation")
            # Restore RNG after the debug reference so the real packed forward
            # receives exactly the same stochastic-depth/dropout draw.  Keep
            # autograd enabled here: CUDA SDPA can choose a different numerical
            # kernel under ``no_grad`` than it uses for the actual detector
            # training forward.  The reference is detached immediately after
            # the matched-dispatch comparison input is materialized, so it
            # never contributes gradients or runtime-cost evidence.
            with torch.random.fork_rng(devices=[source.device.index or 0], enabled=True):
                dense_reference = self.model.backbone.forward_native_dense_reference(
                    selected_native,
                    route["indices"],
                    source_grid_hw=source_grid_hw,
                    use_absolute_position=self.absolute_position_enabled,
                ).detach()
        selected_features = self.model.backbone.forward_native_packed(
            selected_native,
            route["indices"],
            source_grid_hw=source_grid_hw,
            use_absolute_position=self.absolute_position_enabled,
        )
        packed_invocations_after = int(self.model.backbone.native_packed_forward_invocations)
        packed_invocation_delta = packed_invocations_after - packed_invocations_before
        if packed_invocation_delta != 1:
            raise RuntimeError("GeoRoute actual route did not execute exactly one packed VideoMAE forward")
        if self.p0_dense_reference_check:
            reference = dense_reference.to(dtype=selected_features.dtype)
            difference = (selected_features.detach() - reference).abs()
            max_abs_error = float(difference.max().item())
            mean_abs_error = float(difference.mean().item())
            tolerance = 1e-4
            if not bool(
                torch.allclose(
                    selected_features.detach(),
                    reference,
                    rtol=tolerance,
                    atol=tolerance,
                )
            ):
                raise RuntimeError("GeoRoute native packed all-token output disagrees with its dense P0 reference: " f"max_abs_error={max_abs_error:.8g}")
            dense_reference_audit = {
                "enabled": True,
                "reference_heavy_backbone_forward_count": 1,
                "real_route_heavy_backbone_forward_count": packed_invocation_delta,
                "max_abs_error": max_abs_error,
                "mean_abs_error": mean_abs_error,
                "rtol_atol": tolerance,
                "reference_autograd_mode": "enabled_matches_real_packed_forward",
                "passed": True,
                "cost_scope": "p0_debug_only_excluded_from_runtime_cost",
            }
        selected_features = selected_features * route["st_gate"].to(selected_features.dtype).unsqueeze(-1)
        if self.pooling_mode == "uniform_selected":
            selected_scores = torch.zeros_like(route["selected_aggregation_logits"])
        else:
            selected_scores = route["selected_aggregation_logits"]
        selected_coordinates = self._selected_native_coordinates(
            route["indices"],
            source_grid_hw=source_grid_hw,
        ).to(dtype=selected_features.dtype)
        diagnostic_telemetry = None
        if self.diagnostic_telemetry_enabled:
            diagnostic_telemetry = self._diagnostic_route_telemetry(
                route=route,
                roi_logits=roi_logits,
                residual_logits=residual_logits,
                valid_patch_mask=valid_patch_mask,
                selected_coordinates=selected_coordinates,
                geometry=routing_geometry,
                original_geometry=geometry,
                source_grid_hw=source_grid_hw,
                policy_temperature=self.policy_temperature,
            )
            if self.training:
                for role, logits in (
                    ("roi", roi_logits),
                    ("residual", residual_logits),
                ):
                    entry = diagnostic_telemetry["branch_gradient"][role]
                    if not entry["applicable"]:
                        continue

                    def _capture_branch_gradient(
                        gradient: torch.Tensor,
                        *,
                        destination: dict[str, Any] = entry,
                    ) -> torch.Tensor:
                        detached = gradient.detach().float()
                        destination.update(
                            {
                                "observed": True,
                                "shape": list(gradient.shape),
                                "dtype": str(gradient.dtype),
                                "finite": bool(torch.isfinite(detached).all().item()),
                                "l2_norm": float(detached.norm().item()),
                                "max_abs": float(detached.abs().max().item()),
                            }
                        )
                        return gradient

                    logits.register_hook(_capture_branch_gradient)
        intermediate = self.sparse_adapter(
            selected_features,
            selected_scores,
            routing_geometry,
            selected_coordinates,
            use_absolute_coordinates=self.absolute_coordinates_enabled,
            use_roi_relative_coordinates=self.roi_relative_coordinates_enabled,
            use_geometry_projection=self.geometry_projection_enabled,
            pooling_mode=self.pooling_mode,
        )
        output = deterministic_linear_2x(intermediate)
        if output.shape != (
            source.shape[0],
            int(self.model.backbone.embed_dims),
            self.output_length,
        ):
            raise RuntimeError("GeoRoute violated the AdaTAD [B,384,768] backbone contract")
        if masks is not None:
            if masks.shape != (output.shape[0], output.shape[-1]):
                raise ValueError("GeoRoute masks must match the detector time axis")
            output = output * masks.to(output.device).unsqueeze(1).detach().to(output.dtype)

        if self.training:
            if self._pending_regularization is not None or self._pending_score_function is not None:
                raise RuntimeError("GeoRoute pending training losses were not consumed exactly once")
            self._pending_regularization = {"geometry": geometry_regularization}
            if self.policy_estimator == "score_function":
                log_prob = route["ordered_log_prob"]
                if log_prob is None:
                    raise RuntimeError("score-function GeoRoute route did not emit an ordered log-probability")
                self._pending_score_function = {"ordered_log_prob": log_prob}
            else:
                self._pending_score_function = None
        else:
            self._pending_regularization = None
            self._pending_score_function = None

        packed = dict(self.model.backbone.latest_native_packed_summary or {})
        sorted_indices = route["indices"]
        if sorted_indices.shape[-1] > 1:
            duplicate_count = int((sorted_indices[..., 1:] == sorted_indices[..., :-1]).sum().item())
            unique_counts = 1 + (sorted_indices[..., 1:] != sorted_indices[..., :-1]).sum(dim=-1)
        else:
            duplicate_count = 0
            unique_counts = torch.ones_like(sorted_indices[..., 0])
        self.latest_georoute_audit = {
            "schema_version": GEOROUTE_BACKBONE_SCHEMA,
            "routing_schema": route["schema_version"],
            "route_mode": self.route_mode,
            "policy_estimator": self.policy_estimator,
            "scout_autocast_enabled": False,
            "scout_compute_dtype": str(residual_logits.dtype),
            "policy_temperature": self.policy_temperature,
            "score_function_weight": self.score_function_weight,
            "score_function_baseline_momentum": (self.score_function_baseline_momentum),
            "score_function_temporal_reduction": (
                self.score_function_temporal_reduction
            ),
            "geometry_smoothness_weight": self.geometry_smoothness_weight,
            "area_prior_weight": self.area_prior_weight,
            "full_frame_size_penalty_enabled": False,
            "geometry_extent_floor_mode": self.roi_extent_floor_mode,
            "geometry_extent_floor_cells": (
                self.roi_extent_floor_cells
                if self.roi_extent_floor_mode == "native_cells"
                else None
            ),
            "geometry_min_extent_wh": (
                None
                if geometry_min_extent_wh is None
                else list(geometry_min_extent_wh)
            ),
            "geometry_max_extent_wh": [
                self.max_roi_extent,
                self.max_roi_extent,
            ],
            "estimator_claim": "biased_straight_through"
            if self.policy_estimator == "straight_through"
            else "score_function_candidate"
            if self.policy_estimator == "score_function"
            else "no_policy_gradient",
            "shared_backbone_instances": 1,
            "heavy_backbone_forward_count": packed_invocation_delta,
            "native_packed_invocation_counter_before": packed_invocations_before,
            "native_packed_invocation_counter_after": packed_invocations_after,
            "source_input_shape": list(source.shape),
            "source_grid_hw": list(source_grid_hw),
            "source_padding_bottom_right": [0, 0],
            "source_ignored_remainder_bottom_right": list(ignored_remainder),
            "valid_patch_count_min": int(valid_patch_mask.sum(dim=-1).min().item()),
            "valid_patch_count_max": int(valid_patch_mask.sum(dim=-1).max().item()),
            "native_tubelet_shape": list(native.shape),
            "selected_native_tubelet_shape": list(selected_native.shape),
            "intermediate_shape": list(intermediate.shape),
            "output_shape": list(output.shape),
            "geometry_shape": list(geometry.shape),
            "geometry_stride_tubelets": self.geometry_stride_tubelets,
            "geometry_temporal_shift_tubelets": (
                self.geometry_temporal_shift_tubelets
            ),
            "absolute_position_enabled": self.absolute_position_enabled,
            "absolute_coordinates_enabled": self.absolute_coordinates_enabled,
            "roi_relative_coordinates_enabled": (self.roi_relative_coordinates_enabled),
            "geometry_projection_enabled": self.geometry_projection_enabled,
            "diagnostic_telemetry_enabled": self.diagnostic_telemetry_enabled,
            "pooling_mode": self.pooling_mode,
            "adapter_mode": self.adapter_mode,
            "geometry_side_channel": self.geometry_side_channel,
            "learned_geometry_enabled": learned_geometry_enabled,
            "learned_residual_enabled": bool(
                self.route_mode in {"free", "hybrid"}
                or (
                    self.route_mode in STRUCTURED_ROUTE_MODES
                    and self.structured_residual_tokens > 0
                )
            ),
            "free_control_is_roi_free": bool(self.route_mode != "free" or (self.route_mode == "free" and not self.geometry_side_channel)),
            "route_logits_used_for_pooling": self.pooling_mode == "route_score_ablation",
            "geometry_min": float(geometry.detach().min().item()),
            "geometry_max": float(geometry.detach().max().item()),
            "target_k": int(route["target_k"]),
            "item_count": int(route["item_count"]),
            "selected_unique_count_min": int(unique_counts.min().item()),
            "selected_unique_count_max": int(unique_counts.max().item()),
            "selected_duplicate_count": duplicate_count,
            "role_counts": dict(route["role_counts"]),
            "route_rng": dict(route.get("route_rng", {})),
            "branch_log_probabilities": {
                role: self._detached_tensor_statistics(value)
                for role, value in route.get(
                    "branch_log_probabilities",
                    {},
                ).items()
            },
            "packed": packed,
            "dense_native_reference": dense_reference_audit,
            "uses_grid_sample": False,
            "uses_resized_local_crop": False,
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        if self.amp_diagnostic_enabled:
            self.latest_georoute_audit["amp_diagnostic_enabled"] = True
            self.latest_georoute_audit["route_logits"] = {
                "roi": self._detached_tensor_statistics(roi_logits),
                "residual": self._detached_tensor_statistics(residual_logits),
            }
        if diagnostic_telemetry is not None:
            self.latest_georoute_audit["diagnostic_telemetry"] = diagnostic_telemetry
        return output.to(torch.float32)

    def consume_training_auxiliary_losses(
        self,
        *,
        masks: torch.Tensor,
        gt_segments: Sequence[torch.Tensor],
        gt_labels: Sequence[torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.training or self._pending_regularization is None:
            raise RuntimeError("GeoRoute regularization requires one preceding training forward")
        regularization = self._pending_regularization.pop("geometry")
        r3_budget = self._pending_regularization.pop("r3_budget", None)
        if self._pending_regularization:
            raise RuntimeError("GeoRoute published an unknown auxiliary loss")
        self._pending_regularization = None
        if self.latest_georoute_audit is not None:
            self.latest_georoute_audit["geometry_regularization"] = float(regularization.detach().item())
        if self.route_mode not in DYNAMIC_ROUTE_MODES:
            if r3_budget is None:
                return {"georoute_geometry_regularization_loss": regularization}
            return {
                "georoute_geometry_regularization_loss": regularization,
                "georoute_r3_augmented_lagrangian_loss": r3_budget,
            }

        if self._pending_dynamic_auxiliary is None:
            raise RuntimeError(
                "dynamic SCNR auxiliary losses require one preceding training forward"
            )
        pending = self._pending_dynamic_auxiliary
        self._pending_dynamic_auxiliary = None
        auxiliary_logits = pending["auxiliary_logits"]
        proxy_logits = pending["proxy_logits"]
        if auxiliary_logits.shape != proxy_logits.shape:
            raise RuntimeError("dynamic auxiliary and proxy logits must share shape")
        target = _temporal_class_occupancy_targets(
            gt_segments,
            gt_labels,
            batch_size=int(auxiliary_logits.shape[0]),
            num_classes=self.dynamic_aux_num_classes,
            output_length=int(auxiliary_logits.shape[-1]),
            detector_length=self.dynamic_aux_detector_length,
            device=auxiliary_logits.device,
            dtype=auxiliary_logits.dtype,
        )
        valid = F.interpolate(
            masks.to(auxiliary_logits.device).unsqueeze(1).to(auxiliary_logits.dtype),
            size=auxiliary_logits.shape[-1],
            mode="nearest",
        )
        denominator = valid.sum().clamp_min(1.0) * float(
            self.dynamic_aux_num_classes
        )
        auxiliary_raw = (
            F.binary_cross_entropy_with_logits(
                auxiliary_logits,
                target,
                reduction="none",
            )
            * valid
        ).sum() / denominator
        proxy_raw = (
            F.binary_cross_entropy_with_logits(
                proxy_logits,
                target,
                reduction="none",
            )
            * valid
        ).sum() / denominator
        proxy_weight = dynamic_proxy_weight_at_step(
            pending["successful_update"],
            initial_weight=self.dynamic_proxy_initial_weight,
            anneal_start=self.dynamic_proxy_anneal_start,
            anneal_end=self.dynamic_proxy_anneal_end,
        )
        if self.latest_georoute_audit is not None:
            self.latest_georoute_audit.update(
                {
                    "dynamic_auxiliary_raw": float(auxiliary_raw.detach().item()),
                    "dynamic_auxiliary_weight": self.dynamic_aux_weight,
                    "dynamic_proxy_raw": float(proxy_raw.detach().item()),
                    "dynamic_proxy_weight": proxy_weight,
                    "dynamic_proxy_active": proxy_weight > 0.0,
                    "dynamic_proxy_anneal_start": self.dynamic_proxy_anneal_start,
                    "dynamic_proxy_anneal_end": self.dynamic_proxy_anneal_end,
                    "dynamic_auxiliary_gt_scope": "fit_only_not_route_input",
                }
            )
        return {
            "georoute_geometry_regularization_loss": regularization,
            "georoute_dynamic_auxiliary_loss": (
                self.dynamic_aux_weight * auxiliary_raw
            ),
            "georoute_dynamic_soft_proxy_loss": proxy_weight * proxy_raw,
        }

    def consume_detector_policy_loss(
        self,
        *,
        detector_losses: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.training:
            return {}
        required_detector_keys = {"cls_loss", "reg_loss"}
        observed_detector_keys = {str(name) for name in detector_losses}
        if observed_detector_keys != required_detector_keys:
            raise ValueError(
                "GeoRoute policy risk accepts exactly cls_loss and reg_loss; "
                f"observed {sorted(observed_detector_keys)}"
            )
        tensor_losses = [
            (name, detector_losses[name])
            for name in sorted(required_detector_keys)
        ]
        if any(
            not torch.is_tensor(value)
            or value.ndim != 0
            or not bool(torch.isfinite(value).item())
            for _name, value in tensor_losses
        ):
            raise ValueError(
                "GeoRoute detector policy hook requires finite scalar cls/reg losses"
            )
        detector_cost = sum(value for _name, value in tensor_losses)
        if detector_cost.ndim != 0 or not bool(torch.isfinite(detector_cost).item()):
            raise ValueError("GeoRoute detector policy risk must be one finite scalar")
        gradient_decomposition_payload = getattr(
            self,
            "_gradient_decomposition_payload",
            None,
        )
        if gradient_decomposition_payload is not None:
            gradient_decomposition_payload.update(
                {
                    "detector_cost": detector_cost.detach(),
                    "detector_loss_keys": tuple(
                        sorted(name for name, _value in tensor_losses)
                    ),
                }
            )
        if self._pending_score_function is None:
            return {}
        if bool(self.score_function_baseline_initialized.item()):
            baseline = self.score_function_baseline
        else:
            baseline = torch.zeros_like(detector_cost)
        baseline_for_policy = baseline.detach().clone()
        ordered_log_prob = self._pending_score_function["ordered_log_prob"]
        policy_loss = score_function_policy_loss(
            detector_cost=detector_cost,
            ordered_log_prob=ordered_log_prob,
            baseline=baseline,
            weight=self.score_function_weight,
            temporal_reduction=self.score_function_temporal_reduction,
        )
        if gradient_decomposition_payload is not None:
            gradient_decomposition_payload.update(
                {
                    "baseline": baseline_for_policy,
                    "advantage": (
                        detector_cost.detach().to(torch.float32)
                        - baseline_for_policy.to(torch.float32)
                    ),
                    "policy_loss": policy_loss.detach(),
                }
            )
        with torch.no_grad():
            reward = detector_cost.detach()
            if bool(self.score_function_baseline_initialized.item()):
                self.score_function_baseline.mul_(self.score_function_baseline_momentum).add_(reward * (1.0 - self.score_function_baseline_momentum))
            else:
                self.score_function_baseline.copy_(reward)
                self.score_function_baseline_initialized.fill_(True)
        self._pending_score_function = None
        if self.latest_georoute_audit is not None:
            self.latest_georoute_audit["score_function_reward"] = float(detector_cost.detach().item())
            self.latest_georoute_audit["score_function_baseline"] = float(
                baseline_for_policy.item()
            )
            detector_binding = {
                "detector_loss_keys": sorted(name for name, _value in tensor_losses),
                "detector_cost_finite": bool(torch.isfinite(detector_cost).item()),
                "policy_objective_sign": "positive_(detector_loss-baseline)*log_probability_for_risk_minimization",
            }
            if self.amp_diagnostic_enabled:
                advantage = (
                    detector_cost.detach().to(torch.float32)
                    - baseline_for_policy.to(torch.float32)
                )
                detached_log_probability = ordered_log_prob.detach().to(
                    torch.float32
                )
                if self.score_function_temporal_reduction == "sum":
                    joint_log_probability = detached_log_probability.sum(dim=1)
                else:
                    joint_log_probability = detached_log_probability.mean(
                        dim=1
                    )
                detector_binding.update(
                    {
                        "temporal_reduction": (
                            self.score_function_temporal_reduction
                        ),
                        "detector_losses": {
                            name: self._detached_tensor_statistics(value)
                            for name, value in tensor_losses
                        },
                        "detector_cost": self._detached_tensor_statistics(
                            detector_cost
                        ),
                        "baseline": self._detached_tensor_statistics(
                            baseline_for_policy
                        ),
                        "advantage": self._detached_tensor_statistics(
                            advantage
                        ),
                        "ordered_log_prob": (
                            self._detached_tensor_statistics(ordered_log_prob)
                        ),
                        "joint_log_probability": (
                            self._detached_tensor_statistics(
                                joint_log_probability
                            )
                        ),
                        "policy_loss": self._detached_tensor_statistics(
                            policy_loss
                        ),
                    }
                )
            self.latest_georoute_audit[
                "score_function_detector_binding"
            ] = detector_binding
        return {"georoute_score_function_loss": policy_loss}

    def peek_gradient_decomposition_payload(self) -> Mapping[str, Any]:
        if not self.gradient_decomposition_enabled:
            raise RuntimeError("gradient decomposition is not enabled")
        if self._gradient_decomposition_payload is None:
            raise RuntimeError("no gradient decomposition payload is pending")
        return self._gradient_decomposition_payload

    def clear_gradient_decomposition_payload(self) -> None:
        if not self.gradient_decomposition_enabled:
            raise RuntimeError("gradient decomposition is not enabled")
        if self._gradient_decomposition_payload is None:
            raise RuntimeError("gradient decomposition payload was already cleared")
        self._gradient_decomposition_payload = None
