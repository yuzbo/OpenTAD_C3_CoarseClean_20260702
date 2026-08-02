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
    POLICY_ESTIMATORS,
    ROUTE_MODES,
    SCORE_FUNCTION_TEMPORAL_REDUCTIONS,
    STRUCTURED_ROUTE_MODES,
    decode_continuous_geometry,
    interpolate_temporal_knots,
    native_cell_extent_floor,
    native_patch_centers,
    roi_logits_from_geometry,
    score_function_policy_loss,
    select_exact_k,
    select_fixed_quota_structured_exact_k,
)
from .native_crop_wrapper import deterministic_linear_2x


GEOROUTE_BACKBONE_SCHEMA = "georoute_native_packed_backbone_v6"


class GeoRouteScout(nn.Module):
    """Low-cost global observer that predicts geometry and residual saliency."""

    def __init__(self, channels: int = 48) -> None:
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

    def forward(self, scout: torch.Tensor, *, source_grid_hw: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        if scout.ndim != 5 or scout.shape[1] != 3:
            raise ValueError("scout must be [B,3,T,H,W]")
        features = self.stem(scout)
        geometry_logits = self.geometry_head(features.mean(dim=(-1, -2))).transpose(1, 2)
        residual = self.residual_head(features).squeeze(1)
        batch, tubelets, scout_h, scout_w = map(int, residual.shape)
        target_h, target_w = map(int, source_grid_hw)
        residual = F.interpolate(
            residual.reshape(batch * tubelets, 1, scout_h, scout_w),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch, tubelets, target_h * target_w)
        return geometry_logits, residual


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


def _normalize_uint8_video(value: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if value.dtype != torch.uint8:
        raise TypeError("GeoRoute source/scout inputs must remain uint8 before native patch gather")
    return (value.to(torch.float32) - mean) / std


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
        self.policy_estimator = str(getattr(custom_cfg, "georoute_policy_estimator", "straight_through"))
        self.policy_temperature = float(getattr(custom_cfg, "georoute_policy_temperature", 0.5))
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
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unsupported GeoRoute route mode {self.route_mode!r}")
        if self.policy_estimator not in POLICY_ESTIMATORS:
            raise ValueError(f"unsupported GeoRoute estimator {self.policy_estimator!r}")
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
        super().__init__(cfg)
        if int(self.model.backbone.patch_size) != self.patch_size:
            raise ValueError("GeoRoute patch size must match the loaded VideoMAE")
        if bool(getattr(self.model.backbone, "with_cp", False)):
            raise ValueError("GeoRoute requires VideoMAE with_cp=False for one-forward accounting")
        self._freeze_shared_backbone_except_adapters()
        self.scout = GeoRouteScout(channels=48)
        self._configure_scout_trainability()
        self.sparse_adapter = GeoRouteSparseTemporalAdapter(channels=int(self.model.backbone.embed_dims))
        self.register_buffer("source_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1))
        self.register_buffer("source_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1))
        self.register_buffer("score_function_baseline", torch.zeros(()))
        self.register_buffer("score_function_baseline_initialized", torch.zeros((), dtype=torch.bool))
        self._successful_update_index: int | None = None
        self._pending_regularization: dict[str, torch.Tensor] | None = None
        self._pending_score_function: dict[str, torch.Tensor] | None = None
        self._gradient_decomposition_payload: dict[str, Any] | None = None
        self.latest_georoute_audit: dict[str, Any] | None = None

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

    def forward(self, frames, masks=None):
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
        self._pending_regularization = None
        if self.latest_georoute_audit is not None:
            self.latest_georoute_audit["geometry_regularization"] = float(regularization.detach().item())
        return {"georoute_geometry_regularization_loss": regularization}

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
