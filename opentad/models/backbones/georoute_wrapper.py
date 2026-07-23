"""Native-token GeoRoute backbone for a single-heavy-forward AdaTAD path."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm

from .backbone_wrapper import BackboneWrapper
from .georoute_routing import (
    GEOROUTE_ROUTING_SCHEMA,
    POLICY_ESTIMATORS,
    ROUTE_MODES,
    decode_continuous_geometry,
    interpolate_temporal_knots,
    native_patch_centers,
    roi_logits_from_geometry,
    score_function_policy_loss,
    select_exact_k,
)
from .native_crop_wrapper import deterministic_linear_2x


GEOROUTE_BACKBONE_SCHEMA = "georoute_native_packed_backbone_v1"


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
            nn.Conv3d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False),
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
        if use_absolute_coordinates:
            # [absolute x/y, coordinate relative to the current ROI].  The
            # latter exposes the structured region prior to the aggregator
            # without changing the native token support or pixel scale.
            relative = (
                selected_coordinates - geometry[:, :, None, :2]
            ) / geometry[:, :, None, 2:].clamp_min(1e-6)
            coordinate_features = torch.cat((selected_coordinates, relative), dim=-1)
            selected_features = selected_features + self.coordinate_projection(coordinate_features)
        weights = torch.softmax(selected_scores, dim=-1).unsqueeze(-1)
        pooled = (weights * selected_features).sum(dim=2)
        pooled = self.norm(pooled + self.geometry_projection(geometry))
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
) -> tuple[torch.Tensor, tuple[int, int], tuple[int, int]]:
    """Pad only the source boundary and view it as native VideoMAE tubelets."""

    if source.ndim != 5 or source.shape[1] != 3:
        raise ValueError("source must be [B,3,T,H,W]")
    batch, channels, frames, height, width = map(int, source.shape)
    if channels != 3 or frames % int(tubelet_size):
        raise ValueError("source must contain RGB frames divisible by VideoMAE tubelet size")
    pad_bottom = (-height) % int(patch_size)
    pad_right = (-width) % int(patch_size)
    if pad_bottom or pad_right:
        # CUDA does not implement ``replicate`` padding for uint8 tensors.
        # Flattening only the independent batch/time axes lets us append the
        # final row/column directly while preserving each source pixel's dtype
        # and value. This is boundary replication, never a resize or an
        # interpolation.
        frame_images = source.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            channels,
            height,
            width,
        )
        if pad_bottom:
            frame_images = torch.cat(
                (
                    frame_images,
                    frame_images[..., -1:, :].expand(-1, -1, pad_bottom, -1),
                ),
                dim=-2,
            )
        if pad_right:
            frame_images = torch.cat(
                (
                    frame_images,
                    frame_images[..., :, -1:].expand(-1, -1, -1, pad_right),
                ),
                dim=-1,
            )
        padded_images = frame_images
        source = (
            padded_images.reshape(
                batch,
                frames,
                channels,
                height + pad_bottom,
                width + pad_right,
            )
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    padded_height, padded_width = map(int, source.shape[-2:])
    grid_height = padded_height // int(patch_size)
    grid_width = padded_width // int(patch_size)
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
        .reshape(batch, tubelets, grid_height * grid_width, channels, tubelet_size, patch_size, patch_size)
        .contiguous()
    )
    return native, (grid_height, grid_width), (pad_bottom, pad_right)


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
        self.roi_temperature = float(getattr(custom_cfg, "georoute_roi_temperature", 0.25))
        self.geometry_stride_tubelets = int(
            getattr(custom_cfg, "georoute_geometry_stride_tubelets", 1)
        )
        self.absolute_position_enabled = bool(
            getattr(custom_cfg, "georoute_absolute_position_enabled", True)
        )
        self.absolute_coordinates_enabled = bool(
            getattr(custom_cfg, "georoute_absolute_coordinates_enabled", True)
        )
        # A causal control can expose exactly the same learned geometry to the
        # aggregation adapter while keeping a fixed token lattice.  It tests
        # whether a gain comes from spatial *selection*, rather than merely
        # adding an extra geometry-conditioned feature pathway.
        self.geometry_side_channel = bool(
            getattr(custom_cfg, "georoute_geometry_side_channel", False)
        )
        self.min_roi_extent = float(getattr(custom_cfg, "georoute_min_roi_extent", 0.20))
        self.max_roi_extent = float(getattr(custom_cfg, "georoute_max_roi_extent", 1.00))
        self.geometry_smoothness_weight = float(getattr(custom_cfg, "georoute_geometry_smoothness_weight", 0.0))
        self.area_prior_weight = float(getattr(custom_cfg, "georoute_area_prior_weight", 0.0))
        self.area_prior = float(getattr(custom_cfg, "georoute_area_prior", 0.30))
        self.score_function_weight = float(getattr(custom_cfg, "georoute_score_function_weight", 1.0))
        self.score_function_baseline_momentum = float(
            getattr(custom_cfg, "georoute_score_function_baseline_momentum", 0.95)
        )
        # This switch is intentionally P0-only.  It runs a dense numerical
        # reference before the real packed call and is forbidden in ordinary
        # development/paper cells so it can never be mistaken for model cost.
        self.p0_dense_reference_check = bool(
            getattr(custom_cfg, "georoute_p0_dense_reference_check", False)
        )
        self.output_length = int(getattr(custom_cfg, "georoute_output_length", self.window_size))
        self.max_batch_size = int(getattr(custom_cfg, "georoute_max_batch_size", 1))
        if self.route_mode not in ROUTE_MODES:
            raise ValueError(f"unsupported GeoRoute route mode {self.route_mode!r}")
        if self.policy_estimator not in POLICY_ESTIMATORS:
            raise ValueError(f"unsupported GeoRoute estimator {self.policy_estimator!r}")
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
        if self.geometry_stride_tubelets <= 0:
            raise ValueError("GeoRoute geometry stride must be positive")
        if self.p0_dense_reference_check and self.route_mode != "dense":
            raise ValueError("GeoRoute dense numerical reference is valid only for route_mode='dense'")
        if self.geometry_side_channel and self.route_mode not in {"uniform", "random"}:
            raise ValueError(
                "GeoRoute geometry-side-channel control is valid only for fixed uniform/random routes"
            )
        super().__init__(cfg)
        if int(self.model.backbone.patch_size) != self.patch_size:
            raise ValueError("GeoRoute patch size must match the loaded VideoMAE")
        if bool(getattr(self.model.backbone, "with_cp", False)):
            raise ValueError("GeoRoute requires VideoMAE with_cp=False for one-forward accounting")
        self._freeze_shared_backbone_except_adapters()
        self.scout = GeoRouteScout(channels=48)
        self.sparse_adapter = GeoRouteSparseTemporalAdapter(channels=int(self.model.backbone.embed_dims))
        self.register_buffer("source_mean", torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1, 1))
        self.register_buffer("source_std", torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1, 1))
        self.register_buffer("score_function_baseline", torch.zeros(()))
        self.register_buffer("score_function_baseline_initialized", torch.zeros((), dtype=torch.bool))
        self._successful_update_index: int | None = None
        self._pending_regularization: dict[str, torch.Tensor] | None = None
        self._pending_score_function: dict[str, torch.Tensor] | None = None
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
        if not isinstance(frames, Mapping) or set(frames) != {self.source_key, self.scout_key}:
            raise ValueError(
                "GeoRoute inputs are fail-closed; expected exactly "
                f"{sorted((self.source_key, self.scout_key))}"
            )
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
        return centres.view(1, 1, -1, 2).expand(indices.shape[0], indices.shape[1], -1, 2).gather(
            2, gather_index
        )

    def _regularization(self, geometry: torch.Tensor) -> torch.Tensor:
        smoothness = geometry.new_zeros(())
        if geometry.shape[1] > 1:
            smoothness = (geometry[:, 1:] - geometry[:, :-1]).square().mean()
        area = (geometry[..., 2] * geometry[..., 3]).mean()
        area_loss = (area - self.area_prior).square()
        return self.geometry_smoothness_weight * smoothness + self.area_prior_weight * area_loss

    def forward(self, frames, masks=None):
        source, scout = self._validate_inputs(frames)
        self.set_norm_layer()
        native, source_grid_hw, padding_bottom_right = extract_native_tubelets(
            source,
            patch_size=self.patch_size,
            tubelet_size=self.tubelet_size,
        )
        if self.route_mode in {"dense", "uniform", "random"} and not self.geometry_side_channel:
            batch_size = int(source.shape[0])
            tubelets = self.window_size // self.tubelet_size
            item_count = int(source_grid_hw[0]) * int(source_grid_hw[1])
            geometry_logits = torch.zeros((batch_size, tubelets, 4), device=source.device)
            residual_logits = torch.zeros((batch_size, tubelets, item_count), device=source.device)
        else:
            normalized_scout = _normalize_uint8_video(scout, self.source_mean, self.source_std)
            geometry_logits, residual_logits = self.scout(
                normalized_scout,
                source_grid_hw=source_grid_hw,
            )
        geometry = decode_continuous_geometry(
            interpolate_temporal_knots(
                geometry_logits,
                stride=self.geometry_stride_tubelets,
            ),
            min_extent=self.min_roi_extent,
            max_extent=self.max_roi_extent,
        )
        # Fixed lattice and random controls must not obtain an accidental
        # geometry cue through the aggregation adapter.
        if self.route_mode in {"dense", "uniform", "random"} and not self.geometry_side_channel:
            geometry = torch.ones_like(geometry)
            geometry[..., :2] = 0.5
        roi_logits = roi_logits_from_geometry(
            geometry,
            grid_height=source_grid_hw[0],
            grid_width=source_grid_hw[1],
            temperature=self.roi_temperature,
        )
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
            random_seed=self.random_seed,
        )
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
            if not bool(torch.allclose(selected_features.detach(), reference, rtol=tolerance, atol=tolerance)):
                raise RuntimeError(
                    "GeoRoute native packed all-token output disagrees with its dense P0 reference: "
                    f"max_abs_error={max_abs_error:.8g}"
                )
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
        selected_scores = route["selected_aggregation_logits"]
        selected_coordinates = self._selected_native_coordinates(
            route["indices"],
            source_grid_hw=source_grid_hw,
        ).to(dtype=selected_features.dtype)
        intermediate = self.sparse_adapter(
            selected_features,
            selected_scores,
            geometry,
            selected_coordinates,
            use_absolute_coordinates=self.absolute_coordinates_enabled,
        )
        output = deterministic_linear_2x(intermediate)
        if output.shape != (source.shape[0], int(self.model.backbone.embed_dims), self.output_length):
            raise RuntimeError("GeoRoute violated the AdaTAD [B,384,768] backbone contract")
        if masks is not None:
            if masks.shape != (output.shape[0], output.shape[-1]):
                raise ValueError("GeoRoute masks must match the detector time axis")
            output = output * masks.to(output.device).unsqueeze(1).detach().to(output.dtype)

        if self.training:
            if self._pending_regularization is not None or self._pending_score_function is not None:
                raise RuntimeError("GeoRoute pending training losses were not consumed exactly once")
            self._pending_regularization = {"geometry": self._regularization(geometry)}
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
            duplicate_count = int(
                (sorted_indices[..., 1:] == sorted_indices[..., :-1]).sum().item()
            )
            unique_counts = 1 + (sorted_indices[..., 1:] != sorted_indices[..., :-1]).sum(dim=-1)
        else:
            duplicate_count = 0
            unique_counts = torch.ones_like(sorted_indices[..., 0])
        self.latest_georoute_audit = {
            "schema_version": GEOROUTE_BACKBONE_SCHEMA,
            "routing_schema": GEOROUTE_ROUTING_SCHEMA,
            "route_mode": self.route_mode,
            "policy_estimator": self.policy_estimator,
            "estimator_claim": "biased_straight_through" if self.policy_estimator == "straight_through" else "score_function_candidate" if self.policy_estimator == "score_function" else "no_policy_gradient",
            "shared_backbone_instances": 1,
            "heavy_backbone_forward_count": packed_invocation_delta,
            "native_packed_invocation_counter_before": packed_invocations_before,
            "native_packed_invocation_counter_after": packed_invocations_after,
            "source_input_shape": list(source.shape),
            "source_grid_hw": list(source_grid_hw),
            "source_padding_bottom_right": list(padding_bottom_right),
            "native_tubelet_shape": list(native.shape),
            "selected_native_tubelet_shape": list(selected_native.shape),
            "intermediate_shape": list(intermediate.shape),
            "output_shape": list(output.shape),
            "geometry_shape": list(geometry.shape),
            "geometry_stride_tubelets": self.geometry_stride_tubelets,
            "absolute_position_enabled": self.absolute_position_enabled,
            "absolute_coordinates_enabled": self.absolute_coordinates_enabled,
            "geometry_side_channel": self.geometry_side_channel,
            "geometry_min": float(geometry.detach().min().item()),
            "geometry_max": float(geometry.detach().max().item()),
            "target_k": int(route["target_k"]),
            "item_count": int(route["item_count"]),
            "selected_unique_count_min": int(unique_counts.min().item()),
            "selected_unique_count_max": int(unique_counts.max().item()),
            "selected_duplicate_count": duplicate_count,
            "role_counts": dict(route["role_counts"]),
            "packed": packed,
            "dense_native_reference": dense_reference_audit,
            "uses_grid_sample": False,
            "uses_resized_local_crop": False,
            "uses_gt_for_route": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
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
        if not self.training or self._pending_score_function is None:
            return {}
        detector_cost = sum(value for value in detector_losses.values() if torch.is_tensor(value))
        if detector_cost.ndim != 0:
            raise ValueError("GeoRoute detector policy hook requires scalar detector losses")
        if bool(self.score_function_baseline_initialized.item()):
            baseline = self.score_function_baseline
        else:
            baseline = torch.zeros_like(detector_cost)
        policy_loss = score_function_policy_loss(
            detector_cost=detector_cost,
            ordered_log_prob=self._pending_score_function["ordered_log_prob"],
            baseline=baseline,
            weight=self.score_function_weight,
        )
        with torch.no_grad():
            reward = detector_cost.detach()
            if bool(self.score_function_baseline_initialized.item()):
                self.score_function_baseline.mul_(self.score_function_baseline_momentum).add_(
                    reward * (1.0 - self.score_function_baseline_momentum)
                )
            else:
                self.score_function_baseline.copy_(reward)
                self.score_function_baseline_initialized.fill_(True)
        self._pending_score_function = None
        if self.latest_georoute_audit is not None:
            self.latest_georoute_audit["score_function_reward"] = float(detector_cost.detach().item())
            self.latest_georoute_audit["score_function_baseline"] = float(baseline.detach().item())
            self.latest_georoute_audit["score_function_detector_binding"] = {
                "detector_loss_keys": sorted(
                    str(name) for name, value in detector_losses.items() if torch.is_tensor(value)
                ),
                "detector_cost_finite": bool(torch.isfinite(detector_cost).item()),
                "policy_objective_sign": "positive_(detector_loss-baseline)*log_probability_for_risk_minimization",
            }
        return {"georoute_score_function_loss": policy_loss}
