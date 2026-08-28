from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
import torch.nn as nn

from ..builder import TOKEN_COMPRESSORS
from .true_time_residual import TrueTimeFeatureResidual
from .tubelet_coreset import merge_discarded_scout_context


@TOKEN_COMPRESSORS.register_module()
class PhysicalTimeCoresetReconstructor(nn.Module):
    """Reconstruct sparse native-tubelet features on the physical tubelet grid."""

    def __init__(
        self,
        target_len: int = 384,
        feature_dim: int = 384,
        scout_hidden_dim: int = 96,
        context_temperature: float = 0.7,
        time_hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.target_len = int(target_len)
        self.feature_dim = int(feature_dim)
        self.scout_hidden_dim = int(scout_hidden_dim)
        self.context_temperature = float(context_temperature)
        if min(self.target_len, self.feature_dim, self.scout_hidden_dim) <= 0:
            raise ValueError("physical reconstruction dimensions must be positive")
        self.context_projector = nn.Sequential(
            nn.Linear(self.scout_hidden_dim, self.scout_hidden_dim),
            nn.GELU(),
            nn.Linear(self.scout_hidden_dim, self.feature_dim),
        )
        nn.init.zeros_(self.context_projector[-1].weight)
        nn.init.zeros_(self.context_projector[-1].bias)
        self.time_residual = TrueTimeFeatureResidual(
            feature_dim=self.feature_dim,
            hidden_dim=int(time_hidden_dim),
            descriptor_mode="actual",
        )
        self.last_summary: dict[str, Any] = {}

    @staticmethod
    def _metadata_rows(
        metas: Sequence[Mapping[str, Any]] | None,
        *,
        batch_size: int,
    ) -> list[dict[str, Any]]:
        if metas is None:
            raise ValueError("physical reconstruction requires native tubelet metadata")
        if len(metas) != batch_size:
            raise ValueError("physical reconstruction metadata must match batch size")
        rows = [dict(meta) for meta in metas]
        if not all(isinstance(meta, Mapping) for meta in metas):
            raise ValueError("physical reconstruction metadata rows must be mappings")
        return rows

    def _recycle_scout_context(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        output = features.clone()
        for batch_idx, meta in enumerate(metas):
            active = int(masks[batch_idx].long().sum().item())
            selected = torch.as_tensor(
                meta.get("duca_native_tubelet_indices", []),
                device=features.device,
                dtype=torch.long,
            )
            if selected.numel() != active:
                raise ValueError("selected tubelet metadata must match active heavy tokens")
            valid_count = int(meta.get("duca_native_tubelet_valid_len", 0))
            hidden = meta.get("duca_native_tubelet_scout_hidden")
            scores = meta.get("duca_native_tubelet_scores")
            if not torch.is_tensor(hidden) or not torch.is_tensor(scores):
                raise ValueError("physical reconstruction requires detached scout hidden and scores")
            hidden = hidden.to(device=features.device, dtype=features.dtype).detach()
            scores = scores.to(device=features.device, dtype=features.dtype).detach()
            if hidden.shape != (valid_count, self.scout_hidden_dim):
                raise ValueError("scout hidden metadata has the wrong physical-grid shape")
            if scores.shape != (valid_count,):
                raise ValueError("scout score metadata has the wrong physical-grid shape")
            difference = merge_discarded_scout_context(
                hidden,
                scores,
                selected,
                valid_count=valid_count,
                temperature=self.context_temperature,
            )
            residual = self.context_projector(difference).transpose(0, 1)
            output[batch_idx, :, :active] = output[batch_idx, :, :active] + residual
        return output

    def _apply_true_time_residual(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> torch.Tensor:
        time_metas = []
        for batch_idx, meta in enumerate(metas):
            active = int(masks[batch_idx].long().sum().item())
            selected = [int(value) for value in meta.get("duca_native_tubelet_indices", [])]
            if len(selected) != active:
                raise ValueError("true-time residual selected positions do not match active tokens")
            time_metas.append(
                {
                    "selected_axis_to_true_time_dense_index": selected,
                    "truetime_dense_valid_len": int(meta.get("duca_native_tubelet_valid_len", 0)),
                }
            )
        return self.time_residual(features, masks, time_metas)

    def _interpolate_to_original_tubelet_grid(
        self,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, channels, _ = features.shape
        reconstructed = features.new_zeros((batch, channels, self.target_len))
        reconstructed_mask = torch.zeros(
            (batch, self.target_len), device=features.device, dtype=torch.bool
        )
        for batch_idx, meta in enumerate(metas):
            active = int(masks[batch_idx].long().sum().item())
            valid_count = int(meta.get("duca_native_tubelet_valid_len", 0))
            if not 0 < valid_count <= self.target_len:
                raise ValueError("physical tubelet valid length is outside the target grid")
            positions = torch.as_tensor(
                meta.get("duca_native_tubelet_indices", []),
                device=features.device,
                dtype=torch.long,
            )
            if positions.numel() != active or active <= 0:
                raise ValueError("physical reconstruction requires active selected tubelets")
            if bool((positions[1:] <= positions[:-1]).any().item()):
                raise ValueError("selected tubelet positions must be strictly increasing")
            if int(positions[0].item()) != 0 or int(positions[-1].item()) != valid_count - 1:
                raise ValueError("physical reconstruction requires the first and last valid anchors")
            query = torch.arange(valid_count, device=features.device, dtype=torch.long)
            right = torch.searchsorted(positions, query, right=False).clamp(max=active - 1)
            left = (right - 1).clamp(min=0)
            exact = positions[right] == query
            left = torch.where(exact, right, left)
            left_pos = positions[left].to(dtype=features.dtype)
            right_pos = positions[right].to(dtype=features.dtype)
            denominator = (right_pos - left_pos).clamp_min(1.0)
            weight = (query.to(dtype=features.dtype) - left_pos) / denominator
            weight = torch.where(exact, torch.zeros_like(weight), weight)
            sparse = features[batch_idx, :, :active]
            values = sparse[:, left] * (1.0 - weight[None, :]) + sparse[:, right] * weight[None, :]
            reconstructed[batch_idx, :, :valid_count] = values
            reconstructed_mask[batch_idx, :valid_count] = True
        return reconstructed, reconstructed_mask

    @staticmethod
    def _map_gt_to_physical_tubelet_grid(gt_segments):
        if gt_segments is None:
            return None
        mapped = []
        for segments in gt_segments:
            if segments is None:
                mapped.append(None)
            elif torch.is_tensor(segments):
                mapped.append(segments * 0.5)
            else:
                mapped.append(torch.as_tensor(segments, dtype=torch.float32) * 0.5)
        return mapped

    @staticmethod
    def _replace_inference_time_metadata(
        metas: Sequence[Mapping[str, Any]],
        masks: torch.Tensor,
    ) -> list[dict[str, Any]]:
        output = []
        for batch_idx, source in enumerate(metas):
            meta = dict(source)
            valid_count = int(masks[batch_idx].long().sum().item())
            original_stride = float(meta.get("snippet_stride", 1.0))
            meta["duca_original_frame_snippet_stride"] = original_stride
            meta["snippet_stride"] = original_stride * 2.0
            meta["detector_prediction_inverse_map_required"] = False
            meta["detector_output_coordinate_space"] = "physical_tubelet_grid"
            meta["irregular_native_axis"] = True
            meta["irregular_dense_valid_len"] = valid_count
            meta["irregular_selected_valid_len"] = valid_count
            meta["truetime_dense_len"] = int(masks.shape[1])
            meta["truetime_dense_valid_len"] = valid_count
            meta["selected_axis_to_true_time_dense_index"] = list(range(valid_count))
            meta["duca_physical_tubelet_grid"] = True
            meta.pop("duca_native_tubelet_scout_hidden", None)
            meta.pop("duca_native_tubelet_scores", None)
            output.append(meta)
        return output

    def _forward(
        self,
        *,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas: Sequence[Mapping[str, Any]] | None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
        if features.ndim != 3 or int(features.shape[1]) != self.feature_dim:
            raise ValueError(
                f"physical reconstructor expects [B,{self.feature_dim},T] features"
            )
        masks = masks.to(device=features.device, dtype=torch.bool)
        if masks.shape != (features.shape[0], features.shape[2]):
            raise ValueError("physical reconstructor masks must align with sparse features")
        rows = self._metadata_rows(metas, batch_size=int(features.shape[0]))
        recycled = self._recycle_scout_context(features, masks, rows)
        timed = self._apply_true_time_residual(recycled, masks, rows)
        reconstructed, reconstructed_mask = self._interpolate_to_original_tubelet_grid(
            timed, masks, rows
        )
        output_metas = self._replace_inference_time_metadata(rows, reconstructed_mask)
        self.last_summary = {
            "source_tokens": [int(value) for value in masks.long().sum(dim=1).detach().cpu().tolist()],
            "target_tokens": [
                int(value) for value in reconstructed_mask.long().sum(dim=1).detach().cpu().tolist()
            ],
            "target_len": self.target_len,
            "context_recycling": True,
            "physical_time_residual": True,
            "physical_grid_interpolation": True,
        }
        return reconstructed, reconstructed_mask, output_metas

    def forward_train(
        self,
        *,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas,
        gt_segments,
        gt_labels,
    ) -> dict[str, Any]:
        output, output_mask, output_metas = self._forward(
            features=features, masks=masks, metas=metas
        )
        return {
            "features": output,
            "masks": output_mask,
            "metas": output_metas,
            "gt_segments": self._map_gt_to_physical_tubelet_grid(gt_segments),
            "gt_labels": gt_labels,
            "losses": {},
        }

    def forward_test(
        self,
        *,
        features: torch.Tensor,
        masks: torch.Tensor,
        metas,
    ) -> dict[str, Any]:
        output, output_mask, output_metas = self._forward(
            features=features, masks=masks, metas=metas
        )
        return {"features": output, "masks": output_mask, "metas": output_metas}


__all__ = ["PhysicalTimeCoresetReconstructor"]
