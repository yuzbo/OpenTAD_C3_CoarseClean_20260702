from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_wrapper import BackboneWrapper
from .native_crop_wrapper import (
    deterministic_linear_2x,
    flatten_chunk_tubelets,
)


D2S_VIDEOMAE_BACKBONE_SCHEMA = "d2s_temporal_zoom_shared_videomae_v1"


def compute_chunk_saliency(
    global_features: torch.Tensor,
    *,
    chunk_num: int = 48,
    alpha: float = 0.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compute per-chunk saliency from global scout features.
    global_features: [B, C, T_intermediate] where T_intermediate = chunk_num * 8 (e.g. 384)
    Uses hybrid mean+max pooling over chunk tubelets to prevent boundary impulse dilution.
    Returns: [B, chunk_num] saliency scores in [0, 1].
    """
    B, C, T = global_features.shape
    tubelets_per_chunk = T // chunk_num
    feats = global_features.permute(0, 2, 1)  # [B, T, C]

    # 1. Temporal transition gradient: Delta(t) = 1 - cosine_sim(f_t, f_{t-1})
    norm_feats = F.normalize(feats, p=2, dim=-1, eps=eps)  # [B, T, C]
    cos_sim = (norm_feats[:, 1:] * norm_feats[:, :-1]).sum(dim=-1)  # [B, T-1]
    delta = 1.0 - cos_sim  # [B, T-1]
    delta = torch.cat([delta[:, :1], delta], dim=1)  # [B, T]

    # 2. Actionness / energy norm: E(t) = ||f_t||_2
    energy = feats.norm(p=2, dim=-1)  # [B, T]
    min_energy = energy.min(dim=-1, keepdim=True)[0]
    max_energy = energy.max(dim=-1, keepdim=True)[0]
    norm_energy = (energy - min_energy) / (max_energy - min_energy + eps)  # [B, T]

    # 3. Combined tubelet saliency: S(t) = alpha * norm_energy + (1 - alpha) * delta
    saliency = alpha * norm_energy + (1.0 - alpha) * delta  # [B, T]

    # 4. Hybrid mean + max pooling to chunk level (protects single-frame boundary spikes)
    reshaped_saliency = saliency.view(B, chunk_num, tubelets_per_chunk)
    mean_saliency = reshaped_saliency.mean(dim=-1)
    max_saliency = reshaped_saliency.max(dim=-1)[0]
    chunk_saliency = 0.5 * mean_saliency + 0.5 * max_saliency  # [B, chunk_num]

    return chunk_saliency


class D2STemporalZoomBackboneWrapper(BackboneWrapper):
    """
    Dynamic Dual-Speed (D2S) Temporal Zoom Backbone Wrapper.
    Passes a full-video global scout view (96x96) through VideoMAE to detect
    temporal transitions and salient action boundaries, and passes the native-resolution
    local crop (128x128) through the shared VideoMAE backbone with 384-tubelet Adapter compatibility.
    Saliency-guided adaptive fusion weights local high-res features on active/boundary segments.
    0 extra trainable parameters, 100% shared VideoMAE backbone.
    """

    def __init__(self, cfg):
        custom_cfg = getattr(cfg, "custom", None)
        if custom_cfg is None:
            raise ValueError("D2STemporalZoomBackboneWrapper requires custom config block")

        self.global_key = str(getattr(custom_cfg, "global_key", "global"))
        self.local_key = str(getattr(custom_cfg, "local_key", "local"))
        self.expected_global_size = int(getattr(custom_cfg, "global_size", 96))
        self.expected_local_size = int(getattr(custom_cfg, "local_size", 128))
        self.total_chunks = int(getattr(custom_cfg, "total_chunks", 48))
        self.burst_chunks = int(getattr(custom_cfg, "burst_chunks", 16))
        self.saliency_alpha = float(getattr(custom_cfg, "saliency_alpha", 0.5))
        self.expected_intermediate_length = int(
            getattr(custom_cfg, "intermediate_length", 384)
        )
        self.output_length = int(getattr(custom_cfg, "output_length", 768))

        if self.burst_chunks <= 0 or self.burst_chunks > self.total_chunks:
            raise ValueError(
                f"burst_chunks must be in [1, {self.total_chunks}], got {self.burst_chunks}"
            )
        if self.output_length != 2 * self.expected_intermediate_length:
            raise ValueError("D2S wrapper requires exact 2x temporal expansion")

        super().__init__(cfg)
        self.latest_d2s_audit: dict[str, Any] | None = None

    def _validate_view(self, tensor: torch.Tensor, *, name: str, size: int) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"D2S {name} view must collate to torch.Tensor")
        if tensor.ndim != 6:
            raise ValueError(
                f"D2S {name} view must be [B,N,C,T,H,W], got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != 1 or tensor.shape[2] != 3:
            raise ValueError(
                f"D2S {name} view requires N=1 and RGB channels, got {tuple(tensor.shape)}"
            )
        if tensor.shape[-2:] != (size, size):
            raise ValueError(
                f"D2S {name} view expected {size}x{size}, got {tuple(tensor.shape[-2:])}"
            )
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"D2S {name} view must enter the wrapper as uint8; got {tensor.dtype}"
            )

    def _run_shared_backbone(self, frames: torch.Tensor) -> torch.Tensor:
        if self.freeze_backbone:
            with torch.no_grad():
                if self.use_temporal_checkpointing:
                    return self.temporal_checkpointing(
                        frames,
                        self.temporal_checkpointing_chunk_num,
                        self.temporal_checkpointing_chunk_dim,
                    )
                return self.model.backbone(frames)
        if self.use_temporal_checkpointing:
            return self.temporal_checkpointing(
                frames,
                self.temporal_checkpointing_chunk_num,
                self.temporal_checkpointing_chunk_dim,
            )
        return self.model.backbone(frames)

    def _encode_view(self, tensor: torch.Tensor, *, name: str) -> tuple[torch.Tensor, dict]:
        source_batch = int(tensor.shape[0])
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(tensor),
            data_samples=None,
            training=False,
        )
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]
        if frames.ndim != 6:
            raise RuntimeError(
                f"D2S pre-processing produced invalid {name} shape {tuple(frames.shape)}"
            )
        flattened_batches, num_segs = frames.shape[:2]
        if flattened_batches != source_batch * self.total_chunks:
            raise RuntimeError(
                f"D2S {name} expected {self.total_chunks} chunks per sample, "
                f"got flattened_batches={flattened_batches} batch={source_batch}"
            )
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())
        if isinstance(features, (tuple, list)):
            raise TypeError("D2S wrapper requires one VideoMAE feature tensor")
        if features.ndim != 5:
            raise RuntimeError(
                f"D2S shared backbone must return [B,C,T,H,W], got {tuple(features.shape)}"
            )
        features = features.unflatten(0, sizes=(flattened_batches, num_segs))
        features = features.mean(dim=(1, 4, 5))
        features = flatten_chunk_tubelets(
            features,
            source_batch=source_batch,
            chunk_num=self.total_chunks,
        )
        if features.shape[-1] != self.expected_intermediate_length:
            raise RuntimeError(
                f"D2S {name} produced temporal length {features.shape[-1]}, "
                f"expected {self.expected_intermediate_length}"
            )
        return features, {
            "input_shape": list(tensor.shape),
            "normalized_shape": list(frames.shape),
            "feature_map_shape": list(features.shape),
            "runtime_token_grid_hw": [
                int(tensor.shape[-2] // 16),
                int(tensor.shape[-1] // 16),
            ],
        }

    def forward(self, frames, masks=None):
        if not isinstance(frames, Mapping):
            raise TypeError("D2STemporalZoomBackboneWrapper requires dict with global/local uint8 views")
        if set(frames) != {self.global_key, self.local_key}:
            raise ValueError(
                f"D2S expected keys {sorted((self.global_key, self.local_key))}, got {sorted(frames)}"
            )

        global_view = frames[self.global_key]
        local_view = frames[self.local_key]
        self._validate_view(global_view, name="global", size=self.expected_global_size)
        self._validate_view(local_view, name="local", size=self.expected_local_size)

        self.set_norm_layer()

        # Step 1: Global scout stream forward (low-res full temporal coverage)
        global_features, global_audit = self._encode_view(global_view, name="global")
        B, C, T_inter = global_features.shape

        # Step 2: Native local crop stream forward (high-res full 384-tubelet Adapter compatibility)
        local_features, local_audit = self._encode_view(local_view, name="local")

        # Step 3: Compute temporal saliency & select burst chunks
        chunk_saliency = compute_chunk_saliency(
            global_features,
            chunk_num=self.total_chunks,
            alpha=self.saliency_alpha,
        )  # [B, total_chunks]

        # Top-K burst selection per sample
        _, topk_indices = torch.topk(
            chunk_saliency, k=self.burst_chunks, dim=-1, largest=True, sorted=True
        )
        selected_chunk_indices = torch.sort(topk_indices, dim=-1)[0]  # [B, burst_chunks]

        # Step 4: Saliency-guided adaptive fusion (vectorized on-device)
        tubelets_per_chunk = self.expected_intermediate_length // self.total_chunks
        chunk_mask = torch.zeros(
            B, self.total_chunks, device=global_features.device, dtype=torch.bool
        )
        chunk_mask.scatter_(1, selected_chunk_indices, True)  # [B, total_chunks]
        burst_mask = chunk_mask.repeat_interleave(tubelets_per_chunk, dim=1).unsqueeze(1)  # [B, 1, T_inter]

        # On burst chunks: 0.5 * (global + local)
        # On non-burst chunks: global
        fused = torch.where(
            burst_mask,
            0.5 * (global_features + local_features),
            global_features,
        )

        # Step 5: 2x temporal interpolation to length 768
        fused = deterministic_linear_2x(fused)

        if masks is not None:
            if masks.shape != (fused.shape[0], fused.shape[-1]):
                raise ValueError("D2S mask does not match the detector time axis")
            fused = fused * masks.unsqueeze(1).detach().to(fused.dtype)

        self.latest_d2s_audit = {
            "schema_version": D2S_VIDEOMAE_BACKBONE_SCHEMA,
            "shared_backbone_instances": 1,
            "total_chunks": self.total_chunks,
            "burst_chunks": self.burst_chunks,
            "selected_chunk_ratio": self.burst_chunks / self.total_chunks,
            "intermediate_shape": list(global_features.shape),
            "output_shape": list(fused.shape),
            "global": global_audit,
            "local": local_audit,
            "selected_chunk_indices": selected_chunk_indices.detach().cpu().tolist(),
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
        }
        return fused.to(torch.float32)


__all__ = [
    "D2S_VIDEOMAE_BACKBONE_SCHEMA",
    "D2STemporalZoomBackboneWrapper",
    "compute_chunk_saliency",
]
