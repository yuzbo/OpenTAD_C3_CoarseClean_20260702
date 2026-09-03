from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn

from .backbone_wrapper import BackboneWrapper


NATIVE_CROP_BACKBONE_SCHEMA = "native_crop_shared_videomae_v1"


def deterministic_linear_2x(value: torch.Tensor) -> torch.Tensor:
    """Match 2x linear interpolation with align_corners=False."""

    if value.ndim != 3 or value.shape[-1] < 1:
        raise ValueError("deterministic_linear_2x expects a non-empty [B,C,T] tensor")
    if value.shape[-1] == 1:
        return torch.cat((value, value), dim=-1)
    left = value[..., :-1]
    right = value[..., 1:]
    between = torch.stack(
        (
            0.75 * left + 0.25 * right,
            0.25 * left + 0.75 * right,
        ),
        dim=-1,
    ).flatten(start_dim=-2)
    return torch.cat((value[..., :1], between, value[..., -1:]), dim=-1)


def flatten_chunk_tubelets(
    features: torch.Tensor,
    *,
    source_batch: int,
    chunk_num: int,
) -> torch.Tensor:
    """Preserve chronological chunk-major, tubelet-minor ordering."""

    if features.ndim != 3:
        raise ValueError("chunk features must be [B*chunks,C,tubelets]")
    if int(features.shape[0]) != int(source_batch) * int(chunk_num):
        raise ValueError("chunk feature count does not match source batch geometry")
    channels = int(features.shape[1])
    tubelets = int(features.shape[2])
    return (
        features.reshape(source_batch, chunk_num, channels, tubelets)
        .permute(0, 2, 1, 3)
        .flatten(start_dim=2)
    )


class NativeCropFeatureFusion(nn.Module):
    """Fuse two aligned temporal feature streams without changing channels."""

    SUPPORTED_MODES = {"fixed_mean", "global_only", "local_only"}

    def __init__(self, mode: str = "fixed_mean"):
        super().__init__()
        if mode not in self.SUPPORTED_MODES:
            raise ValueError(
                f"unsupported Native-Crop fusion mode {mode!r}; "
                f"expected one of {sorted(self.SUPPORTED_MODES)}"
            )
        self.mode = mode

    def forward(self, global_features: torch.Tensor, local_features: torch.Tensor) -> torch.Tensor:
        if global_features.shape != local_features.shape:
            raise ValueError(
                "Native-Crop branches must align before fusion; "
                f"global={tuple(global_features.shape)} local={tuple(local_features.shape)}"
            )
        if self.mode == "global_only":
            return global_features
        if self.mode == "local_only":
            return local_features
        return 0.5 * (global_features + local_features)


class NativeCropBackboneWrapper(BackboneWrapper):
    """Run two source-derived views through one shared VideoMAE backbone."""

    def __init__(self, cfg):
        custom_cfg = cfg.custom
        self.global_key = str(getattr(custom_cfg, "native_crop_global_key", "global"))
        self.local_key = str(getattr(custom_cfg, "native_crop_local_key", "local"))
        self.expected_global_size = int(
            getattr(custom_cfg, "native_crop_global_size", 96)
        )
        self.expected_local_size = int(
            getattr(custom_cfg, "native_crop_local_size", 128)
        )
        self.chunk_num = int(getattr(custom_cfg, "native_crop_chunk_num", 48))
        self.expected_intermediate_length = int(
            getattr(custom_cfg, "native_crop_intermediate_length", 384)
        )
        self.output_length = int(
            getattr(custom_cfg, "native_crop_output_length", 768)
        )
        self.fusion_mode = str(
            getattr(custom_cfg, "native_crop_fusion_mode", "fixed_mean")
        )
        if self.chunk_num <= 0:
            raise ValueError("native_crop_chunk_num must be positive")
        if self.output_length != 2 * self.expected_intermediate_length:
            raise ValueError(
                "the audited Native-Crop wrapper requires exact 2x temporal output"
            )
        super().__init__(cfg)
        self.fusion = NativeCropFeatureFusion(self.fusion_mode)
        self.latest_native_crop_audit = None

    def _validate_view(self, tensor: torch.Tensor, *, name: str, size: int) -> None:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Native-Crop {name} view must collate to torch.Tensor")
        if tensor.ndim != 6:
            raise ValueError(
                f"Native-Crop {name} view must be [B,N,C,T,H,W], got {tuple(tensor.shape)}"
            )
        if tensor.shape[1] != 1 or tensor.shape[2] != 3:
            raise ValueError(
                f"Native-Crop {name} view requires N=1 and RGB channels, got "
                f"{tuple(tensor.shape)}"
            )
        if tensor.shape[-2:] != (size, size):
            raise ValueError(
                f"Native-Crop {name} view expected {size}x{size}, got "
                f"{tuple(tensor.shape[-2:])}"
            )
        if tensor.dtype != torch.uint8:
            raise TypeError(
                f"Native-Crop {name} view must enter the backbone wrapper as uint8; "
                f"got {tensor.dtype}"
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
                f"Native-Crop pre-processing produced invalid {name} shape "
                f"{tuple(frames.shape)}"
            )
        flattened_batches, num_segs = frames.shape[:2]
        if flattened_batches != source_batch * self.chunk_num:
            raise RuntimeError(
                f"Native-Crop {name} expected {self.chunk_num} chunks per sample, "
                f"got flattened_batches={flattened_batches} batch={source_batch}"
            )
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())
        if isinstance(features, (tuple, list)):
            raise TypeError("Native-Crop v1 requires one VideoMAE feature tensor")
        if features.ndim != 5:
            raise RuntimeError(
                f"Native-Crop shared backbone must return [B,C,T,H,W], got "
                f"{tuple(features.shape)}"
            )
        features = features.unflatten(
            0, sizes=(flattened_batches, num_segs)
        )
        features = features.mean(dim=(1, 4, 5))
        features = flatten_chunk_tubelets(
            features,
            source_batch=source_batch,
            chunk_num=self.chunk_num,
        )
        if features.shape[-1] != self.expected_intermediate_length:
            raise RuntimeError(
                f"Native-Crop {name} produced temporal length {features.shape[-1]}, "
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
            raise TypeError(
                "NativeCropBackboneWrapper requires a mapping with global/local uint8 views"
            )
        if set(frames) != {self.global_key, self.local_key}:
            raise ValueError(
                "Native-Crop input keys are fail-closed; expected exactly "
                f"{sorted((self.global_key, self.local_key))}, got {sorted(frames)}"
            )
        global_view = frames[self.global_key]
        local_view = frames[self.local_key]
        self._validate_view(
            global_view, name="global", size=self.expected_global_size
        )
        self._validate_view(local_view, name="local", size=self.expected_local_size)
        if global_view.shape[:4] != local_view.shape[:4]:
            raise ValueError(
                "Native-Crop global/local views must preserve the same dense time axis"
            )

        self.set_norm_layer()
        global_features, global_audit = self._encode_view(
            global_view, name="global"
        )
        local_features, local_audit = self._encode_view(local_view, name="local")
        fused = self.fusion(global_features, local_features)
        fused = deterministic_linear_2x(fused)
        if fused.shape != (
            global_view.shape[0],
            global_features.shape[1],
            self.output_length,
        ):
            raise RuntimeError(
                f"Native-Crop detector feature contract violated: {tuple(fused.shape)}"
            )
        if masks is not None:
            if masks.shape != (fused.shape[0], fused.shape[-1]):
                raise ValueError("Native-Crop mask does not match the detector time axis")
            fused = fused * masks.unsqueeze(1).detach().to(fused.dtype)
        self.latest_native_crop_audit = {
            "schema_version": NATIVE_CROP_BACKBONE_SCHEMA,
            "shared_backbone_instances": 1,
            "fusion_mode": self.fusion_mode,
            "intermediate_shape": list(global_features.shape),
            "output_shape": list(fused.shape),
            "global": global_audit,
            "local": local_audit,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_oracle": False,
            "uses_test_evidence": False,
        }
        return fused.to(torch.float32)
