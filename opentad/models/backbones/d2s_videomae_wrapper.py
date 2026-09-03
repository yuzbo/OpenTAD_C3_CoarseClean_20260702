# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone_wrapper import BackboneWrapper


class D2STemporalZoomBackboneWrapper(BackboneWrapper):
    """D2S-TAD Backbone: G96 Global Carrier + Dynamic K=16 U128 Local Refresh.

    1. Runs G96 (48 chunks, 96x96) through shared VideoMAE backbone -> G [B, 384, 384].
    2. Dynamic router computes representation shift Delta(t) across adjacent G chunks
       and selects dynamic Top-16 high-change chunks.
    3. TRUE PHYSICAL SKIP: Gathers ONLY the 16 selected chunks from source uint8 video,
       crops 128x128 center box, and executes VideoMAE. The remaining 32 chunks are
       never processed by local operators.
       Token budget: 48 * 8 * 36 + 16 * 8 * 64 = 13,824 + 8,192 = 22,016 tokens (57.33% of dense).
    4. Computes a dense residual R. Plain D2S returns G+R; PA-TAD receives the
       explicit {G,R} bundle and injects R only into L0/L1.
    """

    AUDIT_SCHEMA = "d2s_temporal_zoom_shared_videomae_v2"

    def __init__(self, cfg):
        custom_cfg = cfg.custom
        self.global_key = str(getattr(custom_cfg, "global_key", "global"))
        self.source_key = str(getattr(custom_cfg, "source_key", "source"))
        self.global_size = int(getattr(custom_cfg, "global_size", 96))
        self.local_size = int(getattr(custom_cfg, "local_size", 128))
        self.total_chunks = int(getattr(custom_cfg, "total_chunks", 48))
        self.burst_chunks = int(getattr(custom_cfg, "burst_chunks", 16))
        self.tubelets_per_chunk = int(getattr(custom_cfg, "tubelets_per_chunk", 8))
        self.intermediate_length = self.total_chunks * self.tubelets_per_chunk  # 384
        self.output_length = int(getattr(custom_cfg, "output_length", 768))
        self.return_feature_bundle = bool(
            getattr(custom_cfg, "return_feature_bundle", False)
        )
        self.source_height = int(getattr(custom_cfg, "source_height", 180))
        self.source_width = int(getattr(custom_cfg, "source_width", 320))

        if self.total_chunks <= 0 or self.burst_chunks <= 0 or self.burst_chunks > self.total_chunks:
            raise ValueError(
                "D2S requires 0 < burst_chunks <= total_chunks; "
                f"got k={self.burst_chunks}, total={self.total_chunks}"
            )

        if self.output_length != 2 * self.intermediate_length:
            raise ValueError("D2S requires exact 2x temporal output geometry")
        if self.local_size > min(self.source_height, self.source_width):
            raise ValueError("D2S local crop does not fit the frozen source geometry")

        # Frozen source-native center crop. It is applied after Top-K routing.
        y0 = (self.source_height - self.local_size) // 2
        x0 = (self.source_width - self.local_size) // 2
        self.crop_box = [x0, y0, x0 + self.local_size, y0 + self.local_size]

        super().__init__(cfg)

        channels = int(getattr(custom_cfg, "channels", 384))
        self.proj_local = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj_global = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.eye_(self.proj_local.weight.squeeze(-1))
        nn.init.zeros_(self.proj_local.bias)
        nn.init.eye_(self.proj_global.weight.squeeze(-1))
        nn.init.zeros_(self.proj_global.bias)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.latest_d2s_audit = None

    @staticmethod
    def _flatten_chunk_features(
        features: torch.Tensor, *, source_batch: int, chunk_count: int
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("D2S chunk features must be [B*chunks,C,tubelets]")
        if features.shape[0] != source_batch * chunk_count:
            raise ValueError("D2S feature count does not match chunk geometry")
        channels = int(features.shape[1])
        return (
            features.reshape(source_batch, chunk_count, channels, features.shape[-1])
            .permute(0, 2, 1, 3)
            .flatten(start_dim=2)
            .contiguous()
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

    def _encode_global_view(self, global_view: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.is_tensor(global_view):
            raise TypeError("D2S global view must be a torch.Tensor")
        if global_view.ndim != 6:
            raise ValueError(
                "D2S global view must be [B, 1, 3, T, H, W]; "
                f"got shape {tuple(global_view.shape)}"
            )
        if global_view.shape[1:3] != (1, 3):
            raise ValueError("D2S global view requires N=1 and RGB channels")
        if global_view.shape[-2:] != (self.global_size, self.global_size):
            raise ValueError("D2S global view changed its frozen spatial geometry")
        if global_view.shape[3] != self.total_chunks * 16:
            raise ValueError("D2S global view changed its 48x16-frame geometry")
        if global_view.dtype != torch.uint8:
            raise TypeError("D2S global view must enter the wrapper as uint8")
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(global_view),
            data_samples=None,
            training=False,
        )
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        flattened_batches, num_segs = frames.shape[:2]
        expected_batches = int(global_view.shape[0]) * self.total_chunks
        if flattened_batches != expected_batches:
            raise RuntimeError(
                "D2S global pre-processing did not produce one item per chunk"
            )
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())
        if isinstance(features, (tuple, list)) or features.ndim != 5:
            raise RuntimeError("D2S requires one [B,C,T,H,W] VideoMAE feature tensor")
        features = features.unflatten(0, sizes=(flattened_batches, num_segs))
        features = features.mean(dim=(1, 4, 5))
        B = int(global_view.shape[0])
        G_dense = self._flatten_chunk_features(
            features, source_batch=B, chunk_count=self.total_chunks
        )
        if G_dense.shape[-1] != self.intermediate_length:
            raise RuntimeError("D2S global carrier temporal length changed")
        channels = int(G_dense.shape[1])
        G_chunk = (
            G_dense.reshape(B, channels, self.total_chunks, self.tubelets_per_chunk)
            .mean(dim=-1)
            .transpose(1, 2)
        )  # [B, 48, C]
        return G_dense, G_chunk

    def _route_chunks(self, G_chunk: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute representation shift Delta(t) across adjacent G chunks."""
        B, num_chunks, C = G_chunk.shape
        norm_G = F.normalize(G_chunk, p=2, dim=-1)
        cos_sim = (norm_G[:, 1:] * norm_G[:, :-1]).sum(dim=-1)  # [B, 47]
        d_t = 1.0 - cos_sim  # [B, 47]

        d_left = torch.cat([d_t[:, :1], d_t], dim=-1)
        d_right = torch.cat([d_t, d_t[:, -1:]], dim=-1)
        repr_shift = 0.5 * (d_left + d_right)  # [B, 48]

        # Top-K selection by representation shift
        sorted_indices = torch.argsort(
            repr_shift, dim=-1, descending=True, stable=True
        )
        selected_indices = sorted_indices[:, : self.burst_chunks]
        selected_indices, _ = torch.sort(selected_indices, dim=-1)

        router_outputs = {
            "repr_shift": repr_shift,
            "selected_indices": selected_indices,
        }
        return selected_indices, router_outputs

    def _encode_selected_local_chunks(
        self,
        source_tensor: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Physical skip: Crop and encode ONLY the selected K=16 chunks."""
        if not torch.is_tensor(source_tensor):
            raise TypeError("D2S source view must be a torch.Tensor")
        if source_tensor.ndim != 6:
            raise ValueError(
                "D2S source view must be [B, 1, 3, T, H, W]; "
                f"got shape {tuple(source_tensor.shape)}"
            )
        if source_tensor.dtype != torch.uint8:
            raise TypeError("D2S source view must remain uint8 until selected gather")
        if source_tensor.device.type != "cpu":
            raise ValueError("D2S source view must stay on CPU until Top-K gather")
        B, num_clips, C_in, T_total, H_src, W_src = source_tensor.shape
        if (num_clips, C_in, T_total, H_src, W_src) != (
            1,
            3,
            self.total_chunks * 16,
            self.source_height,
            self.source_width,
        ):
            raise ValueError("D2S source view changed its frozen NCTHW geometry")
        if int(selected_indices.shape[0]) != B:
            raise ValueError("D2S route batch does not match source batch")
        x0, y0, x1, y1 = self.crop_box
        K = self.burst_chunks
        selected_cpu = selected_indices.detach().to(device="cpu")

        local_chunk_list = []
        for b in range(B):
            for k_idx in range(K):
                chunk_id = int(selected_cpu[b, k_idx])
                start_f = chunk_id * 16
                end_f = start_f + 16
                chunk_crop = source_tensor[b, :, :, start_f:end_f, y0:y1, x0:x1]  # [1, 3, 16, 128, 128]
                local_chunk_list.append(chunk_crop)

        local_tensor = torch.stack(local_chunk_list, dim=0)
        if local_tensor.shape != (B * K, 1, 3, 16, self.local_size, self.local_size):
            raise RuntimeError("D2S selected local tensor has invalid geometry")
        local_tensor = local_tensor.to(selected_indices.device, non_blocking=True)
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(local_tensor),
            data_samples=None,
            training=False,
        )
        flattened_batches, num_segs = frames.shape[:2]
        if flattened_batches != B * K:
            raise RuntimeError("D2S local pre-processing changed selected chunk count")
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())
        if isinstance(features, (tuple, list)) or features.ndim != 5:
            raise RuntimeError("D2S local branch requires one 5D feature tensor")
        features = features.unflatten(0, sizes=(flattened_batches, num_segs))
        features = features.mean(dim=(1, 4, 5))  # [B*K, C, 8]
        L_sel = self._flatten_chunk_features(
            features, source_batch=B, chunk_count=K
        )
        return L_sel

    def forward(self, frames, masks=None):
        if not isinstance(frames, Mapping):
            raise TypeError("D2S requires a mapping with global and source views")
        required_keys = {self.global_key, self.source_key}
        if not required_keys.issubset(frames):
            raise ValueError(
                f"D2S input is missing required keys {sorted(required_keys - set(frames))}"
            )
        global_view = frames[self.global_key]
        source_view = frames[self.source_key]

        self.set_norm_layer()

        # 1. Forward G96 global carrier (all 48 chunks)
        G, G_chunk = self._encode_global_view(global_view)
        B, C, T_tubelets = G.shape

        # 2. Router selection (K=16 chunks)
        selected_indices, router_outputs = self._route_chunks(G_chunk)

        # 3. Physical Skip: U128 Local Refresh (ONLY 16 chunks executed)
        L_sel = self._encode_selected_local_chunks(source_view, selected_indices)
        P_L_sel = self.proj_local(L_sel)

        G_4d = G.reshape(B, C, self.total_chunks, self.tubelets_per_chunk)
        gather_index = selected_indices[:, None, :, None].expand(
            B, C, self.burst_chunks, self.tubelets_per_chunk
        )
        G_sel = torch.gather(G_4d, dim=2, index=gather_index).flatten(start_dim=2)
        P_G_sel = self.proj_global(G_sel)
        R_sel = self.gamma * (P_L_sel - P_G_sel)

        R_sel_4d = R_sel.reshape(
            B, C, self.burst_chunks, self.tubelets_per_chunk
        )
        R = torch.zeros_like(G_4d).scatter(
            dim=2, index=gather_index, src=R_sel_4d
        ).flatten(start_dim=2)

        # 4. Interpolate to 768 detector temporal resolution
        G_out = F.interpolate(G, size=self.output_length, mode="nearest")
        R_out = F.interpolate(R, size=self.output_length, mode="nearest")
        if masks is not None:
            if masks.shape != (B, self.output_length):
                raise ValueError("D2S mask does not match the detector time axis")
            valid = masks.unsqueeze(1).detach().to(G_out.dtype)
            G_out = G_out * valid
            R_out = R_out * valid
        G_out = G_out.to(torch.float32)
        R_out = R_out.to(torch.float32)
        self.latest_d2s_audit = {
            "schema_version": self.AUDIT_SCHEMA,
            "global_chunks_executed": B * self.total_chunks,
            "local_chunks_executed": B * self.burst_chunks,
            "local_chunks_skipped": B * (self.total_chunks - self.burst_chunks),
            "global_tokens_per_sample": self.total_chunks
            * self.tubelets_per_chunk
            * (self.global_size // 16) ** 2,
            "local_tokens_per_sample": self.burst_chunks
            * self.tubelets_per_chunk
            * (self.local_size // 16) ** 2,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_test_evidence": False,
        }
        bundle = {
            "global_features": G_out,
            "residual_features": R_out,
            "feats": G_out + R_out,
        }
        return bundle if self.return_feature_bundle else bundle["feats"]
