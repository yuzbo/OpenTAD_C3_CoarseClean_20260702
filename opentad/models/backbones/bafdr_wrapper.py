# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import BACKBONES
from .backbone_wrapper import BackboneWrapper
from .native_crop_wrapper import deterministic_linear_2x

BAFDR_BACKBONE_SCHEMA = "bafdr_k16_shared_videomae_v1"


def flatten_chunk_tubelets(
    features: torch.Tensor,
    *,
    source_batch: int,
    chunk_num: int,
) -> torch.Tensor:
    """Preserve chronological chunk-major, tubelet-minor ordering."""
    if features.ndim != 3:
        raise ValueError("chunk features must be [B*chunks, C, tubelets]")
    if int(features.shape[0]) != int(source_batch) * int(chunk_num):
        raise ValueError(
            f"chunk feature count {features.shape[0]} does not match source batch geometry {source_batch}x{chunk_num}"
        )
    channels = int(features.shape[1])
    tubelets = int(features.shape[2])
    return (
        features.reshape(source_batch, chunk_num, channels, tubelets)
        .permute(0, 2, 1, 3)
        .flatten(start_dim=2)
    )


class BAFDRRouterHead(nn.Module):
    """Boundary-aware fixed-capacity router head for 48 chunks."""

    def __init__(self, in_channels: int = 384, hidden_channels: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels)
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(hidden_channels, 4, kernel_size=1)

    def forward(self, chunk_descriptors: torch.Tensor) -> torch.Tensor:
        """
        Args:
            chunk_descriptors: [B, 48, C] chunk features
        Returns:
            logits: [B, 4, 48] (actionness, start, end, residual_gate)
        """
        x = self.norm(chunk_descriptors)  # [B, 48, C]
        x = x.transpose(1, 2)  # [B, C, 48]
        x = self.act(self.conv1(x))  # [B, hidden, 48]
        logits = self.conv2(x)  # [B, 4, 48]
        return logits


@BACKBONES.register_module()
class BAFDRBackboneWrapper(BackboneWrapper):
    """BA-FDR Backbone: G96 Global Carrier + Fixed K=16 U128 Local Refresh + Residual Scatter.
    
    1. Runs G96 (48 chunks, 96x96) through shared VideoMAE backbone -> G [B, 384, 384].
    2. Boundary-aware router computes scores and selects fixed Top-16 chunks.
    3. TRUE PHYSICAL SKIP: Gathers ONLY the 16 selected chunks from source uint8 video,
       crops fixed 128x128 A0 center box [96, 26, 224, 154], and executes VideoMAE.
       The remaining 32 chunks are NEVER processed by local operators.
    4. Computes residual R_sel = gamma * gate * (P_L(L) - P_G(G)) and scatters into
       all-zero dense tensor R [B, 384, 384]. Fused carrier Z = G + R.
    """

    def __init__(self, cfg):
        custom_cfg = cfg.custom
        self.global_key = str(getattr(custom_cfg, "bafdr_global_key", "global"))
        self.source_key = str(getattr(custom_cfg, "bafdr_source_key", "source"))
        self.global_size = int(getattr(custom_cfg, "bafdr_global_size", 96))
        self.local_size = int(getattr(custom_cfg, "bafdr_local_size", 128))
        self.chunk_num = int(getattr(custom_cfg, "bafdr_chunk_num", 48))
        self.k_chunks = int(getattr(custom_cfg, "bafdr_k_chunks", 16))
        self.tubelets_per_chunk = int(getattr(custom_cfg, "bafdr_tubelets_per_chunk", 8))
        self.intermediate_length = self.chunk_num * self.tubelets_per_chunk  # 384
        self.output_length = int(getattr(custom_cfg, "bafdr_output_length", 768))
        self.uniform_mode = bool(getattr(custom_cfg, "bafdr_uniform_mode", False))
        self.return_bundle = bool(getattr(custom_cfg, "bafdr_return_bundle", True))

        # Center crop canonical box for 180x320 source: [x0, y0, x1, y1] = [96, 26, 224, 154]
        self.crop_box = [96, 26, 224, 154]

        super().__init__(cfg)

        channels = int(getattr(custom_cfg, "bafdr_channels", 384))
        self.router = BAFDRRouterHead(in_channels=channels, hidden_channels=128)

        # Identity-initialized 1x1 channel projections
        self.proj_local = nn.Conv1d(channels, channels, kernel_size=1)
        self.proj_global = nn.Conv1d(channels, channels, kernel_size=1)
        nn.init.eye_(self.proj_local.weight.squeeze(-1))
        nn.init.zeros_(self.proj_local.bias)
        nn.init.eye_(self.proj_global.weight.squeeze(-1))
        nn.init.zeros_(self.proj_global.bias)

        # Learnable global scale scalar initialized to 0
        self.gamma = nn.Parameter(torch.zeros(1))

        self.latest_bafdr_audit = None

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

    def _encode_global_view(self, global_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward all 48 chunks of G96.
        Returns:
            G: [B, C, 384] tubelet features
            G_chunk: [B, 48, C] chunk descriptors
        """
        source_batch = int(global_tensor.shape[0])
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(global_tensor),
            data_samples=None,
            training=False,
        )
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        flattened_batches, num_segs = frames.shape[:2]  # [B*48, 1]
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())  # [B*48, C, 8, 6, 6]
        features = features.unflatten(0, sizes=(flattened_batches, num_segs))
        features = features.mean(dim=(1, 4, 5))  # [B*48, C, 8]

        G = flatten_chunk_tubelets(
            features,
            source_batch=source_batch,
            chunk_num=self.chunk_num,
        )  # [B, C, 384]

        # Chunk descriptors: mean pool over 8 tubelets per chunk
        G_chunk = G.view(source_batch, G.shape[1], self.chunk_num, self.tubelets_per_chunk).mean(dim=-1)  # [B, C, 48]
        G_chunk = G_chunk.transpose(1, 2).contiguous()  # [B, 48, C]
        return G, G_chunk

    def _route_chunks(self, G_chunk: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute boundary-aware routing scores and select top-16 chunks."""
        B, num_chunks, C = G_chunk.shape
        logits = self.router(G_chunk)  # [B, 4, 48]

        actionness_logits = logits[:, 0]  # [B, 48]
        start_logits = logits[:, 1]  # [B, 48]
        end_logits = logits[:, 2]  # [B, 48]
        residual_gate_logits = logits[:, 3]  # [B, 48]

        # Symmetric cosine distance between adjacent chunk descriptors
        norm_G = F.normalize(G_chunk, p=2, dim=-1)  # [B, 48, C]
        cos_sim = (norm_G[:, 1:] * norm_G[:, :-1]).sum(dim=-1)  # [B, 47]
        d_t = 1.0 - cos_sim  # [B, 47]

        # Boundary padding to length 48
        d_left = torch.cat([d_t[:, :1], d_t], dim=-1)  # [B, 48]
        d_right = torch.cat([d_t, d_t[:, -1:]], dim=-1)  # [B, 48]
        repr_shift = 0.5 * (d_left + d_right)  # [B, 48]

        # Normalize across 48 chunks per sample to [0, 1]
        min_v = repr_shift.min(dim=-1, keepdim=True)[0]
        max_v = repr_shift.max(dim=-1, keepdim=True)[0]
        rank01_shift = (repr_shift - min_v) / (max_v - min_v + 1e-6)

        scores = (
            0.40 * rank01_shift
            + 0.30 * torch.sigmoid(start_logits)
            + 0.30 * torch.sigmoid(end_logits)
        )  # [B, 48]

        if self.uniform_mode:
            # Deterministic uniform 16 chunks: [0, 3, 6, ..., 45]
            step = num_chunks // self.k_chunks
            uniform_idx = torch.arange(0, num_chunks, step, device=G_chunk.device)[: self.k_chunks]
            selected_indices = uniform_idx.unsqueeze(0).expand(B, -1)  # [B, 16]
        else:
            # Peak-first stable top-K
            max_pool = F.max_pool1d(scores.unsqueeze(1), kernel_size=3, stride=1, padding=1).squeeze(1)
            is_peak = scores >= max_pool
            priority = is_peak.float() * 10.0 + scores

            # Stable sorting: argmax priority descending
            sorted_indices = torch.argsort(priority, dim=-1, descending=True)
            selected_indices = sorted_indices[:, : self.k_chunks]
            # Maintain chronological ascending order within each sample
            selected_indices, _ = torch.sort(selected_indices, dim=-1)

        router_outputs = {
            "logits": logits,
            "actionness_logits": actionness_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "residual_gate_logits": residual_gate_logits,
            "scores": scores,
            "selected_indices": selected_indices,
        }
        return selected_indices, router_outputs

    def _encode_selected_local_chunks(
        self,
        source_tensor: torch.Tensor,
        selected_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Physical skip: Crop and encode ONLY the selected K=16 chunks.
        Args:
            source_tensor: [B, 1, 3, 768, 180, 320] uint8 raw frames
            selected_indices: [B, 16] chunk indices in {0, ..., 47}
        Returns:
            L_sel: [B, C, 16*8] local features for selected chunks
        """
        B, _, C_in, T_total, H_src, W_src = source_tensor.shape
        x0, y0, x1, y1 = self.crop_box
        K = self.k_chunks

        # Gather selected 16-frame chunks and crop to [128, 128]
        # Shape per chunk: [1, 3, 16, 128, 128]
        local_chunk_list = []
        for b in range(B):
            for k_idx in range(K):
                chunk_id = int(selected_indices[b, k_idx].item())
                start_f = chunk_id * 16
                end_f = start_f + 16
                chunk_crop = source_tensor[b, :, :, start_f:end_f, y0:y1, x0:x1]  # [1, 3, 16, 128, 128]
                local_chunk_list.append(chunk_crop)

        # Batch: [B*K, 1, 3, 16, 128, 128] uint8
        local_tensor = torch.stack(local_chunk_list, dim=0)

        # Data preprocessor: normalize to float
        frames, _ = self.model.data_preprocessor.preprocess(
            self.tensor_to_list(local_tensor),
            data_samples=None,
            training=False,
        )
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        flattened_batches, num_segs = frames.shape[:2]  # [B*K, 1]
        features = self._run_shared_backbone(frames.flatten(0, 1).contiguous())  # [B*K, C, 8, 8, 8]
        features = features.unflatten(0, sizes=(flattened_batches, num_segs))
        features = features.mean(dim=(1, 4, 5))  # [B*K, C, 8]

        # Reshape to [B, C, K*8]
        channels = features.shape[1]
        L_sel = (
            features.reshape(B, K, channels, self.tubelets_per_chunk)
            .permute(0, 2, 1, 3)
            .flatten(start_dim=2)
            .contiguous()
        )  # [B, C, 128]
        return L_sel

    def forward(self, frames, masks=None):
        if isinstance(frames, Mapping):
            global_view = frames.get(self.global_key, frames.get("global"))
            source_view = frames.get(self.source_key, frames.get("source"))
        else:
            global_view = frames
            source_view = None

        if global_view is None:
            raise ValueError("BA-FDR requires global view tensor")

        self.set_norm_layer()

        # 1. Forward G96 global carrier (all 48 chunks)
        G, G_chunk = self._encode_global_view(global_view)
        B, C, T_tubelets = G.shape

        # 2. Router selection (K=16 chunks)
        selected_indices, router_outputs = self._route_chunks(G_chunk)

        # 3. True Physical Skip: U128 Local Refresh (ONLY 16 chunks executed)
        if source_view is not None:
            L_sel = self._encode_selected_local_chunks(source_view, selected_indices)
        else:
            # Fallback for test / feature-only input: gather from G
            L_sel = torch.zeros(B, C, self.k_chunks * self.tubelets_per_chunk, device=G.device, dtype=G.dtype)

        # 4. Gather global tubelets corresponding to selected chunks
        # selected_indices: [B, 16], each chunk has 8 tubelets -> [B, 128] tubelet indices
        tubelet_offsets = torch.arange(self.tubelets_per_chunk, device=G.device).view(1, 1, self.tubelets_per_chunk)
        selected_tubelets = (
            (selected_indices.unsqueeze(-1) * self.tubelets_per_chunk + tubelet_offsets)
            .flatten(start_dim=1)
        )  # [B, 128]

        G_sel = torch.gather(G, 2, selected_tubelets.unsqueeze(1).expand(-1, C, -1))  # [B, C, 128]

        # 5. Channel projections & residual difference
        R_diff = self.proj_local(L_sel) - self.proj_global(G_sel)  # [B, C, 128]

        # 6. Residual gate & learnable scale
        gate_chunk = torch.sigmoid(torch.gather(router_outputs["residual_gate_logits"], 1, selected_indices))  # [B, 16]
        gate_tubelets = (
            gate_chunk.unsqueeze(-1)
            .expand(-1, -1, self.tubelets_per_chunk)
            .flatten(start_dim=1)
            .unsqueeze(1)
        )  # [B, 1, 128]

        R_sel = self.gamma * gate_tubelets * R_diff  # [B, C, 128]

        # 7. Scatter into all-zero dense residual tensor R [B, C, 384]
        R = torch.zeros_like(G)
        R.scatter_(2, selected_tubelets.unsqueeze(1).expand(-1, C, -1), R_sel)

        # 8. Fused dense carrier: Z = G + R
        Z = G + R

        # 9. Expand to detector length (exact 2x temporal expansion, length 768)
        G_768 = deterministic_linear_2x(G)
        R_768 = deterministic_linear_2x(R)
        Z_768 = deterministic_linear_2x(Z)

        if masks is not None:
            if masks.shape != (B, self.output_length):
                raise ValueError("Mask does not match the detector time axis")
            m = masks.unsqueeze(1).detach().to(Z_768.dtype)
            G_768 = G_768 * m
            R_768 = R_768 * m
            Z_768 = Z_768 * m

        self.latest_bafdr_audit = {
            "schema_version": BAFDR_BACKBONE_SCHEMA,
            "shared_backbone_instances": 1,
            "k_chunks": self.k_chunks,
            "total_chunks": self.chunk_num,
            "executed_local_chunks": B * self.k_chunks,
            "unselected_local_chunks": B * (self.chunk_num - self.k_chunks),
            "uniform_mode": self.uniform_mode,
            "selected_indices": selected_indices.detach().cpu().tolist(),
            "gamma_value": float(self.gamma.item()),
            "output_shape": list(Z_768.shape),
        }

        if self.return_bundle:
            return {
                "feats": Z_768.to(torch.float32),
                "global_features": G_768.to(torch.float32),
                "residual_features": R_768.to(torch.float32),
                "fused_features": Z_768.to(torch.float32),
                "router_outputs": router_outputs,
                "selected_indices": selected_indices,
                "executed_local_chunks": B * self.k_chunks,
                "unselected_local_chunks": B * (self.chunk_num - self.k_chunks),
            }
        return Z_768.to(torch.float32)
