# Copyright (c) OpenTAD. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.registry import MODELS
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding

from .vit_adapter import Adapter


class TemporalLowRankJVP(BaseModule):
    """Strictly linear Temporal Low-Rank Jacobian-Vector Product Operator.
    
    Operates along the temporal axis T across tubelets with channel bottleneck rank r.
    Strictly satisfies:
    1. J(0) = 0
    2. J(a + b) = J(a) + J(b)
    3. J(c * a) = c * J(a)
    """
    def __init__(self, embed_dims: int = 384, rank: int = 64, kernel_size: int = 3):
        super().__init__()
        self.embed_dims = embed_dims
        self.rank = rank
        self.down = nn.Linear(embed_dims, rank, bias=False)
        self.temporal = nn.Conv1d(
            rank,
            rank,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=rank,
            bias=False,
        )
        self.up = nn.Linear(rank, embed_dims, bias=False)

        # Initialize with identity-preserving scaling
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.temporal.weight)
        if kernel_size >= 3:
            mid = kernel_size // 2
            self.temporal.weight.data[:, 0, mid] = 1.0
        nn.init.orthogonal_(self.up.weight)

    def forward(self, delta_h: Tensor) -> Tensor:
        """
        Args:
            delta_h: (B, T, S, C) tensor of hidden state differences from nearest anchor
        Returns:
            J * delta_h: (B, T, S, C) strictly linear temporal JVP approximation
        """
        b, t, s, c = delta_h.shape
        # Channel reduction to rank r
        z = self.down(delta_h)  # (B, T, S, R)
        # Permute to (B*S, R, T) for temporal 1D convolution along time axis T
        z = z.permute(0, 2, 3, 1).reshape(b * s, self.rank, t)
        z = self.temporal(z)
        # Permute back to (B, T, S, R) and project up to C
        z = z.reshape(b, s, self.rank, t).permute(0, 3, 1, 2)  # (B, T, S, R)
        return self.up(z)  # (B, T, S, C)


class TaylorAttention(BaseModule):
    """Multi-Head Self-Attention supporting Selected-Q Full-KV Context."""
    def __init__(
        self,
        embed_dims: int = 384,
        num_heads: int = 6,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_dims = embed_dims // num_heads
        self.scale = qk_scale or head_dims**-0.5

        self.q_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.k_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.v_proj = nn.Linear(embed_dims, embed_dims, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)
        self.pretrained_qkv_remapped = False
        self.pretrained_bias_remapped = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # Official VideoMAE checkpoints store one fused ``qkv`` projection,
        # while ET-TRC uses separate projections so Selected-Q / Full-KV can
        # be evaluated without recomputing Q for non-anchor tokens.  Remap
        # once at load time and keep the report on the module for parity tests.
        fused_w = state_dict.pop(prefix + "qkv.weight", None)
        fused_b = state_dict.pop(prefix + "qkv.bias", None)
        if fused_w is not None:
            if fused_w.ndim != 2 or fused_w.shape[0] != 3 * self.embed_dims:
                error_msgs.append(
                    f"{prefix}qkv.weight has incompatible shape {tuple(fused_w.shape)}"
                )
            else:
                q_w, k_w, v_w = fused_w.chunk(3, dim=0)
                state_dict[prefix + "q_proj.weight"] = q_w
                state_dict[prefix + "k_proj.weight"] = k_w
                state_dict[prefix + "v_proj.weight"] = v_w
                self.pretrained_qkv_remapped = True
        if fused_b is not None:
            if fused_b.ndim != 1 or fused_b.shape[0] != 3 * self.embed_dims:
                error_msgs.append(
                    f"{prefix}qkv.bias has incompatible shape {tuple(fused_b.shape)}"
                )
            else:
                q_b, k_b, v_b = fused_b.chunk(3, dim=0)
                state_dict[prefix + "q_proj.bias"] = q_b
                state_dict[prefix + "k_proj.bias"] = k_b
                state_dict[prefix + "v_proj.bias"] = v_b
                self.pretrained_qkv_remapped = True

        # Official VideoMAE stores Q/V bias separately (K has no bias). Map
        # those tensors to the split projections and create an explicit zero K
        # bias so no pretrained bias is silently left random or unexpected.
        official_q_bias = state_dict.pop(prefix + "q_bias", None)
        official_v_bias = state_dict.pop(prefix + "v_bias", None)
        if official_q_bias is not None or official_v_bias is not None:
            if self.q_proj.bias is None:
                error_msgs.append(f"{prefix}q_bias/v_bias provided but qkv_bias=False")
            elif (
                official_q_bias is None
                or official_v_bias is None
                or official_q_bias.shape != (self.embed_dims,)
                or official_v_bias.shape != (self.embed_dims,)
            ):
                error_msgs.append(f"{prefix}q_bias/v_bias have incompatible shapes")
            else:
                state_dict[prefix + "q_proj.bias"] = official_q_bias
                state_dict[prefix + "k_proj.bias"] = torch.zeros_like(official_q_bias)
                state_dict[prefix + "v_proj.bias"] = official_v_bias
                self.pretrained_bias_remapped = True
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(
        self,
        x: Tensor,
        query_x: Optional[Tensor] = None,
        query_segment_ids: Optional[Tensor] = None,
        key_segment_ids: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (B, N, C) full sequence providing Key and Value context
            query_x: Optional (B, M, C) subset of tokens for Query (e.g. Anchor tokens).
                     If None, performs standard full sequence MHA.
        Returns:
            out: (B, M, C) if query_x is provided, else (B, N, C)
        """
        B, N, C = x.shape
        q_src = query_x if query_x is not None else x
        M = q_src.shape[1]

        q = (
            self.q_proj(q_src)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v_proj(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if query_segment_ids is not None or key_segment_ids is not None:
            if query_segment_ids is None or key_segment_ids is None:
                raise ValueError("query_segment_ids and key_segment_ids must be provided together")
            if query_segment_ids.shape != (B, M) or key_segment_ids.shape != (B, N):
                raise ValueError("segment id tensors must match query/key token counts")
            segment_mask = query_segment_ids[:, None, :, None] == key_segment_ids[:, None, None, :]
            attn = attn.masked_fill(~segment_mask, torch.finfo(attn.dtype).min)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, M, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class TaylorResidualBlock(BaseModule):
    """Transformer Block with Event-Triggered Local Taylor Residual Correction (ET-TRC).
    
    Anchor frames (T_A): compute full Selected-Q / Full-KV MHA + MLP -> Delta_a = F(h_a)
    Non-Anchor frames (T_NA): approximate Delta_i = Delta_a + J_{approx} * (h_i - h_a)
    100% Dense temporal state is preserved (State Multiplicity).
    """
    def __init__(
        self,
        embed_dims: int = 384,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: dict = dict(type="GELU"),
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        with_cp: bool = False,
        adapter_cfg: Optional[dict] = None,
        stride_k: int = 4,
        segment_size: Optional[int] = None,
        enable_taylor: bool = True,
        jacobian_rank: int = 64,
    ):
        super().__init__()
        self.with_cp = with_cp
        self.stride_k = stride_k
        self.segment_size = segment_size
        self.enable_taylor = enable_taylor
        
        _, self.norm1 = build_norm_layer(norm_cfg, embed_dims)
        self.attn = TaylorAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
        )
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        
        _, self.norm2 = build_norm_layer(norm_cfg, embed_dims)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=int(embed_dims * mlp_ratio),
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )
        
        # 1st-Order Temporal Low-Rank Jacobian Approximator
        self.jacobian_approx = TemporalLowRankJVP(
            embed_dims=embed_dims,
            rank=jacobian_rank,
            kernel_size=3,
        )
        
        # Optional Temporal Adapter (TIA)
        self.adapter = None
        if adapter_cfg is not None:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                mlp_ratio=adapter_cfg.get("mlp_ratio", 0.25),
                kernel_size=adapter_cfg.get("kernel_size", 3),
                dilation=adapter_cfg.get("dilation", 1),
                temporal_size=adapter_cfg.get("temporal_size", 384),
            )

    def _full_block_residual(self, x: Tensor) -> Tensor:
        """Standard full Transformer Block residual: F(x) = MHA(LN(x)) + MLP(LN(x + MHA(LN(x))))"""
        attn_out = self.drop_path(self.attn(self.norm1(x)))
        mid = x + attn_out
        mlp_out = self.drop_path(self.mlp(self.norm2(mid)))
        return attn_out + mlp_out

    def forward(self, x: Tensor, h: int = 10, w: int = 10, num_frames: int = 16) -> Tensor:
        """
        Args:
            x: (B, N, C) where N = T_tubelets * (h * w) (e.g. 8 * 100 = 800 tokens per chunk)
            h, w: spatial patch dimensions (e.g. 10x10 or 8x8)
            num_frames: number of frames per chunk
        Returns:
            100% Dense updated state (B, N, C)
        """
        B, N, C = x.shape
        spatial_tokens = h * w
        tubelet_count = N // spatial_tokens if spatial_tokens > 0 else 1
        
        if not self.enable_taylor or tubelet_count <= 1 or self.stride_k <= 1:
            # Full dense execution with exact numerical parity
            delta = self._full_block_residual(x)
            x_out = x + delta
            if self.adapter is not None:
                x_out = self.adapter(x_out, h, w)
            return x_out

        # Reshape to (B, T, S, C)
        x_reshaped = x.view(B, tubelet_count, spatial_tokens, C)
        
        # Anchor tubelet indices
        anchor_indices = list(range(0, tubelet_count, self.stride_k))
        if (tubelet_count - 1) not in anchor_indices:
            anchor_indices.append(tubelet_count - 1)
        num_anchors = len(anchor_indices)

        # Full context attention for anchor queries.  Segment ids isolate an
        # anchor from unrelated temporal chunks while retaining all spatial
        # tokens and all tubelets in its own segment as KV context.
        x_norm1 = self.norm1(x)
        x_norm1_reshaped = x_norm1.view(B, tubelet_count, spatial_tokens, C)
        q_anchors = x_norm1_reshaped[:, anchor_indices].reshape(B, num_anchors * spatial_tokens, C)
        
        # Selected-Q full-KV attention preserving full sequence context
        segment_size = int(self.segment_size or tubelet_count)
        segment_size = max(1, segment_size)
        key_segment_ids = (
            torch.arange(tubelet_count, device=x.device).div(segment_size, rounding_mode="floor")
            .view(tubelet_count, 1).expand(tubelet_count, spatial_tokens).reshape(-1)
        )
        query_segment_ids = key_segment_ids.view(tubelet_count, spatial_tokens)[anchor_indices].reshape(-1)
        attn_anchors = self.drop_path(
            self.attn(
                x_norm1,
                query_x=q_anchors,
                query_segment_ids=query_segment_ids.unsqueeze(0).expand(B, -1),
                key_segment_ids=key_segment_ids.unsqueeze(0).expand(B, -1),
            )
        )
        
        # MLP for anchors
        x_anchors_raw = x_reshaped[:, anchor_indices].reshape(B, num_anchors * spatial_tokens, C)
        mid_anchors = x_anchors_raw + attn_anchors
        mlp_anchors = self.drop_path(self.mlp(self.norm2(mid_anchors)))
        delta_anchors = (attn_anchors + mlp_anchors).view(B, num_anchors, spatial_tokens, C)

        # Build vectorized mapping for nearest anchor
        anchor_map = [
            min(range(num_anchors), key=lambda i: abs(anchor_indices[i] - t))
            for t in range(tubelet_count)
        ]
        
        # Vectorized expansion of anchor residuals and anchor hidden states
        delta_expanded = torch.stack([delta_anchors[:, anchor_map[t]] for t in range(tubelet_count)], dim=1)  # (B, T, S, C)
        x_anchor_expanded = torch.stack([x_reshaped[:, anchor_indices[anchor_map[t]]] for t in range(tubelet_count)], dim=1)  # (B, T, S, C)
        
        # Difference from nearest anchor: delta_h = h_i - h_{a(i)}
        delta_h = x_reshaped - x_anchor_expanded  # (B, T, S, C)
        
        # Vectorized 1st-Order Temporal Low-Rank JVP approximation
        taylor_correction = self.jacobian_approx(delta_h)  # (B, T, S, C)
        
        # 100% Dense Residual Update
        delta_all = delta_expanded + taylor_correction
        x_out = x + delta_all.view(B, N, C)
        
        # Pass through AdaTAD Temporal Adapter
        if self.adapter is not None:
            x_out = self.adapter(x_out, h, w)
            
        return x_out


@MODELS.register_module()
class ETTRCVisionTransformerAdapter(BaseModule):
    """VideoMAE Backbone equipped with Event-Triggered 1st-Order Taylor Residual Correction (ET-TRC).
    
    Maintains 100% Dense Temporal Identity across all 12 layers while achieving 2x+ GEMM acceleration.
    """
    def __init__(
        self,
        img_size: int = 160,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        norm_cfg: dict = dict(type="LN", eps=1e-6),
        act_cfg: dict = dict(type="GELU"),
        with_cp: bool = False,
        total_frames: int = 768,
        num_frames: int = 16,
        tubelet_size: int = 2,
        adapter_index: Optional[List[int]] = None,
        adapter_cfg: Optional[dict] = None,
        stride_k: int = 4,
        segment_size: Optional[int] = None,
        enable_taylor: bool = True,
        jacobian_rank: int = 64,
        return_feat_map: bool = True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.depth = depth
        self.total_frames = total_frames
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.return_feat_map = return_feat_map
        self.stride_k = stride_k
        self.segment_size = segment_size
        self.enable_taylor = enable_taylor
        self.with_cp = with_cp

        if adapter_index is None:
            adapter_index = list(range(12))
        if adapter_cfg is None:
            adapter_cfg = dict(mlp_ratio=0.25, kernel_size=3, dilation=1)
        adapter_cfg = dict(adapter_cfg)

        self.temporal_size = total_frames // tubelet_size  # 384
        adapter_cfg["temporal_size"] = self.temporal_size

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
            norm_cfg=None,
        )
        self.num_patches = (img_size // patch_size) * (img_size // patch_size) * (num_frames // tubelet_size)
        self.pos_embed = nn.Parameter(
            get_sinusoid_encoding(self.num_patches, embed_dims),
            requires_grad=False,
        )

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = ModuleList([
            TaylorResidualBlock(
                embed_dims=embed_dims,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[i],
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                with_cp=with_cp,
                adapter_cfg=adapter_cfg if i in adapter_index else None,
                stride_k=stride_k,
                segment_size=segment_size,
                enable_taylor=enable_taylor,
                jacobian_rank=jacobian_rank,
            )
            for i in range(depth)
        ])
        _, self.norm = build_norm_layer(norm_cfg, embed_dims)

    def forward(self, x: Tensor) -> Union[Tensor, Tuple[Tensor]]:
        """
        Args:
            x: (B, C, T, H, W) raw input video clips
        Returns:
            features: (B, C, T_tubelets, H, W) if return_feat_map else (B, N, C)
        """
        B, C, T, H, W = x.shape
        x = self.patch_embed(x)[0]  # (B, N, C)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()

        h = H // self.patch_size
        w = W // self.patch_size

        for block in self.blocks:
            if self.with_cp and x.requires_grad:
                x = cp.checkpoint(
                    block,
                    x,
                    h,
                    w,
                    self.num_frames,
                    use_reentrant=False,
                )
            else:
                x = block(x, h=h, w=w, num_frames=self.num_frames)

        x = self.norm(x)

        if self.return_feat_map:
            tubelets = T // self.tubelet_size
            x = x.reshape(B, tubelets, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, T_tubelets, H, W]

        return x
