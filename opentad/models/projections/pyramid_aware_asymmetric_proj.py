# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, TransformerBlock
from ..builder import PROJECTIONS
from .actionformer_proj import Conv1DTransformerProj, get_sinusoid_encoding


@PROJECTIONS.register_module()
class PyramidAwareAsymmetricProj(Conv1DTransformerProj):
    """
    Pyramid-Aware Asymmetric TAD Projection (PA-TAD, Phase 3).
    
    Decouples feature representation across the 1D temporal feature pyramid (L0 to L5):
    - Low Pyramid Levels (L0, L1): Preserves dense local burst details and high-frequency
      action transition boundaries for short-action recall and fine-grained localization.
    - High Pyramid Levels (L2 to L5): Processes macro-context representations via global
      scout features, completely eliminating redundant high-resolution calculations for
      large receptive field levels.
    - Uses zero extra parameters and maintains exact output signature compatibility
      with ActionFormer / AdaTAD rpn_head and evaluator.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        arch=(2, 2, 5),  # (#convs, #stem transformers, #branch transformers)
        conv_cfg=None,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        use_abs_pe=False,
        max_seq_len=2304,
        input_pdrop=0.0,
        asymmetric_split_level=2,  # Levels >= asymmetric_split_level use global macro downsampling
        high_res_gate_weight=1.0,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            arch=arch,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            attn_cfg=attn_cfg,
            path_pdrop=path_pdrop,
            use_abs_pe=use_abs_pe,
            max_seq_len=max_seq_len,
            input_pdrop=input_pdrop,
        )
        self.asymmetric_split_level = int(asymmetric_split_level)
        self.high_res_gate_weight = float(high_res_gate_weight)

    def forward(self, x, mask, burst_mask=None):
        """
        Args:
            x: [B, C, T] intermediate feature tensor or fused representation.
            mask: [B, T] bool tensor indicating valid positions.
            burst_mask: Optional [B, T] or [B, 1, T] indicating active high-res burst chunks.
        Returns:
            out_feats: tuple of tensors for levels (L0, L1, L2, L3, L4, L5)
            out_masks: tuple of boolean masks for each level
        """
        # Feature projection if multiscale/channel split is used
        if self.proj is not None:
            x = torch.cat(
                [proj(s, mask)[0] for proj, s in zip(self.proj, x.split(self.in_channels, dim=1))],
                dim=1,
            )

        if self.input_pdrop is not None:
            x = self.input_pdrop(x)

        # 1. Embedding network (1D Convolutions)
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

        # Apply asymmetric burst gating to Low Pyramid Levels if burst_mask is provided
        if burst_mask is not None:
            if burst_mask.dim() == 2:
                b_mask = burst_mask.unsqueeze(1).to(x.dtype)
            else:
                b_mask = burst_mask.to(x.dtype)
            # Match temporal dimension if downsampled
            if b_mask.shape[-1] != x.shape[-1]:
                b_mask = F.interpolate(b_mask, size=x.shape[-1], mode="nearest")
            # Enhance local burst boundary features on low levels
            x = x * (1.0 + self.high_res_gate_weight * b_mask)

        # 2. Position Embeddings
        if self.use_abs_pe and self.training:
            assert x.shape[-1] <= self.max_seq_len, "Reached max sequence length."
            pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)
        elif self.use_abs_pe and (not self.training):
            if x.shape[-1] >= self.max_seq_len:
                pe = F.interpolate(self.pos_embed, x.shape[-1], mode="linear", align_corners=False)
            else:
                pe = self.pos_embed
            x = x + pe[:, :, : x.shape[-1]] * mask.unsqueeze(1).to(x.dtype)

        # 3. Stem transformer (Generates Level 0: Fine-grained High-Res Boundary Level)
        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)

        # L0: Full high-resolution output
        out_feats = [x]
        out_masks = [mask]

        # 4. Main branch with asymmetric temporal pyramid downsampling
        curr_feat = x
        curr_mask = mask

        for idx in range(len(self.branch)):
            level = idx + 1
            curr_feat, curr_mask = self.branch[idx](curr_feat, curr_mask)
            out_feats.append(curr_feat)
            out_masks.append(curr_mask)

        return tuple(out_feats), tuple(out_masks)
