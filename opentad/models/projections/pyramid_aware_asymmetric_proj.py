# Copyright (c) OpenTAD. All rights reserved.
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..bricks import ConvModule, TransformerBlock
from ..builder import PROJECTIONS
from .actionformer_proj import Conv1DTransformerProj, get_sinusoid_encoding


@PROJECTIONS.register_module()
class PyramidAwareAsymmetricProj(Conv1DTransformerProj):
    """Pyramid-Aware Asymmetric TAD Projection (PA-TAD).

    Decouples feature representation across the 1D temporal feature pyramid (L0 to L5):
    1. Global carrier G constructs all pyramid levels L0..L5 via standard Transformer branches.
    2. Residual features R are injected ONLY into low pyramid levels (L0, L1):
       - L0 = P0(G) + Q0(R)  (Q0: 1x1 Conv1d)
       - L1 = P1(G) + Q1(R)  (Q1: Stride-2 1D Conv matching L1 temporal geometry)
    3. High pyramid levels L2..L5 are strictly bitwise invariant to R:
       - L2..L5 = P2(G)..P5(G)
       (Completely eliminates redundant local residual propagation across high-level macro contexts).
    """

    def __init__(
        self,
        in_channels: int = 384,
        out_channels: int = 384,
        arch=(2, 2, 5),
        conv_cfg=None,
        norm_cfg=None,
        attn_cfg=None,
        path_pdrop=0.0,
        use_abs_pe=False,
        max_seq_len=2304,
        input_pdrop=0.0,
        asymmetric_split_level=2,
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

        # Lightweight residual injectors for L0 and L1 (in_channels -> out_channels)
        self.q0_inj = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.q1_inj = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)

        # Initialize injectors with identity mappings on center taps
        nn.init.zeros_(self.q0_inj.weight)
        nn.init.zeros_(self.q0_inj.bias)
        nn.init.zeros_(self.q1_inj.weight)
        nn.init.zeros_(self.q1_inj.bias)
        diagonal = min(in_channels, out_channels)
        self.q0_inj.weight.data[:diagonal, :diagonal, 0].fill_diagonal_(1.0)
        center = self.q1_inj.kernel_size[0] // 2
        self.q1_inj.weight.data[:diagonal, :diagonal, center].fill_diagonal_(1.0)

    def _forward_branch(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for idx in range(len(self.embed)):
            x, mask = self.embed[idx](x, mask)

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

        for idx in range(len(self.stem)):
            x, mask = self.stem[idx](x, mask)
        return x, mask

    def forward(
        self,
        x: Union[torch.Tensor, Dict[str, torch.Tensor]],
        mask: torch.Tensor,
        burst_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Tuple[torch.Tensor, ...], Tuple[torch.Tensor, ...]]:
        """
        Args:
            x: Tensor [B, C, T] or dict bundle containing 'global_features' and 'residual_features'
            mask: [B, T] bool mask
            burst_mask: Optional unused parameter for backwards interface compatibility
        Returns:
            out_feats: (L0, L1, L2, L3, L4, L5)
            out_masks: (M0, M1, M2, M3, M4, M5)
        """
        if isinstance(x, dict):
            g_feat = x.get("global_features", x.get("feats"))
            r_feat = x.get("residual_features", torch.zeros_like(g_feat))
        else:
            g_feat = x
            r_feat = torch.zeros_like(x)

        if self.proj is not None:
            g_feat = torch.cat(
                [proj(s, mask)[0] for proj, s in zip(self.proj, g_feat.split(self.in_channels, dim=1))],
                dim=1,
            )
            r_feat = torch.cat(
                [proj(s, mask)[0] for proj, s in zip(self.proj, r_feat.split(self.in_channels, dim=1))],
                dim=1,
            )

        if self.input_pdrop is not None:
            g_feat = self.input_pdrop(g_feat)

        # 1. Forward global branch through stem to get P0(G)
        p0_g, m0 = self._forward_branch(g_feat, mask)

        # 2. Inject residual R into Level 0: L0 = P0(G) + Q0(R)
        r_l0 = self.q0_inj(r_feat) * m0.unsqueeze(1).to(r_feat.dtype)
        l0 = p0_g + r_l0

        out_feats = [l0]
        out_masks = [m0]

        # 3. Main branch levels L1..L5
        # L1 receives P1(G) + Q1(R)
        curr_feat, curr_mask = self.branch[0](p0_g, m0)
        r_l1 = self.q1_inj(r_feat)
        if r_l1.shape[-1] != curr_feat.shape[-1]:
            r_l1 = F.interpolate(r_l1, size=curr_feat.shape[-1], mode="nearest")
        r_l1 = r_l1 * curr_mask.unsqueeze(1).to(r_l1.dtype)
        l1 = curr_feat + r_l1

        out_feats.append(l1)
        out_masks.append(curr_mask)

        # L2..L5 are strictly derived from global branch P1(G) (Bitwise invariant to R)
        for idx in range(1, len(self.branch)):
            curr_feat, curr_mask = self.branch[idx](curr_feat, curr_mask)
            out_feats.append(curr_feat)
            out_masks.append(curr_mask)

        return tuple(out_feats), tuple(out_masks)
