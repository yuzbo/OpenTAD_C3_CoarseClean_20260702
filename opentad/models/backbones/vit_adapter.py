# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from torch import Tensor, nn
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.bricks.transformer import FFN, PatchEmbed
from mmengine.registry import MODELS
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import constant_init, trunc_normal_init
from mmaction.utils import ConfigType, OptConfigType
from mmaction.models.backbones.vit_mae import get_sinusoid_encoding

from .georoute_routing import build_apm32_temporal_plan


class Adapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        kernel_size: int = 3,
        dilation: int = 1,
        temporal_size: int = 384,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # temporal depth-wise convolution
        self.temporal_size = temporal_size
        self.dwconv = nn.Conv1d(
            hidden_dims,
            hidden_dims,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size // 2) * dilation,
            dilation=dilation,
            groups=hidden_dims,
        )
        self.conv = nn.Conv1d(hidden_dims, hidden_dims, 1)
        self.dwconv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / kernel_size))
        self.dwconv.bias.data.zero_()
        self.conv.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / hidden_dims))
        self.conv.bias.data.zero_()

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)

        # temporal depth-wise convolution
        B, N, C = x.shape  # 48, 8*10*10, 384
        attn = x.reshape(-1, self.temporal_size, h, w, x.shape[-1])  # [b,t,h,w,c]  [1,384,10,10,384]
        attn = attn.permute(0, 2, 3, 4, 1).flatten(0, 2)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.dwconv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = self.conv(attn)  # [b*h*w,c,t] [1*10*10,384,384]
        attn = attn.unflatten(0, (-1, h, w)).permute(0, 4, 1, 2, 3)  # [b,t,h,w,c] [1,384,10,10,384]
        attn = attn.reshape(B, N, C)
        x = x + attn

        x = self.up_proj(x)
        return x * self.gamma + inputs

    @staticmethod
    def _lookup_selected_neighbor(
        source_values: Tensor,
        source_indices: Tensor,
        query_indices: Tensor,
    ) -> Tensor:
        """Match exact spatial lineage between adjacent tubelets."""

        if source_values.ndim != 4:
            raise ValueError("source_values must be [B,T,K,C]")
        if source_indices.shape != source_values.shape[:3]:
            raise ValueError("source_indices must match source_values [B,T,K]")
        if query_indices.shape != source_indices.shape:
            raise ValueError("query_indices must match source_indices")
        if source_indices.dtype != torch.long:
            raise TypeError("packed spatial indices must be torch.long")
        select_count = int(source_indices.shape[-1])
        positions = torch.searchsorted(
            source_indices.contiguous(),
            query_indices.contiguous(),
        )
        bounded = positions.clamp(max=select_count - 1)
        hit = (positions < select_count) & (
            source_indices.gather(-1, bounded) == query_indices
        )
        gathered = source_values.gather(
            2,
            bounded.unsqueeze(-1).expand(
                *bounded.shape,
                int(source_values.shape[-1]),
            ),
        )
        return gathered * hit.unsqueeze(-1).to(dtype=source_values.dtype)

    def forward_native_packed(
        self,
        inputs: Tensor,
        dense_mask: Tensor,
        spatial_indices: Tensor,
        *,
        grid_height: int,
        grid_width: int,
    ) -> Tensor:
        """Apply the original Adapter parameters only to selected lineages.

        Unselected carrier positions are exact identity bypasses. A temporal
        neighbor contributes only when the same native spatial index is selected
        in the adjacent tubelet. Full-K selection is numerically equivalent to
        :meth:`forward`.
        """

        if inputs.ndim != 3:
            raise ValueError("packed Adapter inputs must be [Bchunk,L,C]")
        if dense_mask.dtype != torch.bool:
            raise TypeError("packed Adapter dense_mask must be bool")
        if dense_mask.shape != inputs.shape[:2]:
            raise ValueError("packed Adapter dense_mask must match inputs")
        if spatial_indices.ndim != 3:
            raise ValueError("spatial_indices must be [B,T,K]")
        if spatial_indices.dtype != torch.long:
            raise TypeError("spatial_indices must be torch.long")

        batch_size, total_tubelets, select_count = map(
            int,
            spatial_indices.shape,
        )
        channels = int(inputs.shape[-1])
        spatial_tokens = int(grid_height) * int(grid_width)
        if select_count <= 0:
            raise ValueError("packed Adapter requires K>0")
        if spatial_tokens <= 0 or int(inputs.shape[1]) % spatial_tokens:
            raise ValueError(
                "packed carrier token count is not divisible by spatial grid"
            )
        tubelets_per_chunk = int(inputs.shape[1]) // spatial_tokens
        if total_tubelets % tubelets_per_chunk:
            raise ValueError(
                "total tubelets must be divisible by tubelets per chunk"
            )
        chunk_count = total_tubelets // tubelets_per_chunk
        if int(inputs.shape[0]) != batch_size * chunk_count:
            raise ValueError("packed carrier batch/chunk layout is inconsistent")
        if total_tubelets != int(self.temporal_size):
            raise ValueError(
                "packed Adapter temporal axis differs from pretrained Adapter"
            )
        if int(dense_mask.sum().item()) != (
            batch_size * total_tubelets * select_count
        ):
            raise ValueError(
                "packed Adapter mask does not contain exact B*T*K selections"
            )
        if select_count > 1 and not bool(
            (spatial_indices[..., 1:] > spatial_indices[..., :-1]).all().item()
        ):
            raise ValueError(
                "packed Adapter indices must be strictly increasing"
            )

        selected_inputs = inputs[dense_mask].reshape(
            batch_size,
            total_tubelets,
            select_count,
            channels,
        )
        hidden = self.act(self.down_proj(selected_inputs))
        previous = torch.zeros_like(hidden)
        following = torch.zeros_like(hidden)
        if total_tubelets > 1:
            previous[:, 1:] = self._lookup_selected_neighbor(
                hidden[:, :-1],
                spatial_indices[:, :-1],
                spatial_indices[:, 1:],
            )
            following[:, :-1] = self._lookup_selected_neighbor(
                hidden[:, 1:],
                spatial_indices[:, 1:],
                spatial_indices[:, :-1],
            )

        if (
            self.dwconv.kernel_size != (3,)
            or self.dwconv.dilation != (1,)
            or self.dwconv.stride != (1,)
            or self.dwconv.padding != (1,)
        ):
            raise RuntimeError(
                "coordinate-lineage Adapter requires the original "
                "kernel_size=3,dilation=1,stride=1,padding=1"
            )
        if self.conv.kernel_size != (1,):
            raise RuntimeError(
                "coordinate-lineage Adapter requires pointwise Conv1d"
            )
        kernel = self.dwconv.weight[:, 0, :]
        temporal = (
            previous * kernel[:, 0].view(1, 1, 1, -1)
            + hidden * kernel[:, 1].view(1, 1, 1, -1)
            + following * kernel[:, 2].view(1, 1, 1, -1)
        )
        if self.dwconv.bias is not None:
            temporal = temporal + self.dwconv.bias.view(1, 1, 1, -1)
        temporal = F.linear(
            temporal,
            self.conv.weight[:, :, 0],
            self.conv.bias,
        )
        selected_output = (
            self.up_proj(hidden + temporal) * self.gamma + selected_inputs
        )
        output = inputs.clone()
        output[dense_mask] = selected_output.reshape(-1, channels)
        return output

    def forward_native_ragged(
        self,
        inputs: Tensor,
        tubelet_indices: Tensor,
        spatial_indices: Tensor,
        *,
        total_tubelets: int,
        grid_height: int,
        grid_width: int,
    ) -> Tensor:
        """Apply Adapter parameters to a flat, padding-free selected union.

        Every selected token retains its global tubelet and native spatial
        lineage.  Adjacent temporal contributions are present only when the
        same spatial index was genuinely selected; no placeholder token is
        created for an empty tubelet.
        """

        if inputs.ndim != 3:
            raise ValueError("ragged Adapter inputs must be [B,S,C]")
        if tubelet_indices.shape != inputs.shape[:2]:
            raise ValueError("ragged tubelet indices must match [B,S]")
        if spatial_indices.shape != inputs.shape[:2]:
            raise ValueError("ragged spatial indices must match [B,S]")
        if tubelet_indices.dtype != torch.long or spatial_indices.dtype != torch.long:
            raise TypeError("ragged Adapter provenance indices must be torch.long")
        batch_size, selected_count, channels = map(int, inputs.shape)
        spatial_tokens = int(grid_height) * int(grid_width)
        if selected_count <= 0:
            raise ValueError("ragged Adapter requires a positive window budget")
        if int(total_tubelets) != int(self.temporal_size):
            raise ValueError(
                "ragged Adapter temporal axis differs from pretrained Adapter"
            )
        if spatial_tokens <= 0:
            raise ValueError("ragged Adapter native grid must be positive")
        if bool(
            (
                (tubelet_indices < 0)
                | (tubelet_indices >= int(total_tubelets))
                | (spatial_indices < 0)
                | (spatial_indices >= spatial_tokens)
            ).any().item()
        ):
            raise ValueError("ragged Adapter provenance falls outside the native lattice")
        physical_indices = tubelet_indices * spatial_tokens + spatial_indices
        if selected_count > 1 and not bool(
            (physical_indices[:, 1:] > physical_indices[:, :-1]).all().item()
        ):
            raise ValueError(
                "ragged Adapter physical indices must be strictly increasing"
            )

        hidden = self.act(self.down_proj(inputs))
        position_lattice = torch.full(
            (batch_size, int(total_tubelets), spatial_tokens),
            -1,
            device=inputs.device,
            dtype=torch.long,
        )
        batch_indices = torch.arange(
            batch_size,
            device=inputs.device,
            dtype=torch.long,
        ).view(batch_size, 1).expand(batch_size, selected_count)
        selected_positions = torch.arange(
            selected_count,
            device=inputs.device,
            dtype=torch.long,
        ).view(1, selected_count).expand(batch_size, selected_count)
        position_lattice[
            batch_indices,
            tubelet_indices,
            spatial_indices,
        ] = selected_positions

        def _neighbor(delta: int) -> Tensor:
            neighbor_tubelet = tubelet_indices + int(delta)
            in_bounds = (
                (neighbor_tubelet >= 0)
                & (neighbor_tubelet < int(total_tubelets))
            )
            bounded_tubelet = neighbor_tubelet.clamp(
                min=0,
                max=int(total_tubelets) - 1,
            )
            positions = position_lattice[
                batch_indices,
                bounded_tubelet,
                spatial_indices,
            ]
            present = in_bounds & (positions >= 0)
            gathered = hidden.gather(
                1,
                positions.clamp_min(0).unsqueeze(-1).expand(
                    batch_size,
                    selected_count,
                    int(hidden.shape[-1]),
                ),
            )
            return gathered * present.unsqueeze(-1).to(dtype=hidden.dtype)

        previous = _neighbor(-1)
        following = _neighbor(1)
        if (
            self.dwconv.kernel_size != (3,)
            or self.dwconv.dilation != (1,)
            or self.dwconv.stride != (1,)
            or self.dwconv.padding != (1,)
        ):
            raise RuntimeError(
                "coordinate-lineage Adapter requires the original "
                "kernel_size=3,dilation=1,stride=1,padding=1"
            )
        if self.conv.kernel_size != (1,):
            raise RuntimeError(
                "coordinate-lineage Adapter requires pointwise Conv1d"
            )
        kernel = self.dwconv.weight[:, 0, :]
        temporal = (
            previous * kernel[:, 0].view(1, 1, -1)
            + hidden * kernel[:, 1].view(1, 1, -1)
            + following * kernel[:, 2].view(1, 1, -1)
        )
        if self.dwconv.bias is not None:
            temporal = temporal + self.dwconv.bias.view(1, 1, -1)
        temporal = F.linear(
            temporal,
            self.conv.weight[:, :, 0],
            self.conv.bias,
        )
        return self.up_proj(hidden + temporal) * self.gamma + inputs


class PlainAdapter(BaseModule):
    def __init__(
        self,
        embed_dims: int,
        mlp_ratio: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__()

        hidden_dims = int(embed_dims * mlp_ratio)

        # adapter projection
        self.down_proj = nn.Linear(embed_dims, hidden_dims)
        self.act = nn.GELU()
        self.up_proj = nn.Linear(hidden_dims, embed_dims)
        self.gamma = nn.Parameter(torch.ones(1))
        trunc_normal_init(self.down_proj, std=0.02, bias=0)
        constant_init(self.up_proj, 0)  # the last projection layer is initialized to 0

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        inputs = x

        # down and up projection
        x = self.down_proj(x)
        x = self.act(x)
        x = self.up_proj(x)
        return x * self.gamma + inputs


class TubeletTokenRedundancyAux(BaseModule):
    """Local-only tubelet/token redundancy auditor for VideoMAE tokens.

    The first R28 use is deliberately non-destructive: it scores temporal
    tubelet groups using their spatial-token statistics, records a proposed
    tubelet keep mask, and returns the dense token sequence unchanged.
    """

    valid_modes = ("identity", "shadow", "deterministic_tubelet_cap")

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "identity",
        route_unit: str = "temporal_tubelet_group",
        keep_ratio: float = 1.0,
        min_keep_tubelets: int = 1,
        route_pattern: str = "round_linspace",
        spatial_pool: str = "energy_std",
        forbid_spatial_crop: bool = True,
        local_audit_only: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported tubelet redundancy mode: {mode}")
        if route_unit != "temporal_tubelet_group":
            raise ValueError("R28 v0 only supports temporal_tubelet_group routing")
        if route_pattern != "round_linspace":
            raise ValueError("R28 v0 only supports round_linspace route_pattern")
        keep_ratio = float(keep_ratio)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        if int(min_keep_tubelets) <= 0:
            raise ValueError("min_keep_tubelets must be positive")
        if not bool(forbid_spatial_crop):
            raise ValueError("R28 v0 forbids spatial patch crop")
        if not bool(local_audit_only):
            raise ValueError("R28 v0 is local_audit_only")

        self.enabled = bool(enabled)
        self.mode = mode
        self.route_unit = route_unit
        self.keep_ratio = keep_ratio
        self.min_keep_tubelets = int(min_keep_tubelets)
        self.route_pattern = route_pattern
        self.spatial_pool = spatial_pool
        self.forbid_spatial_crop = bool(forbid_spatial_crop)
        self.local_audit_only = bool(local_audit_only)
        self.last_summary = None

    @staticmethod
    def _round_linspace_indices(length: int, count: int, device: torch.device) -> Tensor:
        length = int(length)
        count = min(int(count), length)
        if count <= 0 or length <= 0:
            raise ValueError("length and count must be positive")
        if count >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        if count == 1:
            return torch.zeros(1, device=device, dtype=torch.long)

        selected: List[int] = []
        for idx in range(count):
            pos = int(round(float(idx) * float(length - 1) / float(count - 1)))
            if pos not in selected:
                selected.append(pos)
        filler = 0
        while len(selected) < count:
            if filler not in selected:
                selected.append(filler)
            filler += 1
        return torch.tensor(sorted(selected[:count]), device=device, dtype=torch.long)

    def _shape_contract(self, x: Tensor, h: int, w: int) -> Tuple[int, int]:
        if x.ndim != 3:
            raise ValueError("tubelet redundancy aux expects x with shape [B, N, C]")
        spatial_tokens = int(h) * int(w)
        if spatial_tokens <= 0:
            raise ValueError("spatial token count must be positive")
        if int(x.shape[1]) % spatial_tokens != 0:
            raise ValueError("token length must be divisible by h*w")
        temporal_tubelets = int(x.shape[1]) // spatial_tokens
        if temporal_tubelets <= 0:
            raise ValueError("temporal tubelet count must be positive")
        return temporal_tubelets, spatial_tokens

    def summarize(self, x: Tensor, h: int, w: int) -> Dict[str, object]:
        temporal_tubelets, spatial_tokens = self._shape_contract(x, h, w)
        keep_count = max(
            self.min_keep_tubelets,
            int(math.ceil(float(temporal_tubelets) * float(self.keep_ratio))),
        )
        keep_count = min(keep_count, temporal_tubelets)

        token_energy = x.detach().float().pow(2).mean(dim=-1)
        token_energy = token_energy.reshape(int(x.shape[0]), temporal_tubelets, spatial_tokens)
        spatial_energy_mean = token_energy.mean(dim=-1)
        spatial_energy_std = token_energy.std(dim=-1, unbiased=False)
        redundancy_score = 1.0 / (1.0 + spatial_energy_std)

        proposed_mask = torch.zeros(
            (int(x.shape[0]), temporal_tubelets),
            dtype=torch.bool,
            device=x.device,
        )
        if self.mode in ("identity", "shadow"):
            proposed_mask[:] = True
        else:
            keep_idx = self._round_linspace_indices(temporal_tubelets, keep_count, x.device)
            proposed_mask[:, keep_idx] = True

        return {
            "schema_version": "tubelet_token_redundancy_aux_summary_v0",
            "enabled": self.enabled,
            "mode": self.mode,
            "route_unit": self.route_unit,
            "route_pattern": self.route_pattern,
            "local_audit_only": self.local_audit_only,
            "spatial_patch_crop_allowed": False,
            "spatial_crop_forbidden": self.forbid_spatial_crop,
            "dense_output_preserved": True,
            "compute_route_applied": False,
            "runtime_flops_claim_allowed": False,
            "batch_size": int(x.shape[0]),
            "token_length": int(x.shape[1]),
            "channels": int(x.shape[2]),
            "temporal_tubelets": temporal_tubelets,
            "spatial_tokens_per_tubelet": spatial_tokens,
            "keep_ratio": float(self.keep_ratio),
            "proposed_keep_count": int(proposed_mask[0].sum().item()),
            "effective_dense_token_count": int(x.shape[1]),
            "proposed_tubelet_keep_mask": proposed_mask,
            "spatial_energy_mean": spatial_energy_mean,
            "spatial_energy_std": spatial_energy_std,
            "redundancy_score": redundancy_score,
        }

    def forward(self, x: Tensor, h: int, w: int) -> Tensor:
        if not self.enabled:
            self.last_summary = None
            return x
        self.last_summary = self.summarize(x, h, w)
        return x


class PackedTubeletRuntimeRoute(BaseModule):
    """Opt-in packed temporal-tubelet execution inside ViT forward.

    This route is intentionally narrow. It only handles temporal-tubelet groups
    and keeps all spatial patches within a selected tubelet. Packed execution
    is applied inside each transformer block's attention/MLP subpath, then the
    selected outputs are scattered back before Adapter convolution sees the
    tensor. It is disabled by default and exists to make the R30 packed-runtime
    proof executable through the production backbone forward path.
    """

    valid_modes = ("deterministic_tubelet_cap",)
    valid_scatter_modes = ("identity", "zero")

    def __init__(
        self,
        enabled: bool = False,
        mode: str = "deterministic_tubelet_cap",
        route_unit: str = "temporal_tubelet_group",
        keep_ratio: float = 1.0,
        min_keep_tubelets: int = 1,
        route_pattern: str = "round_linspace",
        forbid_spatial_crop: bool = True,
        local_forward_only: bool = True,
        require_no_adapter_blocks: bool = False,
        allow_training_mode: bool = False,
        scatter_unselected: str = "identity",
        **kwargs,
    ) -> None:
        super().__init__()
        if mode not in self.valid_modes:
            raise ValueError(f"unsupported packed tubelet route mode: {mode}")
        if route_unit != "temporal_tubelet_group":
            raise ValueError("packed tubelet route only supports temporal_tubelet_group")
        if route_pattern != "round_linspace":
            raise ValueError("packed tubelet route only supports round_linspace route_pattern")
        keep_ratio = float(keep_ratio)
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError("keep_ratio must lie in (0, 1]")
        if int(min_keep_tubelets) <= 0:
            raise ValueError("min_keep_tubelets must be positive")
        if not bool(forbid_spatial_crop):
            raise ValueError("packed tubelet route forbids spatial patch crop")
        if not bool(local_forward_only):
            raise ValueError("packed tubelet route is local_forward_only")
        if scatter_unselected not in self.valid_scatter_modes:
            raise ValueError(f"unsupported packed tubelet scatter mode: {scatter_unselected}")

        self.enabled = bool(enabled)
        self.mode = mode
        self.route_unit = route_unit
        self.keep_ratio = keep_ratio
        self.min_keep_tubelets = int(min_keep_tubelets)
        self.route_pattern = route_pattern
        self.forbid_spatial_crop = bool(forbid_spatial_crop)
        self.local_forward_only = bool(local_forward_only)
        self.require_no_adapter_blocks = bool(require_no_adapter_blocks)
        self.allow_training_mode = bool(allow_training_mode)
        self.scatter_unselected = scatter_unselected
        self.last_summary = None

    @staticmethod
    def _shape_contract(x: Tensor, h: int, w: int) -> Tuple[int, int]:
        if x.ndim != 3:
            raise ValueError("packed tubelet route expects x with shape [B, N, C]")
        spatial_tokens = int(h) * int(w)
        if spatial_tokens <= 0:
            raise ValueError("spatial token count must be positive")
        if int(x.shape[1]) % spatial_tokens != 0:
            raise ValueError("token length must be divisible by h*w")
        temporal_tubelets = int(x.shape[1]) // spatial_tokens
        if temporal_tubelets <= 0:
            raise ValueError("temporal tubelet count must be positive")
        return temporal_tubelets, spatial_tokens

    @staticmethod
    def _round_linspace_indices(length: int, count: int, device: torch.device) -> Tensor:
        return TubeletTokenRedundancyAux._round_linspace_indices(length, count, device)

    def _tubelet_mask(self, x: Tensor, h: int, w: int) -> Tuple[Tensor, int, int]:
        temporal_tubelets, spatial_tokens = self._shape_contract(x, h, w)
        keep_count = max(
            self.min_keep_tubelets,
            int(math.ceil(float(temporal_tubelets) * float(self.keep_ratio))),
        )
        keep_count = min(keep_count, temporal_tubelets)
        keep_idx = self._round_linspace_indices(temporal_tubelets, keep_count, x.device)
        tubelet_mask = torch.zeros(
            (int(x.shape[0]), temporal_tubelets),
            dtype=torch.bool,
            device=x.device,
        )
        tubelet_mask[:, keep_idx] = True
        return tubelet_mask, temporal_tubelets, spatial_tokens

    @staticmethod
    def _expand_tubelet_mask(tubelet_mask: Tensor, spatial_tokens: int) -> Tensor:
        if tubelet_mask.ndim != 2:
            raise ValueError("tubelet_mask must have shape [B, T]")
        return tubelet_mask.unsqueeze(-1).expand(-1, -1, int(spatial_tokens)).reshape(
            int(tubelet_mask.shape[0]),
            int(tubelet_mask.shape[1]) * int(spatial_tokens),
        )

    @staticmethod
    def _pack_tokens(x: Tensor, dense_mask: Tensor) -> Tensor:
        if dense_mask.shape != x.shape[:2]:
            raise ValueError("dense_mask must match x batch/token shape")
        per_batch_counts = dense_mask.sum(dim=1)
        if int(per_batch_counts.min().item()) <= 0:
            raise ValueError("packed tubelet route requires at least one selected token per sample")
        if not torch.equal(per_batch_counts, per_batch_counts[:1].expand_as(per_batch_counts)):
            raise ValueError("packed tubelet route requires equal selected-token count across batch")
        return x[dense_mask].reshape(int(x.shape[0]), int(per_batch_counts[0].item()), int(x.shape[2]))

    def _scatter_tokens(self, base: Tensor, packed: Tensor, dense_mask: Tensor) -> Tensor:
        if self.scatter_unselected == "identity":
            scattered = base.clone()
        elif self.scatter_unselected == "zero":
            scattered = packed.new_zeros(base.shape)
        else:
            raise ValueError(f"unsupported packed tubelet scatter mode: {self.scatter_unselected}")
        scattered[dense_mask] = packed.reshape(-1, int(packed.shape[-1]))
        return scattered

    @staticmethod
    def _run_blocks(blocks, x: Tensor, h: int, w: int, dense_mask: Tensor, stats: Dict[str, int]) -> Tensor:
        out = x
        for block in blocks:
            if bool(getattr(block, "use_adapter", False)):
                stats["adapter_forward_count"] += 1
            out = block(out, h, w, packed_dense_mask=dense_mask, packed_stats=stats)
        return out

    def forward(self, x: Tensor, blocks, h: int, w: int, *, training: bool = False) -> Tensor:
        if not self.enabled:
            self.last_summary = None
            return x
        if bool(training) and not self.allow_training_mode:
            raise ValueError("packed tubelet route is local-forward-only and forbids training mode")
        adapter_block_count = sum(1 for block in blocks if bool(getattr(block, "use_adapter", False)))
        if self.require_no_adapter_blocks and adapter_block_count:
            raise ValueError("packed tubelet route requires adapter-free blocks for R31")

        tubelet_mask, temporal_tubelets, spatial_tokens = self._tubelet_mask(x, h, w)
        dense_mask = self._expand_tubelet_mask(tubelet_mask, spatial_tokens)
        packed = self._pack_tokens(x, dense_mask)
        stats = {
            "packed_attention_forward_count": 0,
            "packed_mlp_forward_count": 0,
            "adapter_forward_count": 0,
        }
        out = self._run_blocks(blocks, x, h, w, dense_mask, stats)

        selected_output = out[dense_mask].reshape(int(out.shape[0]), int(packed.shape[1]), int(out.shape[2]))
        selected_output_finite = bool(torch.isfinite(selected_output).all().item())
        scattered_output_finite = bool(torch.isfinite(out).all().item())
        if self.scatter_unselected == "identity" and adapter_block_count == 0:
            unselected_identity = bool(torch.allclose(out[~dense_mask], x[~dense_mask]))
        else:
            unselected_identity = None

        self.last_summary = {
            "schema_version": "packed_tubelet_runtime_route_summary_v0",
            "enabled": True,
            "mode": self.mode,
            "route_unit": self.route_unit,
            "route_pattern": self.route_pattern,
            "local_forward_only": self.local_forward_only,
            "production_forward_changed": True,
            "training_mode_allowed": self.allow_training_mode,
            "adapter_blocks_supported": not self.require_no_adapter_blocks,
            "adapter_block_count": int(adapter_block_count),
            "adapter_dense_contract_preserved": not self.require_no_adapter_blocks,
            "dense_scatter_before_adapter": bool(adapter_block_count and not self.require_no_adapter_blocks),
            "spatial_patch_crop_allowed": False,
            "spatial_filtering_allowed": False,
            "arbitrary_spatial_patch_filtering_allowed": False,
            "scatter_unselected": self.scatter_unselected,
            "dense_token_shape": list(x.shape),
            "packed_token_shape": list(packed.shape),
            "dense_output_shape": list(out.shape),
            "selected_output_shape": list(selected_output.shape),
            "temporal_tubelets": int(temporal_tubelets),
            "spatial_tokens_per_tubelet": int(spatial_tokens),
            "selected_tubelets": int(tubelet_mask[0].sum().item()),
            "selected_dense_tokens": int(dense_mask[0].sum().item()),
            "has_strict_token_saving": bool(packed.shape[1] < x.shape[1]),
            "true_packed_compute_enabled": True,
            "packed_attention_executed_in_forward": True,
            "packed_mlp_executed_in_forward": True,
            "scatter_back_executed": True,
            "packed_attention_forward_count": int(stats["packed_attention_forward_count"]),
            "packed_mlp_forward_count": int(stats["packed_mlp_forward_count"]),
            "adapter_forward_count": int(stats["adapter_forward_count"]),
            "unselected_positions_identity_bypass_without_adapter": unselected_identity,
            "selected_output_finite": selected_output_finite,
            "scattered_output_finite": scattered_output_finite,
            "measured_runtime": False,
            "measured_flops": False,
            "runtime_flops_claim_allowed": False,
            "metric_claim_allowed": False,
            "paper_claim_allowed": False,
        }
        return out


class Attention(BaseModule):
    """Multi-head Self-attention.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop_rate: float = 0.0,
        drop_rate: float = 0.0,
        init_cfg: OptConfigType = None,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads

        self.scale = qk_scale or head_embed_dims**-0.5

        if qkv_bias:
            self._init_qv_bias()

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(drop_rate)

    def _init_qv_bias(self) -> None:
        self.q_bias = nn.Parameter(torch.zeros(self.embed_dims))
        self.v_bias = nn.Parameter(torch.zeros(self.embed_dims))

    def _project_qkv(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Project a full token sequence with the pretrained QKV weights."""
        B, N, _ = x.shape
        if hasattr(self, "q_bias"):
            k_bias = torch.zeros_like(self.v_bias, requires_grad=False)
            qkv_bias = torch.cat((self.q_bias, k_bias, self.v_bias))
            qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        else:
            qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the attention block, same size as inputs.
        """
        B, N, _ = x.shape
        q, k, v = self._project_qkv(x)

        # standard self-attention
        # q = q * self.scale
        # attn = q @ k.transpose(-2, -1)
        # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        # fast attention
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def forward_with_column_mean(
        self,
        x: Tensor,
        *,
        query_chunk_size: int,
    ) -> Tuple[Tensor, Tensor]:
        """Return attention output and A-MoD's incoming-attention score."""
        if x.ndim != 3:
            raise ValueError("A-MoD attention expects [B,N,C]")
        if int(query_chunk_size) <= 0:
            raise ValueError("A-MoD query_chunk_size must be positive")
        if float(self.attn_drop.p) != 0.0:
            raise ValueError("A-MoD requires zero attention dropout")

        batch_size, token_count, _ = x.shape
        q, k, v = self._project_qkv(x)
        key_transpose = k.transpose(-2, -1)
        column_sum = torch.zeros(
            batch_size,
            self.num_heads,
            token_count,
            device=x.device,
            dtype=torch.float32,
        )
        output_chunks = []
        for start in range(0, token_count, int(query_chunk_size)):
            stop = min(start + int(query_chunk_size), token_count)
            logits = (q[:, :, start:stop] * self.scale) @ key_transpose
            probabilities = logits.softmax(dim=-1)
            output_chunks.append(probabilities @ v)
            column_sum.add_(probabilities.float().sum(dim=-2))

        output = torch.cat(output_chunks, dim=-2)
        output = output.transpose(1, 2).reshape(batch_size, token_count, self.embed_dims)
        output = self.proj_drop(self.proj(output))
        column_mean = column_sum.mean(dim=1).div(float(token_count))
        return output, column_mean

    def forward_query_context(self, query: Tensor, context: Tensor) -> Tensor:
        """Attention with a shorter query and K/V context using base qkv weights.

        This is deliberately parameter-free: it slices the existing qkv
        projection and retains the original q/v biases and output projection.
        With equal lengths it is numerically identical to ``forward``.
        """
        if query.ndim != 3 or context.ndim != 3 or query.shape[0] != context.shape[0]:
            raise ValueError("query/context must be [B,N,C] with matching batch")
        if query.shape[-1] != self.embed_dims or context.shape[-1] != self.embed_dims:
            raise ValueError("query/context channel dimension must match attention")
        channels = int(self.embed_dims)
        q_bias = self.q_bias if hasattr(self, "q_bias") else None
        v_bias = self.v_bias if hasattr(self, "v_bias") else None
        q = F.linear(query, self.qkv.weight[:channels], q_bias)
        k = F.linear(context, self.qkv.weight[channels : 2 * channels], None)
        v = F.linear(context, self.qkv.weight[2 * channels :], v_bias)
        q = q.reshape(query.shape[0], query.shape[1], self.num_heads, -1).transpose(1, 2)
        k = k.reshape(context.shape[0], context.shape[1], self.num_heads, -1).transpose(1, 2)
        v = v.reshape(context.shape[0], context.shape[1], self.num_heads, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        out = out.transpose(1, 2).reshape(query.shape[0], query.shape[1], self.embed_dims)
        return self.proj_drop(self.proj(out))


class Block(BaseModule):
    """The basic block in the Vision Transformer.

    Args:
        embed_dims (int): Dimensions of embedding.
        num_heads (int): Number of parallel attention heads.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        act_cfg (dict or ConfigDict): Config for activation layer in FFN.
            Defaults to `dict(type='GELU')`.
        norm_cfg (dict or ConfigDict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        init_cfg (dict or ConfigDict, optional): The Config
            for initialization. Defaults to None.
    """

    def __init__(
        self,
        embed_dims: int,
        num_heads: int,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        act_cfg: ConfigType = dict(type="GELU"),
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        init_cfg: OptConfigType = None,
        with_cp: bool = False,
        use_adapter: bool = False,
        adapter_mlp_ratio: float = 0.25,
        temporal_size: int = 384,
        **kwargs,
    ) -> None:
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp
        self.use_adapter = use_adapter

        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = Attention(
            embed_dims,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            drop_rate=drop_rate,
        )

        self.drop_path = nn.Identity()
        if drop_path_rate > 0.0:
            self.drop_path = DropPath(drop_path_rate)
        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]

        mlp_hidden_dim = int(embed_dims * mlp_ratio)
        self.mlp = FFN(
            embed_dims=embed_dims,
            feedforward_channels=mlp_hidden_dim,
            act_cfg=act_cfg,
            ffn_drop=drop_rate,
            add_identity=False,
        )

        if self.use_adapter:
            self.adapter = Adapter(
                embed_dims=embed_dims,
                kernel_size=3,
                dilation=1,
                temporal_size=temporal_size,
                mlp_ratio=adapter_mlp_ratio,
            )

    @staticmethod
    def _pack_selected_tokens(x: Tensor, dense_mask: Tensor) -> Tensor:
        if dense_mask.shape != x.shape[:2]:
            raise ValueError("packed_dense_mask must match x batch/token shape")
        per_batch_counts = dense_mask.sum(dim=1)
        if int(per_batch_counts.min().item()) <= 0:
            raise ValueError("packed block path requires at least one selected token per sample")
        if not torch.equal(per_batch_counts, per_batch_counts[:1].expand_as(per_batch_counts)):
            raise ValueError("packed block path requires equal selected-token count across batch")
        return x[dense_mask].reshape(int(x.shape[0]), int(per_batch_counts[0].item()), int(x.shape[2]))

    @staticmethod
    def _stable_amod_topk_indices(scores: Tensor, selected_count: int) -> Tensor:
        if scores.ndim != 2:
            raise ValueError("A-MoD scores must be [B,N]")
        token_count = int(scores.shape[1])
        if int(selected_count) <= 0 or int(selected_count) > token_count:
            raise ValueError("A-MoD selected_count must be within [1,N]")
        ranked = torch.argsort(scores, dim=-1, descending=True, stable=True)
        return ranked[:, : int(selected_count)].sort(dim=-1).values

    def forward_dense_with_amod_score(
        self,
        x: Tensor,
        h: int,
        w: int,
        *,
        query_chunk_size: int,
    ) -> Tuple[Tensor, Tensor]:
        """Run one dense block and expose its attention-received score."""

        def _inner_forward(value: Tensor) -> Tuple[Tensor, Tensor]:
            attention_output, score = self.attn.forward_with_column_mean(
                self.norm1(value),
                query_chunk_size=int(query_chunk_size),
            )
            value = value + self.drop_path(attention_output)
            value = value + self.drop_path(self.mlp(self.norm2(value)))
            if self.use_adapter:
                value = self.adapter(value, h, w)
            return value, score.detach()

        if self.with_cp and x.requires_grad:
            return cp.checkpoint(_inner_forward, x, use_reentrant=False)
        return _inner_forward(x)

    def forward_amod(
        self,
        x: Tensor,
        h: int,
        w: int,
        *,
        scores: Tensor,
        capacity: float,
    ) -> Tuple[Tensor, Tensor]:
        """Run Attention+MLP only on exact top-K tokens, then dense Adapter."""
        if scores.shape != x.shape[:2]:
            raise ValueError("A-MoD scores must match the full token carrier")
        if not 0.0 < float(capacity) <= 1.0:
            raise ValueError("A-MoD capacity must be within (0,1]")
        selected_count = max(1, int(round(int(x.shape[1]) * float(capacity))))
        selected_indices = self._stable_amod_topk_indices(
            scores.detach(), selected_count
        )
        gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, int(x.shape[-1]))
        selected = torch.gather(x, dim=1, index=gather_index)

        def _selected_forward(value: Tensor) -> Tensor:
            value = value + self.drop_path(self.attn(self.norm1(value)))
            return value + self.drop_path(self.mlp(self.norm2(value)))

        if self.with_cp and selected.requires_grad:
            selected = cp.checkpoint(_selected_forward, selected, use_reentrant=False)
        else:
            selected = _selected_forward(selected)
        output = x.clone().scatter(1, gather_index, selected)
        if self.use_adapter:
            output = self.adapter(output, h, w)
        return output, selected_indices.detach()

    def _packed_attention_mlp_forward(
        self,
        x: Tensor,
        dense_mask: Tensor,
        packed_stats: Optional[Dict[str, int]],
    ) -> Tensor:
        selected = self._pack_selected_tokens(x, dense_mask)
        selected = selected + self.drop_path(self.attn(self.norm1(selected)))
        selected = selected + self.drop_path(self.mlp(self.norm2(selected)))
        if packed_stats is not None:
            packed_stats["packed_attention_forward_count"] = int(packed_stats.get("packed_attention_forward_count", 0)) + 1
            packed_stats["packed_mlp_forward_count"] = int(packed_stats.get("packed_mlp_forward_count", 0)) + 1
        out = x.clone()
        out[dense_mask] = selected.reshape(-1, int(selected.shape[-1]))
        return out

    @staticmethod
    def _previous_spatial_block_input(
        x: Tensor,
        tubelet_indices: Tensor,
        spatial_indices: Tensor,
        *,
        total_tubelets: int,
        spatial_tokens: int,
    ) -> Tensor:
        """Return detached previous-tubelet inputs at the same spatial index."""
        if tubelet_indices.shape != x.shape[:2] or spatial_indices.shape != x.shape[:2]:
            raise ValueError("refresh lineage must match ragged token shape")
        batch_size, selected_count, channels = map(int, x.shape)
        lattice = x.new_zeros(
            (batch_size, int(total_tubelets), int(spatial_tokens), channels)
        )
        valid = torch.zeros(
            (batch_size, int(total_tubelets), int(spatial_tokens)),
            device=x.device,
            dtype=torch.bool,
        )
        batch = torch.arange(batch_size, device=x.device).view(-1, 1).expand(
            batch_size, selected_count
        )
        lattice[batch, tubelet_indices, spatial_indices] = x
        valid[batch, tubelet_indices, spatial_indices] = True
        previous_tubelet = (tubelet_indices - 1).clamp_min(0)
        gathered = lattice[
            batch,
            previous_tubelet,
            spatial_indices,
        ].detach()
        hit = (tubelet_indices > 0) & valid[
            batch,
            previous_tubelet,
            spatial_indices,
        ]
        return torch.where(hit.unsqueeze(-1), gathered, x)

    def _ragged_refresh_attention_mlp_forward(
        self,
        x: Tensor,
        bucket_positions: List[Tensor],
        refresh_mask: Tensor,
        tubelet_indices: Tensor,
        spatial_indices: Tensor,
        *,
        total_tubelets: int,
        spatial_tokens: int,
        refresh_mode: str,
        refresh_alpha: Optional[Tensor],
        packed_stats: Optional[Dict[str, int]],
    ) -> Tensor:
        if refresh_mode not in {"mod32_kv", "rc32_kv"}:
            raise ValueError("unsupported refresh-KV mode")
        if refresh_mask.shape != x.shape[:2] or refresh_mask.dtype != torch.bool:
            raise ValueError("refresh_mask must be bool and match ragged tokens")
        context = x
        if refresh_mode == "rc32_kv":
            if refresh_alpha is None or refresh_alpha.numel() != 1:
                raise ValueError("RC32 requires one scalar for every block")
            carry = self._previous_spatial_block_input(
                x,
                tubelet_indices,
                spatial_indices,
                total_tubelets=total_tubelets,
                spatial_tokens=spatial_tokens,
            )
            mixed = carry + torch.sigmoid(refresh_alpha) * (x - carry)
            context = torch.where(refresh_mask.unsqueeze(-1), x, mixed)

        flat_context = context.reshape(-1, int(x.shape[-1]))
        flat_mask = refresh_mask.reshape(-1)
        out = flat_context.clone()
        visited = torch.zeros(
            int(flat_context.shape[0]),
            device=x.device,
            dtype=torch.bool,
        )
        for positions in bucket_positions:
            flattened_positions = positions.reshape(-1)
            if bool(visited.gather(0, flattened_positions).any().item()):
                raise RuntimeError("ragged refresh attention buckets overlap")
            visited.scatter_(0, flattened_positions, True)
            selected_context = flat_context[positions]
            selected_mask = flat_mask[positions]
            selected_out = selected_context.clone()
            per_row_queries = selected_mask.sum(dim=1)
            if int(per_row_queries.min().item()) <= 0:
                raise RuntimeError("ragged refresh buckets require nonzero queries")
            rows, kv_tokens = map(int, positions.shape)
            for query_count in torch.unique(per_row_queries, sorted=True):
                query_tokens = int(query_count.item())
                row_indices = torch.nonzero(
                    per_row_queries == query_count,
                    as_tuple=False,
                ).flatten()
                row_context = selected_context.index_select(0, row_indices)
                row_mask = selected_mask.index_select(0, row_indices)
                query = row_context[row_mask].reshape(
                    int(row_indices.numel()),
                    query_tokens,
                    int(x.shape[-1]),
                )
                query = query + self.drop_path(
                    self.attn.forward_query_context(
                        self.norm1(query),
                        self.norm1(row_context),
                    )
                )
                query = query + self.drop_path(self.mlp(self.norm2(query)))
                row_out = row_context.clone()
                row_out[row_mask] = query.reshape(-1, int(x.shape[-1]))
                selected_out.index_copy_(0, row_indices, row_out)
                if packed_stats is not None:
                    subgroup_rows = int(row_indices.numel())
                    packed_stats["ragged_attention_bucket_call_count"] += 1
                    packed_stats["ragged_mlp_bucket_call_count"] += 1
                    packed_stats["executed_attention_tokens"] += (
                        subgroup_rows * query_tokens
                    )
                    packed_stats["executed_kv_tokens"] += subgroup_rows * kv_tokens
                    packed_stats["executed_attention_pairs"] += (
                        subgroup_rows * query_tokens * kv_tokens
                    )
                    packed_stats["executed_mlp_tokens"] += (
                        subgroup_rows * query_tokens
                    )
            out[positions] = selected_out
        if not bool(visited.all().item()):
            raise RuntimeError("ragged refresh buckets omitted a support token")
        return out.reshape_as(x)

    def _ragged_attention_mlp_forward(
        self,
        x: Tensor,
        bucket_positions: List[Tensor],
        packed_stats: Optional[Dict[str, int]],
    ) -> Tensor:
        if x.ndim != 3:
            raise ValueError("ragged block inputs must be [B,S,C]")
        if not bucket_positions:
            raise ValueError("ragged block requires at least one non-empty clip")
        flat = x.reshape(-1, int(x.shape[-1]))
        out = flat.clone()
        visited = torch.zeros(
            int(flat.shape[0]),
            device=x.device,
            dtype=torch.bool,
        )
        for positions in bucket_positions:
            if positions.ndim != 2 or positions.shape[1] <= 0:
                raise ValueError("ragged bucket positions must be non-empty [R,L]")
            if positions.dtype != torch.long:
                raise TypeError("ragged bucket positions must be torch.long")
            flattened_positions = positions.reshape(-1)
            if bool(visited.gather(0, flattened_positions).any().item()):
                raise RuntimeError("ragged attention buckets overlap")
            visited.scatter_(0, flattened_positions, True)
            selected = flat[positions]
            selected = selected + self.drop_path(self.attn(self.norm1(selected)))
            selected = selected + self.drop_path(self.mlp(self.norm2(selected)))
            out[positions] = selected
            if packed_stats is not None:
                rows, tokens = map(int, positions.shape)
                packed_stats["ragged_attention_bucket_call_count"] = int(
                    packed_stats.get("ragged_attention_bucket_call_count", 0)
                ) + 1
                packed_stats["ragged_mlp_bucket_call_count"] = int(
                    packed_stats.get("ragged_mlp_bucket_call_count", 0)
                ) + 1
                packed_stats["executed_attention_tokens"] = int(
                    packed_stats.get("executed_attention_tokens", 0)
                ) + rows * tokens
                packed_stats["executed_attention_pairs"] = int(
                    packed_stats.get("executed_attention_pairs", 0)
                ) + rows * tokens * tokens
                packed_stats["executed_mlp_tokens"] = int(
                    packed_stats.get("executed_mlp_tokens", 0)
                ) + rows * tokens
        if not bool(visited.all().item()):
            raise RuntimeError("ragged attention buckets omitted a selected token")
        return out.reshape_as(x)

    def forward_native_ragged(
        self,
        x: Tensor,
        *,
        bucket_positions: List[Tensor],
        tubelet_indices: Tensor,
        spatial_indices: Tensor,
        total_tubelets: int,
        grid_height: int,
        grid_width: int,
        packed_stats: Optional[Dict[str, int]] = None,
        refresh_mask: Optional[Tensor] = None,
        refresh_mode: str = "full64",
        refresh_alpha: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute one block on true clip-ragged selected-token sequences."""

        checkpoint_active = bool(self.with_cp and x.requires_grad)

        def _inner_forward(
            value: Tensor,
            alpha_value: Optional[Tensor] = None,
        ) -> Tensor:
            # Reentrant checkpoint executes its first pass without autograd and
            # replays the block with autograd during backward.  Record the
            # physical ledger only on the first pass so recomputation cannot be
            # mistaken for a second heavy execution.
            active_stats = (
                None
                if checkpoint_active and torch.is_grad_enabled()
                else packed_stats
            )
            if refresh_mask is None:
                value = self._ragged_attention_mlp_forward(
                    value,
                    bucket_positions,
                    active_stats,
                )
            else:
                value = self._ragged_refresh_attention_mlp_forward(
                    value,
                    bucket_positions,
                    refresh_mask,
                    tubelet_indices,
                    spatial_indices,
                    total_tubelets=total_tubelets,
                    spatial_tokens=int(grid_height) * int(grid_width),
                    refresh_mode=refresh_mode,
                    refresh_alpha=alpha_value,
                    packed_stats=active_stats,
                )
            if self.use_adapter:
                value = self.adapter.forward_native_ragged(
                    value,
                    tubelet_indices,
                    spatial_indices,
                    total_tubelets=total_tubelets,
                    grid_height=grid_height,
                    grid_width=grid_width,
                )
                if active_stats is not None:
                    active_stats["ragged_adapter_forward_count"] = int(
                        active_stats.get("ragged_adapter_forward_count", 0)
                    ) + 1
                    active_stats["executed_adapter_tokens"] = int(
                        active_stats.get("executed_adapter_tokens", 0)
                    ) + int(value.shape[0]) * int(value.shape[1])
            return value

        if checkpoint_active:
            if refresh_alpha is None:
                return cp.checkpoint(_inner_forward, x, use_reentrant=True)
            return cp.checkpoint(
                _inner_forward,
                x,
                refresh_alpha,
                use_reentrant=True,
            )
        return _inner_forward(x, refresh_alpha)

    def forward(
        self,
        x: Tensor,
        h,
        w,
        packed_dense_mask: Optional[Tensor] = None,
        packed_spatial_indices: Optional[Tensor] = None,
        packed_stats: Optional[Dict[str, int]] = None,
    ) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data with size of (B, N, C).
        Returns:
            Tensor: The output of the transformer block, same size as inputs.
        """

        def _inner_forward(x):
            """Forward wrapper for utilizing checkpoint."""
            if packed_dense_mask is None:
                x = x + self.drop_path(self.attn(self.norm1(x)))
                x = x + self.drop_path(self.mlp(self.norm2(x)))
            else:
                x = self._packed_attention_mlp_forward(x, packed_dense_mask, packed_stats)

            if self.use_adapter:
                if packed_dense_mask is None or packed_spatial_indices is None:
                    # The older temporal-tubelet packed route deliberately
                    # preserves its dense Adapter behavior. GeoRoute supplies
                    # native spatial lineage and takes the sparse path below.
                    x = self.adapter(x, h, w)
                else:
                    x = self.adapter.forward_native_packed(
                        x,
                        packed_dense_mask,
                        packed_spatial_indices,
                        grid_height=int(h),
                        grid_width=int(w),
                    )
                    if packed_stats is not None:
                        packed_stats["packed_adapter_forward_count"] = int(
                            packed_stats.get("packed_adapter_forward_count", 0)
                        ) + 1
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x


@MODELS.register_module()
class VisionTransformerAdapter(BaseModule):
    """Vision Transformer with support for patch or hybrid CNN input stage. An
    impl of `VideoMAE: Masked Autoencoders are Data-Efficient Learners for
    Self-Supervised Video Pre-Training <https://arxiv.org/pdf/2203.12602.pdf>`_

    We add the checkpointing and frozen stage to the original VisionTransformer.

    Args:
        img_size (int or tuple): Size of input image.
            Defaults to 224.
        patch_size (int): Spatial size of one patch. Defaults to 16.
        in_channels (int): The number of channels of he input.
            Defaults to 3.
        embed_dims (int): Dimensions of embedding. Defaults to 768.
        depth (int): number of blocks in the transformer.
            Defaults to 12.
        num_heads (int): Number of parallel attention heads in
            TransformerCoder. Defaults to 12.
        mlp_ratio (int): The ratio between the hidden layer and the
            input layer in the FFN. Defaults to 4.
        qkv_bias (bool): If True, add a learnable bias to q and v.
            Defaults to True.
        qk_scale (float, optional): Override default qk scale of
            ``head_dim ** -0.5`` if set. Defaults to None.
        drop_rate (float): Dropout ratio of output. Defaults to 0.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Defaults to 0.
        drop_path_rate (float): Dropout ratio of the residual branch.
            Defaults to 0.
        norm_cfg (dict or Configdict): Config for norm layers.
            Defaults to `dict(type='LN', eps=1e-6)`.
        num_frames (int): Number of frames in the video. Defaults to 16.
        tubelet_size (int): Temporal size of one patch. Defaults to 2.
        use_mean_pooling (bool): If True, take the mean pooling over all
            positions. Defaults to True.
        pretrained (str, optional): Name of pretrained model. Default: None.
        return_feat_map (bool): If True, return the feature in the shape of
            `[B, C, T, H, W]`. Defaults to False.
        init_cfg (dict or list[dict]): Initialization config dict. Defaults to
            ``[
            dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
            dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
            ]``.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dims: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: int = 4.0,
        qkv_bias: bool = True,
        qk_scale: int = None,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_cfg: ConfigType = dict(type="LN", eps=1e-6),
        num_frames: int = 16,  # frames per attention
        tubelet_size: int = 2,
        use_mean_pooling: int = True,
        pretrained: Optional[str] = None,
        return_feat_map: bool = False,
        with_cp: bool = False,
        adapter_mlp_ratio: float = 0.25,
        total_frames: int = 768,
        adapter_index: list = [3, 5, 7, 11],
        tubelet_token_redundancy_aux: Optional[Dict] = None,
        tubelet_packed_runtime_route: Optional[Dict] = None,
        chronotransport: Optional[Dict] = None,
        amod: Optional[Dict] = None,
        init_cfg: Optional[Union[Dict, List[Dict]]] = [
            dict(type="TruncNormal", layer="Linear", std=0.02, bias=0.0),
            dict(type="Constant", layer="LayerNorm", val=1.0, bias=0.0),
        ],
        **kwargs,
    ) -> None:
        if pretrained:
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        super().__init__(init_cfg=init_cfg)

        self.with_cp = with_cp

        self.embed_dims = embed_dims
        self.patch_size = patch_size
        self.latest_tubelet_token_redundancy_summary = None
        self.latest_tubelet_packed_runtime_summary = None
        self.latest_chronotransport_summary = None
        self.latest_native_packed_summary = None
        self.latest_amod_summary = None
        # Runtime evidence only.  GeoRoute records a before/after delta around
        # the actual packed call so P0 does not merely trust a hand-written
        # "one forward" field in a summary dictionary.
        self.native_packed_forward_invocations = 0
        self.native_ragged_forward_invocations = 0
        self.chronotransport_checkpoint_loaded = False
        self.chronotransport_allow_legacy_checkpoint = False

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type="Conv3d",
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
            padding=(0, 0, 0),
            dilation=(1, 1, 1),
        )

        grid_size = img_size // patch_size
        num_patches = grid_size**2 * (num_frames // tubelet_size)
        self.grid_size = (grid_size, grid_size)

        # sine-cosine positional embeddings
        pos_embed = get_sinusoid_encoding(num_patches, embed_dims)
        self.register_buffer("pos_embed", pos_embed)

        self.pos_drop = nn.Dropout(p=drop_rate)
        self.tubelet_token_redundancy_aux = None
        if tubelet_token_redundancy_aux is not None:
            self.tubelet_token_redundancy_aux = TubeletTokenRedundancyAux(**dict(tubelet_token_redundancy_aux))
        self.tubelet_packed_runtime_route = None
        if tubelet_packed_runtime_route is not None:
            self.tubelet_packed_runtime_route = PackedTubeletRuntimeRoute(**dict(tubelet_packed_runtime_route))

        self.chronotransport = None
        if chronotransport is not None:
            from ..chronotransport import ChronoTransportRuntime

            chronotransport_cfg = dict(chronotransport)
            self.chronotransport_allow_legacy_checkpoint = bool(
                chronotransport_cfg.pop("allow_legacy_checkpoint", False)
            )
            if int(total_frames) % int(num_frames) != 0:
                raise ValueError("ChronoTransport requires total_frames divisible by num_frames")
            expected = {
                "embed_dims": int(embed_dims),
                "depth": int(depth),
                "chunks_per_window": int(total_frames) // int(num_frames),
            }
            for key, value in expected.items():
                configured = chronotransport_cfg.pop(key, value)
                if int(configured) != int(value):
                    raise ValueError(f"ChronoTransport {key} must equal backbone value {value}")
            self.chronotransport = ChronoTransportRuntime(**expected, **chronotransport_cfg)
            self.register_load_state_dict_post_hook(self._chronotransport_load_state_dict_post_hook)

        packed_enabled = bool(
            self.tubelet_packed_runtime_route is not None
            and self.tubelet_packed_runtime_route.enabled
        )
        chronotransport_enabled = bool(
            self.chronotransport is not None and self.chronotransport.enabled
        )
        if packed_enabled and chronotransport_enabled:
            raise ValueError(
                "tubelet_packed_runtime_route and ChronoTransport are mutually exclusive"
            )

        self.amod_config = None
        if amod is not None and bool(dict(amod).get("enabled", True)):
            amod_cfg = dict(amod)
            expected_dense = (0, 2, 4, 6, 8, 10)
            expected_sparse = (1, 3, 5, 7, 9, 11)
            if int(depth) != 12:
                raise ValueError("paper-exact VideoMAE A-MoD requires 12 blocks")
            if float(attn_drop_rate) != 0.0:
                raise ValueError("paper-exact VideoMAE A-MoD requires zero attention dropout")
            if tubelet_token_redundancy_aux is not None or packed_enabled or chronotransport_enabled:
                raise ValueError(
                    "A-MoD is mutually exclusive with ROI/packed/ChronoTransport routes"
                )
            capacity = float(amod_cfg.get("capacity", 0.5))
            if not 0.0 < capacity <= 1.0:
                raise ValueError("A-MoD capacity must be within (0,1]")
            dense_blocks = tuple(amod_cfg.get("dense_block_indices", expected_dense))
            sparse_blocks = tuple(amod_cfg.get("amod_block_indices", expected_sparse))
            if dense_blocks != expected_dense or sparse_blocks != expected_sparse:
                raise ValueError("A-MoD must alternate dense-even and sparse-odd blocks")
            query_chunk_size = int(amod_cfg.get("query_chunk_size", 128))
            if query_chunk_size <= 0:
                raise ValueError("A-MoD query_chunk_size must be positive")
            if amod_cfg.get("routing_score", "preceding_dense_attention_column_mean") != (
                "preceding_dense_attention_column_mean"
            ):
                raise ValueError("A-MoD routing score must come from the preceding dense block")
            if amod_cfg.get("unselected_update", "identity_bypass") != "identity_bypass":
                raise ValueError("A-MoD unselected tokens must identity-bypass Attention+MLP")
            self.amod_config = {
                "schema_version": amod_cfg.get(
                    "schema_version", "zoomtoken_videomae_amod_paper_exact_v001"
                ),
                "capacity": capacity,
                "dense_block_indices": dense_blocks,
                "amod_block_indices": sparse_blocks,
                "query_chunk_size": query_chunk_size,
                "routing_score": "preceding_dense_attention_column_mean",
                "unselected_update": "identity_bypass",
            }

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = ModuleList(
            [
                Block(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=dpr[i],
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    init_cfg=init_cfg,
                    use_adapter=i in adapter_index,
                    adapter_mlp_ratio=adapter_mlp_ratio,
                    temporal_size=total_frames // tubelet_size,
                )
                for i in range(depth)
            ]
        )

        if use_mean_pooling:
            self.norm = nn.Identity()
            self.fc_norm = build_norm_layer(norm_cfg, embed_dims)[1]
        else:
            self.norm = build_norm_layer(norm_cfg, embed_dims)[1]
            self.fc_norm = None

        self.return_feat_map = return_feat_map

        num_vit_param = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "adapter" not in name and not name.startswith("chronotransport.")
        )
        num_adapter_param = sum(
            p.numel()
            for name, p in self.named_parameters()
            if "adapter" in name and not name.startswith("chronotransport.")
        )
        num_chronotransport_param = sum(
            p.numel() for name, p in self.named_parameters() if name.startswith("chronotransport.")
        )
        ratio = num_adapter_param / max(1, num_vit_param) * 100
        print(
            "ViT's param: {}, Adapter's params: {}, ChronoTransport params: {}, adapter ratio: {:2.1f}%".format(
                num_vit_param, num_adapter_param, num_chronotransport_param, ratio
            )
        )

    def _chronotransport_load_state_dict_post_hook(self, module, incompatible_keys) -> None:
        del module
        if self.chronotransport is None:
            return
        missing = [
            key for key in list(incompatible_keys.missing_keys) if "chronotransport." in key
        ]
        loaded = not missing
        self.chronotransport_checkpoint_loaded = loaded
        self.chronotransport.set_checkpoint_loaded(loaded)
        if missing and self.chronotransport_allow_legacy_checkpoint:
            for key in missing:
                incompatible_keys.missing_keys.remove(key)

    def _native_packed_position_embedding(self, grid_height: int, grid_width: int) -> Tensor:
        """Interpolate positional embeddings for a native source patch lattice."""

        if int(grid_height) <= 0 or int(grid_width) <= 0:
            raise ValueError("native packed grid dimensions must be positive")
        base_height, base_width = self.grid_size
        spatial_base = int(base_height) * int(base_width)
        if self.pos_embed.shape[1] % spatial_base:
            raise RuntimeError("VideoMAE positional embedding has an invalid spatial layout")
        temporal_tokens = int(self.pos_embed.shape[1]) // spatial_base
        pos_embed = self.pos_embed.reshape(1, temporal_tokens, base_height, base_width, self.embed_dims)
        pos_embed = pos_embed.permute(0, 1, 4, 2, 3).flatten(0, 1)
        if (grid_height, grid_width) != self.grid_size:
            pos_embed = F.interpolate(
                pos_embed,
                size=(int(grid_height), int(grid_width)),
                mode="bicubic",
                align_corners=False,
            )
        return (
            pos_embed.reshape(1, temporal_tokens, self.embed_dims, grid_height, grid_width)
            .permute(0, 1, 3, 4, 2)
            .flatten(1, 3)
        )

    def _prepare_native_packed_lattice(
        self,
        native_tubelets: Tensor,
        spatial_indices: Tensor,
        *,
        source_grid_hw: Tuple[int, int],
        use_absolute_position: bool = True,
    ) -> Tuple[Tensor, Tensor, Tensor, Dict[str, int]]:
        """Embed native selected tubelets and reconstruct their source lattice.

        Both the packed runtime and the P0 dense numerical reference call this
        helper.  Keeping their patch embedding, position table and scatter
        setup identical makes a full-token P0 comparison meaningful instead
        of comparing two different VideoMAE inputs.
        """

        if native_tubelets.ndim != 7:
            raise ValueError("native_tubelets must be [B,T,K,3,tubelet,patch,patch]")
        if spatial_indices.ndim != 3:
            raise ValueError("spatial_indices must be [B,T,K]")
        if tuple(native_tubelets.shape[:3]) != tuple(spatial_indices.shape):
            raise ValueError("native tubelets and spatial indices must share [B,T,K]")
        if native_tubelets.shape[3] != 3:
            raise ValueError("native packed input requires RGB tubelets")
        if native_tubelets.shape[4] != 2 or native_tubelets.shape[5:] != (self.patch_size, self.patch_size):
            raise ValueError("native packed input must use VideoMAE 2x16x16 tubelets")
        if self.with_cp:
            raise ValueError(
                "GeoRoute native packed execution requires with_cp=False so the one-heavy-forward ledger remains exact"
            )
        if self.tubelet_packed_runtime_route is not None and self.tubelet_packed_runtime_route.enabled:
            raise RuntimeError("native packed execution cannot combine with tubelet_packed_runtime_route")
        if self.chronotransport is not None and self.chronotransport.enabled:
            raise RuntimeError("native packed execution cannot combine with ChronoTransport")

        batch_size, total_tubelets, select_count = map(int, spatial_indices.shape)
        if select_count <= 0:
            raise ValueError("native packed execution requires at least one token per tubelet")
        grid_height, grid_width = map(int, source_grid_hw)
        spatial_tokens = grid_height * grid_width
        if spatial_tokens <= 0:
            raise ValueError("source_grid_hw must contain positive dimensions")
        if bool(((spatial_indices < 0) | (spatial_indices >= spatial_tokens)).any().item()):
            raise ValueError("native packed spatial indices fall outside the source patch lattice")
        if select_count > 1:
            ordered_indices = torch.sort(spatial_indices, dim=-1).values
            if bool((ordered_indices[..., 1:] == ordered_indices[..., :-1]).any().item()):
                raise ValueError("native packed execution received duplicate spatial indices")

        base_height, base_width = self.grid_size
        temporal_per_chunk = int(self.pos_embed.shape[1]) // (int(base_height) * int(base_width))
        if total_tubelets % temporal_per_chunk:
            raise ValueError("native packed tubelet count must be divisible by one VideoMAE clip")
        chunk_count = total_tubelets // temporal_per_chunk

        self._freeze_layers()
        embedded = self.patch_embed(
            native_tubelets.reshape(-1, 3, 2, self.patch_size, self.patch_size)
        )[0]
        if embedded.shape[1:] != (1, self.embed_dims):
            raise RuntimeError("native patch embedder did not produce one token per gathered tubelet")
        embedded = embedded.squeeze(1).reshape(
            batch_size,
            chunk_count,
            temporal_per_chunk,
            select_count,
            self.embed_dims,
        )
        per_chunk_indices = spatial_indices.reshape(
            batch_size,
            chunk_count,
            temporal_per_chunk,
            select_count,
        )

        if use_absolute_position:
            position = self._native_packed_position_embedding(grid_height, grid_width)
        else:
            # This is an explicit ablation, not a hidden coordinate fallback:
            # neither the dense lattice nor the selected tokens receive the
            # pretrained absolute spatial position table in this path.
            position = embedded.new_zeros(
                1,
                temporal_per_chunk * spatial_tokens,
                self.embed_dims,
            )
        dense = position.reshape(1, 1, temporal_per_chunk, spatial_tokens, self.embed_dims).expand(
            batch_size,
            chunk_count,
            temporal_per_chunk,
            spatial_tokens,
            self.embed_dims,
        ).clone()
        scatter_index = per_chunk_indices.unsqueeze(-1).expand_as(embedded)
        dense.scatter_(3, scatter_index, embedded + dense.gather(3, scatter_index))
        dense_mask = torch.zeros(
            batch_size,
            chunk_count,
            temporal_per_chunk,
            spatial_tokens,
            device=spatial_indices.device,
            dtype=torch.bool,
        ).scatter_(3, per_chunk_indices, True)
        x = self.pos_drop(dense.reshape(batch_size * chunk_count, -1, self.embed_dims))
        packed_mask = dense_mask.reshape(batch_size * chunk_count, -1)
        metadata = {
            "batch_size": batch_size,
            "total_tubelets": total_tubelets,
            "select_count": select_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "spatial_tokens": spatial_tokens,
            "chunk_count": chunk_count,
            "temporal_per_chunk": temporal_per_chunk,
            "absolute_position_enabled": int(bool(use_absolute_position)),
        }
        return x, packed_mask, scatter_index, metadata

    @staticmethod
    def _gather_native_selected_output(
        x: Tensor,
        scatter_index: Tensor,
        metadata: Dict[str, int],
    ) -> Tensor:
        dense_output = x.reshape(
            metadata["batch_size"],
            metadata["chunk_count"],
            metadata["temporal_per_chunk"],
            metadata["spatial_tokens"],
            x.shape[-1],
        )
        return dense_output.gather(3, scatter_index).reshape(
            metadata["batch_size"],
            metadata["total_tubelets"],
            metadata["select_count"],
            x.shape[-1],
        )

    def forward_native_packed(
        self,
        native_tubelets: Tensor,
        spatial_indices: Tensor,
        *,
        source_grid_hw: Tuple[int, int],
        use_absolute_position: bool = True,
    ) -> Tensor:
        """Run one VideoMAE pass on exact-K native ``2x16x16`` tubelets.

        The visual input is never resized or sampled by coordinates.  The
        gathered tubelets are embedded with the same pretrained Conv3d patch
        embedder, scattered into their native dense lattice, then the existing
        packed block path executes attention/MLP only for selected locations.
        Adapter convolutions follow the selected native spatial lineages, so
        attention, MLP and Adapter all bypass unselected carrier positions.
        """

        self.native_packed_forward_invocations += 1
        x, packed_mask, scatter_index, metadata = self._prepare_native_packed_lattice(
            native_tubelets,
            spatial_indices,
            source_grid_hw=source_grid_hw,
            use_absolute_position=use_absolute_position,
        )

        stats: Dict[str, int] = {
            "heavy_backbone_forward_count": 1,
            "packed_attention_forward_count": 0,
            "packed_mlp_forward_count": 0,
            "dense_adapter_forward_count": 0,
            "packed_adapter_forward_count": 0,
        }
        for block in self.blocks:
            x = block(
                x,
                metadata["grid_height"],
                metadata["grid_width"],
                packed_dense_mask=packed_mask,
                packed_spatial_indices=spatial_indices,
                packed_stats=stats,
            )
        x = self.norm(x)
        selected = self._gather_native_selected_output(x, scatter_index, metadata)
        self.latest_native_packed_summary = {
            "schema_version": "videomae_native_packed_v2",
            "heavy_backbone_forward_count": 1,
            "batch_size": metadata["batch_size"],
            "total_tubelets": metadata["total_tubelets"],
            "chunks_per_window": metadata["chunk_count"],
            "tubelets_per_chunk": metadata["temporal_per_chunk"],
            "source_grid_hw": [metadata["grid_height"], metadata["grid_width"]],
            "spatial_tokens_per_tubelet": metadata["spatial_tokens"],
            "selected_tokens_per_tubelet": metadata["select_count"],
            "absolute_position_enabled": bool(metadata["absolute_position_enabled"]),
            "selected_attention_tokens_per_chunk": metadata["temporal_per_chunk"] * metadata["select_count"],
            "dense_tokens_per_chunk": metadata["temporal_per_chunk"] * metadata["spatial_tokens"],
            "packed_attention_forward_count": stats["packed_attention_forward_count"],
            "packed_mlp_forward_count": stats["packed_mlp_forward_count"],
            "dense_adapter_forward_count": stats["dense_adapter_forward_count"],
            "packed_adapter_forward_count": stats["packed_adapter_forward_count"],
            "adapter_execution": "coordinate_lineage_packed",
        }
        return selected

    def _prepare_native_ragged_tokens(
        self,
        selected_native_tubelets: Tensor,
        physical_indices: Tensor,
        *,
        total_tubelets: int,
        source_grid_hw: Tuple[int, int],
        use_absolute_position: bool = True,
        refresh_mode: str = "full64",
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor], Dict[str, object]]:
        """Patch-embed and bucket one padding-free global physical-token union."""

        if selected_native_tubelets.ndim != 6:
            raise ValueError(
                "ragged native tubelets must be [B,S,3,tubelet,patch,patch]"
            )
        if physical_indices.ndim != 2:
            raise ValueError("ragged physical indices must be [B,S]")
        if tuple(selected_native_tubelets.shape[:2]) != tuple(physical_indices.shape):
            raise ValueError("ragged native tubelets and indices must share [B,S]")
        if physical_indices.dtype != torch.long:
            raise TypeError("ragged physical indices must be torch.long")
        if selected_native_tubelets.shape[2] != 3:
            raise ValueError("ragged native input requires RGB tubelets")
        if (
            selected_native_tubelets.shape[3] != 2
            or selected_native_tubelets.shape[4:]
            != (self.patch_size, self.patch_size)
        ):
            raise ValueError("ragged native input must use VideoMAE 2x16x16 tubelets")
        if (
            self.tubelet_packed_runtime_route is not None
            and self.tubelet_packed_runtime_route.enabled
        ):
            raise RuntimeError(
                "native ragged execution cannot combine with tubelet_packed_runtime_route"
            )
        if self.chronotransport is not None and self.chronotransport.enabled:
            raise RuntimeError(
                "native ragged execution cannot combine with ChronoTransport"
            )
        temporal_alignment_modes = {
            "apm32_ctx64",
            "cur32_ctx64",
            "apm_c32_full64",
        }
        if (
            refresh_mode in temporal_alignment_modes
            and self.amod_config is not None
        ):
            raise RuntimeError("APM32/CUR32/FULL64 cannot combine with strict A-MoD")

        batch_size, selected_count = map(int, physical_indices.shape)
        if selected_count <= 0:
            raise ValueError("native ragged execution requires a positive window budget")
        grid_height, grid_width = map(int, source_grid_hw)
        spatial_tokens = grid_height * grid_width
        if spatial_tokens <= 0:
            raise ValueError("source_grid_hw must contain positive dimensions")
        physical_capacity = int(total_tubelets) * spatial_tokens
        if bool(
            ((physical_indices < 0) | (physical_indices >= physical_capacity))
            .any()
            .item()
        ):
            raise ValueError("ragged physical index falls outside the source lattice")
        if selected_count > 1 and not bool(
            (physical_indices[:, 1:] > physical_indices[:, :-1]).all().item()
        ):
            raise ValueError(
                "ragged physical indices must be strictly increasing and unique"
            )

        base_height, base_width = self.grid_size
        temporal_per_chunk = int(self.pos_embed.shape[1]) // (
            int(base_height) * int(base_width)
        )
        if int(total_tubelets) <= 0 or int(total_tubelets) % temporal_per_chunk:
            raise ValueError(
                "ragged total tubelets must be divisible by one VideoMAE clip"
            )
        chunk_count = int(total_tubelets) // temporal_per_chunk
        tubelet_indices = torch.div(
            physical_indices,
            spatial_tokens,
            rounding_mode="floor",
        )
        spatial_indices = physical_indices.remainder(spatial_tokens)
        clip_indices = torch.div(
            tubelet_indices,
            temporal_per_chunk,
            rounding_mode="floor",
        )
        local_tubelet_indices = tubelet_indices.remainder(temporal_per_chunk)

        self._freeze_layers()
        embedded = self.patch_embed(
            selected_native_tubelets.reshape(
                -1,
                3,
                2,
                self.patch_size,
                self.patch_size,
            )
        )[0]
        if embedded.shape[1:] != (1, self.embed_dims):
            raise RuntimeError(
                "native patch embedder did not produce one token per ragged tubelet"
            )
        embedded = embedded.squeeze(1).reshape(
            batch_size,
            selected_count,
            self.embed_dims,
        )
        temporal_plan = None
        carrier = embedded
        if refresh_mode in temporal_alignment_modes:
            if selected_count != int(total_tubelets) * 64:
                raise ValueError(
                    "APM temporal modes require exact K64 support for every tubelet"
                )
            expected_tubelets = torch.arange(
                int(total_tubelets),
                device=tubelet_indices.device,
                dtype=torch.long,
            ).repeat_interleave(64).view(1, -1).expand(batch_size, -1)
            if not torch.equal(tubelet_indices, expected_tubelets):
                raise ValueError(
                    "APM temporal modes require exactly 64 ordered tokens per tubelet"
                )
            temporal_plan = build_apm32_temporal_plan(
                embedded.reshape(
                    batch_size,
                    int(total_tubelets),
                    64,
                    self.embed_dims,
                ),
                spatial_indices.reshape(batch_size, int(total_tubelets), 64),
                grid_height=grid_height,
                grid_width=grid_width,
            )
            if refresh_mode in {"apm32_ctx64", "apm_c32_full64"}:
                plan_previous = temporal_plan["matched_previous_slot"]
                plan_retained = temporal_plan["retained_mask"]
                plan_alpha = temporal_plan["alpha"]
                if not all(
                    isinstance(value, torch.Tensor)
                    for value in (plan_previous, plan_retained, plan_alpha)
                ):
                    raise RuntimeError("APM32 plan lost its tensor carrier fields")
                embedded_lattice = embedded.reshape(
                    batch_size,
                    int(total_tubelets),
                    64,
                    self.embed_dims,
                )
                previous_lattice = embedded_lattice[:, :-1]
                current_lattice = embedded_lattice[:, 1:]
                previous = previous_lattice.gather(
                    2,
                    plan_previous[:, 1:].clamp_min(0).unsqueeze(-1).expand(
                        -1,
                        -1,
                        -1,
                        self.embed_dims,
                    ),
                ).detach()
                mixed = previous + plan_alpha[:, 1:].to(embedded.dtype).unsqueeze(
                    -1
                ) * (
                    current_lattice - previous
                )
                carrier_tail = torch.where(
                    plan_retained[:, 1:].unsqueeze(-1),
                    mixed,
                    current_lattice,
                )
                carrier = torch.cat(
                    (embedded_lattice[:, :1], carrier_tail),
                    dim=1,
                ).reshape_as(embedded)
        if use_absolute_position:
            position = self._native_packed_position_embedding(
                grid_height,
                grid_width,
            ).reshape(
                1,
                temporal_per_chunk,
                spatial_tokens,
                self.embed_dims,
            )
            position = position.expand(batch_size, -1, -1, -1)
            batch_indices = torch.arange(
                batch_size,
                device=physical_indices.device,
                dtype=torch.long,
            ).view(batch_size, 1).expand(batch_size, selected_count)
            selected_position = position[
                batch_indices,
                local_tubelet_indices,
                spatial_indices,
            ]
        else:
            selected_position = embedded.new_zeros(embedded.shape)
        x = self.pos_drop(carrier + selected_position)

        clip_counts = torch.zeros(
            (batch_size, chunk_count),
            device=physical_indices.device,
            dtype=torch.long,
        ).scatter_add_(
            1,
            clip_indices,
            torch.ones_like(clip_indices),
        )
        if not bool((clip_counts.sum(dim=-1) == selected_count).all().item()):
            raise RuntimeError("ragged clip ledger omitted a selected token")
        flat_counts = clip_counts.reshape(-1)
        clip_offsets = torch.cat(
            (
                torch.zeros(1, device=physical_indices.device, dtype=torch.long),
                flat_counts.cumsum(dim=0),
            )
        )
        bucket_positions: List[Tensor] = []
        bucket_layout: List[Dict[str, int]] = []
        positive_lengths = torch.unique(flat_counts[flat_counts > 0], sorted=True)
        for length_tensor in positive_lengths:
            length = int(length_tensor.item())
            rows = torch.nonzero(flat_counts == length, as_tuple=False).flatten()
            positions = clip_offsets.index_select(0, rows).unsqueeze(-1) + torch.arange(
                length,
                device=physical_indices.device,
                dtype=torch.long,
            ).view(1, length)
            bucket_positions.append(positions)
            bucket_layout.append(
                {
                    "tokens_per_clip": length,
                    "clip_count": int(rows.numel()),
                }
            )
        if not bucket_positions:
            raise RuntimeError("positive window budget produced no non-empty clip")
        covered = sum(int(value.numel()) for value in bucket_positions)
        if covered != batch_size * selected_count:
            raise RuntimeError("ragged clip buckets do not cover exact selected B")

        attention_pairs_per_window = clip_counts.square().sum(dim=-1)
        metadata: Dict[str, object] = {
            "batch_size": batch_size,
            "total_tubelets": int(total_tubelets),
            "window_budget": selected_count,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "spatial_tokens": spatial_tokens,
            "chunk_count": chunk_count,
            "temporal_per_chunk": temporal_per_chunk,
            "absolute_position_enabled": bool(use_absolute_position),
            "clip_indices": clip_indices,
            "clip_counts": clip_counts,
            "attention_pairs_per_window": attention_pairs_per_window,
            "bucket_layout": bucket_layout,
        }
        if temporal_plan is not None:
            refresh_tensor = temporal_plan["refresh_mask"]
            fallback_tensor = temporal_plan["fallback_mask"]
            forced_first_tensor = temporal_plan["forced_first_mask"]
            matched_tensor = temporal_plan["matched_mask"]
            retained_tensor = temporal_plan["retained_mask"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (
                    refresh_tensor,
                    fallback_tensor,
                    forced_first_tensor,
                    matched_tensor,
                    retained_tensor,
                )
            ):
                raise RuntimeError("APM32 plan lost its tensor ledger")
            metadata["temporal_refresh_mask"] = refresh_tensor.reshape(
                batch_size,
                selected_count,
            )
            metadata["temporal_carrier_mask"] = retained_tensor.reshape(
                batch_size,
                selected_count,
            )
            metadata["temporal_alignment_ledger"] = {
                "schema_version": temporal_plan["schema_version"],
                "carrier_mode": refresh_mode,
                "memory_tensor": "pre_position_patch_embedding",
                "memory_lifetime_tubelets": 1,
                "clip_reset_tubelets": int(temporal_plan["clip_tubelets"]),
                "similarity_threshold": float(
                    temporal_plan["similarity_threshold"]
                ),
                "search_radius": int(temporal_plan["search_radius"]),
                "matched_tokens": int(matched_tensor.sum().item()),
                "retained_tokens": int(retained_tensor.sum().item()),
                "refreshed_tokens": int(refresh_tensor.sum().item()),
                "fallback_tubelets": int(fallback_tensor.sum().item()),
                "forced_first_tubelets": int(forced_first_tensor.sum().item()),
                "normal_tubelets": int((~fallback_tensor).sum().item()),
                "total_tubelets": int(fallback_tensor.numel()),
                "new_trainable_parameters": 0,
                "previous_memory_detached": True,
                "current_position_restored": True,
                "future_tubelet_access": False,
            }
        return (
            x,
            tubelet_indices,
            spatial_indices,
            bucket_positions,
            metadata,
        )

    def forward_native_ragged(
        self,
        selected_native_tubelets: Tensor,
        physical_indices: Tensor,
        *,
        total_tubelets: int,
        source_grid_hw: Tuple[int, int],
        use_absolute_position: bool = True,
        refresh_mask: Optional[Tensor] = None,
        refresh_mode: str = "full64",
        refresh_alpha: Optional[Tensor] = None,
    ) -> Tensor:
        """Execute one true clip-ragged VideoMAE pass with zero dummy tokens."""

        self.native_ragged_forward_invocations += 1
        (
            x,
            tubelet_indices,
            spatial_indices,
            bucket_positions,
            metadata,
        ) = self._prepare_native_ragged_tokens(
            selected_native_tubelets,
            physical_indices,
            total_tubelets=total_tubelets,
            source_grid_hw=source_grid_hw,
            use_absolute_position=use_absolute_position,
            refresh_mode=refresh_mode,
        )
        batch_size = int(metadata["batch_size"])
        window_budget = int(metadata["window_budget"])
        selected_total = batch_size * window_budget
        temporal_modes = {"apm32_ctx64", "cur32_ctx64", "apm_c32_full64"}
        temporal_sparse_modes = {"apm32_ctx64", "cur32_ctx64"}
        temporal_carrier_mask = None
        if refresh_mode in temporal_modes:
            if refresh_mask is not None or refresh_alpha is not None:
                raise ValueError(
                    "APM temporal modes derive their frozen alignment mask internally"
                )
            temporal_refresh_mask = metadata.get("temporal_refresh_mask")
            temporal_carrier_mask = metadata.get("temporal_carrier_mask")
            if not isinstance(temporal_refresh_mask, torch.Tensor) or not isinstance(
                temporal_carrier_mask,
                torch.Tensor,
            ):
                raise RuntimeError("APM temporal alignment plan is missing")
            if refresh_mode in temporal_sparse_modes:
                refresh_mask = temporal_refresh_mask
        if refresh_mode not in {
            "full64",
            "drop32",
            "mod32_kv",
            "rc32_kv",
            "dsr6_kv",
            "apm32_ctx64",
            "cur32_ctx64",
            "apm_c32_full64",
        }:
            raise ValueError("unsupported ZoomToken refresh execution mode")
        if refresh_mode in {"full64", "drop32", "apm_c32_full64"}:
            if refresh_mask is not None or refresh_alpha is not None:
                raise ValueError("FULL64 execution must use the ordinary ragged path")
            if refresh_mode == "apm_c32_full64":
                support_counts = torch.zeros(
                    (batch_size, int(metadata["total_tubelets"])),
                    device=x.device,
                    dtype=torch.long,
                ).scatter_add_(1, tubelet_indices, torch.ones_like(tubelet_indices))
                carrier_counts = torch.zeros_like(support_counts).scatter_add_(
                    1,
                    tubelet_indices,
                    temporal_carrier_mask.to(torch.long),
                )
                if not bool((support_counts == 64).all().item()):
                    raise ValueError("APM-C32/FULL64 requires exact K64 support")
                if not bool(((carrier_counts == 0) | (carrier_counts == 32)).all().item()):
                    raise ValueError(
                        "APM-C32/FULL64 requires C32 memory or current-only fallback"
                    )
        else:
            if (
                refresh_mask is None
                or refresh_mask.dtype != torch.bool
                or refresh_mask.shape != x.shape[:2]
            ):
                raise ValueError(
                    "refresh-KV arms require one bool refresh mask over K64"
                )
            support_counts = torch.zeros(
                (batch_size, int(metadata["total_tubelets"])),
                device=x.device,
                dtype=torch.long,
            ).scatter_add_(1, tubelet_indices, torch.ones_like(tubelet_indices))
            refresh_counts = torch.zeros_like(support_counts).scatter_add_(
                1,
                tubelet_indices,
                refresh_mask.to(torch.long),
            )
            if not bool((support_counts == 64).all().item()):
                raise ValueError(
                    "refresh-KV arms require exact K64 support per tubelet"
                )
            if refresh_mode in temporal_modes:
                if not bool(
                    ((refresh_counts == 32) | (refresh_counts == 64)).all().item()
                ):
                    raise ValueError(
                        "APM32/CUR32 require K32 refresh or K64 fallback per tubelet"
                    )
            elif not bool((refresh_counts == 32).all().item()):
                raise ValueError(
                    "MOD32/RC32/DSR6 require exact K32 refresh per tubelet"
                )
            if refresh_mode == "mod32_kv" and refresh_alpha is not None:
                raise ValueError("MOD32-KV has no temporal carry parameter")
            if refresh_mode == "dsr6_kv" and refresh_alpha is not None:
                raise ValueError("DSR6-KV has no temporal carry parameter")
            if refresh_mode == "rc32_kv" and (
                refresh_alpha is None
                or refresh_alpha.ndim != 1
                or int(refresh_alpha.numel()) != len(self.blocks)
            ):
                raise ValueError("RC32-KV requires one scalar carry mix per block")
        stats: Dict[str, int] = {
            "heavy_backbone_forward_count": 1,
            "executed_patch_tokens": selected_total,
            "ragged_attention_bucket_call_count": 0,
            "ragged_mlp_bucket_call_count": 0,
            "ragged_adapter_forward_count": 0,
            "executed_attention_tokens": 0,
            "executed_kv_tokens": 0,
            "executed_attention_pairs": 0,
            "executed_mlp_tokens": 0,
            "executed_adapter_tokens": 0,
            "dense_adapter_forward_count": 0,
        }
        if refresh_mode == "dsr6_kv" and len(self.blocks) != 12:
            raise ValueError("DSR6-KV requires the frozen 12-block VideoMAE-S")
        for block_index, block in enumerate(self.blocks):
            block_refresh_mask = refresh_mask
            block_refresh_mode = refresh_mode
            block_refresh_alpha = (
                refresh_alpha[block_index]
                if refresh_mode == "rc32_kv"
                else None
            )
            if refresh_mode == "dsr6_kv":
                if block_index < 6:
                    block_refresh_mask = None
                    block_refresh_mode = "full64"
                else:
                    block_refresh_mode = "mod32_kv"
            elif refresh_mode in temporal_sparse_modes:
                block_refresh_mode = "mod32_kv"
            x = block.forward_native_ragged(
                x,
                bucket_positions=bucket_positions,
                tubelet_indices=tubelet_indices,
                spatial_indices=spatial_indices,
                total_tubelets=int(metadata["total_tubelets"]),
                grid_height=int(metadata["grid_height"]),
                grid_width=int(metadata["grid_width"]),
                packed_stats=stats,
                refresh_mask=block_refresh_mask,
                refresh_mode=block_refresh_mode,
                refresh_alpha=block_refresh_alpha,
            )
        x = self.norm(x)

        clip_counts = metadata["clip_counts"]
        attention_pairs_per_window = metadata["attention_pairs_per_window"]
        clip_indices = metadata["clip_indices"]
        if not isinstance(clip_counts, torch.Tensor) or not isinstance(
            attention_pairs_per_window,
            torch.Tensor,
        ) or not isinstance(clip_indices, torch.Tensor):
            raise RuntimeError("ragged execution metadata lost its tensor ledger")
        full_attention_pairs = int(attention_pairs_per_window.sum().item())
        if refresh_mask is None:
            expected_attention_pairs = full_attention_pairs * len(self.blocks)
            expected_kv_tokens = selected_total * len(self.blocks)
            refresh_tokens_per_window = window_budget
        else:
            refresh_clip_counts = torch.zeros_like(clip_counts).scatter_add_(
                1,
                clip_indices,
                refresh_mask.to(torch.long),
            )
            refresh_attention_pairs = int(
                (clip_counts * refresh_clip_counts).sum().item()
            )
            if refresh_mode == "dsr6_kv":
                expected_attention_pairs = (
                    full_attention_pairs * 6 + refresh_attention_pairs * 6
                )
                expected_kv_tokens = selected_total * 6
            else:
                expected_attention_pairs = (
                    refresh_attention_pairs * len(self.blocks)
                )
                expected_kv_tokens = selected_total * len(self.blocks)
            refresh_tokens_by_batch = refresh_mask.sum(dim=1)
            refresh_tokens_per_window = int(refresh_tokens_by_batch[0].item())
        if stats["executed_patch_tokens"] != selected_total:
            raise RuntimeError("ragged patch execution count differs from selected B")
        if stats["executed_attention_pairs"] != expected_attention_pairs:
            raise RuntimeError("ragged attention-pair ledger differs from execution")
        if refresh_mask is None:
            kv_ledger_valid = stats["executed_kv_tokens"] in {
                0,
                expected_kv_tokens,
            }
        else:
            kv_ledger_valid = stats["executed_kv_tokens"] == expected_kv_tokens
        if not kv_ledger_valid:
            raise RuntimeError("ragged KV-token ledger differs from execution")
        depth_schedule_summary = (
            {
                "full_update_block_count": 6,
                "refresh_update_block_count": 6,
            }
            if refresh_mode == "dsr6_kv"
            else {}
        )
        self.latest_native_packed_summary = {
            "schema_version": "videomae_native_ragged_v1",
            "execution_mode": "true_clip_ragged_no_padding",
            "refresh_execution_mode": refresh_mode,
            **depth_schedule_summary,
            "heavy_backbone_forward_count": 1,
            "batch_size": batch_size,
            "total_tubelets": int(metadata["total_tubelets"]),
            "chunks_per_window": int(metadata["chunk_count"]),
            "tubelets_per_chunk": int(metadata["temporal_per_chunk"]),
            "source_grid_hw": [
                int(metadata["grid_height"]),
                int(metadata["grid_width"]),
            ],
            "spatial_tokens_per_tubelet": int(metadata["spatial_tokens"]),
            "window_token_budget": window_budget,
            "requested_physical_tokens_per_window": window_budget,
            "unique_physical_tokens_per_window": window_budget,
            "padded_heavy_tokens_per_window": 0,
            "executed_patch_tokens_per_window": window_budget,
            "executed_patch_tokens_total": selected_total,
            "refresh_query_tokens_per_window": refresh_tokens_per_window,
            "refresh_query_tokens_per_window_by_batch": (
                refresh_mask.sum(dim=1).detach().cpu().tolist()
                if refresh_mask is not None
                else [window_budget] * batch_size
            ),
            "kv_context_tokens_per_window": window_budget,
            "absolute_position_enabled": bool(
                metadata["absolute_position_enabled"]
            ),
            "clip_token_counts": clip_counts.detach().cpu().tolist(),
            "empty_clip_count": int((clip_counts == 0).sum().item()),
            "attention_pairs_per_window": (
                attention_pairs_per_window.detach().cpu().tolist()
            ),
            "attention_pairs_all_blocks": stats["executed_attention_pairs"],
            "ragged_bucket_layout": list(metadata["bucket_layout"]),
            "ragged_buckets_per_block": len(bucket_positions),
            "ragged_attention_bucket_call_count": stats[
                "ragged_attention_bucket_call_count"
            ],
            "ragged_mlp_bucket_call_count": stats[
                "ragged_mlp_bucket_call_count"
            ],
            "ragged_adapter_forward_count": stats[
                "ragged_adapter_forward_count"
            ],
            "executed_attention_tokens_all_blocks": stats[
                "executed_attention_tokens"
            ],
            "executed_kv_tokens_all_blocks": stats["executed_kv_tokens"],
            "executed_mlp_tokens_all_blocks": stats["executed_mlp_tokens"],
            "executed_adapter_tokens_all_blocks": stats[
                "executed_adapter_tokens"
            ],
            "dense_adapter_forward_count": 0,
            "adapter_execution": "coordinate_lineage_true_ragged",
        }
        temporal_alignment_ledger = metadata.get("temporal_alignment_ledger")
        if temporal_alignment_ledger is not None:
            if not isinstance(temporal_alignment_ledger, dict):
                raise RuntimeError("APM32/CUR32 temporal ledger is malformed")
            self.latest_native_packed_summary["temporal_alignment"] = dict(
                temporal_alignment_ledger
            )
            if isinstance(temporal_carrier_mask, torch.Tensor):
                self.latest_native_packed_summary[
                    "memory_carrier_tokens_per_window_by_batch"
                ] = temporal_carrier_mask.sum(dim=1).detach().cpu().tolist()
        return x

    def forward_native_dense_reference(
        self,
        native_tubelets: Tensor,
        spatial_indices: Tensor,
        *,
        source_grid_hw: Tuple[int, int],
        use_absolute_position: bool = True,
    ) -> Tensor:
        """P0-only dense reference for an all-native-token packed route.

        This method intentionally performs a second *debug* forward and must
        never be enabled in a train/eval result cell.  It runs the same native
        lattice preparation as :meth:`forward_native_packed`, but executes the
        ordinary dense attention/MLP branch.  Its only purpose is to establish
        numerical agreement when every native patch is selected.
        """

        x, packed_mask, scatter_index, metadata = self._prepare_native_packed_lattice(
            native_tubelets,
            spatial_indices,
            source_grid_hw=source_grid_hw,
            use_absolute_position=use_absolute_position,
        )
        if not bool(packed_mask.all().item()):
            raise ValueError("dense native reference requires every source patch to be selected")
        self._freeze_layers()
        for block in self.blocks:
            x = block(x, metadata["grid_height"], metadata["grid_width"])
        x = self.norm(x)
        return self._gather_native_selected_output(x, scatter_index, metadata)

    def forward(self, x: Tensor) -> Tensor:
        """Defines the computation performed at every call.

        Args:
            x (Tensor): The input data.
        Returns:
            Tensor: The feature of the input
                samples extracted by the backbone.
        """
        self._freeze_layers()

        b, _, _, h, w = x.shape
        h //= self.patch_size
        w //= self.patch_size
        x = self.patch_embed(x)[0]
        if self.tubelet_token_redundancy_aux is not None:
            x = self.tubelet_token_redundancy_aux(x, h, w)
            self.latest_tubelet_token_redundancy_summary = self.tubelet_token_redundancy_aux.last_summary

        if (h, w) != self.grid_size:
            pos_embed = self.pos_embed.reshape(-1, *self.grid_size, self.embed_dims)
            pos_embed = pos_embed.permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).flatten(1, 2)
            pos_embed = pos_embed.reshape(1, -1, self.embed_dims)
        else:
            pos_embed = self.pos_embed

        x = x + pos_embed
        x = self.pos_drop(x)

        packed_enabled = bool(
            self.tubelet_packed_runtime_route is not None
            and self.tubelet_packed_runtime_route.enabled
        )
        chronotransport_enabled = bool(
            self.chronotransport is not None and self.chronotransport.enabled
        )
        amod_enabled = self.amod_config is not None
        if sum((packed_enabled, chronotransport_enabled, amod_enabled)) > 1:
            raise RuntimeError(
                "A-MoD, tubelet_packed_runtime_route, and ChronoTransport are mutually exclusive"
            )
        if amod_enabled:
            previous_dense_score = None
            selected_count = None
            dense_blocks = self.amod_config["dense_block_indices"]
            sparse_blocks = self.amod_config["amod_block_indices"]
            for block_index, block in enumerate(self.blocks):
                if block_index in dense_blocks:
                    x, previous_dense_score = block.forward_dense_with_amod_score(
                        x,
                        h,
                        w,
                        query_chunk_size=self.amod_config["query_chunk_size"],
                    )
                elif block_index in sparse_blocks:
                    if previous_dense_score is None:
                        raise RuntimeError("A-MoD block has no preceding dense attention score")
                    x, selected_indices = block.forward_amod(
                        x,
                        h,
                        w,
                        scores=previous_dense_score,
                        capacity=self.amod_config["capacity"],
                    )
                    selected_count = int(selected_indices.shape[1])
                    previous_dense_score = None
                else:
                    raise RuntimeError("A-MoD schedule omitted a VideoMAE block")
            self.latest_amod_summary = {
                "schema_version": self.amod_config["schema_version"],
                "token_count": int(x.shape[1]),
                "capacity": self.amod_config["capacity"],
                "selected_tokens_per_amod_block": selected_count,
                "dense_block_indices": list(dense_blocks),
                "amod_block_indices": list(sparse_blocks),
                "routing_score": self.amod_config["routing_score"],
                "unselected_update": self.amod_config["unselected_update"],
                "adapter_execution": "dense_full_token_grid",
                "temporal_state_reuse": False,
                "metric_claim_allowed": False,
                "cost_claim_allowed": False,
            }
            self.latest_tubelet_packed_runtime_summary = None
            self.latest_chronotransport_summary = None
        elif chronotransport_enabled:
            x = self.chronotransport(x, self.blocks, h, w)
            summary = dict(self.chronotransport.latest_summary or {})
            summary["checkpoint_loaded"] = self.chronotransport_checkpoint_loaded
            summary["legacy_checkpoint_allowed"] = self.chronotransport_allow_legacy_checkpoint
            self.latest_chronotransport_summary = summary
            self.latest_tubelet_packed_runtime_summary = None
        elif packed_enabled:
            x = self.tubelet_packed_runtime_route(x, self.blocks, h, w, training=self.training)
            self.latest_tubelet_packed_runtime_summary = self.tubelet_packed_runtime_route.last_summary
            self.latest_chronotransport_summary = None
        else:
            self.latest_amod_summary = None
            self.latest_tubelet_packed_runtime_summary = None
            self.latest_chronotransport_summary = None
            for blk in self.blocks:
                x = blk(x, h, w)

        x = self.norm(x)

        if self.return_feat_map:
            x = x.reshape(b, -1, h, w, self.embed_dims)
            x = x.permute(0, 4, 1, 2, 3)
            return x

        if self.fc_norm is not None:
            return self.fc_norm(x.mean(1))

        return x[:, 0]

    def _freeze_layers(self):
        """Prevent all the parameters not in the adapters"""

        # freeze patch_embed
        self.patch_embed.eval()
        for m in self.patch_embed.modules():
            for param in m.parameters():
                param.requires_grad = False

        # freeze blocks except the adapter's parameters
        for block in self.blocks:
            for m, n in block.named_children():
                if "adapter" not in m and m != "drop_path":
                    n.eval()
                    for param in n.parameters():
                        param.requires_grad = False
