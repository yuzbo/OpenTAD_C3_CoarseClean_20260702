from typing import Optional, Union, Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from ..builder import MODELS
from .deform_conv1d import deform_conv1d


def inverse_piecewise_linear_1d(
    timestamps: Tensor,  # [B, T_in] non-decreasing
    target_times: Tensor,  # [B, K, T_out]
) -> Tensor:  # Returns fractional token indices [B, K, T_out]
    """Maps physical target timestamps to continuous fractional token coordinates via piecewise linear interpolation.

    Boundary handling:
    Targets beyond the first/last endpoint are linearly extrapolated along the first/last interval rate,
    producing fractional coordinates (<0 or >T_in-1) that cleanly map into torchvision deform_conv2d zero-padding domain.
    """
    B, T_in = timestamps.shape
    if T_in < 2:
        raise ValueError(f"timestamps must have at least 2 points along temporal dimension, got {T_in}")
    device = timestamps.device
    dtype = timestamps.dtype

    B, K, T_out = target_times.shape
    flat_targets = target_times.reshape(B, -1)  # [B, K*T_out]

    # Search for right bracket index: timestamps[b, idx-1] <= target < timestamps[b, idx]
    right_idx = torch.searchsorted(timestamps.contiguous(), flat_targets.contiguous(), right=True)
    right_idx = right_idx.clamp(1, T_in - 1)
    left_idx = right_idx - 1

    t_left = torch.gather(timestamps, 1, left_idx)
    t_right = torch.gather(timestamps, 1, right_idx)
    dt = (t_right - t_left).clamp_min(1e-6)

    alpha = (flat_targets - t_left) / dt
    fractional_idx = left_idx.to(dtype=dtype) + alpha.to(dtype=dtype)

    return fractional_idx.reshape(B, K, T_out)


@MODELS.register_module()
class ContinuousTimeScaleAdaptiveConv1d(nn.Module):
    """Continuous-Time Scale-Adaptive 1D Convolution.

    Maintains constant physical temporal receptive field across non-uniformly sampled frames.
    Given discrete feature points with physical timestamps, calculates exact fractional offset:
        offset_{j,i} = u_{j,i}^fractional - u_{j,i}^nominal
    where torchvision kernel slot i in {0, ..., K-1} has nominal input position:
        u_{j,i}^nominal = j * stride - padding + i * dilation
    and physical center corresponds to slot i = half_k:
        center(j) = j * stride - padding + half_k * dilation
    with relative tap offset tap_i = i - half_k in {-half_k, ..., +half_k}:
        tau_target(j, i) = tau(center(j)) + tap_i * dilation * ref_delta_t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        ref_delta_t: float = 1.0,
        enable_learned_modulation: bool = False,
        offset_hidden_dim: int = 16,
    ):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd for symmetric physical receptive field")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.ref_delta_t = float(ref_delta_t)
        self.enable_learned_modulation = bool(enable_learned_modulation)

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter("bias", None)

        if self.enable_learned_modulation:
            self.offset_net = nn.Sequential(
                nn.Linear(1, offset_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(offset_hidden_dim, kernel_size),
                nn.Tanh(),  # Bounded learned residual offset
            )
        else:
            self.offset_net = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        if self.offset_net is not None:
            nn.init.zeros_(self.offset_net[-2].weight)
            nn.init.zeros_(self.offset_net[-2].bias)

    def forward(
        self,
        x: Tensor,
        delta_t: Optional[Tensor] = None,
        temporal_positions: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Forward pass with continuous-time geometry scale adaptation.

        Args:
            x: Input feature tensor [B, C, T_in].
            delta_t: Optional per-position physical time interval [B, T_in].
            temporal_positions: Optional physical timestamps [B, T_in].
            mask: Optional boolean valid mask [B, T_in].
        """
        B, C, T_in = x.shape

        # Calculate exact convolution output length
        T_out = int(math.floor((T_in + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) / self.stride) + 1)
        device = x.device
        dtype = x.dtype

        half_k = self.kernel_size // 2
        slots = torch.arange(self.kernel_size, device=device, dtype=dtype)  # [K]: 0, 1, ..., K-1
        taps = slots - half_k  # [K]: -half_k, ..., +half_k
        out_indices = torch.arange(T_out, device=device, dtype=dtype)  # [T_out]

        # Nominal input index for torchvision deform_conv2d kernel slot i: j * stride - padding + i * dilation
        nominal_indices = (
            out_indices[None, None, :] * float(self.stride)
            - float(self.padding)
            + slots[None, :, None] * float(self.dilation)
        )  # [1, K, T_out]

        # Center input index for each output step j (corresponding to slot i = half_k)
        center_input_idx = (
            out_indices * float(self.stride)
            - float(self.padding)
            + float(half_k * self.dilation)
        ).clamp(0, T_in - 1).long()  # [T_out]

        if temporal_positions is not None:
            # Center physical timestamp for each output step: [B, 1, T_out]
            center_tau = torch.gather(temporal_positions, 1, center_input_idx.unsqueeze(0).expand(B, -1)).unsqueeze(1)
            # Target physical timestamp for tap i: tau_target(j, i) = center_tau(j) + tap_i * dilation * ref_delta_t
            target_tau = center_tau + taps[None, :, None] * float(self.dilation) * self.ref_delta_t  # [B, K, T_out]

            # Fractional input token index from inverse piecewise linear mapping
            fractional_idx = inverse_piecewise_linear_1d(temporal_positions, target_tau)  # [B, K, T_out]
            geom_offset = fractional_idx - nominal_indices  # [B, K, T_out]
        else:
            if delta_t is None:
                delta_t = torch.full((B, T_in), self.ref_delta_t, device=device, dtype=dtype)

            # Sample delta_t at output step centers
            delta_t_out = torch.gather(delta_t, 1, center_input_idx.unsqueeze(0).expand(B, -1))  # [B, T_out]

            # Scale factor = ref_delta_t / delta_t
            scale = (self.ref_delta_t / delta_t_out.clamp_min(1e-4)).clamp(0.1, 10.0)  # [B, T_out]
            geom_offset = taps[None, :, None] * float(self.dilation) * (scale[:, None, :] - 1.0)  # [B, K, T_out]

        if self.offset_net is not None:
            # Learned bounded residual offset
            effective_scale = (geom_offset.abs().mean(dim=1, keepdim=True) + 1.0).permute(0, 2, 1)  # [B, T_out, 1]
            residual = self.offset_net(effective_scale).permute(0, 2, 1)  # [B, K, T_out]
            total_offset = geom_offset + residual
        else:
            total_offset = geom_offset

        out = deform_conv1d(
            input=x,
            offset=total_offset,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            mask=None,
        )

        if mask is not None:
            mask_bool = mask.to(device=out.device, dtype=torch.bool)
            if mask_bool.shape[-1] != out.shape[-1]:
                mask_out = F.interpolate(mask_bool.unsqueeze(1).float(), size=out.shape[-1], mode="nearest").squeeze(1).bool()
            else:
                mask_out = mask_bool
            out = out * mask_out.unsqueeze(1).to(dtype=out.dtype)

        return out
