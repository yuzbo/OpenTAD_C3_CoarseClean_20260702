import hashlib
import random

import torch
from torch import nn
from torch.nn import functional as F


class NativeGridSparseQuerySelector(nn.Module):
    """Select a fixed query budget without changing the FPN coordinate grid."""

    _SUPPORTED_POLICIES = ("stratified_uniform", "video_hash_random")

    def __init__(self, budget, policy="stratified_uniform", hash_seed=1234567891):
        super().__init__()
        if type(budget) is not int or budget <= 0:
            raise ValueError("sparse-head budget must be a positive integer")
        if policy not in self._SUPPORTED_POLICIES:
            raise ValueError(
                "unsupported sparse-head policy: {:s}".format(str(policy))
            )
        if type(hash_seed) is not int:
            raise ValueError("sparse-head hash_seed must be an integer")
        self.budget = budget
        self.policy = policy
        self.hash_seed = hash_seed

    @staticmethod
    def allocate_level_quotas(valid_lengths, budget):
        """Largest-remainder allocation with deterministic level-order ties."""
        valid_lengths = [int(x) for x in valid_lengths]
        if any(x < 0 for x in valid_lengths):
            raise ValueError("valid FPN lengths must be non-negative")
        total = sum(valid_lengths)
        if total <= budget:
            return valid_lengths
        scaled = [budget * x for x in valid_lengths]
        quotas = [x // total for x in scaled]
        remaining = budget - sum(quotas)
        order = sorted(
            range(len(valid_lengths)),
            key=lambda idx: (-(scaled[idx] % total), idx),
        )
        for idx in order[:remaining]:
            quotas[idx] += 1
        if sum(quotas) != budget:
            raise RuntimeError("failed to allocate the exact sparse-head budget")
        if any(q > n for q, n in zip(quotas, valid_lengths)):
            raise RuntimeError("allocated quota exceeds a valid FPN level")
        return quotas

    @staticmethod
    def _stratified_uniform_ranks(valid_length, quota, device):
        if quota == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if quota == valid_length:
            return torch.arange(valid_length, dtype=torch.long, device=device)
        ranks = torch.arange(quota, dtype=torch.long, device=device)
        return ((2 * ranks + 1) * valid_length) // (2 * quota)

    def _video_hash_random_ranks(
        self, valid_length, quota, video_id, level_idx, device
    ):
        if quota == 0:
            return torch.empty(0, dtype=torch.long, device=device)
        if quota == valid_length:
            return torch.arange(valid_length, dtype=torch.long, device=device)
        seed_text = "{:d}|{:s}|{:d}|{:d}".format(
            self.hash_seed, str(video_id), level_idx, valid_length
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big"
        )
        ranks = sorted(random.Random(seed).sample(range(valid_length), quota))
        return torch.as_tensor(ranks, dtype=torch.long, device=device)

    def forward(self, fpn_masks, video_ids=None):
        if len(fpn_masks) == 0:
            raise ValueError("sparse-head selection requires at least one FPN level")
        batch_size = fpn_masks[0].shape[0]
        for mask in fpn_masks:
            if mask.dtype != torch.bool or mask.ndim != 3 or mask.shape[1] != 1:
                raise ValueError("FPN masks must be boolean [B, 1, T] tensors")
            if mask.shape[0] != batch_size:
                raise ValueError("all FPN masks must have the same batch size")
        if self.policy == "video_hash_random":
            if video_ids is None or len(video_ids) != batch_size:
                raise ValueError(
                    "video_hash_random requires one stable video_id per sample"
                )

        selected_masks = [torch.zeros_like(mask) for mask in fpn_masks]
        for batch_idx in range(batch_size):
            valid_indices = [
                mask[batch_idx, 0].nonzero(as_tuple=True)[0]
                for mask in fpn_masks
            ]
            valid_lengths = [indices.numel() for indices in valid_indices]
            quotas = self.allocate_level_quotas(valid_lengths, self.budget)
            for level_idx, (indices, valid_length, quota) in enumerate(
                zip(valid_indices, valid_lengths, quotas)
            ):
                if self.policy == "stratified_uniform":
                    ranks = self._stratified_uniform_ranks(
                        valid_length, quota, indices.device
                    )
                else:
                    ranks = self._video_hash_random_ranks(
                        valid_length,
                        quota,
                        video_ids[batch_idx],
                        level_idx,
                        indices.device,
                    )
                selected_masks[level_idx][batch_idx, 0, indices[ranks]] = True
        return tuple(selected_masks)


def _masked_conv_spec(masked_conv):
    conv = masked_conv.conv
    kernel_size = conv.kernel_size[0]
    if (
        conv.stride != (1,)
        or conv.dilation != (1,)
        or conv.groups != 1
        or kernel_size % 2 != 1
        or conv.padding != (kernel_size // 2,)
    ):
        raise ValueError(
            "exact native-grid sparse execution only supports stride-1, "
            "dilation-1, groups-1 odd-kernel heads"
        )
    return conv, kernel_size // 2


def _expand_valid_positions(positions, radius, valid_mask):
    if positions.numel() == 0:
        return positions
    offsets = torch.arange(
        -radius, radius + 1, dtype=torch.long, device=positions.device
    )
    candidates = positions[:, None] + offsets[None, :]
    in_bounds = torch.logical_and(candidates >= 0, candidates < valid_mask.numel())
    candidates = torch.unique(candidates[in_bounds], sorted=True)
    return candidates[valid_mask[candidates]]


def _sparse_conv_at_positions(dense_input, output_positions, masked_conv):
    """Evaluate one official Conv1d only at requested physical output indices."""
    conv, radius = _masked_conv_spec(masked_conv)
    if output_positions.numel() == 0:
        return dense_input.new_zeros((0, conv.out_channels))
    time_size = dense_input.shape[-1]
    offsets = torch.arange(
        -radius, radius + 1, dtype=torch.long, device=output_positions.device
    )
    gather_indices = output_positions[:, None] + offsets[None, :]
    in_bounds = torch.logical_and(gather_indices >= 0, gather_indices < time_size)
    gather_indices = gather_indices.clamp(0, time_size - 1)
    patches = dense_input[:, gather_indices].permute(1, 0, 2).contiguous()
    patches = patches * in_bounds[:, None, :].to(patches.dtype)
    values = F.conv1d(
        patches,
        conv.weight,
        conv.bias,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
    )
    return values.squeeze(-1)


def _scatter_physical(values, positions, time_size, out_channels=None):
    if out_channels is None:
        out_channels = values.shape[-1]
    dense = values.new_zeros((out_channels, time_size))
    if positions.numel() == 0:
        return dense
    indices = positions[None, :].expand(out_channels, -1)
    return dense.scatter(1, indices, values.transpose(0, 1))


def _sparse_stack_single_sample(
    dense_input,
    valid_mask,
    selected_mask,
    hidden_layers,
    norms,
    activation,
    final_layer,
):
    layers = list(hidden_layers) + [final_layer]
    selected_positions = selected_mask.nonzero(as_tuple=True)[0]
    layer_positions = [None] * len(layers)
    required = selected_positions
    for layer_idx in range(len(layers) - 1, -1, -1):
        layer_positions[layer_idx] = required
        if layer_idx > 0:
            _, radius = _masked_conv_spec(layers[layer_idx])
            required = _expand_valid_positions(required, radius, valid_mask)

    current_dense = dense_input
    for layer_idx, (layer, positions) in enumerate(
        zip(layers, layer_positions)
    ):
        values = _sparse_conv_at_positions(current_dense, positions, layer)
        if layer_idx < len(hidden_layers):
            if values.numel() > 0:
                values = norms[layer_idx](
                    values.transpose(0, 1).unsqueeze(0)
                ).squeeze(0).transpose(0, 1)
                values = activation(values)
            current_dense = _scatter_physical(
                values,
                positions,
                dense_input.shape[-1],
                out_channels=layer.conv.out_channels,
            )
    return _scatter_physical(
        values,
        selected_positions,
        dense_input.shape[-1],
        out_channels=final_layer.conv.out_channels,
    )


def run_sparse_cls_head(cls_head, fpn_feats, fpn_masks, selected_masks):
    outputs = []
    for feat, mask, selected in zip(fpn_feats, fpn_masks, selected_masks):
        per_sample = []
        for batch_idx in range(feat.shape[0]):
            per_sample.append(
                _sparse_stack_single_sample(
                    feat[batch_idx],
                    mask[batch_idx, 0],
                    selected[batch_idx, 0],
                    cls_head.head,
                    cls_head.norm,
                    cls_head.act,
                    cls_head.cls_head,
                )
            )
        outputs.append(torch.stack(per_sample, dim=0))
    return tuple(outputs)


def run_sparse_reg_head(reg_head, fpn_feats, fpn_masks, selected_masks):
    outputs = []
    for level_idx, (feat, mask, selected) in enumerate(
        zip(fpn_feats, fpn_masks, selected_masks)
    ):
        per_sample = []
        for batch_idx in range(feat.shape[0]):
            per_sample.append(
                _sparse_stack_single_sample(
                    feat[batch_idx],
                    mask[batch_idx, 0],
                    selected[batch_idx, 0],
                    reg_head.head,
                    reg_head.norm,
                    reg_head.act,
                    reg_head.offset_head,
                )
            )
        raw = torch.stack(per_sample, dim=0)
        outputs.append(F.relu(reg_head.scale[level_idx](raw)))
    return tuple(outputs)


def _conv_macs(masked_conv, output_count):
    conv, _ = _masked_conv_spec(masked_conv)
    return (
        int(output_count)
        * conv.out_channels
        * conv.in_channels
        * conv.kernel_size[0]
    )


def estimate_sparse_head_macs(head, fpn_masks, selected_masks, final_attr):
    """Count executed multiply-accumulates for a dense or native sparse head."""
    final_layer = getattr(head, final_attr)
    layers = list(head.head) + [final_layer]
    dense_macs = 0
    sparse_macs = 0
    for mask, selected in zip(fpn_masks, selected_masks):
        batch_size, _, time_size = mask.shape
        for layer in layers:
            dense_macs += batch_size * _conv_macs(layer, time_size)
        for batch_idx in range(batch_size):
            required = selected[batch_idx, 0].nonzero(as_tuple=True)[0]
            layer_positions = [None] * len(layers)
            for layer_idx in range(len(layers) - 1, -1, -1):
                layer_positions[layer_idx] = required
                if layer_idx > 0:
                    _, radius = _masked_conv_spec(layers[layer_idx])
                    required = _expand_valid_positions(
                        required, radius, mask[batch_idx, 0]
                    )
            for layer, positions in zip(layers, layer_positions):
                sparse_macs += _conv_macs(layer, positions.numel())
    return {"dense_macs": dense_macs, "sparse_macs": sparse_macs}


def build_sparse_head_execution_receipt(
    cls_head,
    reg_head,
    fpn_masks,
    selected_masks,
    budget,
    policy,
    training_loss_support,
):
    if type(budget) is not int or budget <= 0:
        raise ValueError("receipt budget must be a positive integer")
    if training_loss_support != "selected_native_grid_queries":
        raise ValueError(
            "receipt must declare selected_native_grid_queries training support"
        )
    valid_counts = []
    selected_counts = []
    for batch_idx in range(fpn_masks[0].shape[0]):
        valid_per_level = [
            int(mask[batch_idx, 0].sum().item()) for mask in fpn_masks
        ]
        selected_per_level = [
            int(mask[batch_idx, 0].sum().item()) for mask in selected_masks
        ]
        if any(s > v for s, v in zip(selected_per_level, valid_per_level)):
            raise RuntimeError("selected query lies outside the valid FPN grid")
        expected = min(budget, sum(valid_per_level))
        if sum(selected_per_level) != expected:
            raise RuntimeError("selected query count violates the fixed budget")
        valid_counts.append(valid_per_level)
        selected_counts.append(selected_per_level)

    cls_macs = estimate_sparse_head_macs(
        cls_head, fpn_masks, selected_masks, "cls_head"
    )
    reg_macs = estimate_sparse_head_macs(
        reg_head, fpn_masks, selected_masks, "offset_head"
    )
    dense_macs = cls_macs["dense_macs"] + reg_macs["dense_macs"]
    sparse_macs = cls_macs["sparse_macs"] + reg_macs["sparse_macs"]
    return {
        "schema_version": "actionformer_native_grid_sparse_head_execution_v1",
        "policy": policy,
        "budget": budget,
        "training_loss_support": training_loss_support,
        "coordinate_grid": "original_full_video_fpn_physical_indices",
        "valid_counts_per_sample_level": valid_counts,
        "selected_counts_per_sample_level": selected_counts,
        "selected_count_contract_pass": True,
        "classification_head_macs": cls_macs,
        "regression_head_macs": reg_macs,
        "combined_dense_macs": dense_macs,
        "combined_sparse_macs": sparse_macs,
        "theoretical_head_mac_fraction": (
            float(sparse_macs) / float(dense_macs)
            if dense_macs > 0
            else 0.0
        ),
        "wall_clock_claim_allowed": False,
    }
