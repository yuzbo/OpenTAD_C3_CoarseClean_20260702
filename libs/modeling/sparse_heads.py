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
        self.audit_callback = None

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
        selected_masks = tuple(selected_masks)
        if self.audit_callback is not None:
            self.audit_callback(fpn_masks, selected_masks, video_ids)
        return selected_masks


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
    patches = _gather_conv_patches(dense_input, output_positions, radius)
    return _conv_patches(patches, conv)


def _gather_conv_patches(dense_input, output_positions, radius):
    """Gather exact physical Conv1d neighborhoods without executing the Conv."""
    kernel_size = 2 * radius + 1
    if output_positions.numel() == 0:
        return dense_input.new_zeros((0, dense_input.shape[0], kernel_size))
    time_size = dense_input.shape[-1]
    offsets = torch.arange(
        -radius, radius + 1, dtype=torch.long, device=output_positions.device
    )
    gather_indices = output_positions[:, None] + offsets[None, :]
    in_bounds = torch.logical_and(gather_indices >= 0, gather_indices < time_size)
    gather_indices = gather_indices.clamp(0, time_size - 1)
    patches = dense_input[:, gather_indices].permute(1, 0, 2).contiguous()
    patches = patches * in_bounds[:, None, :].to(patches.dtype)
    return patches


def _conv_patches(patches, conv):
    if patches.shape[0] == 0:
        return patches.new_zeros((0, conv.out_channels))
    return F.linear(
        patches.reshape(patches.shape[0], -1),
        conv.weight.reshape(conv.out_channels, -1),
        conv.bias,
    )


def _scatter_physical(values, positions, time_size, out_channels=None):
    if out_channels is None:
        out_channels = values.shape[-1]
    dense = values.new_zeros((out_channels, time_size))
    if positions.numel() == 0:
        return dense
    indices = positions[None, :].expand(out_channels, -1)
    return dense.scatter(1, indices, values.transpose(0, 1))


def _sparse_layer_positions(valid_mask, selected_mask, layers):
    selected_positions = selected_mask.nonzero(as_tuple=True)[0]
    layer_positions = [None] * len(layers)
    required = selected_positions
    for layer_idx in range(len(layers) - 1, -1, -1):
        layer_positions[layer_idx] = required
        if layer_idx > 0:
            _, radius = _masked_conv_spec(layers[layer_idx])
            required = _expand_valid_positions(required, radius, valid_mask)
    return selected_positions, layer_positions


def build_native_grid_sparse_execution_plan(
    fpn_feats,
    fpn_masks,
    selected_masks,
    layers,
):
    """Build one physical-position plan shared by classification/regression."""
    dense_inputs, valid_masks, sparse_masks, batch_sizes = _flatten_sparse_jobs(
        fpn_feats, fpn_masks, selected_masks
    )
    if not dense_inputs:
        raise ValueError("sparse execution requires at least one FPN job")

    layers = list(layers)
    layer_radii = tuple(_masked_conv_spec(layer)[1] for layer in layers)
    guard_size = max(layer_radii, default=0)
    time_sizes = tuple(int(dense_input.shape[-1]) for dense_input in dense_inputs)
    packed_dense_parts = []
    packed_valid_parts = []
    packed_selected_parts = []
    job_offsets = []
    running_offset = 0
    for job_idx, (dense_input, valid_mask, selected_mask) in enumerate(
        zip(dense_inputs, valid_masks, sparse_masks)
    ):
        job_offsets.append(running_offset)
        packed_dense_parts.append(dense_input)
        packed_valid_parts.append(valid_mask)
        packed_selected_parts.append(selected_mask)
        running_offset += dense_input.shape[-1]
        if job_idx + 1 < len(dense_inputs) and guard_size > 0:
            packed_dense_parts.append(
                dense_input.new_zeros((dense_input.shape[0], guard_size))
            )
            packed_valid_parts.append(
                valid_mask.new_zeros((guard_size,))
            )
            packed_selected_parts.append(
                selected_mask.new_zeros((guard_size,))
            )
            running_offset += guard_size

    packed_dense_input = torch.cat(packed_dense_parts, dim=-1)
    packed_valid_mask = torch.cat(packed_valid_parts, dim=0)
    packed_selected_mask = torch.cat(packed_selected_parts, dim=0)
    _, global_layer_positions = _sparse_layer_positions(
        packed_valid_mask, packed_selected_mask, layers
    )
    per_job_positions = tuple(
        selected_mask.nonzero(as_tuple=True)[0]
        for selected_mask in sparse_masks
    )
    final_counts = tuple(
        int(positions.numel()) for positions in per_job_positions
    )
    if sum(final_counts) != global_layer_positions[-1].numel():
        raise RuntimeError("packed sparse selection-count mismatch")

    layer_plans = []
    for layer_idx, (positions, radius) in enumerate(
        zip(global_layer_positions, layer_radii)
    ):
        offsets = torch.arange(
            -radius, radius + 1,
            dtype=torch.long,
            device=positions.device,
        )
        candidate_positions = positions[:, None] + offsets[None, :]
        if layer_idx == 0:
            present = torch.logical_and(
                candidate_positions >= 0,
                candidate_positions < packed_dense_input.shape[-1],
            )
            gather_indices = candidate_positions.clamp(
                0, packed_dense_input.shape[-1] - 1
            )
        else:
            previous_positions = global_layer_positions[layer_idx - 1]
            if previous_positions.numel() == 0:
                gather_indices = torch.zeros_like(candidate_positions)
                present = torch.zeros_like(
                    candidate_positions, dtype=torch.bool
                )
            else:
                insertion_indices = torch.searchsorted(
                    previous_positions, candidate_positions
                )
                gather_indices = insertion_indices.clamp(
                    max=previous_positions.numel() - 1
                )
                present = torch.logical_and(
                    insertion_indices < previous_positions.numel(),
                    previous_positions[gather_indices] == candidate_positions,
                )
        layer_plans.append(
            {
                "gather_indices": gather_indices,
                "gather_present": present,
                "global_positions": positions,
            }
        )
    first_plan = layer_plans[0]
    first_patches = packed_dense_input[:, first_plan["gather_indices"]]
    first_patches = first_patches.permute(1, 0, 2).contiguous()
    first_patches = first_patches * first_plan["gather_present"][
        :, None, :
    ].to(first_patches.dtype)
    return {
        "batch_sizes": tuple(batch_sizes),
        "final_counts": final_counts,
        "first_patches": first_patches,
        "job_offsets": tuple(job_offsets),
        "layer_plans": tuple(layer_plans),
        "layer_radii": layer_radii,
        "packed_dense_input": packed_dense_input,
        "per_job_selected_positions": per_job_positions,
        "time_sizes": time_sizes,
    }


def _gather_packed_patches(packed_values, layer_plan):
    """Gather a sparse state with plan indices shared by both official heads."""
    if layer_plan["global_positions"].numel() == 0:
        kernel_size = layer_plan["gather_indices"].shape[-1]
        return packed_values.new_zeros(
            (0, packed_values.shape[-1], kernel_size)
        )
    patches = packed_values[layer_plan["gather_indices"]].permute(0, 2, 1)
    return patches.contiguous() * layer_plan["gather_present"][
        :, None, :
    ].to(patches.dtype)


def _sparse_stack_packed(
    execution_plan,
    hidden_layers,
    norms,
    activation,
    final_layer,
):
    """Keep hidden states globally sparse and scatter only final API outputs."""
    layers = list(hidden_layers) + [final_layer]
    layer_radii = tuple(_masked_conv_spec(layer)[1] for layer in layers)
    if layer_radii != execution_plan["layer_radii"]:
        raise ValueError("sparse execution-plan layer geometry mismatch")
    packed_values = None
    for layer_idx, layer in enumerate(layers):
        conv, _ = _masked_conv_spec(layer)
        layer_plan = execution_plan["layer_plans"][layer_idx]
        if layer_idx == 0:
            patches = execution_plan["first_patches"]
        else:
            patches = _gather_packed_patches(
                packed_values,
                layer_plan,
            )
        packed_values = _conv_patches(patches, conv)
        if layer_idx < len(hidden_layers):
            if packed_values.numel() > 0:
                packed_values = norms[layer_idx](
                    packed_values.transpose(0, 1).unsqueeze(0)
                ).squeeze(0).transpose(0, 1)
                packed_values = activation(packed_values)

    final_plan = execution_plan["layer_plans"][-1]
    per_job_values = torch.split(
        packed_values, execution_plan["final_counts"], dim=0
    )
    return [
        _scatter_physical(
            values,
            selected_positions,
            time_size,
            out_channels=final_layer.conv.out_channels,
        )
        for values, selected_positions, time_size in zip(
            per_job_values,
            execution_plan["per_job_selected_positions"],
            execution_plan["time_sizes"],
        )
    ]


def _flatten_sparse_jobs(fpn_feats, fpn_masks, selected_masks):
    if not (
        len(fpn_feats) == len(fpn_masks) == len(selected_masks)
    ):
        raise ValueError("sparse FPN feature/mask tuple lengths must match")
    if len(fpn_feats) == 0:
        raise ValueError("sparse execution requires at least one FPN level")
    dense_inputs = []
    valid_masks = []
    sparse_masks = []
    batch_sizes = []
    for feat, mask, selected in zip(fpn_feats, fpn_masks, selected_masks):
        if feat.ndim != 3:
            raise ValueError("sparse FPN features must be [B, C, T]")
        if (
            mask.dtype != torch.bool
            or selected.dtype != torch.bool
            or mask.ndim != 3
            or selected.ndim != 3
            or mask.shape[1] != 1
            or selected.shape[1] != 1
            or mask.shape != selected.shape
            or mask.shape[0] != feat.shape[0]
            or mask.shape[-1] != feat.shape[-1]
        ):
            raise ValueError(
                "sparse FPN masks must be matching boolean [B, 1, T] tensors"
            )
        batch_sizes.append(feat.shape[0])
        for batch_idx in range(feat.shape[0]):
            dense_inputs.append(feat[batch_idx])
            valid_masks.append(mask[batch_idx, 0])
            sparse_masks.append(selected[batch_idx, 0])
    return dense_inputs, valid_masks, sparse_masks, batch_sizes


def _restore_fpn_batches(flat_outputs, batch_sizes):
    outputs = []
    cursor = 0
    for batch_size in batch_sizes:
        outputs.append(torch.stack(flat_outputs[cursor : cursor + batch_size], dim=0))
        cursor += batch_size
    if cursor != len(flat_outputs):
        raise RuntimeError("packed sparse stack output count mismatch")
    return tuple(outputs)


def run_sparse_cls_head(
    cls_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
    execution_plan=None,
):
    layers = list(cls_head.head) + [cls_head.cls_head]
    if execution_plan is None:
        execution_plan = build_native_grid_sparse_execution_plan(
            fpn_feats, fpn_masks, selected_masks, layers
        )
    flat_outputs = _sparse_stack_packed(
        execution_plan,
        cls_head.head,
        cls_head.norm,
        cls_head.act,
        cls_head.cls_head,
    )
    return _restore_fpn_batches(
        flat_outputs, execution_plan["batch_sizes"]
    )


def run_sparse_reg_head(
    reg_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
    execution_plan=None,
):
    layers = list(reg_head.head) + [reg_head.offset_head]
    if execution_plan is None:
        execution_plan = build_native_grid_sparse_execution_plan(
            fpn_feats, fpn_masks, selected_masks, layers
        )
    flat_outputs = _sparse_stack_packed(
        execution_plan,
        reg_head.head,
        reg_head.norm,
        reg_head.act,
        reg_head.offset_head,
    )
    raw_outputs = _restore_fpn_batches(
        flat_outputs, execution_plan["batch_sizes"]
    )
    outputs = []
    for level_idx, raw in enumerate(raw_outputs):
        outputs.append(F.relu(reg_head.scale[level_idx](raw)))
    return tuple(outputs)


def run_sparse_reg_residual_head(
    reg_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
    execution_plan=None,
):
    """Run the official regression stack as a signed sparse residual.

    The ordinary regression head applies ReLU because it predicts absolute
    left/right distances.  DCSR refines an already-positive dense scaffold, so
    its residual must remain signed until after it is added to that scaffold.
    """
    layers = list(reg_head.head) + [reg_head.offset_head]
    if execution_plan is None:
        execution_plan = build_native_grid_sparse_execution_plan(
            fpn_feats, fpn_masks, selected_masks, layers
        )
    flat_outputs = _sparse_stack_packed(
        execution_plan,
        reg_head.head,
        reg_head.norm,
        reg_head.act,
        reg_head.offset_head,
    )
    raw_outputs = _restore_fpn_batches(
        flat_outputs, execution_plan["batch_sizes"]
    )
    return tuple(
        reg_head.scale[level_idx](raw)
        for level_idx, raw in enumerate(raw_outputs)
    )


def run_sparse_heads(
    cls_head,
    reg_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
):
    """Run both official heads from one shared native-grid execution plan."""
    cls_layers = list(cls_head.head) + [cls_head.cls_head]
    reg_layers = list(reg_head.head) + [reg_head.offset_head]
    cls_radii = tuple(_masked_conv_spec(layer)[1] for layer in cls_layers)
    reg_radii = tuple(_masked_conv_spec(layer)[1] for layer in reg_layers)
    if cls_radii != reg_radii:
        raise ValueError(
            "classification/regression sparse-head geometry must match"
        )
    execution_plan = build_native_grid_sparse_execution_plan(
        fpn_feats, fpn_masks, selected_masks, cls_layers
    )
    cls_outputs = run_sparse_cls_head(
        cls_head,
        fpn_feats,
        fpn_masks,
        selected_masks,
        execution_plan=execution_plan,
    )
    reg_outputs = run_sparse_reg_head(
        reg_head,
        fpn_feats,
        fpn_masks,
        selected_masks,
        execution_plan=execution_plan,
    )
    return cls_outputs, reg_outputs


def run_sparse_residual_heads(
    cls_head,
    reg_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
):
    """Run classification and signed regression residuals from one plan."""
    cls_layers = list(cls_head.head) + [cls_head.cls_head]
    reg_layers = list(reg_head.head) + [reg_head.offset_head]
    cls_radii = tuple(_masked_conv_spec(layer)[1] for layer in cls_layers)
    reg_radii = tuple(_masked_conv_spec(layer)[1] for layer in reg_layers)
    if cls_radii != reg_radii:
        raise ValueError(
            "classification/regression residual-head geometry must match"
        )
    execution_plan = build_native_grid_sparse_execution_plan(
        fpn_feats, fpn_masks, selected_masks, cls_layers
    )
    cls_outputs = run_sparse_cls_head(
        cls_head,
        fpn_feats,
        fpn_masks,
        selected_masks,
        execution_plan=execution_plan,
    )
    reg_outputs = run_sparse_reg_residual_head(
        reg_head,
        fpn_feats,
        fpn_masks,
        selected_masks,
        execution_plan=execution_plan,
    )
    return cls_outputs, reg_outputs


def run_dense_reg_residual_head(
    reg_head,
    fpn_feats,
    fpn_masks,
):
    """Run the official regression stack densely without the final ReLU."""
    if not (
        len(fpn_feats) == len(fpn_masks) == reg_head.fpn_levels
    ):
        raise ValueError("dense residual FPN feature/mask lengths must match")
    out_offsets = tuple()
    for level_idx, (cur_feat, cur_mask) in enumerate(
        zip(fpn_feats, fpn_masks)
    ):
        cur_out = cur_feat
        for layer_idx in range(len(reg_head.head)):
            cur_out, _ = reg_head.head[layer_idx](cur_out, cur_mask)
            cur_out = reg_head.act(reg_head.norm[layer_idx](cur_out))
        cur_offsets, _ = reg_head.offset_head(cur_out, cur_mask)
        out_offsets += (reg_head.scale[level_idx](cur_offsets),)
    return out_offsets


def run_dense_residual_heads(
    cls_head,
    reg_head,
    fpn_feats,
    fpn_masks,
):
    """Run all-query classification and signed regression residuals densely."""
    return (
        cls_head(fpn_feats, fpn_masks),
        run_dense_reg_residual_head(reg_head, fpn_feats, fpn_masks),
    )


def run_dcsr_heads(
    scaffold_cls_head,
    scaffold_reg_head,
    fpn_feats,
    fpn_masks,
    selected_masks,
    residual_cls_head=None,
    residual_reg_head=None,
    residual_enabled=True,
    residual_scale=1.0,
    residual_execution="sparse",
):
    """Dense proposal floor plus sparse expensive residual refinement.

    Dense scaffold outputs and the original valid FPN masks remain the public
    prediction and supervision support.  Sparse outputs are additive residuals
    only; they can never erase an unselected query.
    """
    if type(residual_enabled) is not bool:
        raise ValueError("DCSR residual_enabled must be boolean")
    if not isinstance(residual_scale, (int, float)) or residual_scale <= 0:
        raise ValueError("DCSR residual_scale must be positive")
    if residual_execution not in ("sparse", "dense"):
        raise ValueError("DCSR residual_execution must be sparse or dense")

    scaffold_cls = scaffold_cls_head(fpn_feats, fpn_masks)
    scaffold_reg = scaffold_reg_head(fpn_feats, fpn_masks)
    if not residual_enabled:
        return scaffold_cls, scaffold_reg
    if residual_cls_head is None or residual_reg_head is None:
        raise ValueError("enabled DCSR residual requires both residual heads")

    if residual_execution == "dense":
        if selected_masks is not fpn_masks:
            raise ValueError(
                "dense residual execution requires the original FPN masks"
            )
        residual_cls, residual_reg = run_dense_residual_heads(
            residual_cls_head,
            residual_reg_head,
            fpn_feats,
            fpn_masks,
        )
    else:
        residual_cls, residual_reg = run_sparse_residual_heads(
            residual_cls_head,
            residual_reg_head,
            fpn_feats,
            fpn_masks,
            selected_masks,
        )
    out_cls = []
    out_reg = []
    for (
        base_cls,
        base_reg,
        cls_delta,
        reg_delta,
        valid_mask,
        selected_mask,
    ) in zip(
        scaffold_cls,
        scaffold_reg,
        residual_cls,
        residual_reg,
        fpn_masks,
        selected_masks,
    ):
        if not (
            base_cls.shape == cls_delta.shape
            and base_reg.shape == reg_delta.shape
            and valid_mask.shape == selected_mask.shape
            and base_cls.shape[0] == valid_mask.shape[0]
            and base_cls.shape[-1] == valid_mask.shape[-1]
            and base_reg.shape[0] == valid_mask.shape[0]
            and base_reg.shape[-1] == valid_mask.shape[-1]
        ):
            raise RuntimeError("DCSR scaffold/residual grid shape mismatch")
        # NativeGridSparseQuerySelector constructs selected_mask as a subset
        # of valid_mask, while the sparse executor structurally zero-scatters
        # every unselected residual.  Do not re-check tensor values with
        # ``.item()`` here: that would force a CUDA synchronization in every
        # detector forward and invalidate the later cost study.
        out_cls.append(base_cls + float(residual_scale) * cls_delta)
        out_reg.append(
            F.relu(base_reg + float(residual_scale) * reg_delta)
        )
    return tuple(out_cls), tuple(out_reg)


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


def build_dcsr_head_execution_receipt(
    scaffold_cls_head,
    scaffold_reg_head,
    residual_cls_head,
    residual_reg_head,
    fpn_masks,
    selected_masks,
    budget,
    policy,
    training_loss_support,
):
    """Build a theoretical head-only receipt; never a wall-clock claim."""
    if training_loss_support != "official_all_valid_fpn_queries":
        raise ValueError(
            "DCSR receipt requires official_all_valid_fpn_queries support"
        )
    scaffold_cls = estimate_sparse_head_macs(
        scaffold_cls_head, fpn_masks, fpn_masks, "cls_head"
    )
    scaffold_reg = estimate_sparse_head_macs(
        scaffold_reg_head, fpn_masks, fpn_masks, "offset_head"
    )
    residual_cls = estimate_sparse_head_macs(
        residual_cls_head, fpn_masks, selected_masks, "cls_head"
    )
    residual_reg = estimate_sparse_head_macs(
        residual_reg_head, fpn_masks, selected_masks, "offset_head"
    )
    official_dense_macs = (
        residual_cls["dense_macs"] + residual_reg["dense_macs"]
    )
    scaffold_dense_macs = (
        scaffold_cls["dense_macs"] + scaffold_reg["dense_macs"]
    )
    residual_sparse_macs = (
        residual_cls["sparse_macs"] + residual_reg["sparse_macs"]
    )
    total_dcsr_macs = scaffold_dense_macs + residual_sparse_macs
    valid_counts = []
    selected_counts = []
    for batch_idx in range(fpn_masks[0].shape[0]):
        valid_per_level = [
            int(mask[batch_idx, 0].sum().item()) for mask in fpn_masks
        ]
        selected_per_level = [
            int(mask[batch_idx, 0].sum().item()) for mask in selected_masks
        ]
        expected = min(budget, sum(valid_per_level))
        if sum(selected_per_level) != expected:
            raise RuntimeError("DCSR selected query count violates budget")
        valid_counts.append(valid_per_level)
        selected_counts.append(selected_per_level)
    return {
        "schema_version": "actionformer_dcsr_head_execution_v1",
        "policy": policy,
        "budget": budget,
        "training_loss_support": training_loss_support,
        "unselected_queries_keep_dense_scaffold": True,
        "valid_counts_per_sample_level": valid_counts,
        "selected_counts_per_sample_level": selected_counts,
        "official_dense_head_macs": official_dense_macs,
        "dense_scaffold_macs": scaffold_dense_macs,
        "sparse_residual_macs": residual_sparse_macs,
        "combined_dcsr_head_macs": total_dcsr_macs,
        "theoretical_head_mac_fraction": (
            float(total_dcsr_macs) / float(official_dense_macs)
            if official_dense_macs > 0
            else 0.0
        ),
        "wall_clock_claim_allowed": False,
    }
