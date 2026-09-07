"""Real selected heavy execution using the repository's actual AdaTAD blocks.

The block owns LN/MHSA/MLP/residual/DropPath/TIA. This module changes only
which native tokens enter its heavy sublayers. Parents remain separate batch
rows. Dense TIA is applied after heavy scatter, exactly as vit_adapter.Block.
"""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def parent_drop_masks(block, state):
    """Draw in native parent order, independent of packing and local K.

    Source DropPath draws one value per parent. Drawing separately per bucket
    would reorder those values at padded tails and change counterfactual noise.
    Frozen attention/MLP dropout is in eval mode, so these are the only heavy
    block random draws. Use its actual module to retain its RNG convention.
    """
    probability = getattr(block.drop_path, "drop_prob", 0.)
    if not block.drop_path.training or probability == 0:
        return None
    dtype = torch.get_autocast_gpu_dtype() if state.is_cuda and torch.is_autocast_enabled() else state.dtype
    ones = torch.ones((len(state), 1, 1), device=state.device, dtype=dtype)
    return probability, block.drop_path(ones) != 0, block.drop_path(ones) != 0


def heavy(block, x, drop_masks=None, parents=None):
    def drop(value, index):
        if drop_masks is None:
            return block.drop_path(value)
        return value.div(1 - drop_masks[0]) * drop_masks[index + 1][parents].to(value.dtype)
    x = x + drop(block.attn(block.norm1(x)), 0)
    return x + drop(block.mlp(block.norm2(x)), 1)


def _groups(counts, implementation):
    if implementation == "reference":
        return [([i], int(k)) for i, k in enumerate(counts.tolist()) if k]
    if implementation != "bucket":
        raise ValueError("implementation must be reference or bucket")
    return [(torch.where(counts == k)[0].tolist(), int(k)) for k in counts.unique(sorted=True).tolist() if k]


def sparse_block(block, state, selected, h, w, valid=None, implementation="bucket", trace=None, layer=0):
    """A: true ragged selected tensors, K=0 identity heavy path + dense TIA."""
    if selected.dtype != torch.bool or selected.shape != state.shape[:2]:
        raise ValueError("selection must be a boolean native mask [parents,tokens]")
    if valid is not None:
        selected = selected & valid
        state = state * valid[..., None].to(state.dtype)
    if valid is None and bool(selected.all()):
        # No packing is needed at the full limit. Retain the original strides:
        # source PatchEmbed returns a transposed tensor, and CUDA linear kernels
        # may round differently after an otherwise redundant contiguous gather.
        output = heavy(block, state)
        if trace is not None:
            trace.extend(dict(layer=layer, parent=p, native_tokens=state.shape[1], qkv_tokens=state.shape[1],
                              mlp_tokens=state.shape[1], attention_shape=[block.attn.num_heads, state.shape[1], state.shape[1]],
                              padding_tokens=0) for p in range(len(state)))
        return block.adapter(output, h, w) if block.use_adapter else output
    output = state.clone()
    counts = selected.sum(-1)
    drop_masks = parent_drop_masks(block, state)
    for parents, k in _groups(counts, implementation):
        indices = torch.stack([torch.where(selected[p])[0] for p in parents])
        pidx = torch.tensor(parents, device=state.device)
        packed = state[pidx[:, None], indices]
        updated = heavy(block, packed, drop_masks, pidx)
        output[pidx[:, None], indices] = updated
        if trace is not None:
            trace.extend(dict(layer=layer, parent=p, native_tokens=state.shape[1],
                              qkv_tokens=k, mlp_tokens=k, attention_shape=[block.attn.num_heads, k, k],
                              padding_tokens=0) for p in parents)
    if trace is not None:
        trace.extend(dict(layer=layer, parent=p, native_tokens=state.shape[1], qkv_tokens=0,
                          mlp_tokens=0, attention_shape=[block.attn.num_heads, 0, 0], padding_tokens=0)
                     for p in torch.where(counts == 0)[0].tolist())
    if block.use_adapter:
        output = block.adapter(output, h, w)
    return output if valid is None else output * valid[..., None].to(output.dtype)


class MixedScaleProjection(nn.Module):
    def __init__(self, width, learned=False, scale_embedding=True):
        super().__init__()
        self.projection = nn.Linear(width, width, bias=False) if learned else nn.Identity()
        if learned:
            nn.init.eye_(self.projection.weight)
        self.scale = nn.Parameter(torch.zeros(width)) if scale_embedding else None

    def forward(self, x):
        return self.projection(x) + (self.scale if self.scale is not None else 0)


def mixed_block(block, state, fine, h, w, projection, valid=None, trace=None, layer=0, use_tia=True):
    """C: one coarse OR four fine tokens; scatter only the heavy residual.

    Canonical sorting restores exactly the source token order when all fine.
    A coarse residual goes to each observed native member, preserving identity.
    """
    if h % 2 or w % 2:
        raise ValueError("C requires a 2x2 partition of native patches")
    p, n, d = state.shape
    t = n // (h * w)
    if fine.shape != (p, t, h // 2, w // 2):
        raise ValueError("fine mask must address native 2x2 groups")
    if valid is None and bool(fine.all()) and use_tia:
        return sparse_block(block, state, torch.ones((p, n), dtype=torch.bool, device=state.device),
                            h, w, trace=trace, layer=layer)
    ids = torch.arange(n, device=state.device).reshape(t, h, w)
    groups = ids.reshape(t, h // 2, 2, w // 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4)
    if valid is not None:
        state = state * valid[..., None].to(state.dtype)
    outputs = []
    drop_masks = parent_drop_masks(block, state)
    for parent in range(p):
        entries = []
        for group, is_fine in zip(groups, fine[parent].flatten().tolist()):
            members = group if valid is None else group[valid[parent, group]]
            if not members.numel():
                continue
            if is_fine:
                entries.extend((int(i), i.reshape(1), state[parent, i], True) for i in members)
            else:
                entries.append((int(members[0]), members, projection(state[parent, members].mean(0)), False))
        entries.sort(key=lambda x: x[0])
        out = state[parent].clone()
        if entries:
            packed = torch.stack([entry[2] for entry in entries])[None]
            updated = heavy(block, packed, drop_masks, [parent])[0]
            delta = updated - packed[0]
            # Memberships are disjoint; index_add remains differentiable.
            coarse = [(i, entry) for i, entry in enumerate(entries) if not entry[3]]
            if coarse:
                indices = torch.cat([entry[1] for _, entry in coarse])
                values = torch.cat([delta[i:i + 1].expand(entry[1].numel(), -1) for i, entry in coarse])
                out = out.index_add(0, indices, values)
            fine_entries = [(i, entry) for i, entry in enumerate(entries) if entry[3]]
            if fine_entries:
                out[torch.cat([entry[1] for _, entry in fine_entries])] = updated[[i for i, _ in fine_entries]]
        outputs.append(out)
        if trace is not None:
            k = len(entries)
            trace.append(dict(layer=layer, parent=parent, native_tokens=n, qkv_tokens=k,
                              mlp_tokens=k, attention_shape=[block.attn.num_heads, k, k], padding_tokens=0))
    output = torch.stack(outputs)
    if use_tia and block.use_adapter:
        output = block.adapter(output, h, w)
    return output if valid is None else output * valid[..., None].to(output.dtype)


def mixed_block_bucket(block, state, fine, h, w, projection, valid=None, trace=None, layer=0, use_tia=True):
    """Equivalent C packing without Python loops over individual CUDA tokens."""
    p, n, d = state.shape
    if valid is None and bool(fine.all()) and use_tia:
        return sparse_block(block, state, torch.ones((p, n), dtype=torch.bool, device=state.device),
                            h, w, trace=trace, layer=layer)
    t = n // (h * w)
    ids = torch.arange(n, device=state.device).reshape(t, h, w)
    groups = ids.reshape(t, h // 2, 2, w // 2, 2).permute(0, 1, 3, 2, 4).reshape(-1, 4)
    fine = fine.flatten(1)
    member_valid = torch.ones((p, len(groups), 4), dtype=torch.bool, device=state.device) if valid is None else valid[:, groups]
    if valid is not None:
        state = state * valid[..., None].to(state.dtype)
    members = state[:, groups]
    coarse = projection(members.sum(2) / member_valid.sum(2).clamp_min(1)[..., None])
    menu = torch.cat((coarse[:, :, None], members), 2).flatten(1, 2)
    active = torch.cat(((~fine & member_valid.any(-1))[..., None], fine[..., None] & member_valid), -1).flatten(1)
    keys = torch.cat((groups[:, :1], groups), 1).flatten()
    order = torch.where(active, keys[None], n).argsort(dim=1, stable=True)
    # Slot 0 maps to all four members; fine slots map to their one native id.
    destinations = torch.full((len(groups), 5, 4), -1, device=state.device, dtype=torch.long)
    destinations[:, 0] = groups
    destinations[:, 1:, 0] = groups
    destinations = destinations.flatten(0, 1)
    output = state.clone()
    counts = active.sum(-1)
    drop_masks = parent_drop_masks(block, state)
    for parents, k in _groups(counts, "bucket"):
        pidx = torch.tensor(parents, device=state.device)
        chosen = order[pidx, :k]
        packed = menu[pidx[:, None], chosen]
        updated = heavy(block, packed, drop_masks, pidx)
        delta = updated - packed
        target = destinations[chosen]
        mask = target >= 0
        if valid is not None:
            mask = mask & valid[pidx[:, None, None], target.clamp_min(0)]
        is_fine = chosen.remainder(5) != 0
        coarse_mask = mask & ~is_fine[..., None]
        values = delta[:, :, None].expand(-1, -1, 4, -1) * coarse_mask[..., None].to(delta.dtype)
        update = torch.zeros_like(state[pidx]).scatter_add(1, target.clamp_min(0).flatten(1)[..., None].expand(-1, -1, d), values.flatten(1, 2))
        output[pidx] = state[pidx] + update
        # Fine tokens retain the source residual arithmetic exactly; subtracting
        # then re-adding the same state creates avoidable FP32 cancellation.
        parent_grid = pidx[:, None].expand_as(chosen)
        output[parent_grid[is_fine], target[..., 0][is_fine]] = updated[is_fine]
    if trace is not None:
        trace.extend(dict(layer=layer, parent=parent, native_tokens=n, qkv_tokens=int(k), mlp_tokens=int(k),
                          attention_shape=[block.attn.num_heads, int(k), int(k)], padding_tokens=0)
                     for parent, k in enumerate(counts.tolist()))
    if use_tia and block.use_adapter:
        output = block.adapter(output, h, w)
    return output if valid is None else output * valid[..., None].to(output.dtype)


class NativeEncoder(nn.Module):
    """Wrap an actual VisionTransformerAdapter instance, preserving its weights."""
    def __init__(self, backbone, route="A", implementation="bucket", coarse_variant="mean_shared", active_layers=None, prefix_depth=0):
        super().__init__()
        if route not in {"A", "C", "DENSE", "B"}:
            raise ValueError("unknown native encoder route")
        self.source = backbone
        self.tubelet_size = int(backbone.patch_embed.projection.kernel_size[0])
        self.route, self.implementation = route, implementation
        self.coarse_variant = coarse_variant
        self.active_layers = tuple(range(len(backbone.blocks))) if active_layers is None else tuple(active_layers)
        self.prefix_depth = prefix_depth
        if route == "C":
            self.coarse = nn.ModuleList([MixedScaleProjection(
                backbone.embed_dims, learned=coarse_variant == "learned_projection",
                scale_embedding=coarse_variant != "no_scale_embedding") for _ in backbone.blocks])
        self.trace = []

    def mixed(self, *args, **kwargs):
        return (mixed_block if self.implementation == "reference" else mixed_block_bucket)(*args, **kwargs)

    def embed(self, frames, valid_frames=None, position=True):
        source = self.source
        source._freeze_layers()
        p, _, t, height, width = frames.shape
        h, w = height // source.patch_size, width // source.patch_size
        valid = None
        if valid_frames is not None:
            frames = frames * valid_frames[:, None, :, None, None].to(frames.dtype)
            valid = valid_frames.reshape(p, -1, self.tubelet_size).any(-1)
            valid = valid[:, :, None].expand(-1, -1, h * w).reshape(p, -1)
            if bool(valid.all()):
                valid = None
        x = source.patch_embed(frames)[0]
        if position:
            pe = source.pos_embed
            if (h, w) != source.grid_size:
                pe = pe.reshape(-1, *source.grid_size, source.embed_dims).permute(0, 3, 1, 2)
                pe = torch.nn.functional.interpolate(pe, size=(h, w), mode="bicubic", align_corners=False)
                pe = pe.permute(0, 2, 3, 1).reshape(1, -1, source.embed_dims)
            if pe.shape[1] != x.shape[1]:
                raise ValueError("native parent length must match source PE")
            x = x + pe
        x = source.pos_drop(x)
        if valid is not None:
            x = x * valid[..., None].to(x.dtype)
        return x, valid, h, w

    def forward(self, frames, selection=None, valid_frames=None, position=True):
        x, valid, h, w = self.embed(frames, valid_frames, position)
        self.trace = []
        if self.route in {"A", "B"} and selection is None:
            selection = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        full_native = (valid is None and selection is not None and bool(selection.all())
                       and (self.route in {"A", "B"} or (self.route == "C" and self.coarse_variant != "coarse_update_no_tia")))
        for layer, block in enumerate(self.source.blocks):
            if layer not in self.active_layers:
                continue
            if self.route == "DENSE" or layer < self.prefix_depth or full_native:
                # The actual block already owns its checkpoint wrapper.
                x = block(x, h, w)
                n = x.shape[1]
                self.trace.extend(dict(layer=layer, parent=p, native_tokens=n, qkv_tokens=n, mlp_tokens=n,
                                       attention_shape=[block.attn.num_heads, n, n], padding_tokens=0)
                                  for p in range(x.shape[0]))
                continue
            def run(state, current=block, depth=layer):
                # Checkpoint recomputation must not append duplicate cost samples.
                if self.route == "C":
                    return self.mixed(current, state, selection, h, w, self.coarse[depth], valid,
                                       use_tia=self.coarse_variant != "coarse_update_no_tia")
                return sparse_block(current, state, selection, h, w, valid, self.implementation)
            if block.with_cp and x.requires_grad:
                # Match the source block's reentrant/no-grad forward convention,
                # including eval-mode numerical/gradient checks on CUDA SDPA.
                x = checkpoint(run, x, use_reentrant=True)
                # Trace counts correspond exactly to the compressed forward above.
                if self.route == "C":
                    groups = selection.flatten(1)
                    counts = groups.sum(-1) * 3 + groups.shape[1]
                    if valid is not None:
                        group_valid = valid.reshape(-1, frames.shape[2] // 2, h // 2, 2, w // 2, 2)
                        group_valid = group_valid.any(3).any(-1).flatten(1)
                        counts = (group_valid * (1 + 3 * groups)).sum(-1)
                else:
                    counts = (selection if valid is None else selection & valid).sum(-1)
                self.trace.extend(dict(layer=layer, parent=p, native_tokens=x.shape[1], qkv_tokens=int(k),
                                       mlp_tokens=int(k), attention_shape=[block.attn.num_heads, int(k), int(k)],
                                       padding_tokens=0) for p, k in enumerate(counts.tolist()))
            elif self.route == "C":
                x = self.mixed(block, x, selection, h, w, self.coarse[layer], valid, self.trace, layer,
                                self.coarse_variant != "coarse_update_no_tia")
            else:
                x = sparse_block(block, x, selection, h, w, valid, self.implementation, self.trace, layer)
        x = self.source.norm(x)
        if valid is not None:
            x = x * valid[..., None].to(x.dtype)
        return x.reshape(frames.shape[0], -1, h, w, self.source.embed_dims).permute(0, 4, 1, 2, 3)


def heavy_macs(trace, width, mlp_ratio=4):
    """QKV + output projection + MLP + two attention matmuls; 1 MAC=2 FLOPs."""
    return sum((4 + 2 * mlp_ratio) * item["qkv_tokens"] * width * width
               + 2 * item["qkv_tokens"] ** 2 * width for item in trace)
