from __future__ import annotations

import torch

from opentad.models.backbones.vit_adapter import Adapter


def _nontrivial_adapter(*, temporal_size: int) -> Adapter:
    torch.manual_seed(23)
    adapter = Adapter(
        embed_dims=8,
        mlp_ratio=0.5,
        kernel_size=3,
        dilation=1,
        temporal_size=temporal_size,
    )
    with torch.no_grad():
        adapter.up_proj.weight.normal_(mean=0.0, std=0.1)
        adapter.up_proj.bias.normal_(mean=0.0, std=0.1)
    return adapter


def test_coordinate_lineage_packed_adapter_has_full_k_dense_parity():
    adapter = _nontrivial_adapter(temporal_size=3)
    inputs = torch.randn(1, 3 * 2 * 2, 8)
    dense_mask = torch.ones(1, 12, dtype=torch.bool)
    spatial_indices = torch.arange(4).view(1, 1, 4).expand(1, 3, 4)

    dense = adapter(inputs, 2, 2)
    packed = adapter.forward_native_packed(
        inputs,
        dense_mask,
        spatial_indices,
        grid_height=2,
        grid_width=2,
    )

    assert torch.allclose(packed, dense, atol=1e-6, rtol=1e-5)


def test_coordinate_lineage_packed_adapter_keeps_unselected_positions_exact_identity():
    adapter = _nontrivial_adapter(temporal_size=3)
    inputs = torch.randn(1, 3 * 2 * 2, 8)
    spatial_indices = torch.tensor([[[0, 2], [1, 2], [2, 3]]])
    dense_mask = torch.zeros(1, 3, 4, dtype=torch.bool)
    dense_mask.scatter_(2, spatial_indices, True)
    dense_mask = dense_mask.reshape(1, 12)

    packed = adapter.forward_native_packed(
        inputs,
        dense_mask,
        spatial_indices,
        grid_height=2,
        grid_width=2,
    )

    assert torch.equal(packed[~dense_mask], inputs[~dense_mask])
    assert not torch.equal(packed[dense_mask], inputs[dense_mask])
