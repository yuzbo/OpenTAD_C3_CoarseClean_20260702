import torch

from opentad.models.backbones.georoute_routing import (
    build_refresh_mask,
    detached_spatial_carry,
)


def test_refresh_mask_exact_k32_and_age_priority():
    motion = torch.zeros(1, 2, 64)
    valid = torch.ones_like(motion, dtype=torch.bool)
    age = torch.zeros_like(motion)
    age[..., :2] = 2
    mask = build_refresh_mask(motion, valid, age)
    assert mask.shape == (1, 2, 64)
    assert mask.sum(-1).tolist() == [[32, 32]]
    assert mask[0, 0, :2].all()


def test_carry_same_spatial_index_is_detached_and_causal():
    prev = torch.randn(1, 1, 3, 4, requires_grad=True)
    cur = torch.zeros_like(prev, requires_grad=True)
    indices = torch.tensor([[[1, 3, 5]]])
    out = detached_spatial_carry(cur, prev, indices, indices)
    assert torch.equal(out, prev.detach())
    assert not out.requires_grad


def test_carry_reset_and_missing_lineage_are_zero():
    cur = torch.ones(1, 1, 2, 4)
    indices = torch.tensor([[[0, 2]]])
    assert torch.equal(detached_spatial_carry(cur, None, indices, None), torch.zeros_like(cur))
