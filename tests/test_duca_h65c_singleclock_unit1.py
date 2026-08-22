import torch
from opentad.models.backbones.physical_time import build_canonical_time_residual_bias

def test_bias_shape_formula_and_uniform_zero():
    u = torch.arange(4, dtype=torch.float32).view(1, 4)
    b = build_canonical_time_residual_bias(u, u, 2, 3)
    assert b.shape == (1, 3, 8, 8) and torch.equal(b, torch.zeros_like(b))
    t = torch.tensor([[0., 1., 3., 4.]])
    b = build_canonical_time_residual_bias(t, u, 2, 3)
    assert torch.isfinite(b).all() and b.dtype == t.dtype

def test_temporal_to_spatial_broadcast():
    t = torch.tensor([[0., 1.]])
    u = torch.tensor([[0., 2.]])
    b = build_canonical_time_residual_bias(t, u, 2, 1)
    assert b[0, 0, 0, 3] == b[0, 0, 1, 2]
