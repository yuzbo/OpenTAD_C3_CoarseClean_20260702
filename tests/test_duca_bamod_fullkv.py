import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.vit_adapter import Block


def test_amod_capacity_one_executes_all_tokens():
    block = Block(embed_dims=12, num_heads=3, mlp_ratio=2.0)
    x = torch.randn(1, 8, 12)
    scores = torch.randn(1, 8)
    routed = block.forward_amod(x, 1, 1, scores, capacity=1.0)
    assert routed.shape == x.shape
    assert torch.isfinite(routed).all()


def test_full_kv_unselected_tokens_can_influence_selected_query():
    block = Block(embed_dims=12, num_heads=3, mlp_ratio=2.0)
    x = torch.randn(1, 8, 12)
    scores = torch.tensor([[10.0] + [-10.0] * 7])
    out1 = block.forward_amod(x, 1, 1, scores, capacity=0.5)
    x2 = x.clone()
    # A uniform channel shift is removed by the block's LayerNorm. Perturb a
    # single channel so the full-KV dependency is observable.
    x2[:, 7, 0] += 5.0
    out2 = block.forward_amod(x2, 1, 1, scores, capacity=0.5)
    assert not torch.allclose(out1[:, :4], out2[:, :4])
