import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import TaylorAttention


def test_segment_mask_blocks_cross_segment_keys():
    torch.manual_seed(6)
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    x = torch.randn(1, 4, 12)
    q_seg = torch.tensor([[1]])
    k_seg = torch.tensor([[0, 0, 1, 1]])
    out1 = attn(x, query_x=x[:, 2:3], query_segment_ids=q_seg, key_segment_ids=k_seg)
    changed = x.clone(); changed[:, :2] += 100.0
    out2 = attn(changed, query_x=changed[:, 2:3], query_segment_ids=q_seg, key_segment_ids=k_seg)
    assert torch.allclose(out1, out2, atol=1e-5)
