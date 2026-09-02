import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import TaylorAttention


def test_selected_query_reads_unselected_full_kv():
    torch.manual_seed(5)
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    x = torch.randn(1, 4, 12)
    out1 = attn(x, query_x=x[:, :1])
    x2 = x.clone(); x2[:, 3] += 4.0
    out2 = attn(x2, query_x=x2[:, :1])
    assert not torch.allclose(out1, out2)
