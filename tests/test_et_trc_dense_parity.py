import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import TaylorResidualBlock


def test_disable_taylor_keeps_dense_state_shape():
    block = TaylorResidualBlock(embed_dims=12, num_heads=3, mlp_ratio=2.0, enable_taylor=False)
    x = torch.randn(1, 8, 12)
    out = block(x, h=1, w=1, num_frames=16)
    assert out.shape == x.shape and torch.isfinite(out).all()
