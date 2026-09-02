import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import ETTRCVisionTransformerAdapter


def test_adapter_and_jvp_are_trainable_parameters():
    model = ETTRCVisionTransformerAdapter(embed_dims=12, depth=1, num_heads=3, total_frames=16, num_frames=16, jacobian_rank=4)
    names = {name for name, p in model.named_parameters() if p.requires_grad}
    assert any("adapter" in name for name in names)
    assert any("jacobian_approx" in name for name in names)
