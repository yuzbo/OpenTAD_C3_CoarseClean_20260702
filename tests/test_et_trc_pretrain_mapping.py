import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import TaylorAttention


def test_fused_qkv_is_split_into_real_projections():
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    fused = torch.arange(36 * 12, dtype=torch.float32).reshape(36, 12)
    state = {
        "qkv.weight": fused,
        "qkv.bias": torch.arange(36, dtype=torch.float32),
        "proj.weight": attn.proj.weight.detach().clone(),
        "proj.bias": attn.proj.bias.detach().clone(),
    }
    attn.load_state_dict(state, strict=False)
    assert attn.pretrained_qkv_remapped
    assert torch.equal(attn.q_proj.weight, fused[:12])
    assert torch.equal(attn.k_proj.weight, fused[12:24])
    assert torch.equal(attn.v_proj.weight, fused[24:])


def test_official_videomae_separate_qv_bias_is_mapped():
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    fused = torch.randn(36, 12)
    q_bias = torch.randn(12)
    v_bias = torch.randn(12)
    attn.load_state_dict(
        {
            "qkv.weight": fused,
            "q_bias": q_bias,
            "v_bias": v_bias,
        },
        strict=False,
    )
    assert attn.pretrained_qkv_remapped and attn.pretrained_bias_remapped
    assert torch.equal(attn.q_proj.bias, q_bias)
    assert torch.equal(attn.v_proj.bias, v_bias)
    assert torch.count_nonzero(attn.k_proj.bias).item() == 0
