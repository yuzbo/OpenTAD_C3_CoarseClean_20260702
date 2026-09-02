"""Pretraining, parity, and gradient contracts for ET-TRC."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.et_trc_videomae import TaylorAttention, TaylorResidualBlock


def test_fused_qkv_checkpoint_remap_is_exact():
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    fused_w = torch.arange(36 * 12, dtype=torch.float32).view(36, 12)
    fused_b = torch.arange(36, dtype=torch.float32)
    attn.load_state_dict({"qkv.weight": fused_w, "qkv.bias": fused_b}, strict=False)
    assert torch.equal(attn.q_proj.weight, fused_w[:12])
    assert torch.equal(attn.k_proj.weight, fused_w[12:24])
    assert torch.equal(attn.v_proj.weight, fused_w[24:])
    assert torch.equal(attn.q_proj.bias, fused_b[:12])
    assert torch.equal(attn.k_proj.bias, fused_b[12:24])
    assert torch.equal(attn.v_proj.bias, fused_b[24:])


def test_stride_one_is_dense_parity():
    torch.manual_seed(7)
    dense = TaylorResidualBlock(embed_dims=12, num_heads=3, mlp_ratio=2.0, stride_k=1, enable_taylor=True)
    reference = TaylorResidualBlock(embed_dims=12, num_heads=3, mlp_ratio=2.0, stride_k=4, enable_taylor=False)
    reference.load_state_dict(dense.state_dict(), strict=False)
    dense.eval(); reference.eval()
    x = torch.randn(1, 8, 12)
    assert (dense(x, h=1, w=1) - reference(x, h=1, w=1)).abs().max() < 1e-5


def test_selected_query_depends_on_unselected_kv():
    torch.manual_seed(9)
    attn = TaylorAttention(embed_dims=12, num_heads=3)
    x = torch.randn(1, 4, 12)
    y1 = attn(x, query_x=x[:, :1])
    x[:, 3] += 2.0
    y2 = attn(x, query_x=x[:, :1])
    assert not torch.allclose(y1, y2)


def test_low_rank_residual_has_finite_nonzero_gradient():
    block = TaylorResidualBlock(embed_dims=12, num_heads=3, mlp_ratio=2.0, stride_k=4, segment_size=2)
    x = torch.randn(1, 8, 12, requires_grad=True)
    loss = block(x, h=1, w=1).square().mean()
    loss.backward()
    grads = [p.grad for n, p in block.named_parameters() if "jacobian_approx" in n and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g).item() > 0 for g in grads)


def test_submit_script_binds_clean_correction_checkout():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_zoomtoken_et_trc_n16r4.sbatch"
    text = script.read_text(encoding="utf-8")
    assert "git rev-parse HEAD" in text
    assert "git status --porcelain" in text
    assert "zoomtoken_s2_v3_rpl1_2c8f25fe_src" not in text
    assert "zoomtoken_et_trc_fix_" in text
