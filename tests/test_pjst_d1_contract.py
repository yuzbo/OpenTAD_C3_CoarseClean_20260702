import torch
import inspect
from pathlib import Path

from opentad.models.utils.temporal_grid import build_pjst_pair_metadata


def test_global_pair_scale_and_uniform_identity():
    pos = torch.tensor([[0, 2, 4, 6, 9, 11, 13, 15]], dtype=torch.int64)
    out = build_pjst_pair_metadata(pos, torch.tensor([16]), k=8)
    assert torch.equal(out["pair_valid"], torch.ones(1, 4, dtype=torch.bool))
    assert torch.allclose(out["pair_scale"], torch.tensor([[1., 1., 1., 1.]]))
    assert out["exact_uniform_identity"]


def test_invalid_suffix_is_audit_only():
    pos = torch.tensor([[0, 2, 5, 9, -1, -1, -1, -1]], dtype=torch.int64)
    out = build_pjst_pair_metadata(pos, torch.tensor([10]), k=8)
    assert torch.equal(out["pair_valid"], torch.tensor([[True, True, False, False]]))
    assert torch.equal(out["pair_scale"][0, 2:], torch.ones(2))


def test_k384_pair_layout_and_partial_padding():
    pos = torch.arange(384, dtype=torch.int64).reshape(1, 384)
    out = build_pjst_pair_metadata(pos, torch.tensor([768]), k=384)
    assert out["pair_scale"].shape == (1, 192)
    assert out["pair_valid"].all()
    padded = pos.clone(); padded[:, 380:] = -1
    out = build_pjst_pair_metadata(padded, torch.tensor([768]), k=384)
    assert out["pair_valid"][0, -2:].sum() == 0
    assert torch.equal(out["pair_scale"][0, -2:], torch.ones(2))


def test_forward_signature_and_checkpoint_keyword_reachability():
    adapter = Path("opentad/models/backbones/vit_adapter.py").read_text()
    wrapper = Path("opentad/models/backbones/backbone_wrapper.py").read_text()
    assert "pjst_pair_scale: Optional[Tensor]" in adapter and "pjst_pair_valid: Optional[Tensor]" in adapter
    ckpt = wrapper
    assert "pjst_pair_scale=pjst_pair_scale" in ckpt
    assert "pjst_pair_valid=pjst_pair_valid" in ckpt


def test_mixed_batch_uniform_identity_is_exact():
    uniform = torch.arange(384, dtype=torch.int64).reshape(1, -1)
    irregular = uniform.clone(); irregular[0, 1::2] += 1
    out = build_pjst_pair_metadata(torch.cat([uniform, irregular]), torch.tensor([768, 768]), k=384)
    assert torch.equal(out["pair_scale"][0], torch.ones(192))
    assert torch.equal(out["pair_valid"], torch.ones(2, 192, dtype=torch.bool))
