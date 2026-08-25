import torch

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
