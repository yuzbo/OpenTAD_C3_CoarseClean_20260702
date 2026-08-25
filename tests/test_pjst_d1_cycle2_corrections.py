import inspect
import torch

from opentad.models.backbones.vit_adapter import VisionTransformerAdapter
from opentad.models.utils.temporal_grid import pjst_pair_metadata


def test_forward_accepts_pjst_runtime_kwargs():
    params = inspect.signature(VisionTransformerAdapter.forward).parameters
    assert "pjst_pair_scale" in params and "pjst_pair_valid" in params


def test_packed_metadata_and_mixed_identity_formula():
    actual = torch.tensor([[[2, 2, 2, 2, 2, 2, 2, 2]] * 24, [[1] * 8] * 24])
    canonical = torch.tensor([[[1, 1, 1, 1, 1, 1, 1, 1]] * 24, [[1] * 8] * 24])
    out = pjst_pair_metadata(actual, canonical)
    assert out["packed_pair_scale"].shape == (48, 8)
    assert torch.equal(out["packed_pair_scale"][24], torch.ones(8))
    assert torch.allclose(out["packed_pair_scale"][0], torch.full((8,), .5))
