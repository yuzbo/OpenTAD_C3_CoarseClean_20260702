import torch

from opentad.models.selectors.pc_ot_mras_prebackbone_frame_selector import PCOTMRASPreBackboneFrameSelector


def test_boundary_priority_and_stable_physical_order():
    a = torch.tensor([[.9, .8, .7, .6]])
    b = torch.tensor([[.1, .2, .9, .1]])
    v = torch.ones_like(a, dtype=torch.bool)
    assert PCOTMRASPreBackboneFrameSelector.deterministic_semantic_allocate(a, b, v, 3) == [[0, 1, 2]]


def test_invalid_padding_never_selected():
    a = torch.ones(1, 4); b = torch.zeros(1, 4); v = torch.tensor([[1, 1, 0, 0]], dtype=torch.bool)
    assert PCOTMRASPreBackboneFrameSelector.deterministic_semantic_allocate(a, b, v, 4) == [[0, 1]]
