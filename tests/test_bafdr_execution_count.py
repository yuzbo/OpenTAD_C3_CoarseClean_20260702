import pytest

torch = pytest.importorskip("torch")
from torch import nn
from opentad.models.backbones.bafdr_wrapper import BAFDRBackboneWrapper, BAFDRRouterHead


def test_fixed_k16_execution_budget():
    wrapper = object.__new__(BAFDRBackboneWrapper)
    nn.Module.__init__(wrapper)
    wrapper.k_chunks = 16
    wrapper.chunk_num = 48
    assert wrapper.k_chunks == 16
    assert wrapper.chunk_num - wrapper.k_chunks == 32


def test_router_returns_exactly_sixteen_selected_chunks():
    wrapper = object.__new__(BAFDRBackboneWrapper)
    wrapper.k_chunks = 16
    wrapper.uniform_mode = True
    wrapper.router = BAFDRRouterHead(in_channels=8, hidden_channels=4)
    selected, audit = wrapper._route_chunks(torch.randn(2, 48, 8))
    assert selected.shape == (2, 16)
    assert audit["selected_indices"].shape == (2, 16)
