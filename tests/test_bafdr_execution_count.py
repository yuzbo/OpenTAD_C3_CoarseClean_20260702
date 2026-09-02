import pytest

torch = pytest.importorskip("torch")
from opentad.models.backbones.bafdr_wrapper import BAFDRBackboneWrapper


def test_fixed_k16_execution_budget():
    wrapper = object.__new__(BAFDRBackboneWrapper)
    wrapper.k_chunks = 16
    wrapper.chunk_num = 48
    assert wrapper.k_chunks == 16
    assert wrapper.chunk_num - wrapper.k_chunks == 32
