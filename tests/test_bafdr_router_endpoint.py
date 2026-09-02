import pytest

torch = pytest.importorskip("torch")
from tools.bata.bafdr_k16_fullmatrix_train import compute_router_targets


def test_router_endpoint_uses_last_chunk_for_end_frame():
    _, _, end = compute_router_targets(
        [torch.tensor([[760.0, 768.0]])], window_size=768, num_chunks=48
    )
    assert end.shape == (1, 48)
    assert end[0, 47].item() == 1.0
