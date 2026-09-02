import pytest

torch = pytest.importorskip("torch")
from opentad.models.selectors.dual_phase_frame_selector import DualPhaseFrameSelector


def test_coordinate_contract_is_monotonic_and_round_trippable():
    selector = DualPhaseFrameSelector(total_budget=8, scaffold_budget=4, burst_budget=4)
    out = selector.forward_test(
        torch.randn(1, 3, 8, 16, 16), torch.ones(1, 8, dtype=torch.bool), [{}]
    )
    positions = out["selected_positions"]
    tau = out["temporal_positions"]
    assert positions.shape == (1, 8)
    assert tau.shape == (1, 8)
    assert torch.all(tau[:, 1:] > tau[:, :-1])
    assert torch.isfinite(tau).all()
