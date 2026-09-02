import pytest

torch = pytest.importorskip("torch")
from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d


def test_uniform_affine_grid_is_finite_for_both_modes():
    torch.manual_seed(4)
    x = torch.randn(1, 2, 8)
    tau = torch.arange(8, dtype=torch.float32).view(1, -1)
    v0 = ContinuousTimeScaleAdaptiveConv1d(2, 2, reference_spacing_mode="absolute")
    v1 = ContinuousTimeScaleAdaptiveConv1d(2, 2, reference_spacing_mode="level_nominal")
    v1.load_state_dict(v0.state_dict())
    assert torch.isfinite(v0(x, temporal_positions=tau)).all()
    assert torch.isfinite(v1(x, temporal_positions=tau)).all()
