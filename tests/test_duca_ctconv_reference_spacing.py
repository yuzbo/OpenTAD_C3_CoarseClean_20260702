import pytest

torch = pytest.importorskip("torch")
from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d


def test_absolute_and_level_nominal_modes_are_explicit():
    absolute = ContinuousTimeScaleAdaptiveConv1d(2, 2, reference_spacing_mode="absolute")
    nominal = ContinuousTimeScaleAdaptiveConv1d(2, 2, reference_spacing_mode="level_nominal")
    assert absolute.reference_spacing_mode == "absolute"
    assert nominal.reference_spacing_mode == "level_nominal"


def test_level_nominal_handles_irregular_grid_and_stride_dilation():
    layer = ContinuousTimeScaleAdaptiveConv1d(
        2, 2, kernel_size=3, stride=2, dilation=2, padding=2,
        reference_spacing_mode="level_nominal",
    )
    x = torch.randn(1, 2, 9)
    tau = torch.tensor([[0.0, 1.0, 2.5, 4.0, 6.0, 8.0, 10.5, 13.0, 16.0]])
    out = layer(x, temporal_positions=tau)
    assert out.shape[-1] == 5
    assert torch.isfinite(out).all()
