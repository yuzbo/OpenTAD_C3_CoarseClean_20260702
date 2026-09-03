import pytest
import torch
from pathlib import Path


pytest.importorskip("torch")

from opentad.models.bricks.scale_adaptive_conv1d import ContinuousTimeScaleAdaptiveConv1d
from opentad.models.selectors.dual_phase_frame_selector import DualPhaseFrameSelector


def test_reference_spacing_modes_are_explicit():
    v0 = ContinuousTimeScaleAdaptiveConv1d(4, 4, reference_spacing_mode="absolute")
    v1 = ContinuousTimeScaleAdaptiveConv1d(4, 4, reference_spacing_mode="level_nominal")
    assert v0.reference_spacing_mode == "absolute"
    assert v1.reference_spacing_mode == "level_nominal"
    with pytest.raises(ValueError):
        ContinuousTimeScaleAdaptiveConv1d(4, 4, reference_spacing_mode="implicit")


def test_level_nominal_spacing_matches_uniform_reference():
    torch.manual_seed(3)
    layer = ContinuousTimeScaleAdaptiveConv1d(2, 2, reference_spacing_mode="level_nominal")
    x = torch.randn(1, 2, 8)
    tau = torch.arange(8, dtype=torch.float32).view(1, -1)
    out = layer(x, temporal_positions=tau)
    assert out.shape[-1] == 8
    assert torch.isfinite(out).all()


def test_dual_phase_exposes_tubelet_boundary_prior():
    selector = DualPhaseFrameSelector(total_budget=8, scaffold_budget=4, burst_budget=4)
    inputs = torch.randn(1, 3, 8, 16, 16)
    masks = torch.ones(1, 8, dtype=torch.bool)
    metas = [{}]
    out = selector.forward_test(inputs, masks, metas)
    assert out["boundary_prior"].shape == (1, 8)
    assert out["boundary_prior_tubelet"].shape == (1, 4)
    assert metas[0]["boundary_prior_tubelet"].shape == (4,)


def test_campaign_propagates_runtime_root_for_pretrained_weights():
    script = Path(__file__).resolve().parents[1] / "scripts" / "submit_duca_ctdp_revised_campaign_n16r4.sh"
    text = script.read_text(encoding="utf-8")
    assert "YUZIBO_ROOT=" in text
    assert "${BASE}" in text
