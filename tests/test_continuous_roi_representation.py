import pytest
import torch

from opentad.models.backbones.continuous_roi_wrapper import (
    ContinuousRoiBackboneWrapper,
    ContinuousRoiFeatureFusion,
    auxiliary_loss_weights,
    temporal_class_occupancy_targets,
)


def test_registered_fusion_parameter_count_and_gradient_surface():
    fusion = ContinuousRoiFeatureFusion()
    assert sum(parameter.numel() for parameter in fusion.parameters()) == 594_049
    global_features = torch.randn(2, 384, 9, requires_grad=True)
    local_features = torch.randn(2, 384, 9, requires_grad=True)
    output, alpha = fusion(global_features, local_features)
    assert output.shape == global_features.shape
    assert alpha.shape == (2, 1, 9)
    assert bool(((alpha >= 0.25) & (alpha <= 0.75)).all())
    output.square().mean().backward()
    assert global_features.grad is not None
    assert local_features.grad is not None
    assert float(global_features.grad.abs().sum()) > 0.0
    assert float(local_features.grad.abs().sum()) > 0.0
    for name, parameter in fusion.named_parameters():
        assert parameter.grad is not None, name
        assert bool(torch.isfinite(parameter.grad).all()), name


def test_temporal_class_occupancy_uses_only_temporal_gt_and_labels():
    target = temporal_class_occupancy_targets(
        [torch.tensor([[0.0, 4.0], [6.0, 8.0]])],
        [torch.tensor([2, 4])],
        batch_size=1,
        num_classes=20,
        output_length=4,
        detector_length=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert target.shape == (1, 20, 4)
    assert torch.equal(target[0, 2], torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.equal(target[0, 4], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert float(target.sum()) == 3.0


@pytest.mark.parametrize(
    "update,expected",
    [
        (0, (0.25, 0.50)),
        (800, (0.25, 0.50)),
        (1600, (0.175, 0.35)),
        (2400, (0.10, 0.20)),
        (4800, (0.10, 0.20)),
    ],
)
def test_auxiliary_weight_schedule(update, expected):
    assert auxiliary_loss_weights(update) == pytest.approx(expected)


class _ToySharedBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = torch.nn.Linear(4, 4)
        self.blocks = torch.nn.ModuleList(
            [
                torch.nn.ModuleDict(
                    {
                        "attn": torch.nn.Linear(4, 4),
                        "adapter": torch.nn.Linear(4, 4),
                    }
                )
            ]
        )
        self.norm = torch.nn.LayerNorm(4)
        self.fc_norm = torch.nn.LayerNorm(4)


class _ToyRecognizer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ToySharedBackbone()


def test_u128_freezes_every_shared_core_parameter_and_preserves_fusion_norms():
    wrapper = ContinuousRoiBackboneWrapper.__new__(ContinuousRoiBackboneWrapper)
    torch.nn.Module.__init__(wrapper)
    wrapper.model = _ToyRecognizer()
    wrapper.norm_eval = True
    wrapper.fusion = ContinuousRoiFeatureFusion()

    wrapper._freeze_shared_backbone_except_adapters()
    for name, parameter in wrapper.model.backbone.named_parameters():
        assert parameter.requires_grad is (".adapter." in f".{name}.")
    assert wrapper.trainable_adapter_parameters > 0

    wrapper.set_norm_layer()
    assert wrapper.model.backbone.norm.training is False
    assert wrapper.model.backbone.fc_norm.training is False
    assert wrapper.fusion.global_norm.norm.training is True
    assert all(parameter.requires_grad for parameter in wrapper.fusion.parameters())
