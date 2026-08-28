from __future__ import annotations

import torch

from opentad.models.duca.physical_time_reconstruction import (
    PhysicalTimeCoresetReconstructor,
)


def _meta() -> dict:
    hidden = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [3.0, 1.0], [4.0, 2.0], [5.0, 2.0], [6.0, 3.0], [7.0, 3.0]]
    )
    return {
        "duca_native_tubelet_indices": [0, 2, 4, 7],
        "duca_native_tubelet_valid_len": 8,
        "duca_native_tubelet_scout_hidden": hidden,
        "duca_native_tubelet_scores": torch.linspace(0.0, 1.0, 8),
        "snippet_stride": 1,
        "offset_frames": 0,
        "fps": 25.0,
    }


def test_physical_reconstruction_is_linear_identity_at_zero_residual_init() -> None:
    module = PhysicalTimeCoresetReconstructor(
        target_len=8, feature_dim=4, scout_hidden_dim=2, time_hidden_dim=4
    )
    positions = torch.tensor([0.0, 2.0, 4.0, 7.0])
    sparse = (2.0 * positions + 1.0).reshape(1, 1, 4).repeat(1, 4, 1)
    masks = torch.ones((1, 4), dtype=torch.bool)
    output = module.forward_test(features=sparse, masks=masks, metas=[_meta()])
    expected = (2.0 * torch.arange(8, dtype=torch.float32) + 1.0).reshape(1, 1, 8)
    assert output["features"].shape == (1, 4, 8)
    assert torch.allclose(output["features"], expected.repeat(1, 4, 1), atol=1.0e-6)
    assert output["masks"].all()
    assert output["metas"][0]["snippet_stride"] == 2.0
    assert output["metas"][0]["detector_prediction_inverse_map_required"] is False
    assert "duca_native_tubelet_scout_hidden" not in output["metas"][0]


def test_recycling_adapter_trains_while_scout_context_stays_detached() -> None:
    module = PhysicalTimeCoresetReconstructor(
        target_len=8, feature_dim=4, scout_hidden_dim=2, time_hidden_dim=4
    )
    features = torch.randn(1, 4, 4, requires_grad=True)
    masks = torch.ones((1, 4), dtype=torch.bool)
    output = module.forward_test(features=features, masks=masks, metas=[_meta()])
    output["features"].square().mean().backward()
    final = module.context_projector[-1]
    assert final.weight.grad is not None
    assert torch.isfinite(final.weight.grad).all()
    assert float(final.weight.grad.abs().sum().item()) > 0.0
    assert features.grad is not None and torch.isfinite(features.grad).all()


def test_training_maps_frame_coordinates_to_physical_tubelet_grid() -> None:
    module = PhysicalTimeCoresetReconstructor(
        target_len=8, feature_dim=4, scout_hidden_dim=2, time_hidden_dim=4
    )
    features = torch.randn(1, 4, 4)
    masks = torch.ones((1, 4), dtype=torch.bool)
    segments = [torch.tensor([[2.0, 10.0]])]
    labels = [torch.tensor([3])]
    output = module.forward_train(
        features=features,
        masks=masks,
        metas=[_meta()],
        gt_segments=segments,
        gt_labels=labels,
    )
    assert torch.equal(output["gt_segments"][0], torch.tensor([[1.0, 5.0]]))
    assert output["gt_labels"] is labels
