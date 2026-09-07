import torch

from opentad.models.bricks.dense_temporal_recovery import DenseTemporalRecovery


def test_zero_gate_preserves_initial_output_but_refinement_learns():
    torch.manual_seed(52)
    layer = DenseTemporalRecovery(embed_dims=4, target_grid_size=16, original_window_size=32)
    feats = torch.randn(2, 4, 8)
    centers = torch.linspace(1, 30, 8).repeat(2, 1)
    intervals = torch.stack([centers - 2, centers + 2], dim=-1)
    target = torch.randn(2, 4, 16)
    initial = layer.scatter_triangular(feats, centers, intervals)
    torch.testing.assert_close(layer(feats, centers, intervals), initial, rtol=0, atol=0)
    weight = layer.pwconv.weight.detach().clone()
    optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-2, weight_decay=0)
    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        (layer(feats, centers, intervals) - target).square().mean().backward()
        assert torch.isfinite(layer.residual_gate.grad).all()
        assert layer.residual_gate.grad.abs().sum() > 0
        if step > 0:
            assert layer.pwconv.weight.grad.abs().sum() > 0
            assert layer.dwconv.weight.grad.abs().sum() > 0
        optimizer.step()
    assert not torch.equal(layer.pwconv.weight, weight)
    assert not torch.equal(layer(feats, centers, intervals), initial)


def test_old_double_zero_checkpoint_is_not_silently_reinitialized():
    layer = DenseTemporalRecovery(embed_dims=2, target_grid_size=4, original_window_size=8)
    old_state = layer.state_dict()
    old_state["pwconv.weight"].zero_()
    old_state["pwconv.bias"].zero_()
    layer.load_state_dict(old_state)
    assert not bool(layer.residual_gate.detach().any())
    assert not bool(layer.pwconv.weight.detach().any())
