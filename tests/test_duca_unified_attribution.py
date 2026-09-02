import torch

from opentad.models.duca.feature_attribution import (
    listwise_distribution_loss,
    pairwise_ranking_loss,
    should_refresh_taylor_target,
    signed_feature_taylor_target,
    update_ema_target,
)


def test_signed_feature_taylor_target_matches_detached_formula():
    feature = torch.tensor(
        [[[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]]],
        requires_grad=True,
    )
    weights = torch.tensor([[[2.0, -1.0, 0.25], [1.0, -0.5, 3.0]]])
    objective = (feature * weights).sum()

    target = signed_feature_taylor_target(objective, feature)
    expected = torch.relu(-(weights.detach() * feature.detach()).sum(dim=1))

    assert target.shape == (1, 3)
    assert target.requires_grad is False
    assert torch.allclose(target, expected)


def test_taylor_refresh_period_and_ema_are_successful_update_based():
    assert should_refresh_taylor_target(0, period=4) is True
    assert should_refresh_taylor_target(3, period=4) is False
    assert should_refresh_taylor_target(4, period=4) is True

    old = torch.zeros(2, 3)
    new = torch.ones(2, 3)
    assert torch.allclose(update_ema_target(None, new, decay=0.9), new)
    assert torch.allclose(update_ema_target(old, new, decay=0.75), torch.full((2, 3), 0.25))


def test_taylor_ranking_losses_are_finite_and_order_sensitive():
    scores = torch.tensor([[0.1, 0.3, 2.0, -0.4]], requires_grad=True)
    target = torch.tensor([[0.0, 0.2, 1.0, 0.1]])
    valid = torch.tensor([[True, True, True, False]])

    listwise = listwise_distribution_loss(scores, target, valid)
    pairwise = pairwise_ranking_loss(scores, target, valid)

    assert torch.isfinite(listwise)
    assert torch.isfinite(pairwise)
    assert listwise.item() >= 0.0
    assert pairwise.item() >= 0.0
