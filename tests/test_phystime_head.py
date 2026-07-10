import pytest

torch = pytest.importorskip("torch")

from opentad.models.dense_heads.phystime_head import PhysTimeHead


def _head():
    return PhysTimeHead(
        num_classes=2,
        in_channels=4,
        feat_channels=8,
        num_convs=1,
        regression_ranges_sec=[(0.0, 10.0)],
        loss_normalizer=4.0,
        endpoint_loss_weight=0.5,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    )


def _level_geometry(batch_size=1):
    centers = torch.tensor([[1.0, 3.0]]).repeat(batch_size, 1)
    intervals = torch.tensor([[[0.0, 2.0], [2.0, 4.0]]]).repeat(batch_size, 1, 1)
    return (
        {
            "level": 0,
            "spacing_sec": 2.0,
            "centers_sec": centers,
            "intervals_sec": intervals,
            "widths_sec": torch.full_like(centers, 2.0),
            "valid_mask": torch.ones((batch_size, 2), dtype=torch.bool),
            "coverage_sec": torch.full_like(centers, 2.0),
        },
    )


def test_points_use_absolute_seconds_cell_width_and_seconds_regression_range():
    points = _head().build_physical_points(_level_geometry())

    assert len(points) == 1
    assert points[0].shape == (1, 2, 4)
    assert torch.allclose(points[0][0, :, 0], torch.tensor([1.0, 3.0]))
    assert torch.allclose(points[0][0, :, 1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(points[0][0, :, 2], torch.tensor([10.0, 10.0]))
    assert torch.allclose(points[0][0, :, 3], torch.tensor([2.0, 2.0]))


def test_integrated_event_probability_increases_with_physical_cell_width():
    logits = torch.zeros(1, 2)
    widths = torch.tensor([[0.5, 2.0]])

    probability = PhysTimeHead.integrated_event_probability(logits, widths)

    assert torch.all(probability > 0)
    assert probability[0, 1] > probability[0, 0]
    expected = 1.0 - torch.exp(-torch.nn.functional.softplus(torch.tensor(0.0)) * widths)
    assert torch.allclose(probability, expected)


def test_decode_uses_physical_cell_width_and_orders_segments():
    head = _head()
    points = head.build_physical_points(_level_geometry())
    distances = (torch.tensor([[[0.25, 0.5], [0.5, 0.25]]]),)

    proposals = head.decode_segments(points, distances)

    assert torch.allclose(
        proposals,
        torch.tensor([[[0.5, 2.0], [2.0, 3.5]]]),
    )
    assert torch.all(proposals[..., 1] >= proposals[..., 0])


def test_empty_ground_truth_has_finite_losses():
    head = _head().train()
    features = (torch.randn(1, 4, 2),)
    masks = (torch.tensor([[True, True]]),)
    losses, raw = head.forward_train(
        features,
        masks,
        _level_geometry(),
        gt_segments=[torch.empty(0, 2)],
        gt_labels=[torch.empty(0, dtype=torch.long)],
        return_outputs=True,
    )

    assert set(losses) == {"cls_loss", "reg_loss", "endpoint_loss"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert raw["proposals_sec"].shape == (1, 2, 2)


def test_all_prediction_branches_receive_gradients():
    head = _head().train()
    features = (torch.randn(1, 4, 2, requires_grad=True),)
    masks = (torch.tensor([[True, True]]),)
    losses, _ = head.forward_train(
        features,
        masks,
        _level_geometry(),
        gt_segments=[torch.tensor([[0.5, 3.5]])],
        gt_labels=[torch.tensor([1])],
        return_outputs=True,
    )

    sum(losses.values()).backward()

    assert features[0].grad is not None and features[0].grad.abs().sum().item() > 0
    for branch in (head.cls_head, head.reg_head, head.endpoint_head):
        assert branch.weight.grad is not None
        assert torch.isfinite(branch.weight.grad).all()
        assert branch.weight.grad.abs().sum().item() > 0


def test_forward_test_returns_only_valid_absolute_seconds_predictions():
    head = _head().eval()
    features = (torch.randn(1, 4, 2),)
    masks = (torch.tensor([[True, False]]),)
    geometry = list(_level_geometry())
    geometry[0] = dict(geometry[0], valid_mask=masks[0], coverage_sec=torch.tensor([[2.0, 0.0]]))

    proposals, scores = head.forward_test(features, masks, tuple(geometry))

    assert len(proposals) == len(scores) == 1
    assert proposals[0].shape == (1, 2)
    assert scores[0].shape == (1, 2)
    assert torch.all(proposals[0][:, 1] >= proposals[0][:, 0])
