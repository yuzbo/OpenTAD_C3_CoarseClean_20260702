import pytest

torch = pytest.importorskip("torch")

from opentad.models.dense_heads.support_decoupled_physical_query_head import (
    SupportDecoupledPhysicalQueryHead,
)
from opentad.models.projections.phystime_projection import PhysTimeMeasureProjection


def _head():
    return SupportDecoupledPhysicalQueryHead(
        num_classes=2,
        in_channels=4,
        feat_channels=8,
        num_convs=1,
        regression_ranges_sec=[(0.0, 2.0)],
        loss_normalizer=4.0,
        endpoint_loss_weight=0.25,
        loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
    )


def _geometry(coverage=0.0):
    centers = torch.tensor([[0.5, 1.5, 2.5, 3.5]])
    intervals = torch.tensor([[[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
    return (
        {
            "level": 0,
            "spacing_sec": 1.0,
            "centers_sec": centers,
            "intervals_sec": intervals,
            "widths_sec": torch.ones_like(centers),
            "valid_mask": torch.ones((1, 4), dtype=torch.bool),
            "coverage_sec": torch.full_like(centers, float(coverage)),
        },
    )


def test_signed_center_width_decodes_off_anchor_segments():
    head = _head()
    points = head.build_query_points(_geometry())
    offsets = (torch.tensor([[[1.0, 0.0], [0.0, 0.0], [-1.0, 0.0], [0.0, 0.0]]]),)

    proposals = head.decode_segments(points, offsets)

    assert torch.allclose(proposals[0, 0], torch.tensor([0.5, 2.5]))
    assert torch.allclose(proposals[0, 2], torch.tensor([0.5, 2.5]))
    assert torch.all(proposals[..., 1] >= proposals[..., 0])


def test_short_gt_without_internal_query_still_gets_reserved_assignment():
    head = _head().train()
    points = head.build_query_points(_geometry())
    cls_target, offset_target, segment_target, endpoint_target = head._prepare_targets(
        points,
        _geometry(),
        gt_segments=[torch.tensor([[1.95, 2.05]])],
        gt_labels=[torch.tensor([1])],
    )

    assert int((cls_target.sum(dim=-1) > 0).sum().item()) >= 1
    assert torch.allclose(segment_target[cls_target.sum(dim=-1) > 0][0], torch.tensor([1.95, 2.05]))
    debug = head.collect_debug_state()["target_assignment"][0]
    assert debug["gt_without_assigned_query"] == 0
    assert debug["short_gt_without_assigned_query"] == 0
    assert torch.isfinite(offset_target).all()
    assert torch.isfinite(endpoint_target).all()


def test_loss_backward_reaches_all_sdpq_branches():
    head = _head().train()
    features = (torch.randn(1, 4, 4, requires_grad=True),)
    masks = (torch.ones((1, 4), dtype=torch.bool),)
    losses, raw = head.forward_train(
        features,
        masks,
        _geometry(coverage=1.0),
        gt_segments=[torch.tensor([[1.95, 2.05]])],
        gt_labels=[torch.tensor([1])],
        return_outputs=True,
    )

    sum(losses.values()).backward()

    assert raw["proposals_sec"].shape == (1, 4, 2)
    assert features[0].grad is not None and features[0].grad.abs().sum().item() > 0
    for branch in (head.cls_head, head.reg_head, head.endpoint_head):
        assert branch.weight.grad is not None
        assert torch.isfinite(branch.weight.grad).all()
        assert branch.weight.grad.abs().sum().item() > 0
    assert losses["offset_loss"].item() >= 0


def test_projection_can_keep_uncovered_queries_with_null_evidence():
    projection = PhysTimeMeasureProjection(
        in_channels=2,
        out_channels=4,
        attention_channels=4,
        base_spacing_sec=1.0,
        num_levels=1,
        keep_uncovered_queries=True,
        use_null_evidence=True,
    )
    inputs = torch.randn(1, 2, 1, requires_grad=True)
    masks = torch.tensor([[True]])
    metas = [
        {
            "phystime_timestamps_sec": [0.25],
            "phystime_support_intervals_sec": [[0.0, 0.5]],
            "phystime_duration_sec": 2.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 2.0,
            "phystime_support_provenance": "synthetic_explicit_support",
        }
    ]

    features, level_masks, level_geometry = projection(inputs, masks, metas)
    features[0].sum().backward()

    assert level_masks[0].tolist() == [[True, True]]
    assert level_geometry[0]["coverage_sec"].tolist() == [[0.5, 0.0]]
    assert level_geometry[0]["domain_valid_mask"].tolist() == [[True, True]]
    assert level_geometry[0]["evidence_mask"].tolist() == [[True, False]]
    assert level_geometry[0]["assignment_mask"].tolist() == [[True, True]]
    assert projection.level_attentions[0].null_evidence.grad is not None
    assert torch.isfinite(projection.level_attentions[0].null_evidence.grad).all()


def test_support_overlap_masks_variable_duration_query_padding_exactly():
    projection = PhysTimeMeasureProjection(
        in_channels=2,
        out_channels=4,
        attention_channels=4,
        base_spacing_sec=1.0,
        num_levels=1,
        observation_measure="support_overlap",
        keep_uncovered_queries=True,
        use_null_evidence=True,
    )
    inputs = torch.randn(2, 2, 1)
    masks = torch.ones((2, 1), dtype=torch.bool)
    metas = [
        {
            "phystime_timestamps_sec": [0.25],
            "phystime_support_intervals_sec": [[0.0, 0.5]],
            "phystime_duration_sec": duration,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": duration,
            "phystime_support_provenance": "synthetic_explicit_support",
        }
        for duration in (2.0, 4.0)
    ]

    features, level_masks, level_geometry = projection(inputs, masks, metas)

    domain_valid = level_geometry[0]["domain_valid_mask"]
    padding = ~domain_valid
    assert domain_valid.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
    assert torch.count_nonzero(level_geometry[0]["coverage_sec"][padding]) == 0
    assert not torch.any(level_geometry[0]["evidence_mask"][padding])
    assert not torch.any(level_geometry[0]["assignment_mask"][padding])
    assert not torch.any(level_masks[0][padding])
    assert torch.count_nonzero(features[0].transpose(1, 2)[padding]) == 0


def test_assignment_mask_blocks_uncovered_positive_queries():
    head = _head().train()
    geometry = _geometry()
    geometry[0]["assignment_mask"] = torch.tensor([[False, False, False, False]])
    points = head.build_query_points(geometry)

    cls_target, _, _, _ = head._prepare_targets(
        points,
        geometry,
        gt_segments=[torch.tensor([[1.95, 2.05]])],
        gt_labels=[torch.tensor([1])],
    )

    assert int((cls_target.sum(dim=-1) > 0).sum().item()) == 0
    debug = head.collect_debug_state()["target_assignment"][0]
    assert debug["gt_without_assigned_query"] == 1


def test_reservation_ties_request_stable_annotation_order(monkeypatch):
    head = _head().train()
    geometry = _geometry()
    points = head.build_query_points(geometry)
    observed_stable = []
    original_argsort = torch.argsort

    def recording_argsort(input_tensor, *args, **kwargs):
        observed_stable.append(kwargs.get("stable"))
        return original_argsort(input_tensor, *args, **kwargs)

    monkeypatch.setattr(torch, "argsort", recording_argsort)
    head._prepare_targets(
        points,
        geometry,
        # Both GTs have the same duration and therefore the same candidate
        # count; their input annotation order is the contractual tie-break.
        gt_segments=[torch.tensor([[0.4, 0.6], [2.4, 2.6]])],
        gt_labels=[torch.tensor([0, 1])],
    )

    assert observed_stable == [True]


def test_projection_can_require_evidence_for_assignment_mask():
    projection = PhysTimeMeasureProjection(
        in_channels=2,
        out_channels=4,
        attention_channels=4,
        base_spacing_sec=1.0,
        num_levels=1,
        keep_uncovered_queries=True,
        use_null_evidence=True,
        min_assignment_coverage=1.0e-6,
    )
    inputs = torch.randn(1, 2, 1)
    masks = torch.tensor([[True]])
    metas = [
        {
            "phystime_timestamps_sec": [0.25],
            "phystime_support_intervals_sec": [[0.0, 0.5]],
            "phystime_duration_sec": 2.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 2.0,
            "phystime_support_provenance": "synthetic_explicit_support",
        }
    ]

    _, level_masks, level_geometry = projection(inputs, masks, metas)

    assert level_masks[0].tolist() == [[True, True]]
    assert level_geometry[0]["assignment_mask"].tolist() == [[True, False]]
