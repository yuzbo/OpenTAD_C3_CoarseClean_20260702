import pytest

torch = pytest.importorskip("torch")

from opentad.models.projections.phystime_projection import (
    PhysTimeMeasureProjection,
    SupportIntegratedMeasureAttention,
)


def _geometry(timestamps, supports, duration=2.0):
    count = len(timestamps)
    return {
        "timestamps_sec": torch.tensor([timestamps], dtype=torch.float32),
        "ownership_intervals_sec": torch.tensor([supports], dtype=torch.float32),
        "valid_mask": torch.ones((1, count), dtype=torch.bool),
        "duration_sec": torch.tensor([duration], dtype=torch.float32),
        "domain_start_sec": torch.tensor([0.0], dtype=torch.float32),
        "domain_end_sec": torch.tensor([duration], dtype=torch.float32),
    }


def _query(intervals):
    intervals = torch.tensor([intervals], dtype=torch.float32)
    widths = intervals[..., 1] - intervals[..., 0]
    return {
        "centers_sec": intervals.mean(dim=-1),
        "intervals_sec": intervals,
        "widths_sec": widths,
        "valid_mask": torch.ones(intervals.shape[:2], dtype=torch.bool),
    }


def _set_scalar_identity(module):
    with torch.no_grad():
        module.value_proj.weight.fill_(1.0)
        module.value_proj.bias.zero_()
        module.output_proj.weight.fill_(1.0)
        module.output_proj.bias.zero_()


def test_constant_kernel_is_invariant_to_support_refinement():
    attention = SupportIntegratedMeasureAttention(
        in_channels=1,
        out_channels=1,
        attention_channels=1,
        content_logits=False,
        relative_time_logits=False,
    )
    _set_scalar_identity(attention)
    query = _query([[0.0, 2.0]])

    coarse, coarse_mask, _ = attention(
        torch.tensor([[[3.0]]]),
        _geometry([1.0], [[0.0, 2.0]]),
        query,
    )
    refined, refined_mask, _ = attention(
        torch.tensor([[[3.0], [3.0]]]),
        _geometry([0.5, 1.5], [[0.0, 1.0], [1.0, 2.0]]),
        query,
    )

    assert torch.equal(coarse_mask, refined_mask)
    assert torch.allclose(coarse, refined, atol=1e-6)
    assert coarse.item() == pytest.approx(3.0)


def test_temporal_mass_not_duplicate_count_controls_constant_kernel():
    attention = SupportIntegratedMeasureAttention(
        in_channels=1,
        out_channels=1,
        attention_channels=1,
        content_logits=False,
        relative_time_logits=False,
    )
    _set_scalar_identity(attention)
    query = _query([[0.0, 2.0]])

    output, _, diagnostics = attention(
        torch.tensor([[[2.0], [6.0], [6.0]]]),
        _geometry([0.5, 1.25, 1.75], [[0.0, 1.0], [1.0, 1.5], [1.5, 2.0]]),
        query,
    )

    assert output.item() == pytest.approx(4.0)
    assert diagnostics["coverage_sec"].item() == pytest.approx(2.0)


def test_uncovered_query_returns_finite_zero_and_invalid_mask():
    attention = SupportIntegratedMeasureAttention(2, 4, attention_channels=4)
    geometry = _geometry([0.5], [[0.0, 1.0]], duration=3.0)
    query = _query([[1.5, 2.5]])

    output, query_mask, diagnostics = attention(torch.randn(1, 1, 2), geometry, query)

    assert torch.isfinite(output).all()
    assert output.abs().sum().item() == 0.0
    assert not query_mask.any()
    assert diagnostics["attention_weights"].abs().sum().item() == 0.0


def test_measure_attention_has_finite_observation_and_parameter_gradients():
    attention = SupportIntegratedMeasureAttention(2, 4, attention_channels=4)
    observations = torch.randn(1, 2, 2, requires_grad=True)
    geometry = _geometry([0.5, 1.5], [[0.0, 1.0], [1.0, 2.0]])
    query = _query([[0.0, 1.0], [1.0, 2.0]])

    output, query_mask, _ = attention(observations, geometry, query)
    output[query_mask].sum().backward()

    assert observations.grad is not None
    assert torch.isfinite(observations.grad).all()
    assert observations.grad.abs().sum().item() > 0
    trainable_grads = [parameter.grad for parameter in attention.parameters() if parameter.requires_grad]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in trainable_grads)


def test_projection_builds_every_level_directly_from_original_observations():
    projection = PhysTimeMeasureProjection(
        in_channels=2,
        out_channels=4,
        attention_channels=4,
        base_spacing_sec=0.5,
        num_levels=3,
    )
    inputs = torch.randn(1, 2, 3, requires_grad=True)
    masks = torch.tensor([[True, True, True]])
    metas = [
        {
            "phystime_timestamps_sec": [0.25, 0.75, 1.75],
            "phystime_support_intervals_sec": [[0.0, 0.5], [0.5, 1.0], [1.5, 2.0]],
            "phystime_duration_sec": 2.0,
            "phystime_domain_start_sec": 0.0,
            "phystime_domain_end_sec": 2.0,
            "phystime_support_provenance": "synthetic_explicit_support",
        }
    ]

    features, level_masks, level_geometry = projection(inputs, masks, metas)
    sum(feature[level_mask.unsqueeze(1).expand_as(feature)].sum() for feature, level_mask in zip(features, level_masks)).backward()

    assert len(features) == len(level_masks) == len(level_geometry) == 3
    assert [feature.shape[-1] for feature in features] == [4, 2, 1]
    assert inputs.grad is not None and inputs.grad.abs().sum().item() > 0
    for attention in projection.level_attentions:
        assert attention.value_proj.weight.grad is not None
        assert attention.value_proj.weight.grad.abs().sum().item() > 0
