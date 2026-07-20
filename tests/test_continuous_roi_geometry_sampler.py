import math

import pytest
import torch

from opentad.models.backbones.continuous_roi_geometry import (
    ANCHOR_LOGITS,
    _philox4x32_10,
    anchor_knot_logits,
    common_support_clip_boxes,
    decode_continuous_roi_logits,
    exogenous_common_support_logits,
    family_for_sample,
    interpolate_knot_logits,
    stateless_philox_uniform,
    temporal_filter_knot_logits,
)
from opentad.models.backbones.continuous_roi_sampler import (
    sample_continuous_roi,
    sample_continuous_roi_runtime,
)


def test_philox_matches_random123_zero_vector():
    assert _philox4x32_10((0, 0, 0, 0), (0, 0)) == (
        0x6627E8D5,
        0xE169C58D,
        0xBC57AC4C,
        0x9B00DBD8,
    )


def test_stateless_philox_is_retry_stable_and_key_sensitive():
    first = stateless_philox_uniform((12, 4), key_parts=(3407, 11, 0, 19, 64))
    retry = stateless_philox_uniform((12, 4), key_parts=(3407, 11, 0, 19, 64))
    changed = stateless_philox_uniform((12, 4), key_parts=(3407, 12, 0, 19, 64))
    assert torch.equal(first, retry)
    assert not torch.equal(first, changed)
    assert bool(((first > 0.0) & (first < 1.0)).all())


def test_registered_family_cycle_is_exactly_balanced():
    counts = {name: 0 for name in ("anchor", "fixed_size", "variable_size")}
    for update in range(4800):
        for batch_slot in range(2):
            counts[family_for_sample(update, batch_slot)] += 1
    assert counts == {"anchor": 3200, "fixed_size": 3200, "variable_size": 3200}


def test_anchor_decoder_matches_source_native_128_square():
    boxes = decode_continuous_roi_logits(anchor_knot_logits(batch_size=2))
    expected = torch.tensor([0.5, 0.5, 0.4, 128.0 / 180.0])
    assert torch.allclose(boxes, expected.view(1, 1, 4), atol=1e-6, rtol=0.0)


def test_decoder_is_analytic_and_in_bounds_for_extreme_logits():
    logits = torch.tensor(
        [
            [-100.0, -100.0, -100.0, -100.0],
            [100.0, 100.0, 100.0, 100.0],
        ]
    )
    boxes = decode_continuous_roi_logits(logits)
    left_top = boxes[:, :2] - 0.5 * boxes[:, 2:]
    right_bottom = boxes[:, :2] + 0.5 * boxes[:, 2:]
    assert bool((left_top >= -1e-6).all())
    assert bool((right_bottom <= 1.0 + 1e-6).all())
    area = boxes[:, 2] * boxes[:, 3]
    pixel_ratio = boxes[:, 2] * (16.0 / 9.0) / boxes[:, 3]
    assert bool((area >= 0.18 - 1e-6).all())
    assert bool((area <= 0.36 + 1e-6).all())
    assert bool((pixel_ratio >= 0.75 - 1e-6).all())
    assert bool((pixel_ratio <= 2.25 + 1e-6).all())


def test_temporal_filter_and_interpolation_preserve_constant_anchor():
    anchor = anchor_knot_logits(batch_size=1)
    filtered = temporal_filter_knot_logits(anchor, passes=2)
    clips = interpolate_knot_logits(filtered, clips=48)
    assert torch.equal(filtered, anchor)
    assert torch.equal(clips, anchor[:, :1].expand(1, 48, 4))


def test_common_support_retry_and_family_interventions():
    kwargs = dict(
        training_seed=3407,
        successful_update=1,
        sample_keys=[101, 102],
        window_starts=[0, 128],
    )
    first, families = exogenous_common_support_logits(**kwargs)
    retry, retry_families = exogenous_common_support_logits(**kwargs)
    assert torch.equal(first, retry)
    assert families == retry_families == ("variable_size", "anchor")
    anchor = torch.tensor(ANCHOR_LOGITS)
    assert torch.equal(first[1], anchor.view(1, 4).expand(12, 4))
    assert not torch.equal(first[0, :, 2:], anchor[2:].view(1, 2).expand(12, 2))
    boxes, _, _ = common_support_clip_boxes(**kwargs)
    assert boxes.shape == (2, 48, 4)


def _coordinate_source(height=9, width=13):
    y = torch.arange(height, dtype=torch.float32).view(1, 1, 1, height, 1)
    x = torch.arange(width, dtype=torch.float32).view(1, 1, 1, 1, width)
    plane = x + 10.0 * y
    return plane.expand(1, 3, 2, height, width).clone()


def test_sampler_full_frame_matches_align_corners_false_resize():
    source = _coordinate_source()
    boxes = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]])
    sampled = sample_continuous_roi(
        source,
        boxes,
        output_height=5,
        output_width=7,
        frames_per_clip=2,
    )
    expected = torch.nn.functional.interpolate(
        source.permute(0, 2, 1, 3, 4).reshape(2, 3, 9, 13),
        size=(5, 7),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 2, 3, 5, 7).permute(0, 2, 1, 3, 4).unsqueeze(1)
    assert torch.allclose(sampled, expected, atol=1e-5, rtol=0.0)


def test_runtime_sampler_is_exactly_the_same_operator():
    source = _coordinate_source()
    boxes = torch.tensor([[[0.4, 0.6, 0.5, 0.4]]])
    differentiable = sample_continuous_roi(
        source,
        boxes,
        output_height=6,
        output_width=8,
        frames_per_clip=2,
    )
    runtime = sample_continuous_roi_runtime(
        source,
        boxes,
        output_height=6,
        output_width=8,
        frames_per_clip=2,
    )
    assert torch.equal(differentiable, runtime)


def test_sampler_box_gradient_matches_centered_finite_difference():
    source = _coordinate_source(height=17, width=23)
    boxes = torch.tensor(
        [[[0.45, 0.55, 0.5, 0.4]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    weights = torch.linspace(
        -1.0,
        1.0,
        6 * 8,
        dtype=torch.float32,
    ).reshape(1, 1, 1, 1, 6, 8)

    def objective(value):
        return (
            sample_continuous_roi(
                source,
                value,
                output_height=6,
                output_width=8,
                frames_per_clip=2,
            )
            * weights
        ).sum()

    objective(boxes).backward()
    analytic = boxes.grad.detach().clone()
    epsilon = 1e-3
    numeric = torch.zeros_like(analytic)
    for channel in range(4):
        plus = boxes.detach().clone()
        minus = boxes.detach().clone()
        plus[..., channel] += epsilon
        minus[..., channel] -= epsilon
        numeric[..., channel] = (objective(plus) - objective(minus)) / (2.0 * epsilon)
    assert bool(torch.isfinite(analytic).all())
    assert torch.allclose(analytic, numeric, atol=0.6, rtol=0.03)


@pytest.mark.parametrize(
    "bad_box",
    [
        [0.5, 0.5, 0.0, 0.5],
        [0.1, 0.5, 0.5, 0.5],
        [math.nan, 0.5, 0.5, 0.5],
    ],
)
def test_sampler_rejects_invalid_geometry(bad_box):
    with pytest.raises(ValueError):
        sample_continuous_roi(
            _coordinate_source(),
            torch.tensor([[bad_box]]),
            output_height=4,
            output_width=4,
            frames_per_clip=2,
        )
