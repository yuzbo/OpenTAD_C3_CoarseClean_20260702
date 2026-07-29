from __future__ import annotations

import pytest
import torch

from opentad.models.backbones.georoute_wrapper import (
    GeoRouteBackboneWrapper,
    GeoRouteScout,
    GeoRouteSparseTemporalAdapter,
    extract_native_tubelets,
)
from opentad.models.backbones.georoute_routing import (
    GEOROUTE_ROUTING_SCHEMA,
    decode_continuous_geometry,
    native_patch_centers,
    ordered_plackett_luce_log_prob,
    interpolate_temporal_knots,
    score_function_policy_loss,
    select_exact_k,
)


def _logits(*, batch: int = 2, tubelets: int = 3, patches: int = 20):
    torch.manual_seed(11)
    roi = torch.randn(batch, tubelets, patches, requires_grad=True)
    residual = torch.randn(batch, tubelets, patches, requires_grad=True)
    return roi, residual


def test_continuous_geometry_is_in_bounds_and_differentiable():
    logits = torch.tensor(
        [[[0.4, -0.5, 1.2, -0.8], [-0.3, 0.9, -1.0, 0.6]]],
        requires_grad=True,
    )

    geometry = decode_continuous_geometry(logits, min_extent=0.15, max_extent=0.8)

    assert geometry.shape == (1, 2, 4)
    assert torch.all(geometry[..., 2:] >= 0.15)
    assert torch.all(geometry[..., 2:] <= 0.8)
    assert torch.all(geometry[..., :2] - geometry[..., 2:] / 2 >= 0.0)
    assert torch.all(geometry[..., :2] + geometry[..., 2:] / 2 <= 1.0)
    geometry.square().sum().backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_native_patch_centers_are_row_major_and_normalized():
    centers = native_patch_centers(2, 3, device=torch.device("cpu"), dtype=torch.float32)

    assert centers.shape == (6, 2)
    assert torch.equal(centers[0], torch.tensor([1.0 / 6.0, 0.25]))
    assert torch.equal(centers[-1], torch.tensor([5.0 / 6.0, 0.75]))
    assert torch.all((centers > 0.0) & (centers < 1.0))


def test_native_tubelet_gather_preserves_the_btk_video_layout():
    """The [B,T,K] route index must expand to native video's seven dimensions."""

    native = torch.arange(1 * 2 * 5 * 3 * 2 * 2 * 2, dtype=torch.uint8).reshape(1, 2, 5, 3, 2, 2, 2)
    indices = torch.tensor([[[4, 1, 3], [0, 2, 4]]], dtype=torch.long)

    gathered = GeoRouteBackboneWrapper._gather_selected_native_tubelets(None, native, indices)

    assert gathered.shape == (1, 2, 3, 3, 2, 2, 2)
    assert torch.equal(gathered[0, 0, 0], native[0, 0, 4])
    assert torch.equal(gathered[0, 1, 1], native[0, 1, 2])


def test_native_tubelets_floor_crop_nondivisible_ncthw_without_synthetic_pixels():
    """Real 180x320 support is the complete 176x320 native patch lattice."""

    source = torch.arange(1 * 3 * 2 * 180 * 320, dtype=torch.uint8).reshape(
        1,
        3,
        2,
        180,
        320,
    )
    native, grid_hw, ignored, valid_mask = extract_native_tubelets(
        source,
        patch_size=16,
        tubelet_size=2,
    )

    assert grid_hw == (11, 20)
    assert ignored == (4, 0)
    assert native.shape == (1, 1, 220, 3, 2, 16, 16)
    assert native.dtype == torch.uint8
    assert valid_mask.dtype == torch.bool
    assert valid_mask.shape == (1, 1, 220)
    assert bool(valid_mask.all())

    restored = native.reshape(1, 1, 11, 20, 3, 2, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 3, 2, 176, 320)
    assert torch.equal(restored, source[..., :176, :])


def test_native_tubelets_floor_crop_both_uint8_boundaries():
    source = torch.arange(1 * 3 * 2 * 17 * 18, dtype=torch.uint8).reshape(
        1,
        3,
        2,
        17,
        18,
    )
    native, grid_hw, ignored, valid_mask = extract_native_tubelets(
        source,
        patch_size=16,
        tubelet_size=2,
    )

    assert grid_hw == (1, 1)
    assert ignored == (1, 2)
    assert native.shape == (1, 1, 1, 3, 2, 16, 16)
    assert native.dtype == torch.uint8
    assert bool(valid_mask.all())

    restored = native.reshape(1, 1, 1, 1, 3, 2, 16, 16).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(1, 3, 2, 16, 16)
    assert torch.equal(restored, source[..., :16, :16])


@pytest.mark.parametrize(
    ("mode", "estimator", "context_tokens"),
    [
        ("dense", "none", 0),
        ("uniform", "none", 2),
        ("random", "none", 2),
        ("roi", "straight_through", 2),
        ("free", "straight_through", 2),
        ("hybrid", "straight_through", 2),
    ],
)
def test_exact_k_routes_have_no_duplicates_and_a_complete_schema(mode, estimator, context_tokens):
    roi, residual = _logits()
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode=mode,
        tokens_per_tubelet=6,
        context_tokens=context_tokens,
        roi_fraction=0.5,
        training=True,
        estimator=estimator,
        temperature=0.7,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
    )

    expected_k = roi.shape[-1] if mode == "dense" else 6
    assert route["schema_version"] == GEOROUTE_ROUTING_SCHEMA
    assert route["indices"].shape == (2, 3, expected_k)
    assert torch.equal(route["indices"], torch.sort(route["indices"], dim=-1).values)
    assert torch.all(route["selected_mask"].sum(dim=-1) == expected_k)
    assert torch.all(route["indices"] >= 0)
    assert torch.all(route["indices"] < roi.shape[-1])
    assert torch.allclose(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    assert sum(route["role_counts"].values()) == expected_k
    if mode in {"uniform", "random"}:
        assert route["role_counts"][mode] == expected_k


def test_st_gate_is_hard_in_forward_but_has_a_biased_surrogate_gradient():
    roi, residual = _logits(batch=1, tubelets=2, patches=12)
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="hybrid",
        tokens_per_tubelet=6,
        context_tokens=1,
        roi_fraction=0.5,
        training=True,
        estimator="straight_through",
        temperature=0.6,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
    )

    assert torch.equal(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    context = route["ordered_indices"][..., :1]
    context_in_sorted_route = route["indices"] == context
    assert torch.equal(
        route["selected_surrogate"].masked_select(context_in_sorted_route),
        torch.zeros_like(route["selected_surrogate"].masked_select(context_in_sorted_route)),
    )
    route["st_gate"].sum().backward()
    assert roi.grad is not None and residual.grad is not None
    assert torch.isfinite(roi.grad).all() and torch.isfinite(residual.grad).all()
    assert torch.count_nonzero(roi.grad) > 0
    assert torch.count_nonzero(residual.grad) > 0
    assert torch.count_nonzero(roi.grad.gather(-1, context)) == 0
    assert torch.count_nonzero(residual.grad.gather(-1, context)) == 0


def test_score_function_route_has_plackett_luce_gradient_and_loss():
    roi, residual = _logits(batch=1, tubelets=2, patches=9)
    torch.manual_seed(9)
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="roi",
        tokens_per_tubelet=4,
        context_tokens=0,
        roi_fraction=1.0,
        training=True,
        estimator="score_function",
        temperature=0.8,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
    )
    assert route["ordered_log_prob"] is not None
    recomputed = ordered_plackett_luce_log_prob(
        roi,
        route["ordered_indices"],
        temperature=0.8,
    )
    assert torch.allclose(route["ordered_log_prob"], recomputed)

    policy_loss = score_function_policy_loss(
        detector_cost=torch.tensor(2.0, requires_grad=True),
        ordered_log_prob=route["ordered_log_prob"],
        baseline=torch.tensor(0.5),
        weight=0.75,
    )
    policy_loss.backward()
    assert roi.grad is not None
    assert torch.isfinite(roi.grad).all()
    assert torch.count_nonzero(roi.grad) > 0
    assert residual.grad is None


def test_score_function_loss_has_the_exact_risk_gradient_sign_for_one_draw():
    """A finite K=1 case catches an otherwise easy-to-miss REINFORCE sign flip."""

    logits = torch.tensor([[[0.2, -0.3, 0.7]]], requires_grad=True)
    losses = torch.tensor([0.4, 1.1, -0.2])
    probabilities = torch.softmax(logits, dim=-1)
    expected_risk = (probabilities * losses.view(1, 1, -1)).sum()
    expected_gradient = torch.autograd.grad(expected_risk, logits, retain_graph=True)[0]

    # Exact expectation of the implemented score-function objective, with a
    # zero baseline.  This is not a Monte Carlo approximation.
    estimator_expectation = logits.new_zeros(())
    for choice in range(losses.numel()):
        ordered = torch.tensor([[[choice]]])
        log_probability = ordered_plackett_luce_log_prob(
            logits,
            ordered,
            temperature=1.0,
        )
        estimator_expectation = estimator_expectation + probabilities[..., choice].detach() * score_function_policy_loss(
            detector_cost=losses[choice],
            ordered_log_prob=log_probability,
            baseline=torch.zeros(()),
            weight=1.0,
        )
    observed_gradient = torch.autograd.grad(estimator_expectation, logits)[0]
    assert torch.allclose(observed_gradient, expected_gradient, atol=1e-6, rtol=1e-6)


def test_score_function_sums_temporal_log_probability_then_averages_batch():
    loss = score_function_policy_loss(
        detector_cost=torch.tensor(2.0),
        ordered_log_prob=torch.tensor([[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]]),
        baseline=torch.tensor(1.0),
        weight=0.5,
    )
    assert loss.item() == pytest.approx(3.0)


def test_score_function_promotes_amp_likelihood_and_long_temporal_reduction():
    logits = torch.zeros(
        1,
        384,
        220,
        dtype=torch.float16,
        requires_grad=True,
    )
    ordered = torch.arange(64).view(1, 1, 64).expand(1, 384, 64)
    log_probability = ordered_plackett_luce_log_prob(
        logits,
        ordered,
        temperature=0.7,
    )
    loss = score_function_policy_loss(
        detector_cost=torch.tensor(2.0),
        ordered_log_prob=log_probability,
        baseline=torch.tensor(1.0),
        weight=1.0,
    )

    assert log_probability.dtype == torch.float32
    assert loss.dtype == torch.float32
    assert torch.isfinite(log_probability).all()
    assert torch.isfinite(loss)
    assert loss.detach().abs() > torch.finfo(torch.float16).max
    (loss * 256.0).backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0


def test_temporal_knot_interpolation_is_endpoint_aligned_and_differentiable():
    knots = torch.tensor([[[0.0], [1.0], [4.0], [9.0], [16.0]]], requires_grad=True)
    interpolated = interpolate_temporal_knots(knots, stride=2)
    assert torch.equal(interpolated.detach(), torch.tensor([[[0.0], [2.0], [4.0], [10.0], [16.0]]]))
    interpolated.sum().backward()
    assert knots.grad is not None
    assert torch.count_nonzero(knots.grad) > 0


def test_stateless_random_control_is_independent_of_global_rng_state():
    roi, residual = _logits(batch=1, tubelets=2, patches=13)
    torch.manual_seed(1)
    first = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="random",
        tokens_per_tubelet=5,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="none",
        temperature=0.5,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
        random_seed=91,
    )["indices"]
    torch.manual_seed(999999)
    second = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="random",
        tokens_per_tubelet=5,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="none",
        temperature=0.5,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
        random_seed=91,
    )["indices"]
    assert torch.equal(first, second)


def test_score_function_and_training_contracts_fail_closed_when_semantics_are_invalid():
    roi, residual = _logits(batch=1, tubelets=1, patches=8)
    kwargs = dict(
        roi_logits=roi,
        residual_logits=residual,
        tokens_per_tubelet=4,
        context_tokens=1,
        roi_fraction=0.5,
        training=True,
        temperature=0.5,
        valid_mask=torch.ones_like(roi, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="single-family"):
        select_exact_k(mode="hybrid", estimator="score_function", **kwargs)
    with pytest.raises(ValueError, match="explicit gradient estimator"):
        select_exact_k(mode="roi", estimator="none", **kwargs)
    with pytest.raises(ValueError, match="positive temperature"):
        select_exact_k(
            mode="free",
            estimator="straight_through",
            **{**kwargs, "temperature": 0.0},
        )
    with pytest.raises(ValueError, match="duplicate"):
        ordered_plackett_luce_log_prob(
            roi,
            torch.tensor([[[1, 1]]]),
            temperature=1.0,
        )


def test_exact_k_never_selects_invalid_native_patches_and_fails_when_k_is_too_large():
    roi, residual = _logits(batch=1, tubelets=2, patches=8)
    valid_mask = torch.tensor([[[True, False, True, True, False, True, True, False]]]).expand(1, 2, 8)
    route = select_exact_k(
        roi_logits=roi,
        residual_logits=residual,
        mode="free",
        tokens_per_tubelet=5,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="straight_through",
        temperature=0.5,
        valid_mask=valid_mask,
    )
    assert bool(valid_mask.gather(-1, route["indices"]).all())
    assert route["valid_patch_count_min"] == 5
    with pytest.raises(ValueError, match="valid native patch count"):
        select_exact_k(
            roi_logits=roi,
            residual_logits=residual,
            mode="free",
            tokens_per_tubelet=6,
            context_tokens=0,
            roi_fraction=0.0,
            training=True,
            estimator="straight_through",
            temperature=0.5,
            valid_mask=valid_mask,
        )


def test_uniform_selected_pooling_is_independent_of_route_logits():
    torch.manual_seed(29)
    adapter = GeoRouteSparseTemporalAdapter(channels=8)
    selected = torch.randn(1, 3, 4, 8)
    geometry = torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]).expand(1, 3, 4)
    coordinates = torch.rand(1, 3, 4, 2)
    first = adapter(
        selected,
        torch.randn(1, 3, 4),
        geometry,
        coordinates,
        use_absolute_coordinates=True,
        use_roi_relative_coordinates=True,
        use_geometry_projection=True,
        pooling_mode="uniform_selected",
    )
    second = adapter(
        selected,
        torch.randn(1, 3, 4) * 100.0,
        geometry,
        coordinates,
        use_absolute_coordinates=True,
        use_roi_relative_coordinates=True,
        use_geometry_projection=True,
        pooling_mode="uniform_selected",
    )
    assert torch.equal(first, second)


def test_pl_reaches_unselected_logits_while_selected_only_st_does_not():
    roi = torch.zeros(1, 1, 8)
    st_logits = torch.tensor(
        [[[1.7, -0.4, 0.8, 2.1, -1.2, 0.3, 1.1, -0.7]]],
        requires_grad=True,
    )
    st_route = select_exact_k(
        roi_logits=roi,
        residual_logits=st_logits,
        mode="free",
        tokens_per_tubelet=3,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="straight_through",
        temperature=0.7,
        valid_mask=torch.ones_like(st_logits, dtype=torch.bool),
    )
    st_route["st_gate"].sum().backward()
    assert st_logits.grad is not None
    assert torch.count_nonzero(st_logits.grad.masked_select(st_route["selected_mask"])) > 0
    assert torch.count_nonzero(st_logits.grad.masked_select(~st_route["selected_mask"])) == 0

    pl_logits = st_logits.detach().clone().requires_grad_(True)
    torch.manual_seed(37)
    pl_route = select_exact_k(
        roi_logits=roi,
        residual_logits=pl_logits,
        mode="free",
        tokens_per_tubelet=3,
        context_tokens=0,
        roi_fraction=0.0,
        training=True,
        estimator="score_function",
        temperature=0.7,
        valid_mask=torch.ones_like(pl_logits, dtype=torch.bool),
    )
    pl_route["ordered_log_prob"].sum().backward()
    assert pl_logits.grad is not None
    assert torch.isfinite(pl_logits.grad).all()
    assert torch.count_nonzero(pl_logits.grad.masked_select(pl_route["selected_mask"])) > 0
    assert torch.count_nonzero(pl_logits.grad.masked_select(~pl_route["selected_mask"])) > 0


def _legacy_sparse_adapter_forward(
    adapter,
    selected_features,
    selected_scores,
    geometry,
    selected_coordinates,
):
    relative = (selected_coordinates - geometry[:, :, None, :2]) / geometry[:, :, None, 2:].clamp_min(1e-6)
    coordinate_features = torch.cat((selected_coordinates, relative), dim=-1)
    selected_features = selected_features + adapter.coordinate_projection(coordinate_features)
    weights = torch.full_like(
        selected_scores,
        1.0 / float(selected_scores.shape[-1]),
    ).unsqueeze(-1)
    pooled = (weights * selected_features).sum(dim=2)
    pooled = adapter.norm(pooled + adapter.geometry_projection(geometry))
    temporal = adapter.output(adapter.temporal(pooled.transpose(1, 2))).transpose(1, 2)
    return (pooled + temporal).transpose(1, 2)


def test_sparse_adapter_representation_channels_are_independently_isolated():
    torch.manual_seed(41)
    adapter = GeoRouteSparseTemporalAdapter(channels=8)
    selected = torch.randn(1, 3, 4, 8)
    scores = torch.randn(1, 3, 4)
    geometry = (
        torch.tensor(
            [[[0.4, 0.6, 0.7, 0.8]]],
            dtype=torch.float32,
        )
        .expand(1, 3, 4)
        .clone()
        .requires_grad_(True)
    )
    coordinates = torch.rand(1, 3, 4, 2, requires_grad=True)

    disabled = adapter(
        selected,
        scores,
        geometry,
        coordinates,
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    changed = adapter(
        selected,
        scores,
        geometry.detach() + 0.05,
        (coordinates.detach() + 0.1).clamp_max(1.0),
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    assert torch.equal(disabled, changed)
    geometry_grad, coordinate_grad = torch.autograd.grad(
        disabled.sum(),
        (geometry, coordinates),
        allow_unused=True,
    )
    assert geometry_grad is None
    assert coordinate_grad is None

    legacy = _legacy_sparse_adapter_forward(
        adapter,
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
    )
    split_all_enabled = adapter(
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
        use_absolute_coordinates=True,
        use_roi_relative_coordinates=True,
        use_geometry_projection=True,
        pooling_mode="uniform_selected",
    )
    assert torch.allclose(legacy, split_all_enabled, atol=1e-7, rtol=1e-7)

    absolute_only = adapter(
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
        use_absolute_coordinates=True,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    relative_only = adapter(
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=True,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )
    geometry_only = adapter(
        selected,
        scores,
        geometry.detach(),
        coordinates.detach(),
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=True,
        pooling_mode="uniform_selected",
    )
    assert not torch.equal(absolute_only, disabled)
    assert not torch.equal(relative_only, disabled)
    assert not torch.equal(geometry_only, disabled)


def test_free_route_uses_fixed_full_frame_geometry_without_geometry_gradient():
    class FakeScout:
        def __init__(self):
            self.geometry_logits = torch.randn(1, 2, 4, requires_grad=True)
            self.residual_logits = torch.randn(1, 2, 6, requires_grad=True)

        def __call__(self, _scout, *, source_grid_hw):
            assert source_grid_hw == (2, 3)
            return self.geometry_logits, self.residual_logits

    fake = type("FreeRouteFields", (), {})()
    fake.route_mode = "free"
    fake.geometry_side_channel = False
    fake.window_size = 4
    fake.tubelet_size = 2
    fake.source_mean = torch.zeros(1, 3, 1, 1, 1)
    fake.source_std = torch.ones(1, 3, 1, 1, 1)
    fake.scout = FakeScout()
    fake._fixed_full_frame_geometry = GeoRouteBackboneWrapper._fixed_full_frame_geometry
    fields = GeoRouteBackboneWrapper._compute_route_fields(
        fake,
        torch.zeros(1, 3, 4, 8, 8, dtype=torch.uint8),
        source_grid_hw=(2, 3),
    )
    geometry, residual, regularization = fields

    expected = torch.tensor([0.5, 0.5, 1.0, 1.0]).view(1, 1, 4)
    assert torch.equal(geometry, expected.expand(1, 2, 4))
    assert regularization.item() == 0.0
    residual.sum().backward()
    assert fake.scout.residual_logits.grad is not None
    assert fake.scout.geometry_logits.grad is None


def test_route_scout_stays_fp32_inside_outer_autocast():
    fake = type("AutocastRouteFields", (), {})()
    fake.route_mode = "free"
    fake.geometry_side_channel = False
    fake.window_size = 4
    fake.tubelet_size = 2
    fake.source_mean = torch.zeros(1, 3, 1, 1, 1)
    fake.source_std = torch.ones(1, 3, 1, 1, 1)
    fake.scout = GeoRouteScout(channels=8)
    fake._fixed_full_frame_geometry = GeoRouteBackboneWrapper._fixed_full_frame_geometry

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        (
            geometry,
            residual,
            regularization,
        ) = GeoRouteBackboneWrapper._compute_route_fields(
            fake,
            torch.zeros(
                1,
                3,
                4,
                16,
                16,
                dtype=torch.uint8,
            ),
            source_grid_hw=(2, 3),
        )

    assert geometry.dtype == torch.float32
    assert residual.dtype == torch.float32
    assert regularization.dtype == torch.float32
