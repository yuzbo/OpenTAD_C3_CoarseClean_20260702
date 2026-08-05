from __future__ import annotations

import pytest
import torch

from opentad.models.backbones.georoute_routing import (
    GEOROUTE_DYNAMIC_ROUTING_SCHEMA,
    global_sigmoid_budget_projection,
    roi_modifier_from_geometry,
    select_dynamic_global_exact_budget,
)
from opentad.models.backbones.georoute_wrapper import (
    GeoRouteBackboneWrapper,
    GeoRouteScout,
    GeoRouteSparseTemporalAdapter,
    dynamic_proxy_weight_at_step,
)
from opentad.models.backbones.vit_adapter import VisionTransformerAdapter


def test_dynamic_global_route_induces_zero_kt_and_all_three_roles():
    q_base = torch.full((1, 3, 4), -10.0)
    delta_roi = torch.full_like(q_base, -1.0)
    delta_residual = torch.full_like(q_base, -1.0)
    q_base[0, 0, 0] = 10.0
    q_base[0, 0, 1] = 8.0
    q_base[0, 0, 2] = 7.0
    delta_roi[0, 0, 1] = 3.0
    delta_residual[0, 0, 2] = 5.0

    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=3,
        training=False,
        estimator="none",
        temperature=0.7,
        valid_mask=torch.ones_like(q_base, dtype=torch.bool),
    )

    assert route["schema_version"] == GEOROUTE_DYNAMIC_ROUTING_SCHEMA
    assert torch.equal(route["physical_indices"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(route["tubelet_indices"], torch.tensor([[0, 0, 0]]))
    assert torch.equal(route["spatial_indices"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(route["k_per_tubelet"], torch.tensor([[3, 0, 0]]))
    assert torch.equal(route["selected_role_ids"], torch.tensor([[0, 1, 2]]))
    assert torch.equal(route["role_counts_per_window"], torch.tensor([[1, 1, 1]]))
    assert route["role_counts"] == {"context": 1, "roi": 1, "residual": 1}
    assert route["padded_token_count"] == 0
    assert int(route["selected_mask"].sum()) == 3
    assert torch.equal(route["st_gate"], torch.ones_like(route["st_gate"]))
    assert route["soft_probability"] is None


def test_dynamic_route_selects_one_physical_copy_when_role_modifiers_compete():
    q_base = torch.zeros(1, 2, 3)
    delta_roi = torch.full_like(q_base, -2.0)
    delta_residual = torch.full_like(q_base, -2.0)
    delta_roi[0, 1, 1] = 8.0
    delta_residual[0, 1, 1] = 7.0

    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=1,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=torch.ones_like(q_base, dtype=torch.bool),
    )

    assert torch.equal(route["physical_indices"], torch.tensor([[4]]))
    assert torch.equal(route["selected_role_ids"], torch.tensor([[1]]))
    assert int(route["selected_mask"].sum()) == 1


def test_global_soft_budget_is_strict_exact_sum_shift_invariant_and_dense_gradient():
    torch.manual_seed(71)
    scores = torch.randn(2, 3, 5, dtype=torch.float64, requires_grad=True)
    valid = torch.ones_like(scores, dtype=torch.bool)
    valid[0, 2, 4] = False

    probability = global_sigmoid_budget_projection(
        scores,
        valid_mask=valid,
        window_budget=6,
        temperature=0.8,
    )
    shifted = global_sigmoid_budget_projection(
        scores.detach() + 13.0,
        valid_mask=valid,
        window_budget=6,
        temperature=0.8,
    )

    assert torch.equal(probability.masked_select(~valid), torch.zeros(1, dtype=torch.float64))
    assert torch.all(probability.masked_select(valid) > 0.0)
    assert torch.all(probability.masked_select(valid) < 1.0)
    assert torch.allclose(
        probability.sum(dim=(1, 2)),
        torch.tensor([6.0, 6.0], dtype=torch.float64),
        atol=1e-10,
        rtol=0.0,
    )
    assert torch.allclose(probability, shifted, atol=1e-10, rtol=1e-10)

    weights = torch.arange(probability.numel(), dtype=probability.dtype).reshape_as(probability)
    (probability * weights).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()
    assert torch.count_nonzero(scores.grad.masked_select(valid)) == int(valid.sum())
    assert torch.count_nonzero(scores.grad.masked_select(~valid)) == 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_global_soft_budget_promotes_amp_scores_to_strict_fp32(dtype):
    scores = torch.tensor(
        [[[40.0, 20.0, 0.0, -20.0, -40.0]]],
        dtype=dtype,
        requires_grad=True,
    )
    probability = global_sigmoid_budget_projection(
        scores,
        valid_mask=torch.ones_like(scores, dtype=torch.bool),
        window_budget=2,
        temperature=0.5,
    )
    assert probability.dtype == torch.float32
    assert torch.all(probability > 0.0)
    assert torch.all(probability < 1.0)
    assert probability.sum().item() == pytest.approx(2.0, abs=1e-5)
    (probability * torch.arange(5, dtype=probability.dtype)).sum().backward()
    assert scores.grad is not None
    assert torch.isfinite(scores.grad).all()


def test_dynamic_st_forward_is_hard_and_backward_reaches_unselected_candidates():
    torch.manual_seed(73)
    q_base = torch.randn(1, 3, 6, requires_grad=True)
    delta_roi = torch.randn(1, 3, 6, requires_grad=True)
    delta_residual = torch.randn(1, 3, 6, requires_grad=True)
    valid = torch.ones_like(q_base, dtype=torch.bool)

    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=7,
        training=True,
        estimator="straight_through",
        temperature=0.6,
        valid_mask=valid,
    )

    assert torch.equal(route["st_gate"].detach(), torch.ones_like(route["st_gate"]))
    assert torch.allclose(
        route["soft_probability"].sum(dim=(1, 2)),
        torch.tensor([7.0]),
        atol=1e-4,
        rtol=0.0,
    )
    selected_weights = torch.arange(
        1,
        route["st_gate"].numel() + 1,
        dtype=route["st_gate"].dtype,
    ).reshape_as(route["st_gate"])
    (route["st_gate"] * selected_weights).sum().backward()

    for value in (q_base, delta_roi, delta_residual):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0
    assert torch.count_nonzero(q_base.grad.masked_select(~route["selected_mask"])) > 0


def test_signed_roi_modifier_keeps_context_identifiable_outside_the_box():
    geometry = torch.tensor(
        [[[0.5, 0.5, 0.25, 0.25], [0.5, 0.5, 1.0, 1.0]]]
    )
    modifier = roi_modifier_from_geometry(
        geometry,
        grid_height=2,
        grid_width=2,
        temperature=0.5,
    )

    assert torch.all(modifier[:, 0] < 0.0)
    assert torch.all(modifier[:, 1] > 0.0)


def test_signed_roi_modifier_zero_contour_uses_half_of_full_extent():
    # The two patch centres are x={0.25,0.75}; for a decoded full width 0.5,
    # both lie exactly on the horizontal ellipse boundary around cx=0.5.
    geometry = torch.tensor([[[0.5, 0.5, 0.5, 1.0]]])
    modifier = roi_modifier_from_geometry(
        geometry,
        grid_height=1,
        grid_width=2,
        temperature=0.5,
    )
    assert torch.allclose(modifier, torch.zeros_like(modifier), atol=1e-7)


@pytest.mark.parametrize(
    ("budget", "message"),
    [(0, "0 < window budget"), (6, "0 < window budget")],
)
def test_dynamic_route_rejects_zero_or_capacity_saturating_budget(budget, message):
    utility = torch.zeros(1, 2, 3)
    with pytest.raises(ValueError, match=message):
        select_dynamic_global_exact_budget(
            q_base=utility,
            delta_roi=utility,
            delta_residual=utility,
            window_budget=budget,
            training=False,
            estimator="none",
            temperature=0.5,
            valid_mask=torch.ones_like(utility, dtype=torch.bool),
        )


def test_dynamic_training_rejects_none_or_pl_estimator():
    utility = torch.zeros(1, 2, 3)
    common = dict(
        q_base=utility,
        delta_roi=utility,
        delta_residual=utility,
        window_budget=2,
        training=True,
        temperature=0.5,
        valid_mask=torch.ones_like(utility, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="requires straight_through"):
        select_dynamic_global_exact_budget(estimator="none", **common)
    with pytest.raises(ValueError, match="separate ablation"):
        select_dynamic_global_exact_budget(estimator="score_function", **common)


def _tiny_videomae(*, adapter: bool) -> VisionTransformerAdapter:
    torch.manual_seed(79)
    model = VisionTransformerAdapter(
        img_size=32,
        patch_size=16,
        embed_dims=8,
        depth=1,
        num_heads=2,
        mlp_ratio=2.0,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_frames=4,
        tubelet_size=2,
        total_frames=8,
        adapter_index=[0] if adapter else [],
        adapter_mlp_ratio=0.5,
        return_feat_map=True,
        with_cp=False,
    )
    model.eval()
    return model


def test_native_ragged_videomae_has_full_token_packed_parity():
    model = _tiny_videomae(adapter=True)
    native = torch.randn(1, 4, 4, 3, 2, 16, 16)
    spatial_indices = torch.arange(4).view(1, 1, 4).expand(1, 4, 4)
    physical_indices = torch.arange(16).view(1, 16)

    packed = model.forward_native_packed(
        native,
        spatial_indices,
        source_grid_hw=(2, 2),
        use_absolute_position=True,
    )
    ragged = model.forward_native_ragged(
        native.reshape(1, 16, 3, 2, 16, 16),
        physical_indices,
        total_tubelets=4,
        source_grid_hw=(2, 2),
        use_absolute_position=True,
    )

    assert torch.allclose(
        ragged,
        packed.reshape(1, 16, 8),
        atol=1e-5,
        rtol=1e-5,
    )
    summary = model.latest_native_packed_summary
    assert summary["schema_version"] == "videomae_native_ragged_v1"
    assert summary["padded_heavy_tokens_per_window"] == 0
    assert summary["executed_patch_tokens_per_window"] == 16
    assert summary["clip_token_counts"] == [[8, 8]]
    assert summary["attention_pairs_per_window"] == [128]


def test_native_ragged_videomae_skips_empty_clip_without_dummy_execution():
    model = _tiny_videomae(adapter=False)
    selected_native = torch.randn(1, 2, 3, 2, 16, 16)
    physical_indices = torch.tensor([[0, 1]])

    output = model.forward_native_ragged(
        selected_native,
        physical_indices,
        total_tubelets=4,
        source_grid_hw=(2, 2),
        use_absolute_position=True,
    )

    summary = model.latest_native_packed_summary
    assert output.shape == (1, 2, 8)
    assert summary["clip_token_counts"] == [[2, 0]]
    assert summary["empty_clip_count"] == 1
    assert summary["attention_pairs_per_window"] == [4]
    assert summary["attention_pairs_all_blocks"] == 4
    assert summary["requested_physical_tokens_per_window"] == 2
    assert summary["unique_physical_tokens_per_window"] == 2
    assert summary["padded_heavy_tokens_per_window"] == 0
    assert summary["executed_patch_tokens_per_window"] == 2
    assert summary["ragged_attention_bucket_call_count"] == 1
    assert summary["ragged_mlp_bucket_call_count"] == 1


def test_dynamic_scout_policy_heads_do_not_backpropagate_into_observer_stem():
    torch.manual_seed(83)
    scout = GeoRouteScout(channels=8, dynamic_utility=True)
    value = torch.randn(1, 3, 4, 16, 16)

    geometry, q_base, residual, _features = scout.forward_dynamic(
        value,
        source_grid_hw=(2, 2),
    )
    (geometry.sum() + q_base.sum() + residual.sum()).backward()

    assert all(parameter.grad is None for parameter in scout.stem.parameters())
    for head in (scout.geometry_head, scout.base_utility_head, scout.residual_head):
        assert head is not None
        assert any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
            for parameter in head.parameters()
        )

    scout.zero_grad(set_to_none=True)
    auxiliary_head = torch.nn.Conv1d(8, 3, kernel_size=1)
    *_policy, live_features = scout.forward_dynamic(
        value,
        source_grid_hw=(2, 2),
    )
    auxiliary_head(live_features.mean(dim=(-1, -2))).sum().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in scout.stem.parameters()
    )


def test_dynamic_soft_proxy_detaches_scout_features_but_updates_global_policy():
    torch.manual_seed(89)
    scout_features = torch.randn(1, 3, 2, 2, 2, requires_grad=True)
    policy_scores = torch.randn(1, 2, 4, requires_grad=True)
    probability = global_sigmoid_budget_projection(
        policy_scores,
        valid_mask=torch.ones_like(policy_scores, dtype=torch.bool),
        window_budget=3,
        temperature=0.6,
    )
    proxy = GeoRouteBackboneWrapper._dynamic_soft_proxy_features(
        scout_features,
        probability,
        source_grid_hw=(2, 2),
    )
    weights = torch.arange(proxy.numel(), dtype=proxy.dtype).reshape_as(proxy)
    (proxy * weights).sum().backward()

    assert scout_features.grad is None
    assert policy_scores.grad is not None
    assert torch.isfinite(policy_scores.grad).all()
    assert torch.count_nonzero(policy_scores.grad) > 0


def test_masked_zero_ragged_carrier_remains_zero_after_bias_paths():
    torch.manual_seed(97)
    adapter = GeoRouteSparseTemporalAdapter(channels=4)
    with torch.no_grad():
        adapter.temporal.bias.fill_(1.0)
        adapter.output.bias.fill_(2.0)
    selected = torch.randn(1, 2, 4)
    scores = torch.zeros(1, 2)
    geometry = torch.tensor(
        [[[0.5, 0.5, 1.0, 1.0]] * 4],
        dtype=selected.dtype,
    )
    coordinates = torch.tensor([[[0.25, 0.25], [0.75, 0.75]]])
    tubelet_indices = torch.tensor([[0, 2]])

    output, heavy_valid = adapter.forward_ragged(
        selected,
        scores,
        geometry,
        coordinates,
        tubelet_indices,
        use_absolute_coordinates=False,
        use_roi_relative_coordinates=False,
        use_geometry_projection=False,
        pooling_mode="uniform_selected",
    )

    assert torch.equal(
        heavy_valid,
        torch.tensor([[True, False, True, False]]),
    )
    assert torch.equal(output[:, :, 1], torch.zeros_like(output[:, :, 1]))
    assert torch.equal(output[:, :, 3], torch.zeros_like(output[:, :, 3]))


def test_dynamic_physical_gather_preserves_sorted_global_lineage():
    native = torch.arange(1 * 3 * 4 * 1 * 1 * 1 * 1).reshape(
        1,
        3,
        4,
        1,
        1,
        1,
        1,
    )
    selected = GeoRouteBackboneWrapper._gather_selected_native_physical(
        native,
        torch.tensor([[0, 6, 11]]),
    )
    assert torch.equal(selected.flatten(), torch.tensor([0, 6, 11]))
    with pytest.raises(ValueError, match="strictly increasing"):
        GeoRouteBackboneWrapper._gather_selected_native_physical(
            native,
            torch.tensor([[6, 6, 11]]),
        )


def test_dynamic_diagnostic_telemetry_receipts_geometry_roles_and_ragged_cost():
    q_base = torch.full((1, 3, 4), -10.0)
    delta_roi = torch.full_like(q_base, -1.0)
    delta_residual = torch.full_like(q_base, -1.0)
    q_base[0, 0, :3] = torch.tensor([10.0, 8.0, 7.0])
    delta_roi[0, 0, 1] = 3.0
    delta_residual[0, 0, 2] = 5.0
    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=3,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=torch.ones_like(q_base, dtype=torch.bool),
    )
    geometry = torch.tensor(
        [
            [
                [0.50, 0.50, 0.50, 0.50],
                [0.50, 0.50, 1.00, 1.00],
                [0.50, 0.50, 0.75, 0.75],
            ]
        ]
    )
    packed = {
        "schema_version": "videomae_native_ragged_v1",
        "execution_mode": "true_clip_ragged_no_padding",
        "batch_size": 1,
        "total_tubelets": 3,
        "source_grid_hw": [2, 2],
        "spatial_tokens_per_tubelet": 4,
        "window_token_budget": 3,
        "clip_token_counts": [[3, 0]],
        "attention_pairs_per_window": [9],
        "requested_physical_tokens_per_window": 3,
        "unique_physical_tokens_per_window": 3,
        "padded_heavy_tokens_per_window": 0,
        "executed_patch_tokens_per_window": 3,
        "heavy_backbone_forward_count": 1,
        "dense_adapter_forward_count": 0,
        "adapter_execution": "coordinate_lineage_true_ragged",
        "ragged_attention_bucket_call_count": 1,
        "ragged_mlp_bucket_call_count": 1,
    }
    calibration = GeoRouteBackboneWrapper._dynamic_policy_calibration_telemetry(
        route=route,
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        valid_patch_mask=torch.ones_like(q_base, dtype=torch.bool),
    )

    telemetry = GeoRouteBackboneWrapper._dynamic_diagnostic_route_telemetry(
        route=route,
        geometry=geometry,
        source_grid_hw=(2, 2),
        minimum_extent_wh=(0.5, 0.5),
        maximum_extent_wh=(1.0, 1.0),
        packed=packed,
        policy_calibration=calibration,
    )
    telemetry_without_calibration = (
        GeoRouteBackboneWrapper._dynamic_diagnostic_route_telemetry(
            route=route,
            geometry=geometry,
            source_grid_hw=(2, 2),
            minimum_extent_wh=(0.5, 0.5),
            maximum_extent_wh=(1.0, 1.0),
            packed=packed,
        )
    )

    assert (
        telemetry["schema_version"]
        == "georoute_dynamic_diagnostic_window_telemetry_v1"
    )
    assert telemetry["measurement_scope"] == (
        "accuracy_replay_only_excluded_from_timed_cost"
    )
    assert telemetry["k_t"]["values"] == [3, 0, 0]
    assert telemetry["k_t"]["histogram"] == {"0": 2, "3": 1}
    assert telemetry["roles"]["aggregate_counts"] == {
        "context": 1,
        "roi": 1,
        "residual": 1,
    }
    assert telemetry["roles"]["per_tubelet_counts"] == [
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert telemetry["geometry"]["width_floor_saturation_rate"] == pytest.approx(
        1.0 / 3.0
    )
    assert telemetry["geometry"]["height_ceiling_saturation_rate"] == pytest.approx(
        1.0 / 3.0
    )
    assert telemetry["geometry"]["area"]["p50"] == pytest.approx(0.75**2)
    assert telemetry["ragged_execution"]["clip_token_counts"] == [3, 0]
    assert telemetry["ragged_execution"]["attention_pairs"] == 9
    assert telemetry["ragged_execution"]["padded_heavy_tokens"] == 0
    assert telemetry["policy_calibration"]["valid_role_counts"] == {
        "context": 10,
        "roi": 1,
        "residual": 1,
    }
    assert telemetry["policy_calibration"]["selected_role_counts"] == {
        "context": 1,
        "roi": 1,
        "residual": 1,
    }
    assert telemetry["policy_calibration"]["fields"][
        "winner_top1_minus_top2_margin"
    ]["selected"]["p50"] == pytest.approx(3.0)
    assert telemetry["policy_calibration"]["role_target_fractions_used"] is False
    assert "policy_calibration" not in telemetry_without_calibration
    assert telemetry["official_test_opened"] is False

    broken_packed = dict(packed, unique_physical_tokens_per_window=2)
    with pytest.raises(RuntimeError, match="ragged ledger"):
        GeoRouteBackboneWrapper._dynamic_diagnostic_route_telemetry(
            route=route,
            geometry=geometry,
            source_grid_hw=(2, 2),
            minimum_extent_wh=(0.5, 0.5),
            maximum_extent_wh=(1.0, 1.0),
            packed=broken_packed,
        )


def test_dynamic_role_calibration_records_collapse_without_enforcing_quotas():
    q_base = torch.tensor([[[4.0, 3.0], [2.0, 1.0]]])
    delta_roi = torch.full_like(q_base, -1.0)
    delta_residual = torch.full_like(q_base, 2.0)
    valid = torch.ones_like(q_base, dtype=torch.bool)
    route = select_dynamic_global_exact_budget(
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        window_budget=2,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=valid,
    )

    calibration = GeoRouteBackboneWrapper._dynamic_policy_calibration_telemetry(
        route=route,
        q_base=q_base,
        delta_roi=delta_roi,
        delta_residual=delta_residual,
        valid_patch_mask=valid,
    )

    assert calibration["valid_role_counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 4,
    }
    assert calibration["selected_role_counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 2,
    }
    assert calibration["unselected_role_counts"] == {
        "context": 0,
        "roi": 0,
        "residual": 2,
    }
    assert calibration["selected_missing_roles"] == ["context", "roi"]
    assert calibration["selected_dominant_role"] == "residual"
    assert calibration["selected_dominant_role_fraction"] == pytest.approx(1.0)
    assert calibration["fixed_role_quota_used"] is False
    assert calibration["changes_route_or_execution"] is False


def test_dynamic_diagnostic_telemetry_rejects_multi_sample_attribution():
    utility = torch.zeros(2, 2, 2)
    route = select_dynamic_global_exact_budget(
        q_base=utility,
        delta_roi=utility,
        delta_residual=utility,
        window_budget=2,
        training=False,
        estimator="none",
        temperature=0.5,
        valid_mask=torch.ones_like(utility, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="one aligned sample"):
        GeoRouteBackboneWrapper._dynamic_diagnostic_route_telemetry(
            route=route,
            geometry=torch.tensor(
                [
                    [[0.5, 0.5, 0.5, 0.5]] * 2,
                    [[0.5, 0.5, 0.5, 0.5]] * 2,
                ]
            ),
            source_grid_hw=(1, 2),
            minimum_extent_wh=(0.5, 0.5),
            maximum_extent_wh=(1.0, 1.0),
            packed={},
        )


def test_dynamic_proxy_schedule_uses_successful_optimizer_steps():
    common = dict(initial_weight=0.5, anneal_start=10, anneal_end=20)
    assert dynamic_proxy_weight_at_step(0, **common) == pytest.approx(0.5)
    assert dynamic_proxy_weight_at_step(10, **common) == pytest.approx(0.5)
    assert dynamic_proxy_weight_at_step(15, **common) == pytest.approx(0.25)
    assert dynamic_proxy_weight_at_step(20, **common) == pytest.approx(0.0)
    assert dynamic_proxy_weight_at_step(200, **common) == pytest.approx(0.0)


def test_dynamic_auxiliary_loss_consumes_once_and_applies_proxy_annealing():
    auxiliary_logits = torch.randn(1, 2, 4, requires_grad=True)
    proxy_logits = torch.randn(1, 2, 4, requires_grad=True)
    fake = type("DynamicAuxiliaryHarness", (), {})()
    fake.training = True
    fake.route_mode = "dynamic_scnr"
    fake._pending_regularization = {"geometry": torch.zeros((), requires_grad=True)}
    fake._pending_dynamic_auxiliary = {
        "auxiliary_logits": auxiliary_logits,
        "proxy_logits": proxy_logits,
        "successful_update": 15,
    }
    fake.dynamic_aux_num_classes = 2
    fake.dynamic_aux_detector_length = 8
    fake.dynamic_aux_weight = 0.25
    fake.dynamic_proxy_initial_weight = 0.5
    fake.dynamic_proxy_anneal_start = 10
    fake.dynamic_proxy_anneal_end = 20
    fake.latest_georoute_audit = {}

    losses = GeoRouteBackboneWrapper.consume_training_auxiliary_losses(
        fake,
        masks=torch.ones(1, 8, dtype=torch.bool),
        gt_segments=[torch.tensor([[1.0, 5.0]])],
        gt_labels=[torch.tensor([1])],
    )
    assert set(losses) == {
        "georoute_geometry_regularization_loss",
        "georoute_dynamic_auxiliary_loss",
        "georoute_dynamic_soft_proxy_loss",
    }
    assert fake._pending_regularization is None
    assert fake._pending_dynamic_auxiliary is None
    assert fake.latest_georoute_audit["dynamic_proxy_weight"] == pytest.approx(0.25)
    sum(losses.values()).backward()
    assert auxiliary_logits.grad is not None
    assert proxy_logits.grad is not None
    with pytest.raises(RuntimeError, match="preceding training forward"):
        GeoRouteBackboneWrapper.consume_training_auxiliary_losses(
            fake,
            masks=torch.ones(1, 8, dtype=torch.bool),
            gt_segments=[torch.empty(0, 2)],
            gt_labels=[torch.empty(0, dtype=torch.long)],
        )
