"""Approved dynamic ROI + TokenSelect Hybrid Stage-1 development base.

This opt-in config preserves the fixed-K=64 pilot's total native-token budget
(``384 * 64``) only as a matched window-level compute arm.  It does not impose
a per-tubelet K or a context/ROI/residual quota.  A stage runner must still bind
an immutable development split and work directory before any performance run.
"""

_base_ = ["./georoute_adatad_development_base.py"]

window_size = 768
tubelets = window_size // 2
matched_window_budget = tubelets * 64

model = dict(
    backbone=dict(
        custom=dict(
            georoute_route_mode="dynamic_scnr",
            georoute_policy_estimator="straight_through",
            georoute_policy_temperature=0.5,
            georoute_window_token_budget=matched_window_budget,
            georoute_zero_carrier_mode="masked_zero",
            georoute_branch_calibration_mode="none",
            # F/N/Q causal arms (accepted decision).  F keeps both modifiers.
            # N sets ``georoute_dynamic_roi_modifier_enabled=False``; Q sets both
            # modifier switches False so routing reduces to the shared base
            # utility.  Global exact-B, dynamic K_t, ragged execution and the
            # masked-zero carrier are unchanged across the three sparse arms.
            georoute_dynamic_roi_modifier_enabled=True,
            georoute_dynamic_residual_modifier_enabled=True,
            georoute_dynamic_aux_num_classes=20,
            georoute_dynamic_aux_detector_length=window_size,
            georoute_dynamic_aux_weight=0.25,
            georoute_dynamic_proxy_initial_weight=0.50,
            georoute_dynamic_proxy_anneal_start=1600,
            georoute_dynamic_proxy_anneal_end=3200,
            georoute_roi_extent_floor_mode="native_cells",
            georoute_roi_extent_floor_cells=1,
            georoute_geometry_smoothness_weight=0.0,
            georoute_area_prior_weight=0.0,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=False,
            georoute_roi_relative_coordinates_enabled=False,
            georoute_geometry_projection_enabled=False,
            georoute_geometry_side_channel=False,
            georoute_diagnostic_telemetry_enabled=False,
            georoute_role_calibration_telemetry_enabled=False,
            georoute_amp_diagnostic_enabled=False,
            georoute_gradient_decomposition_enabled=False,
            georoute_p0_dense_reference_check=False,
            georoute_pooling_mode="uniform_selected",
        )
    )
)

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="dynamic_aux_head", lr=2e-4, weight_decay=0.05),
            dict(name="sparse_adapter", lr=1e-4, weight_decay=0.05),
            dict(name="scout", lr=2e-4, weight_decay=0.05),
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
        ]
    )
)

solver = dict(static_graph=False)

georoute_protocol = dict(
    schema_version="georoute_dynamic_scnr_stage1_v1",
    route="global-exact-budget-dynamic-roi-token-select-hybrid",
    stage="implementation_and_no_performance_admission",
    decision_unit="native_two_frame_videomae_tubelet_patch",
    exact_window_budget=matched_window_budget,
    per_tubelet_quota=False,
    fixed_context_quota=False,
    dynamic_roles=True,
    branch_calibration="none_historical_default",
    k_t_zero_allowed=True,
    zero_carrier="masked_zero_with_explicit_heavy_valid_mask",
    ragged_execution="true_clip_buckets_without_padding_or_dummy_tokens",
    proxy_scope="backward_only_fit_time_annealed_by_successful_updates",
    learned_null_main=False,
    scout_projection_main=False,
    official_test_open_allowed=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
)

work_dir = "exps/thumos/adatad/georoute_dynamic_scnr_stage1_unbound"
