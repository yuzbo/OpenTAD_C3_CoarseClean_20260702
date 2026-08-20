"""Content-only Q surface for the seed-3407 P1 screen.

Q keeps the approved global exact-B production route and disables both
ROI/residual modifiers.  ``K_t`` is induced by the selected physical-token
union, including legal zero-token tubelets, and the heavy path remains the
no-padding ragged executor with a masked-zero carrier.
"""

_base_ = ["./georoute_dynamic_scnr_stage1_base.py"]

window_size = 768
tubelets = window_size // 2
matched_window_budget = tubelets * 64

model = dict(
    backbone=dict(
        custom=dict(
            georoute_route_mode="dynamic_scnr",
            georoute_policy_estimator="straight_through",
            georoute_window_token_budget=matched_window_budget,
            georoute_zero_carrier_mode="masked_zero",
            georoute_branch_calibration_mode="none",
            georoute_dynamic_roi_modifier_enabled=False,
            georoute_dynamic_residual_modifier_enabled=False,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=False,
            georoute_roi_relative_coordinates_enabled=False,
            georoute_geometry_projection_enabled=False,
            georoute_geometry_side_channel=False,
            georoute_random_seed=3407,
        )
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_p1_dnurq_config_v001",
    arm_surface="Q",
    seed=3407,
    runner_binding_required=True,
    route_mode="dynamic_scnr",
    matched_native_source=True,
    routing_enabled=True,
    official_dense_anchor=False,
    exact_window_budget=matched_window_budget,
    window_budget_is_global=True,
    unique_physical_selection=True,
    q_dynamic_k_t=True,
    k_t_zero_allowed=True,
    fixed_per_tubelet_quota=False,
    fixed_role_quota=False,
    zero_carrier="masked_zero_with_explicit_heavy_valid_mask",
    ragged_execution="true_clip_buckets_without_padding_or_dummy_tokens",
    roi_modifier_enabled=False,
    residual_modifier_enabled=False,
    branch_calibration="none",
    conditional_controls_open=False,
    conditional_modifier_map=dict(
        G=dict(roi=True, residual=False, branch_calibration="none"),
        N=dict(
            roi=False,
            residual=True,
            branch_calibration="residual_window_center",
        ),
        F=dict(
            roi=True,
            residual=True,
            branch_calibration="residual_window_center",
        ),
    ),
    split="Fit/train_to_Gate/development",
    official_test_open_allowed=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    performance_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/zoomtoken_p1_q_seed3407_unbound"
