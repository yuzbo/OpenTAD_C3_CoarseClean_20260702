"""Matched native-source dense surface for the seed-3407 P1 screen.

The official dense reproduction remains the upstream ``DO`` config.  ``DN``
uses the same GeoRoute source preprocessing and detector context as Q, but
disables routing and executes the complete native spatial support.  The formal
development runner must still bind the Fit/Gate population and matched recipe.
"""

_base_ = ["./georoute_adatad_development_base.py"]

window_size = 768
tubelets = window_size // 2
matched_window_budget = tubelets * 64

model = dict(
    backbone=dict(
        custom=dict(
            georoute_route_mode="dense",
            georoute_policy_estimator="none",
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
    arm_surface="DN",
    seed=3407,
    runner_binding_required=True,
    route_mode="dense",
    matched_native_source=True,
    routing_enabled=False,
    full_native_spatial_compute=True,
    official_dense_anchor=False,
    matched_sparse_window_budget=matched_window_budget,
    executed_token_contract="full_native_spatial_support",
    window_token_budget_applies_to_execution=False,
    q_dynamic_k_t=False,
    split="Fit/train_to_Gate/development",
    official_test_open_allowed=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    performance_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/zoomtoken_p1_dn_seed3407_unbound"
