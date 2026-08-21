_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


# Both arms instantiate the same route observer, native/ragged VideoMAE path and
# sparse adapter.  The arm files change only georoute_official_support.
model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="georoute_native_packed_v1",
            georoute_official_support="all_native",
            georoute_official_roi_tokens=64,
            georoute_window_size=768,
            georoute_scout_size=96,
            georoute_patch_size=16,
            georoute_tubelet_size=2,
            georoute_tokens_per_tubelet=64,
            georoute_context_tokens=0,
            georoute_roi_fraction=1.0,
            georoute_route_mode="roi",
            georoute_policy_estimator="straight_through",
            georoute_policy_temperature=0.5,
            georoute_roi_temperature=0.25,
            georoute_geometry_stride_tubelets=1,
            georoute_roi_extent_floor_mode="native_cells",
            georoute_roi_extent_floor_cells=1,
            georoute_max_roi_extent=1.0,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=False,
            georoute_roi_relative_coordinates_enabled=False,
            georoute_geometry_projection_enabled=False,
            georoute_geometry_side_channel=False,
            georoute_pooling_mode="uniform_selected",
            georoute_adapter_mode="coordinate_lineage_packed",
            georoute_geometry_smoothness_weight=0.0,
            georoute_area_prior_weight=0.0,
            georoute_dynamic_roi_modifier_enabled=False,
            georoute_dynamic_residual_modifier_enabled=False,
            georoute_diagnostic_telemetry_enabled=False,
            georoute_role_calibration_telemetry_enabled=False,
            georoute_amp_diagnostic_enabled=False,
            georoute_gradient_decomposition_enabled=False,
            georoute_p0_dense_reference_check=False,
            georoute_output_length=768,
            georoute_max_batch_size=1,
            georoute_random_seed=42,
        )
    )
)

# OpenTAD interprets this as the job-global batch, then divides by two ranks.
solver = dict(
    train=dict(batch_size=2),
    val=dict(batch_size=2),
    test=dict(batch_size=2),
)

optimizer = dict(
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="sparse_adapter", lr=2e-4, weight_decay=0.05),
            dict(name="scout.stem", lr=2e-4, weight_decay=0.05),
            dict(name="scout.geometry_head", lr=2e-4, weight_decay=0.05),
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
        ],
        exclude=[
            "backbone",
            "residual_head",
            "base_utility_head",
            "geometry_projection",
            "coordinate_projection",
        ],
    )
)

official_bc_contract = dict(
    upstream_revision="01c58b9f2370e914150cf94d392208a4e211c053",
    official_reference_job="1245842",
    seed=42,
    rank_count=2,
    local_batch_size=1,
    global_batch_size=2,
    source_grid_hw=(10, 10),
    tubelets_per_window=384,
    arm_b_tokens_per_tubelet=100,
    arm_c_tokens_per_tubelet=64,
    native_materialization="before_any_videomae_heavy_block",
    heavy_backbone_execution="same_true_clip_ragged_no_padding_path",
    support_is_only_scientific_difference=True,
    sparse_adapter="GeoRouteSparseTemporalAdapter_uniform_selected",
    residual_enabled=False,
    dynamic_k_t_enabled=False,
    auxiliary_or_proxy_loss_enabled=False,
    adapter_side_channel_enabled=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
)
