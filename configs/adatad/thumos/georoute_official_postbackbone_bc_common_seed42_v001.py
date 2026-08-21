_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


# Accuracy-only causal matrix.  The heavy VideoMAE execution remains dense in
# both arms, so these cells cannot support a compute or efficiency claim.
model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="georoute_postbackbone_sparse_aggregation_v1",
            georoute_postbackbone_selection="all",
            georoute_postbackbone_window_size=768,
            georoute_postbackbone_chunk_num=48,
            georoute_postbackbone_tubelet_size=2,
            georoute_postbackbone_source_grid_hw=(10, 10),
            georoute_postbackbone_roi_tokens=64,
            georoute_postbackbone_scout_size=96,
            georoute_postbackbone_roi_temperature=0.25,
            georoute_postbackbone_policy_temperature=0.5,
            georoute_postbackbone_min_roi_extent_cells=1,
            georoute_postbackbone_max_roi_extent=1.0,
            georoute_postbackbone_pooling_mode="uniform_selected",
            georoute_postbackbone_absolute_coordinates_enabled=False,
            georoute_postbackbone_roi_relative_coordinates_enabled=False,
            georoute_postbackbone_geometry_projection_enabled=False,
        )
    )
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
    heavy_backbone_execution="untouched_official_dense_videomae_forward",
    selection_application="post_backbone_pre_aggregation",
    sparse_adapter="GeoRouteSparseTemporalAdapter_uniform_selected",
    roi_k=64,
    residual_enabled=False,
    auxiliary_or_proxy_loss_enabled=False,
    adapter_side_channel_enabled=False,
    efficiency_claim_allowed=False,
)
