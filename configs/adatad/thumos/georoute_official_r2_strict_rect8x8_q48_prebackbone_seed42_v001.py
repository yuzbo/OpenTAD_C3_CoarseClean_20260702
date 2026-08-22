_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]

official_bc_arm = "R2"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="strict_rect8x8_q48",
            georoute_tokens_per_tubelet=48,
        )
    )
)

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="sparse_adapter", lr=2e-4, weight_decay=0.05),
            dict(name="scout.stem", lr=2e-4, weight_decay=0.05),
            dict(name="scout.geometry_head", lr=2e-4, weight_decay=0.05),
            dict(name="scout.base_utility_head", lr=2e-4, weight_decay=0.05),
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
        ],
        exclude=[
            "backbone",
            "residual_head",
            "geometry_projection",
            "coordinate_projection",
        ],
    )
)

route_topology = dict(
    source_grid_hw=(10, 10),
    candidate="one_complete_hole_free_8x8_block",
    candidate_tokens=64,
    selected="stable_q_base_top48_inside_candidate",
    tokens_per_tubelet=48,
    q_base_roi_modifier_enabled=False,
    q_base_residual_modifier_enabled=False,
    q_base_geometry_side_channel_enabled=False,
    padding_or_dummy_tokens_allowed=False,
)

workflow = dict(
    checkpoint_interval=5,
    checkpoint_policy="recovery_latest3_plus_final",
    require_successful_update_hook=True,
    schedule_and_ema_on_success_only=True,
    max_amp_retries_per_batch=8,
    fail_on_skipped_update=True,
)
zoomtoken_recovery = dict(
    schema_version="zoomtoken_same_cell_recovery_v001",
    enabled=True,
    interval_epochs=5,
    keep_latest=3,
    save_final=True,
    full_state=True,
    same_cell_only=True,
    unsealed_only=True,
    seal_marker=".zoomtoken_cell_sealed",
)
zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r234_config_v001",
    arm_surface="R2",
    seed=42,
    source_commit=None,
    runner_binding_required=True,
    support="strict_rect8x8_q48",
    support_topology="complete_8x8_candidate_q_base_top48",
    executed_token_contract="raw_native_prepatch_exact_k48_true_ragged",
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
)
work_dir = "exps/thumos/adatad/georoute_official_r2_q48_seed42_unbound"
