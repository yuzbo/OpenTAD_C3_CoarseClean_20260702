_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]

official_bc_arm = "R3"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="continuous_rect_dynamic",
            georoute_r3_soft_membership_temperature=0.025,
            georoute_r3_area_shift_tubelets=0,
        )
    )
)
official_bc_contract = dict(dynamic_k_t_enabled=True)
route_topology = dict(
    source_grid_hw=(10, 10), support="all_hard_members_of_continuous_strict_rectangle",
    min_extent=0.02, max_extent=1.0, initial_center=(0.5, 0.5), initial_extent=(0.8, 0.8),
    inside_topk_or_fill_allowed=False, naturally_dynamic_k_t=True,
    zero_carrier_mode="masked_zero", padding_or_dummy_tokens_allowed=False,
)
r3_budget_contract = dict(
    target_dataset_average_tokens_per_tubelet=64, soft_membership_temperature=0.025,
    augmented_lagrangian_penalty=1.0, dual_initial=0.0, dual_clip=(-4.0, 4.0),
    dual_update="successful_update_mean_at_epoch_end", extra_anti_collapse_loss_enabled=False,
)
workflow = dict(checkpoint_interval=5, checkpoint_policy="recovery_latest3_plus_final", require_successful_update_hook=True, schedule_and_ema_on_success_only=True, max_amp_retries_per_batch=8, fail_on_skipped_update=True)
zoomtoken_recovery = dict(schema_version="zoomtoken_same_cell_recovery_v001", enabled=True, interval_epochs=5, keep_latest=3, save_final=True, full_state=True, same_cell_only=True, unsealed_only=True, seal_marker=".zoomtoken_cell_sealed")
zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r234_config_v001", arm_surface="R3", seed=42,
    source_commit=None, runner_binding_required=True, support="continuous_rect_dynamic",
    support_topology="continuous_strict_rectangle_all_hard_members_dynamic_k_t",
    executed_token_contract="raw_native_prepatch_natural_k_t_true_ragged_masked_zero",
    gt_for_route_allowed=False, teacher_for_route_allowed=False,
    oracle_for_route_allowed=False, raw_prediction_cache_allowed=False,
)
work_dir = "exps/thumos/adatad/georoute_official_r3_continuous_rect_seed42_unbound"
