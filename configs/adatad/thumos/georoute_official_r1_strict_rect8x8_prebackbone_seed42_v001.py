_base_ = ["./georoute_official_prebackbone_bc_common_seed42_v001.py"]

official_bc_arm = "R1"
model = dict(
    backbone=dict(custom=dict(georoute_official_support="strict_rect8x8"))
)

strict_rectangle_topology = dict(
    source_grid_hw=(10, 10),
    candidate_top_left_rows=(0, 1, 2),
    candidate_top_left_cols=(0, 1, 2),
    block_size_hw=(8, 8),
    tokens_per_tubelet=64,
    hole_count=0,
    categorical_temperature=0.5,
    native_identity_preserved=True,
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
    schema_version="zoomtoken_strict_rectangle_r1_config_v001",
    arm_surface="R1",
    seed=42,
    source_commit=None,
    runner_binding_required=True,
    support="strict_rect8x8",
    support_topology="one_complete_hole_free_8x8_block",
    executed_token_contract="raw_native_prepatch_exact_k64_true_ragged",
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_strict_rect8x8_prebackbone_seed42_unbound"
)
