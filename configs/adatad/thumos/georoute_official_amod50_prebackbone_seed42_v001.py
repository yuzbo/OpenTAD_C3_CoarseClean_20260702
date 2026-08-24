_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

official_bc_arm = "AMOD50"

model = dict(
    backbone=dict(
        backbone=dict(
            amod=dict(
                enabled=True,
                schema_version="zoomtoken_videomae_amod_paper_exact_v001",
                capacity=0.5,
                dense_block_indices=(0, 2, 4, 6, 8, 10),
                amod_block_indices=(1, 3, 5, 7, 9, 11),
                query_chunk_size=128,
                routing_score="preceding_dense_attention_column_mean",
                unselected_update="identity_bypass",
            )
        )
    )
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
    schema_version="zoomtoken_videomae_amod_paper_exact_v001",
    arm_surface="AMOD50",
    seed=42,
    source_commit=None,
    runner_binding_required=True,
    support="full_800_token_videomae_grid",
    dense_blocks=(0, 2, 4, 6, 8, 10),
    amod_blocks=(1, 3, 5, 7, 9, 11),
    selected_tokens_per_amod_block=400,
    adapter_execution="dense_full_token_grid",
    temporal_state_reuse=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    future_frame_route_input_allowed=False,
)

official_amod_contract = dict(
    upstream_revision="01c58b9f2370e914150cf94d392208a4e211c053",
    official_reference_job="1245842",
    seed=42,
    rank_count=2,
    local_batch_size=1,
    global_batch_size=2,
    total_tokens=800,
    capacity=0.5,
    routing_score="preceding_dense_attention_column_mean",
    unselected_update="identity_bypass_attention_mlp",
    adapter_execution="dense_all_tokens",
    new_trainable_router=False,
    auxiliary_loss=False,
    temporal_cache=False,
    selector_inclusive_cost_measured=False,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_amod50_prebackbone_seed42_unbound"
)
