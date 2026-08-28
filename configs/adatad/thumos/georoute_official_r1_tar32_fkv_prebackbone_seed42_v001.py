_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-TAR32-FKV"

model = dict(
    backbone=dict(
        backbone=dict(
            tar32_fkv=dict(
                enabled=True,
                schema_version="zoomtoken_r1_tar32_fkv_v001",
                support_tokens_per_tubelet=64,
                selected_tokens_per_tubelet=32,
                dense_block_indices=(0, 2, 4, 6, 8, 10),
                tar32_block_indices=(1, 3, 5, 7, 9, 11),
                query_chunk_size=128,
                routing_score="preceding_dense_attention_column_mean",
            )
        ),
        # The wrapper owns only the physical strict-R1 K64 carrier. TAR32-FKV
        # makes its parameter-free per-block K32 decisions inside VideoMAE.
        custom=dict(
            georoute_official_support="strict_rect8x8",
            zoomtoken_refresh_carry_mode="full64",
            zoomtoken_query_tokens=64,
            zoomtoken_kv_tokens=64,
            zoomtoken_mlp_tokens=64,
        ),
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_tar32_fkv_v001",
    arm_surface="R1-TAR32-FKV",
    seed=42,
    source_commit=None,
    runner_binding_required=True,
    support="strict_rect8x8_k64_tar32_fkv",
    support_topology="one_complete_hole_free_8x8_block",
    dense_blocks=(0, 2, 4, 6, 8, 10),
    tar32_blocks=(1, 3, 5, 7, 9, 11),
    query_tokens_per_tubelet=32,
    kv_tokens_per_tubelet=64,
    mlp_tokens_per_tubelet=32,
    routing_score="preceding_dense_attention_column_mean",
    adapter_execution="full_k64_coordinate_lineage_true_ragged",
    executed_token_contract=(
        "raw_native_prepatch_k64_even_q64kv64mlp64_"
        "odd_q32kv64mlp32_true_ragged"
    ),
    temporal_state_reuse=False,
    new_trainable_router=False,
    auxiliary_loss=False,
    dynamic_token_count=False,
    fallback_route=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    oracle_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    future_frame_route_input_allowed=False,
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    composite_probe="R1-TAR32-FKV",
    strict_rectangle_support_tokens=64,
    dense_block_indices=(0, 2, 4, 6, 8, 10),
    tar32_block_indices=(1, 3, 5, 7, 9, 11),
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    adapter_tokens=64,
    routing_score="preceding_dense_attention_column_mean",
    unselected_update="identity_bypass_attention_mlp",
    temporal_cache=False,
    new_trainable_module=False,
    selector_inclusive_cost_measured=False,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_tar32_fkv_prebackbone_seed42_unbound"
)
