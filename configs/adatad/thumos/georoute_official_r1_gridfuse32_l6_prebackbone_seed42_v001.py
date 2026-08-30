_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]


model = dict(
    backbone=dict(
        backbone=dict(
            gridfuse32_l6=dict(
                schema_version="zoomtoken_gridfuse32_l6_v001",
                enabled=True,
                dense_block_indices=tuple(range(6)),
                fused_block_indices=tuple(range(6, 12)),
                even_pairing="horizontal",
                odd_pairing="vertical",
                native_tokens_per_clip=512,
                merged_tokens_per_clip=256,
                completion="broadcast_residual_delta",
            )
        )
    )
)


gridfuse32_l6_contract = dict(
    schema_version="zoomtoken_gridfuse32_l6_contract_v001",
    unique_task="ZOOMTOKEN-GRIDFUSE32-L6-GATED-v001",
    source_base="2d945e64bdccd09ae2e2916524562e3f388c5a2a",
    seed=42,
    native_support="R1_contiguous_hole_free_8x8_K64",
    tubelets_per_clip=8,
    native_tokens_per_clip=512,
    dense_block_indices=tuple(range(6)),
    fused_block_indices=tuple(range(6, 12)),
    merged_tokens_per_clip=256,
    attention_qkv_execution="all_256_merged_tokens",
    mlp_execution="all_256_merged_tokens",
    native_completion="current_native_residual_plus_broadcast_pair_delta",
    adapter_execution="all_512_restored_native_tokens",
    cross_tubelet_pairing=False,
    cross_clip_state=False,
    content_routing=False,
    new_trainable_parameters=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    training_allowed_only_after_g0=True,
)


gridfuse32_l6_gates = dict(
    g0=dict(
        batch_size=1,
        native_tokens=512,
        candidate_tokens=256,
        embed_dims=384,
        num_heads=6,
        warmup_iterations=100,
        minimum_timed_iterations=500,
        p50_speedup_min=1.35,
        peak_allocated_ratio_max=1.05,
        peak_reserved_ratio_max=1.05,
        p95_report_only=True,
        failure_decision="STOP_GRIDFUSE32_L6_BEFORE_TRAINING",
    ),
    g1=dict(
        enabled_only_after_g0=True,
        seed=42,
        max_epochs=60,
        rank_count=2,
        global_batch_size=2,
        local_batch_size=1,
        reference_final_ema=(69.07, 61.14, 46.57),
        average_map_min=68.57,
        map_06_min=60.64,
        map_07_min=46.07,
        short_action_delta_pp_min=-0.75,
        start_end_boundary_ratio_max=1.05,
    ),
    g2=dict(
        enabled_only_after_g1=True,
        canonical_video_count=211,
        canonical_loader_item_count=792,
        pass_order=("R1", "C", "C", "R1", "C", "R1", "R1", "C"),
        full_passes_per_arm=4,
        full_stack="decode_to_soft_nms",
        p50_ratio_max=0.95,
        gross_energy_ratio_max=0.95,
        peak_allocated_ratio_max=1.05,
        peak_reserved_ratio_max=1.05,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_gridfuse32_l6_prebackbone_seed42_unbound"
)
