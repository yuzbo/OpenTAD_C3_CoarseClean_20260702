_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-DSR6-KV"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="strict_rect8x8",
            zoomtoken_refresh_carry_mode="dsr6_kv",
            zoomtoken_query_tokens=32,
            zoomtoken_kv_tokens=64,
            zoomtoken_mlp_tokens=32,
        )
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_depth_staged_refresh_v001",
    arm_surface="R1-DSR6-KV",
    support="strict_rect8x8_k64_dsr6_refresh32_kv64",
    full_update_blocks=(0, 1, 2, 3, 4, 5),
    refresh_update_blocks=(6, 7, 8, 9, 10, 11),
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    executed_token_contract=(
        "raw_native_prepatch_k64_blocks0to5_full_"
        "blocks6to11_q32_kv64_mlp32_true_ragged"
    ),
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="dsr6_kv",
    strict_rectangle_support_tokens=64,
    full_update_blocks=(0, 1, 2, 3, 4, 5),
    refresh_update_blocks=(6, 7, 8, 9, 10, 11),
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    declared_block_flops_proxy_ratio=0.79055,
    selector_inclusive_cost_measured=False,
    hidden_state_carry=False,
    shallow_transport=False,
    new_trainable_module=False,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_dsr6_kv_prebackbone_seed42_v001"
)
