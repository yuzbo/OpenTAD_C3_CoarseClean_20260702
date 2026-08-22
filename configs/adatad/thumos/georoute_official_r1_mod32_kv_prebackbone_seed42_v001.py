_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]
official_bc_arm = "R1-MOD32-KV"
model = dict(backbone=dict(custom=dict(
    georoute_official_support="strict_rect8x8",
    zoomtoken_refresh_carry_mode="mod32_kv",
    zoomtoken_query_tokens=32, zoomtoken_kv_tokens=64,
    zoomtoken_mlp_tokens=32,
)))
zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_refresh_compute_v001",
    arm_surface="R1-MOD32-KV",
    support="strict_rect8x8_k64_refresh32_kv64",
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    executed_token_contract="raw_native_prepatch_k64_q32_kv64_mlp32_true_ragged",
)
official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="mod32_kv",
    strict_rectangle_support_tokens=64,
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
)
work_dir = "exps/thumos/adatad/georoute_official_r1_mod32_kv_prebackbone_seed42_v001"
