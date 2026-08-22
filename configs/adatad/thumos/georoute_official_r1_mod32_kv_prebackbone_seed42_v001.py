_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]
official_bc_arm = "R1-MOD32-KV"
model = dict(backbone=dict(custom=dict(
    georoute_official_support="strict_rect8x8",
    zoomtoken_refresh_carry_mode="mod32_kv",
    zoomtoken_query_tokens=32, zoomtoken_kv_tokens=64,
    zoomtoken_mlp_tokens=32,
)))
zoomtoken_p1_config = dict(arm_surface="R1-MOD32-KV", support="strict_rect8x8_k32_kv64", qkv_mlp_width=32)
work_dir = "exps/thumos/adatad/georoute_official_r1_mod32_kv_prebackbone_seed42_v001"
