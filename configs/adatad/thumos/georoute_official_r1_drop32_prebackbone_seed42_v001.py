_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]
official_bc_arm = "R1-DROP32"
model = dict(backbone=dict(custom=dict(
    georoute_official_support="strict_rect8x8",
    zoomtoken_refresh_carry_mode="drop32",
    zoomtoken_query_dim=32, zoomtoken_key_dim=32, zoomtoken_value_dim=32,
    zoomtoken_mlp_dim=32,
)))
zoomtoken_p1_config = dict(arm_surface="R1-DROP32", support="strict_rect8x8_k32", qkv_mlp_width=32)
work_dir = "exps/thumos/adatad/georoute_official_r1_drop32_prebackbone_seed42_v001"
