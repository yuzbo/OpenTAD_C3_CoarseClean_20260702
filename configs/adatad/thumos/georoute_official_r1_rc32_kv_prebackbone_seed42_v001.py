_base_ = ["./georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py"]
official_bc_arm = "R1-RC32-KV"
model = dict(backbone=dict(custom=dict(
    zoomtoken_refresh_carry_mode="rc32_kv",
    zoomtoken_temporal_carry=True,
    zoomtoken_carry_detach=True,
    zoomtoken_carry_mix_per_block=True,
)))
zoomtoken_p1_config = dict(arm_surface="R1-RC32-KV", temporal_carry="previous_tubelet_same_spatial_index_detached_block_input")
work_dir = "exps/thumos/adatad/georoute_official_r1_rc32_kv_prebackbone_seed42_v001"
