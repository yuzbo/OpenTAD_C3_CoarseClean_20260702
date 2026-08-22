_base_ = ["./georoute_official_r1_mod32_kv_prebackbone_seed42_v001.py"]
official_bc_arm = "R1-RC32-KV"
model = dict(backbone=dict(custom=dict(
    zoomtoken_refresh_carry_mode="rc32_kv",
    zoomtoken_temporal_carry=True,
    zoomtoken_carry_detach=True,
    zoomtoken_carry_mix_per_block=True,
)))
zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_refresh_compute_v001",
    arm_surface="R1-RC32-KV",
    temporal_carry="previous_tubelet_same_spatial_index_detached_block_input",
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    executed_token_contract="raw_native_prepatch_k64_q32_kv64_mlp32_detached_carry_true_ragged",
)
official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="rc32_kv",
    strict_rectangle_support_tokens=64,
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    temporal_carry="previous_tubelet_same_spatial_index_detached_block_input",
)

optimizer = dict(
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="sparse_adapter", lr=2e-4, weight_decay=0.05),
            dict(name="scout.stem", lr=2e-4, weight_decay=0.05),
            dict(name="scout.geometry_head", lr=2e-4, weight_decay=0.05),
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(
                name="zoomtoken_refresh_carry_alpha",
                lr=2e-4,
                weight_decay=0.0,
            ),
        ],
        exclude=[
            "backbone",
            "residual_head",
            "base_utility_head",
            "geometry_projection",
            "coordinate_projection",
        ],
    )
)
work_dir = "exps/thumos/adatad/georoute_official_r1_rc32_kv_prebackbone_seed42_v001"
