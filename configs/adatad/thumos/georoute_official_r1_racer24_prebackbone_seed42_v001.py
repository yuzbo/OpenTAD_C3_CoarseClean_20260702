_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-RACER24"

model = dict(
    backbone=dict(
        backbone=dict(
            racer24=dict(
                enabled=True,
                schema_version="zoomtoken_racer24_iteration0_v001",
                racer_blocks=(4, 6, 8, 10),
                tubelets_per_clip=8,
                spatial_tokens_per_tubelet=64,
                selected_per_tubelet=24,
                selected_query_tokens_per_clip=192,
                full_kv_tokens_per_clip=512,
                completion="parameter_free_key_residual_entropy",
                router="preceding_dense_residual_plus_adjacent_surprise",
            )
        )
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_racer24_iteration0_config_v001",
    arm_surface="RACER24",
    support="strict_rect8x8_k64_dense512_racer24",
    racer_blocks=(4, 6, 8, 10),
    selected_per_tubelet=24,
    spatial_tokens_per_tubelet=64,
    selected_query_tokens_per_clip=192,
    full_kv_tokens_per_clip=512,
    completion="parameter_free_key_residual_entropy",
    new_trainable_parameters=False,
    auxiliary_loss_enabled=False,
    teacher_enabled=False,
    cross_clip_state_enabled=False,
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="racer24",
    strict_rectangle_support_tokens=64,
    dense_carrier_tokens_per_clip=512,
    racer_blocks=(4, 6, 8, 10),
    query_tokens_per_tubelet=24,
    kv_tokens_per_tubelet=64,
    mlp_tokens_per_tubelet=24,
    selected_query_tokens_per_clip=192,
    full_kv_tokens_per_clip=512,
    adapter_tokens_per_clip=512,
    no_global_topk=True,
    hidden_state_carry=False,
    new_trainable_module=False,
    selector_inclusive_cost_measured=False,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_racer24_prebackbone_seed42_iteration0_unbound"
)
