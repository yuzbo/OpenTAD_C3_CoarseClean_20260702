_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-APM32-CTX64"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="strict_rect8x8",
            zoomtoken_refresh_carry_mode="apm32_ctx64",
            zoomtoken_query_tokens=32,
            zoomtoken_kv_tokens=64,
            zoomtoken_mlp_tokens=32,
            zoomtoken_temporal_carry=False,
            zoomtoken_carry_detach=False,
            zoomtoken_carry_mix_per_block=False,
        )
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_apm32_ctx64_v001",
    arm_surface="R1-APM32-CTX64",
    support="strict_rect8x8_k64_apm_refresh32_ctx64",
    temporal_memory="one_previous_tubelet_pre_position_patch_embedding",
    temporal_memory_detached=True,
    clip_reset_tubelets=8,
    local_search_radius=2,
    mutual_nearest_similarity_threshold=0.80,
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    fallback_query_tokens=64,
    new_trainable_parameters=0,
    executed_token_contract=(
        "raw_native_prepatch_k64_apm32_or_k64_fallback_"
        "ctx64_adapter64_true_ragged"
    ),
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="apm32_ctx64",
    strict_rectangle_support_tokens=64,
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    carrier="aligned_previous_plus_current_residual_before_current_position",
    no_hidden_or_kv_cache=True,
    no_new_module_or_loss=True,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_apm32_ctx64_prebackbone_seed42_v001"
)
