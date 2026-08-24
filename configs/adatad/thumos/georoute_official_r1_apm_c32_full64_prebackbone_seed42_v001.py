_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-APM-C32-FULL64"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="strict_rect8x8",
            zoomtoken_refresh_carry_mode="apm_c32_full64",
            zoomtoken_query_tokens=64,
            zoomtoken_kv_tokens=64,
            zoomtoken_mlp_tokens=64,
            zoomtoken_temporal_carry=False,
            zoomtoken_carry_detach=False,
            zoomtoken_carry_mix_per_block=False,
        )
    )
)

zoomtoken_p1_config = dict(
    schema_version="zoomtoken_r1_apm_c32_full64_v001",
    arm_surface="R1-APM-C32-FULL64",
    support="strict_rect8x8_k64_apm_carrier32_full64",
    temporal_memory="one_previous_tubelet_pre_position_patch_embedding",
    temporal_memory_detached=True,
    clip_reset_tubelets=8,
    local_search_radius=2,
    mutual_nearest_similarity_threshold=0.80,
    memory_carrier_tokens=32,
    deep_update_tokens=64,
    kv_tokens=64,
    adapter_tokens=64,
    fallback_deep_update_tokens=64,
    new_trainable_parameters=0,
    executed_token_contract=(
        "raw_native_prepatch_k64_apm_carrier32_or_current_fallback_"
        "full64_all12_adapter64_true_ragged"
    ),
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="apm_c32_full64",
    strict_rectangle_support_tokens=64,
    memory_carrier_tokens=32,
    query_tokens=64,
    kv_tokens=64,
    mlp_tokens=64,
    adapter_tokens=64,
    carrier="aligned_previous_plus_current_residual_before_current_position",
    no_hidden_or_kv_cache=True,
    no_new_module_or_loss=True,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_apm_c32_full64_prebackbone_seed42_v001"
)
