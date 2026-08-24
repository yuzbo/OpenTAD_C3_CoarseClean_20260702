_base_ = ["./georoute_official_r1_strict_rect8x8_prebackbone_seed42_v001.py"]

official_bc_arm = "R1-CUR32-CTX64"
model = dict(
    backbone=dict(
        custom=dict(
            georoute_official_support="strict_rect8x8",
            zoomtoken_refresh_carry_mode="cur32_ctx64",
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
    schema_version="zoomtoken_r1_cur32_ctx64_v001",
    arm_surface="R1-CUR32-CTX64",
    support="strict_rect8x8_k64_cur_refresh32_ctx64",
    temporal_memory="mapping_only_no_memory_substitution",
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
        "raw_native_prepatch_k64_cur32_or_k64_fallback_"
        "ctx64_adapter64_true_ragged"
    ),
)

official_bc_contract = dict(
    support_is_only_scientific_difference=False,
    temporal_refresh_arm="cur32_ctx64",
    strict_rectangle_support_tokens=64,
    query_tokens=32,
    kv_tokens=64,
    mlp_tokens=32,
    carrier="current_patch_embedding_before_current_position",
    no_hidden_or_kv_cache=True,
    no_new_module_or_loss=True,
)

work_dir = (
    "exps/thumos/adatad/"
    "georoute_official_r1_cur32_ctx64_prebackbone_seed42_v001"
)
