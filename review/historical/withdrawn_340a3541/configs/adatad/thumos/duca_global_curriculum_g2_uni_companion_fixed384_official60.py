_base_ = ["./duca_global_curriculum_g1_protected_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_GLOBAL_CURRICULUM_G2_UNI_COMPANION_FIXED384_OFFICIAL60",
    optimization_variant="p0_initialized_global_policy_with_matched_uniform_companion",
    acquisition_policy="global_structured_topk",
    detector_gradient_bridge="protected_structured_transport",
    detector_gradient_final_weight=0.25,
    detector_gradient_updates="transition_scorer_only_on_learned_rows",
    training_uniform_companion_fraction=0.50,
    learned_row_detector_gradient_normalization="batch_size_over_learned_rows",
    detector_forward_passes_per_batch=1,
    inference_uses_learned_policy_only=True,
    inference_extra_companion_cost=False,
    coarse_probe_training="p0_pretrained_then_frozen",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.50,
        training_uniform_companion_normalize_learned_gradient=True,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_global_curriculum_g2_uni_companion_fixed384_official60"
)
