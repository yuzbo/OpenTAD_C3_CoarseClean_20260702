_base_ = ["./duca_two_stage_pretrained_frozen_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_GLOBAL_CURRICULUM_G1_PROTECTED_FIXED384_OFFICIAL60",
    optimization_variant="p0_initialized_global_policy_with_protected_detector_feedback",
    acquisition_policy="global_structured_topk",
    detector_gradient_bridge="protected_structured_transport",
    detector_gradient_final_weight=0.25,
    detector_gradient_updates="transition_scorer_only",
    coarse_probe_training="p0_pretrained_then_frozen",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        detector_gradient_mode="protected_structured_transport",
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_global_curriculum_g1_protected_fixed384_official60"
)
