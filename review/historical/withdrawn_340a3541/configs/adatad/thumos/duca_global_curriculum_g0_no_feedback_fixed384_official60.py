_base_ = ["./duca_two_stage_pretrained_frozen_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_GLOBAL_CURRICULUM_G0_NO_FEEDBACK_FIXED384_OFFICIAL60",
    optimization_variant="p0_initialized_global_policy_without_detector_feedback",
    acquisition_policy="global_structured_topk",
    detector_gradient_bridge="none",
    detector_gradient_final_weight=0.0,
    detector_gradient_updates="none",
    coarse_probe_training="p0_pretrained_then_frozen",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        detector_gradient_mode="none",
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.0,
            ),
        ),
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_global_curriculum_g0_no_feedback_fixed384_official60"
)
