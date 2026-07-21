_base_ = ["./duca_local_residual_r1_feedback_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_LOCAL_RESIDUAL_R1_UNIFORM_COMPANION_FIXED384_OFFICIAL60",
    optimization_variant="bounded_residual_with_detector_feedback_and_uniform_companion",
    training_uniform_companion_fraction=0.50,
    inference_uses_learned_policy_only=True,
    inference_extra_companion_cost=False,
)


model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.50,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_local_residual_r1_uniform_companion_fixed384_official60"
)
