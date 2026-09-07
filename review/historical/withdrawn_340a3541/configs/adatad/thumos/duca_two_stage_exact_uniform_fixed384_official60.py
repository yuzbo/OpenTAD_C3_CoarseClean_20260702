_base_ = ["./duca_two_stage_joint_fixed384_official60_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_EXACT_UNIFORM_FIXED384_OFFICIAL60",
    frontend_initialization="none",
    detector_pretraining_policy="exact_uniform_k384",
    detector_pretraining_updates=6000,
    policy_uniform_steps=6000,
    policy_ramp_steps=0,
    detector_bridge_delay_steps=6000,
    detector_bridge_ramp_steps=0,
)


model = dict(
    frame_selector=dict(
        detector_gradient_mode="none",
        inference_policy_alpha=0.0,
        loss_weights=dict(
            actionness=0.0,
            transition=0.0,
            transition_boundary=0.0,
        ),
        loss_weight_schedule=dict(
            actionness=dict(start=0.0, end=0.0),
            transition=dict(start=0.0, end=0.0),
            transition_boundary=dict(start=0.0, end=0.0),
            policy_alpha=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_two_stage_exact_uniform_fixed384_official60"
