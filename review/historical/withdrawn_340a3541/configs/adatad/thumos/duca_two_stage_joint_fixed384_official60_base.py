_base_ = ["./duca_protected_e2e_fixed384_official60.py"]


duca_uniform_detector_warmup_steps = 1000
duca_policy_transition_steps = 1500
duca_detector_bridge_warmup_steps = (
    duca_uniform_detector_warmup_steps + duca_policy_transition_steps
)
duca_detector_bridge_transition_steps = 1500


duca_transition_only_contract = dict(
    route="DUCA_TWO_STAGE_SCRATCH_FIXED384_OFFICIAL60",
    stage="uniform_detector_cowarmup_then_joint_detection",
    training_profile="official60",
    frontend_initialization="none",
    detector_pretraining_policy="exact_uniform_k384",
    detector_pretraining_updates=duca_uniform_detector_warmup_steps,
    detector_pretraining_frontend_updates=False,
    detector_pretraining_gradient_clipping_isolation="frontend_losses_exactly_zero",
    detector_total_successful_updates=6000,
    detector_extra_updates_outside_official60=0,
    policy_uniform_steps=duca_uniform_detector_warmup_steps,
    policy_ramp_steps=duca_policy_transition_steps,
    detector_bridge_delay_steps=duca_detector_bridge_warmup_steps,
    detector_bridge_ramp_steps=duca_detector_bridge_transition_steps,
    stage1_frontend_pretraining_loss_weights_selected_on_train_holdout=True,
    official60_uniform_warmup_frontend_loss_weights=dict(
        actionness=0.0,
        transition=0.0,
        transition_boundary=0.0,
    ),
    late_frontend_loss_weights=dict(
        actionness=0.10,
        transition=0.02,
        transition_boundary=2.0,
    ),
    detector_loss_weight=1.0,
    detector_loss_weight_schedule="constant",
    detector_gradient_updates="transition_scorer_only",
    action_head_detector_gradient=False,
    asformer_trunk_detector_gradient=False,
    paper_claim_allowed=False,
)


model = dict(
    selector_train_only=False,
    selector_train_only_skip_detector=False,
    frame_selector=dict(
        training_uniform_companion_fraction=0.0,
        inference_policy_alpha=1.0,
        loss_weights=dict(
            _delete_=True,
            actionness=0.0,
            transition=0.0,
            transition_boundary=0.0,
        ),
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="cosine",
            warmup_steps=duca_uniform_detector_warmup_steps,
            transition_steps=duca_policy_transition_steps,
            actionness=dict(
                start=0.0,
                end=0.10,
                warmup_steps=duca_uniform_detector_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
            transition=dict(
                start=0.0,
                end=0.02,
                warmup_steps=duca_uniform_detector_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
            transition_boundary=dict(
                start=0.0,
                end=2.0,
                warmup_steps=duca_uniform_detector_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=duca_uniform_detector_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=duca_detector_bridge_warmup_steps,
                transition_steps=duca_detector_bridge_transition_steps,
            ),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_two_stage_joint_fixed384_official60_base"
