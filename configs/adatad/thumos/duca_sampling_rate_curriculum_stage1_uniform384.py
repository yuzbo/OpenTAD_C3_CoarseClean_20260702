_base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]

# Stage 1 is deliberately a real full-model uniform-K=384 warmup.  The
# detector learns on the same observations it will see in the uniform control,
# while the ASFormer coarse probe is optimized only by binary actionness and
# transition supervision.  No learned sampling or detector-to-policy bridge is
# enabled in this phase.
duca_stage1_end_epoch = 30
duca_stage1_steps_per_epoch = 100
duca_stage1_total_steps = duca_stage1_end_epoch * duca_stage1_steps_per_epoch

duca_sampling_rate_contract = dict(
    route="DUCA_RATE_CURRICULUM_STAGE1_UNIFORM384",
    task="offline_temporal_action_detection",
    stage="uniform_k384_full_model_coarse_convergence",
    pre_backbone_plugin=True,
    exact_budget=384,
    detector_policy="exact_uniform",
    coarse_supervision=[
        "binary_actionness",
        "state_transition_distribution",
        "transition_boundary_support",
    ],
    detector_gradient_to_selector=False,
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        inference_policy_alpha=0.0,
        training_uniform_companion_fraction=0.0,
        # Stage 1 has no learned-policy rows, so companion-only gradient
        # normalization is inapplicable and must not leak from the joint base.
        training_uniform_companion_normalize_learned_gradient=False,
        detector_gradient_mode="density_transport_st",
        detector_contribution_distillation_weight=0.0,
        coarse_trunk_lr=5.0e-5,
        action_head_lr=1.0e-4,
        transition_scorer_lr=5.0e-5,
        loss_weights=dict(
            actionness=1.0,
            transition=0.50,
            transition_boundary=2.0,
        ),
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="linear",
            warmup_steps=0,
            transition_steps=1,
            actionness=dict(start=1.0, end=1.0),
            transition=dict(start=0.50, end=0.50),
            transition_boundary=dict(start=2.0, end=2.0),
            policy_alpha=dict(start=0.0, end=0.0),
            detector_gradient=dict(start=0.0, end=0.0),
            detector_contribution=dict(start=0.0, end=0.0),
            asformer_adapt=dict(start=0.0, end=0.0),
        ),
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="asformer_full_encoder",
        ),
    ),
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=3,
    max_epoch=duca_stage1_end_epoch,
)

workflow = dict(
    formal_protocol="",
    training_profile="duca_rate_curriculum_stage1_uniform384",
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_eval_interval_anchor_epoch=5,
    val_start_epoch=4,
    intermediate_validation_role="stage1_learning_curve_only",
    intermediate_validation_selects_checkpoint=False,
    end_epoch=duca_stage1_end_epoch,
    formal_successful_update_contract=False,
    expected_train_batches_per_epoch=duca_stage1_steps_per_epoch,
    expected_successful_optimizer_updates=duca_stage1_total_steps,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=duca_stage1_end_epoch - 1,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_29_state_dict_ema",
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage1_uniform384"
