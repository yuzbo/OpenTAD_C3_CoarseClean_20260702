"""H65-60 attribution: mature Stage-1 plus AM-RPCH25 Stage-2."""

_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]


duca_stage2_transition_steps = 2000
duca_stage2_total_steps = 3000

duca_sampling_rate_contract = dict(
    route="DUCA_H65_60_STAGE2_AM_RPCH25",
    stage="mature_stage1_then_area_matched_relative_plateau_cosine_hold",
    stage1_initialization="full_uniform_k384_epoch29_ema_model",
    optimizer_scheduler_amp_state_reset=True,
    curriculum_only_change=True,
    model_change_allowed=False,
)

model = dict(
    frame_selector=dict(
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="cosine",
            warmup_steps=0,
            transition_steps=duca_stage2_transition_steps,
            actionness=dict(
                start=1.0,
                end=0.25,
                warmup_steps=0,
                transition_steps=duca_stage2_transition_steps,
            ),
            transition=dict(
                start=0.50,
                end=0.10,
                warmup_steps=0,
                transition_steps=duca_stage2_transition_steps,
            ),
            transition_boundary=dict(
                start=2.0,
                end=0.25,
                warmup_steps=0,
                transition_steps=duca_stage2_transition_steps,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=0,
                transition_steps=duca_stage2_transition_steps,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=1000,
                transition_steps=1000,
            ),
            detector_contribution=dict(
                start=0.0,
                end=1.0,
                warmup_steps=1000,
                transition_steps=1000,
            ),
            asformer_adapt=dict(
                start=0.0,
                end=1.0,
                warmup_steps=0,
                transition_steps=duca_stage2_transition_steps,
            ),
        ),
    ),
)

scheduler = dict(
    _delete_=True,
    type="RelativeSuccessfulUpdateLR",
    mode="am_rpch25",
    max_epoch=30,
    total_updates=duca_stage2_total_steps,
    warmup_updates=500,
    plateau_updates=1000,
    decay_updates=1000,
    hold_updates=500,
    terminal_factor=0.25,
    horizon_updates=6000,
)

workflow = dict(
    training_profile="duca_h65_60_stage2_am_rpch25",
    checkpoint_interval=5,
    require_resumable_training_state=True,
    end_epoch=30,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=duca_stage2_total_steps,
    primary_checkpoint_epoch=29,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_29_state_dict_ema",
    intermediate_validation_role="learning_curve_only",
    intermediate_validation_selects_checkpoint=False,
    model_initialization=dict(expected_checkpoint_epoch=29),
)

seed = 3407
total_epochs = 30
max_updates = duca_stage2_total_steps
checkpoint_interval_epochs = 5
checkpoint_policy = dict(
    resumable=True,
    keep_latest=3,
    milestones=True,
    final=True,
    final_ema=True,
)
paper_claim_allowed = False
work_dir = "exps/thumos/adatad/duca_h65_60_stage2_am_rpch25"
