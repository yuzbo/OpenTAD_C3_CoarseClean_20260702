"""H65-60 Stage 2: 20-epoch cosine transition plus 20-epoch full joint training."""

_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

duca_stage2_transition_steps = 2000
duca_stage2_total_steps = 4000

duca_sampling_rate_contract = dict(
    route="DUCA_H65_60_STAGE2_TRANSITION20_JOINT20",
    stage="low_lr_joint_rate_adaptation20_then_tad_led_joint20",
    stage1_initialization="full_uniform_k384_epoch19_ema_model",
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
            # Historical H65 uses 1,000 + 2,000 steps. Scaling both intervals
            # by 2/3 preserves their ratio and completes feedback at step 2,000.
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=667,
                transition_steps=1333,
            ),
            detector_contribution=dict(
                start=0.0,
                end=1.0,
                warmup_steps=667,
                transition_steps=1333,
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
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=3,
    max_epoch=40,
)

workflow = dict(
    training_profile="duca_h65_60_stage2_transition20_joint20",
    end_epoch=40,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=duca_stage2_total_steps,
    primary_checkpoint_epoch=39,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_39_state_dict_ema",
    model_initialization=dict(
        expected_checkpoint_epoch=19,
    ),
)

seed = 3407
total_epochs = 40
max_updates = 4000
checkpoint_interval_epochs = 5
checkpoint_policy = dict(
    resumable=True,
    keep_latest=3,
    milestones=True,
    final=True,
    final_ema=True,
)
paper_claim_allowed = False
work_dir = "exps/thumos/adatad/duca_h65_60_stage2_transition20_joint20"

