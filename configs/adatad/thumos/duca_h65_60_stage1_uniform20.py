"""H65-60 Stage 1: the historical H65 uniform warmup compressed to 20 epochs."""

_base_ = ["./duca_sampling_rate_curriculum_stage1_uniform384.py"]

duca_stage1_end_epoch = 20
duca_stage1_steps_per_epoch = 100
duca_stage1_total_steps = 2000

duca_sampling_rate_contract = dict(
    route="DUCA_H65_60_STAGE1_UNIFORM20",
    stage="uniform_k384_full_model_coarse_convergence_compressed20",
    curriculum_only_change=True,
    model_change_allowed=False,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=2,
    max_epoch=duca_stage1_end_epoch,
)

workflow = dict(
    training_profile="duca_h65_60_stage1_uniform20",
    end_epoch=duca_stage1_end_epoch,
    expected_train_batches_per_epoch=duca_stage1_steps_per_epoch,
    expected_successful_optimizer_updates=duca_stage1_total_steps,
    primary_checkpoint_epoch=duca_stage1_end_epoch - 1,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_19_state_dict_ema",
)

seed = 3407
total_epochs = 20
max_updates = 2000
checkpoint_interval_epochs = 5
checkpoint_policy = dict(
    resumable=True,
    keep_latest=3,
    milestones=True,
    final=True,
    final_ema=True,
)
paper_claim_allowed = False
work_dir = "exps/thumos/adatad/duca_h65_60_stage1_uniform20"

