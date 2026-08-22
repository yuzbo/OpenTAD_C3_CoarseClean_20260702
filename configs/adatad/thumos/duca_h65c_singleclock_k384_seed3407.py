"""DUCA-H65C-SINGLECLOCK Unit-1; H65 Stage-2 compatibility, one bias change."""
_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
seed = 3407
model = dict(singleclock=dict(enabled=True, eps=1e-6))
workflow = dict(end_epoch=60, expected_successful_optimizer_updates=6000,
                checkpoint_interval=5, primary_checkpoint="final-ema",
                checkpoint_retention="latest-3+milestones+final+final-ema",
                intermediate_validation_selects_checkpoint=False)
work_dir = "exps/thumos/adatad/duca_h65c_singleclock_k384_seed3407"
