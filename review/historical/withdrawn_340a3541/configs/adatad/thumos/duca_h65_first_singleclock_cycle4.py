_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

single_clock_unit1 = dict(
    k=384, clip_len=16, tubelet_size=2, global_rank_selection=True,
    relative_physical_time_residual=True, relative_residual_init=0.0,
    relative_residual_h=lambda r: r,
    canonical_selection="exact_uniform_positions_once_over_dense_window",
)
model = dict(
    single_clock_admission=True,
    backbone=dict(
        backbone=dict(total_frames=768, num_frames=16, tubelet_size=2,
                      relative_physical_time_residual=True,
                      tubelet_packed_runtime_route=dict(enabled=False)),
        custom=dict(global_rank_selection=True,
                    canonical_selection="exact_uniform_positions_once_over_dense_window"),
    ),
)
packed_route_policy = dict(enabled=False, fail_closed=True)
seed = 3407
total_epochs = 60
max_updates = 6000
checkpoint_interval_epochs = 5
checkpoint_policy = dict(resumable=True, keep_latest=3, milestones=True, final=True, final_ema=True)
paper_claim_allowed = False
work_dir = "exps/thumos/adatad/duca_h65_first_singleclock_cycle4"
