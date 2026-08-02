_base_ = ["./duca_paper_rime_selected_axis_base.py"]


duca_paper_cell = dict(
    arm="duca_fixed_k384",
    detector_backend="ActionFormer",
    train_video_count=200,
    evaluation_video_count=211,
    world_size=2,
    global_batch_size=2,
    successful_updates=6000,
    evaluation_heavy_k=384,
    position_policy="joint_asformer_learned_exact_k",
    dynamic_budget=False,
    coarse_scanner="jointly_optimized_train_only_asformer",
)

work_dir = "exps/thumos/adatad/duca_paper_duca_fixed_k384_full200"
