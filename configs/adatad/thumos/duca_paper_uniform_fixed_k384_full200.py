_base_ = ["./duca_paper_full200_base.py"]


model = dict(
    frame_selector=dict(
        _delete_=True,
        type="DucaProtectedE2EFrameSelector",
        in_channels=3,
        arm="exact_uniform",
        dense_window_size=768,
        budget=384,
        execution_quantum=16,
        detector_coordinate_mode="selected_axis_plugin",
        actionness_source_cfg=None,
        strict_physical_metadata=True,
        forbid_raw_prediction_cache=True,
    ),
    backbone=dict(
        backbone=dict(total_frames=384, with_cp=False),
        custom=dict(
            dynamic_temporal_bucket=True,
            dynamic_temporal_clip_len=16,
        ),
    ),
    projection=dict(max_seq_len=512),
    rpn_head=dict(physical_grid_actionformer=None),
)

duca_paper_cell = dict(
    arm="uniform_fixed_k384",
    detector_backend="ActionFormer",
    train_video_count=200,
    evaluation_video_count=211,
    world_size=2,
    global_batch_size=2,
    successful_updates=6000,
    evaluation_heavy_k=384,
    position_policy="exact_uniform",
    dynamic_budget=False,
)

work_dir = "exps/thumos/adatad/duca_paper_uniform_fixed_k384_full200"
