_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


dense_window_size = 768
duca_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "avg_fps",
    "duration",
    "total_frames",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "frame_inds",
    "duca_stateless_seed",
    "duca_stateless_epoch",
    "duca_stateless_sample_index",
]

dataset = dict(
    train=dict(
        type="DucaStatelessThumosPaddingDataset",
        stateless_seed=3407,
        subset_name="training",
        block_list=None,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=1,
                emit_boundary_validity=True,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="ConvertToTensor",
                keys=[
                    "imgs",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                ],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=[
                    "masks",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                ],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
    val=None,
    test=dict(
        subset_name="validation",
        block_list=None,
        test_mode=True,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=1,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
)

evaluation = dict(
    subset="validation",
    blocked_videos=None,
)

model = dict(
    frame_selector=None,
    backbone=dict(
        backbone=dict(
            total_frames=dense_window_size,
            with_cp=False,
        ),
        custom=dict(
            dynamic_temporal_bucket=False,
        ),
    ),
    projection=dict(max_seq_len=dense_window_size),
    rpn_head=dict(physical_grid_actionformer=None),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
    find_unused_parameters=True,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=60,
)

workflow = dict(
    formal_protocol="duca_paper_full200_actionformer_v1",
    logging_interval=50,
    checkpoint_interval=5,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=6000,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    seal_eval_dataloaders_during_training=True,
    derive_train_loader_contract=True,
)

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/duca_paper_base_do_not_run"
