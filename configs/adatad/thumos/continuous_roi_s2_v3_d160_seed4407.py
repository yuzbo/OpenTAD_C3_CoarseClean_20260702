_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768
seed = 4407

train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="random_trunc",
        trunc_len=window_size,
        trunc_thresh=0.75,
        crop_ratio=[0.9, 1.0],
        scale_factor=1,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(type="FullFrameLetterboxView", output_size=160),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]

evaluation_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="FullFrameLetterboxView", output_size=160),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="imgs", keys=["masks"]),
]

dataset = dict(
    train=dict(subset_name="training", pipeline=train_pipeline),
    val=dict(
        subset_name="validation",
        window_size=window_size,
        window_overlap_ratio=0.5,
        test_mode=True,
        pipeline=evaluation_pipeline,
    ),
    test=dict(
        subset_name="validation",
        window_size=window_size,
        window_overlap_ratio=0.5,
        test_mode=True,
        pipeline=evaluation_pipeline,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=60,
    checkpoint_policy="final_only",
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=60,
    end_epoch=60,
    max_train_iters=100,
    require_successful_update_hook=False,
    schedule_and_ema_on_success_only=True,
    max_amp_retries_per_batch=8,
    fail_on_skipped_update=True,
    fail_on_nonfinite_loss=True,
)

inference = dict(
    load_from_raw_predictions=False,
    save_raw_prediction=True,
    test_epoch=59,
)

continuous_roi_s2_v3_full200_compute = dict(
    protocol="ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FULL200-COMPUTE-PARETO-3X3-v001",
    arm="D160",
    seed=seed,
    training_identities=200,
    successful_updates_per_epoch=100,
    epochs=60,
    total_successful_updates=6000,
    evaluation_videos=211,
    evaluation_ordered_windows=792,
    final_checkpoint="epoch_59/state_dict_ema/update_6000",
    admission_axis="detection_performance_vs_full_model_executed_computation",
)

work_dir = f"exps/thumos/adatad/continuous_roi_s2_v3_d160_seed{seed}"
