_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768

d160_train_pipeline = [
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

d160_val_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="FullFrameLetterboxView", output_size=160),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]

dataset = dict(
    train=dict(subset_name="training", pipeline=d160_train_pipeline),
    val=dict(
        subset_name="training",
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=d160_val_pipeline,
    ),
    test=None,
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=60,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=60,
    end_epoch=60,
    require_successful_update_hook=False,
    schedule_and_ema_on_success_only=True,
    max_amp_retries_per_batch=8,
    fail_on_skipped_update=True,
)

continuous_roi_s2_gate = dict(
    route="spatial-zoom-continuous-roi-s2",
    stage="implementation-one-step-gate",
    precheck_only=True,
    allow_detector_training=False,
    allow_tools_train=False,
    allow_tools_test=False,
    allow_detector_map=False,
    official_test_open_allowed=False,
    learned_crop_policy_allowed=False,
    selector_parameters=0,
    paper_claim_allowed=False,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
work_dir = "exps/thumos/adatad/continuous_roi_s2_d160"
