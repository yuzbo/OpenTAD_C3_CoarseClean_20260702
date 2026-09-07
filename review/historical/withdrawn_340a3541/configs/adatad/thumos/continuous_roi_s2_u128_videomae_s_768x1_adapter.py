_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768

continuous_roi_train_pipeline = [
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
    dict(
        type="ContinuousRoiSourceViews",
        global_size=96,
        output_key="continuous_roi_inputs",
        required_source_height=180,
        required_source_width=320,
    ),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="continuous_roi_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

continuous_roi_val_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="sliding_window",
        scale_factor=1,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(
        type="ContinuousRoiSourceViews",
        global_size=96,
        output_key="continuous_roi_inputs",
        required_source_height=180,
        required_source_width=320,
    ),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="continuous_roi_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

dataset = dict(
    train=dict(subset_name="training", pipeline=continuous_roi_train_pipeline),
    val=dict(
        subset_name="training",
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=continuous_roi_val_pipeline,
    ),
    test=None,
)

model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="continuous_roi_common_support_u128",
            native_crop_global_key="global",
            native_crop_local_key="local",
            native_crop_global_size=96,
            native_crop_local_size=128,
            native_crop_chunk_num=48,
            native_crop_intermediate_length=384,
            native_crop_output_length=window_size,
            continuous_roi_source_key="source",
            continuous_roi_sample_key="sample_key",
            continuous_roi_window_start_key="window_start",
            continuous_roi_boxes_key="roi_clip_boxes",
            continuous_roi_training_seed=3407,
            continuous_roi_source_height=180,
            continuous_roi_source_width=320,
            continuous_roi_knots=12,
            continuous_roi_frames_per_clip=16,
            continuous_roi_local_clips_per_call=4,
            continuous_roi_num_classes=20,
            continuous_roi_detector_length=window_size,
        )
    )
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="fusion", lr=1e-4, weight_decay=0.05),
            dict(name="global_aux_head", lr=1e-4, weight_decay=0.05),
            dict(name="local_aux_head", lr=1e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=60,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=60,
    end_epoch=60,
    require_successful_update_hook=True,
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
work_dir = "exps/thumos/adatad/continuous_roi_s2_u128"
