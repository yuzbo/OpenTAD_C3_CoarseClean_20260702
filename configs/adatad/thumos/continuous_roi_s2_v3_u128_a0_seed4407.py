_base_ = ["./continuous_roi_s2_v3_d160_seed4407.py"]

seed = 4407

train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9, 1.0], scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="NativeCropSourceViews", global_size=96, local_size=128, output_key="native_crop_inputs", allow_local_padding=False),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="native_crop_inputs", keys=["masks", "gt_segments", "gt_labels"]),
]

evaluation_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="NativeCropSourceViews", global_size=96, local_size=128, output_key="native_crop_inputs", allow_local_padding=False),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="native_crop_inputs", keys=["masks"]),
]

dataset = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=evaluation_pipeline),
    test=dict(pipeline=evaluation_pipeline),
)

model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="native_crop_shared_videomae",
            native_crop_global_key="global",
            native_crop_local_key="local",
            native_crop_global_size=96,
            native_crop_local_size=128,
            native_crop_chunk_num=48,
            native_crop_intermediate_length=384,
            native_crop_output_length=768,
            native_crop_fusion_mode="fixed_mean",
        )
    )
)

continuous_roi_s2_v3_full200_compute = dict(
    arm="U128-A0",
    seed=seed,
    crop_policy="source_native_fixed_center",
    canonical_source_hw=[180, 320],
    canonical_crop_xyxy=[96, 26, 224, 154],
    fusion="fixed_mean_parameter_free",
    shared_backbone_instances=1,
    learned_selector_parameters=0,
)
work_dir = f"exps/thumos/adatad/continuous_roi_s2_v3_u128_a0_seed{seed}"
