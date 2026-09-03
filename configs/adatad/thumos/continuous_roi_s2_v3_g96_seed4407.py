_base_ = ["./continuous_roi_s2_v3_d160_seed4407.py"]

seed = 4407

train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9, 1.0], scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="FullFrameLetterboxView", output_size=96),
    dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
]

evaluation_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="FullFrameLetterboxView", output_size=96),
    dict(type="ConvertToTensor", keys=["imgs"]),
    dict(type="Collect", inputs="imgs", keys=["masks"]),
]

dataset = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=evaluation_pipeline),
    test=dict(pipeline=evaluation_pipeline),
)

continuous_roi_s2_v3_full200_compute = dict(arm="G96", seed=seed)
continuous_roi_d2s_v3_full200_compute = dict(arm="G96", seed=seed)
continuous_roi_patad_v3_full200_compute = dict(arm="G96", seed=seed)
work_dir = f"exps/thumos/adatad/continuous_roi_s2_v3_g96_seed{seed}"
