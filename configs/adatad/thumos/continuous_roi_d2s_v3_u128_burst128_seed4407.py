_base_ = ["./continuous_roi_s2_v3_d160_seed4407.py"]

seed = 4407

train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=768, trunc_thresh=0.75, crop_ratio=[0.9, 1.0], scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="ContinuousRoiSourceViews", global_size=96, output_key="d2s_inputs"),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(type="Collect", inputs="d2s_inputs", keys=["masks", "gt_segments", "gt_labels"]),
]

evaluation_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="ContinuousRoiSourceViews", global_size=96, output_key="d2s_inputs"),
    dict(type="ConvertToTensor", keys=[]),
    dict(type="Collect", inputs="d2s_inputs", keys=["masks"]),
]

dataset = dict(
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=evaluation_pipeline),
    test=dict(pipeline=evaluation_pipeline),
)

model = dict(
    backbone=dict(
        custom=dict(
            wrapper_type="d2s_temporal_zoom_shared_videomae",
            global_key="global",
            source_key="source",
            global_size=96,
            local_size=128,
            source_height=180,
            source_width=320,
            total_chunks=48,
            burst_chunks=16,
            tubelets_per_chunk=8,
            intermediate_length=384,
            output_length=768,
            return_feature_bundle=False,
        )
    )
)

optimizer = dict(
    backbone=dict(
        custom=[
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
            dict(name="proj_local", lr=1e-4, weight_decay=0.05),
            dict(name="proj_global", lr=1e-4, weight_decay=0.05),
            dict(name="gamma", lr=1e-4, weight_decay=0.0),
        ]
    )
)

continuous_roi_d2s_v3_full200_compute = dict(
    protocol="ZOOMTOKEN-D2S-TAD-FULL200-COMPUTE-PARETO-3X3-v001",
    arm="D2S-U128-B128",
    seed=seed,
    burst_chunks=16,
    total_chunks=48,
    crop_policy="top16_then_source_native_fixed_center",
    canonical_source_hw=[180, 320],
    canonical_crop_xyxy=[96, 26, 224, 154],
    shared_backbone_instances=1,
    learned_selector_parameters=0,
    learned_residual_parameters=True,
)
work_dir = f"exps/thumos/adatad/continuous_roi_d2s_v3_u128_burst128_seed{seed}"

