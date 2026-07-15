_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768
scale_factor = 1

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 291)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(256, 256), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    val=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.CenterCrop", crop_size=256),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
            ),
        ],
    ),
    test=dict(
        window_size=window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 256)),
            dict(type="mmaction.CenterCrop", crop_size=256),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

spatial_zoom_s1_contract = dict(
    schema_version="spatial_zoom_s1_config_v1",
    gate="S1_spatial_resolution_headroom",
    runtime_resolution=256,
    train_short_side=291,
    temporal_window=768,
    detector_time_grid=768,
    tubelet_points=384,
    fit_gate_manifest_required=True,
    official_test_sealed_until_protocol_freeze=True,
    checkpoint_selection_rule="max_gate_high_tiou_headroom_earliest_epoch_tie",
    training_seeds=[3407, 3408, 3409],
    roi_policy_enabled=False,
    teacher_oracle_enabled=False,
    new_detector_enabled=False,
    paper_claim_allowed=False,
)

work_dir = "exps/thumos/adatad/spatial_zoom_s1_dense256"
