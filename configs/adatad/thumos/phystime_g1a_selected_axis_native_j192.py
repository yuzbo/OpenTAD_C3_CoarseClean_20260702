_base_ = ["./phystime_g1a_physical_metric_native_j192.py"]

raw_observation_count = 384
dense_window_size = 768
scale_factor = 1
transformer_depth = 12
adapter_indices = list(range(transformer_depth))

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                target_len=raw_observation_count,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
            ),
            dict(
                type="BuildPhysTimeRawFrameGeometry",
                convert_gt_to_seconds=True,
                fps_relative_tolerance=0.0125,
                duration_relative_tolerance=0.0125,
                frame_count_relative_tolerance=0.0001,
            ),
            dict(
                type="BuildPhysTimeNativeTubeletGeometry",
                tubelet_size=2,
                chunk_size=16,
                transformer_depth=transformer_depth,
                adapter_indices=adapter_indices,
                adapter_kernel_size=3,
                adapter_dilation=1,
                coordinate_mode="uniform_rank_seconds",
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ]
    ),
    val=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=raw_observation_count,
                source_len=dense_window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
            ),
            dict(
                type="BuildPhysTimeRawFrameGeometry",
                convert_gt_to_seconds=True,
                fps_relative_tolerance=0.0125,
                duration_relative_tolerance=0.0125,
                frame_count_relative_tolerance=0.0001,
            ),
            dict(
                type="BuildPhysTimeNativeTubeletGeometry",
                tubelet_size=2,
                chunk_size=16,
                transformer_depth=transformer_depth,
                adapter_indices=adapter_indices,
                adapter_kernel_size=3,
                adapter_dilation=1,
                coordinate_mode="uniform_rank_seconds",
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ]
    ),
    test=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=raw_observation_count,
                source_len=dense_window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=False,
            ),
            dict(
                type="BuildPhysTimeRawFrameGeometry",
                convert_gt_to_seconds=False,
                fps_relative_tolerance=0.0125,
                duration_relative_tolerance=0.0125,
                frame_count_relative_tolerance=0.0001,
            ),
            dict(
                type="BuildPhysTimeNativeTubeletGeometry",
                tubelet_size=2,
                chunk_size=16,
                transformer_depth=transformer_depth,
                adapter_indices=adapter_indices,
                adapter_kernel_size=3,
                adapter_dilation=1,
                coordinate_mode="uniform_rank_seconds",
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ]
    ),
)

work_dir = "exps/thumos/adatad/phystime_g1a_selected_axis_native_j192"
