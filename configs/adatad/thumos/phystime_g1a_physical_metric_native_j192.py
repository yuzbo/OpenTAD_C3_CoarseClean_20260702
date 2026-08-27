_base_ = ["./selected_axis_adatad_sparse_k384.py"]

raw_observation_count = 384
dense_window_size = 768
native_token_count = 192
chunk_num = 24
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
                coordinate_mode="physical_time_seconds",
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
                coordinate_mode="physical_time_seconds",
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
                coordinate_mode="physical_time_seconds",
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

model = dict(
    native_temporal_geometry=dict(
        enabled=True,
        tubelet_size=2,
        expected_raw_count=raw_observation_count,
        expected_token_count=native_token_count,
        expected_transformer_depth=transformer_depth,
        expected_adapter_indices=adapter_indices,
        expected_adapter_kernel_size=3,
        expected_adapter_dilation=1,
    ),
    backbone=dict(
        backbone=dict(total_frames=raw_observation_count),
        custom=dict(
            strict_temporal_padding_mask=True,
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                )
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
            ],
        ),
    ),
    projection=dict(max_seq_len=native_token_count),
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            eps=1.0e-6,
            positions_key="phystime_g1a_axis_positions_sec",
            selected_count_keys=["phystime_native_valid_count"],
            axis_start_key="phystime_g1a_axis_start_sec",
            axis_end_key="phystime_g1a_axis_end_sec",
        ),
        assignment_debug=dict(enabled=True),
    ),
)

work_dir = "exps/thumos/adatad/phystime_g1a_physical_metric_native_j192"
