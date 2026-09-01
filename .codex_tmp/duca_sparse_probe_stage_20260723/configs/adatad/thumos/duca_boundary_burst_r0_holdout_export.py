_base_ = ["./duca_allocation_ceiling_training_windows.py"]

import os


holdout_block_list = os.environ.get("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST", "")
if not holdout_block_list:
    raise ValueError("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST is required for R0")


allocation_ceiling_training_window_contract = dict(
    purpose="exhaustive_train_holdout_selected_axis_oracle_map",
    action_intersecting_windows_only=False,
    includes_background_windows=True,
    validation_consumed=False,
    test_subset_consumed=False,
)


dataset = dict(
    train=dict(
        block_list=holdout_block_list,
        include_background_windows=True,
        ioa_thresh=1.0e-8,
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
            dict(
                type="ConvertToTensor",
                keys=["imgs", "gt_segments", "gt_labels", "gt_boundary_validity"],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels", "gt_boundary_validity"],
                meta_keys=[
                    "video_name",
                    "data_path",
                    "fps",
                    "avg_fps",
                    "duration",
                    "total_frames",
                    "snippet_stride",
                    "window_start_frame",
                    "frame_inds",
                    "window_size",
                    "offset_frames",
                ],
            ),
        ],
    ),
)


work_dir = "exps/thumos/adatad/duca_boundary_burst_r0_holdout_export"
