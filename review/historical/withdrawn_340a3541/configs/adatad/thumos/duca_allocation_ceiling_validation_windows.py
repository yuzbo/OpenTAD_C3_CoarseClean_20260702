_base_ = ["./duca_allocation_ceiling_physical_grid_evaluator.py"]

import os


dense_window_size = 768
annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    "data/thumos-14/annotations/thumos_14_anno.json",
)
class_map_path = os.environ.get(
    "THUMOS14_CLASS_MAP",
    "data/thumos-14/annotations/category_idx.txt",
)
test_data_path = os.environ.get(
    "THUMOS14_TEST_DATA_PATH",
    "data/thumos-14/raw_data/video",
)


allocation_ceiling_validation_window_contract = dict(
    task="offline_temporal_action_detection",
    subset="validation",
    execution_split="test",
    deterministic_sliding_windows=True,
    includes_background_windows=True,
    gt_available_to_selector=False,
    validation_authorization_required=True,
    actual_decoded_frame_indices_retained=True,
    window_overlap_ratio=0.5,
    selected_axis_gt_remap=False,
)


_metadata_keys = [
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
]


dataset = dict(
    test=dict(
        _delete_=True,
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=None,
        class_map=class_map_path,
        data_path=test_data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
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
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=_metadata_keys,
            ),
        ],
    ),
)


solver = dict(
    test=dict(batch_size=1, num_workers=2),
)


work_dir = "exps/thumos/adatad/duca_allocation_ceiling_validation_windows"
