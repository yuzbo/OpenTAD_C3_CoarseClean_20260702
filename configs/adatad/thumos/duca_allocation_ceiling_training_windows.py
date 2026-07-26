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
train_data_path = os.environ.get(
    "THUMOS14_TRAIN_DATA_PATH",
    "data/thumos-14/raw_data/video",
)


allocation_ceiling_training_window_contract = dict(
    task="offline_temporal_action_detection",
    subset="training",
    deterministic_sliding_windows=True,
    action_intersecting_windows_only=True,
    gt_truncated_to_window=True,
    random_crop=False,
    random_augmentation=False,
    purpose="allocation_family_definition_and_recoverability_only",
    model_training=False,
    validation_consumed=False,
)


dataset = dict(
    train=dict(
        _delete_=True,
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=None,
        class_map=class_map_path,
        data_path=train_data_path,
        filter_gt=False,
        test_mode=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        # SlidingWindowDataset only clips GT when ioa_thresh is strictly
        # positive. A tiny threshold keeps every genuinely intersecting action
        # window while excluding non-intersecting GT from local coordinates.
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
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks", "gt_segments", "gt_labels"],
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


solver = dict(
    train=dict(batch_size=1, num_workers=2),
)


work_dir = "exps/thumos/adatad/duca_allocation_ceiling_training_windows"
