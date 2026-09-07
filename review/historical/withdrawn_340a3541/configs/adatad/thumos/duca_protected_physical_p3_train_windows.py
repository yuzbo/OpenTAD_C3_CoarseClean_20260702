_base_ = ["./duca_protected_physical_e2e_fixed384_official60.py"]


dense_window_size = 768
p3_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "avg_fps",
    "duration",
    "total_frames",
    "snippet_stride",
    "window_start_frame",
    "window_size",
    "offset_frames",
    "frame_inds",
]

dataset = dict(
    train=dict(
        _delete_=True,
        type="ThumosSlidingDataset",
        ann_file="data/thumos-14/annotations/thumos_14_anno.json",
        subset_name="training",
        block_list=None,
        class_map="data/thumos-14/annotations/category_idx.txt",
        data_path="data/thumos-14/raw_data/video",
        filter_gt=False,
        test_mode=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
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
                meta_keys=p3_meta_keys,
            ),
        ],
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
)

duca_p3_contract = dict(
    split="train_only",
    optimizer_step=0,
    seed=3407,
    window_count=48,
    windows_per_duration_stratum=16,
    minimum_padded_windows_per_duration_stratum=4,
    max_windows_per_video=4,
    swaps_per_window=12,
    total_preregistered_swaps=576,
    minimum_effective_swaps=512,
    bootstrap_replicates=2000,
    bootstrap_seed=20260720,
    hard_loss="official_actionformer_cls_plus_reg",
    boundary_source="original_uncropped_annotation",
    boundary_distance_unit="seconds",
    retain_near_zero_gradient_rows=True,
    checkpoint_written=False,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_p3_do_not_train"
