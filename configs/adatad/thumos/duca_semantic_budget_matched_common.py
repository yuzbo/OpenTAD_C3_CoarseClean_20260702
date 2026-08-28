_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

import os


def _required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the DUCA semantic-budget experiment")
    return value


duca_window_table_path = _required("DUCA_WINDOW_TABLE_PATH")
duca_budget_arm = _required("DUCA_WINDOW_BUDGET_ARM")
if duca_budget_arm not in {"fixed384", "semantic", "permuted_control"}:
    raise ValueError(f"unsupported DUCA_WINDOW_BUDGET_ARM={duca_budget_arm}")

dense_window_size = 768
detector_length = 384

duca_window_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "window_end_frame",
    "window_size",
    "offset_frames",
    "duca_split",
    "duca_window_index",
    "duca_window_count",
    "duca_stateless_seed",
    "duca_stateless_epoch",
    "duca_budget_arm",
    "duca_requested_k",
    "duca_effective_k",
    "duca_unique_k",
    "duca_actual_backbone_input_k",
    "duca_actual_backbone_chunks",
    "duca_detector_length",
    "duca_dynamic_compute_realized",
    "duca_acquisition_positions",
    "selected_axis_to_true_time_dense_index",
    "irregular_selected_positions",
    "irregular_selected_count",
    "irregular_selected_valid_len",
    "irregular_dense_valid_len",
    "truetime_dense_len",
    "truetime_dense_valid_len",
    "detector_prediction_inverse_map_required",
    "detector_output_coordinate_space",
    "irregular_native_axis",
    "gt_remapped_to_selected_axis",
    "gt_coordinate_space",
    "duca_window_table_path",
]


def _pipeline(training, with_gt):
    transforms = [
        dict(type="PrepareVideoInfo", format="mp4"),
        dict(type="mmaction.DecordInit", num_threads=4),
        dict(
            type="LoadDucaWindowBudgetFrames",
            table_path=duca_window_table_path,
            arm=duca_budget_arm,
            detector_length=detector_length,
            clip_len=16,
        ),
        dict(type="mmaction.DecordDecode"),
    ]
    if training:
        transforms.extend(
            [
                dict(type="mmaction.Resize", scale=(-1, 182)),
                dict(type="mmaction.RandomResizedCrop"),
                dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
                dict(type="mmaction.Flip", flip_ratio=0.5),
                dict(type="mmaction.ImgAug", transforms="default"),
                dict(type="mmaction.ColorJitter"),
            ]
        )
    else:
        transforms.extend(
            [
                dict(type="mmaction.Resize", scale=(-1, 160)),
                dict(type="mmaction.CenterCrop", crop_size=160),
            ]
        )
    transforms.append(dict(type="mmaction.FormatShape", input_format="NCTHW"))
    tensor_keys = ["imgs"]
    collect_keys = ["masks"]
    if with_gt:
        tensor_keys.extend(["gt_segments", "gt_labels", "gt_boundary_validity"])
        collect_keys.extend(["gt_segments", "gt_labels", "gt_boundary_validity"])
    transforms.extend(
        [
            dict(type="ConvertToTensor", keys=tensor_keys),
            dict(
                type="Collect",
                inputs="imgs",
                keys=collect_keys,
                meta_keys=duca_window_meta_keys,
            ),
        ]
    )
    return transforms


dataset = dict(
    train=dict(
        _delete_=True,
        type="DucaVideoGroupedThumosSlidingDataset",
        ann_file="data/thumos-14/annotations/thumos_14_anno.json",
        subset_name="training",
        block_list=None,
        class_map="data/thumos-14/annotations/category_idx.txt",
        data_path="data/thumos-14/raw_data/video",
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.75,
        include_background_windows=True,
        stateless_seed=3407,
        group_by_video=True,
        pipeline=_pipeline(training=True, with_gt=True),
    ),
    val=dict(
        _delete_=True,
        type="DucaVideoGroupedThumosSlidingDataset",
        ann_file="data/thumos-14/annotations/thumos_14_anno.json",
        subset_name="validation",
        block_list=None,
        class_map="data/thumos-14/annotations/category_idx.txt",
        data_path="data/thumos-14/raw_data/video",
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.75,
        include_background_windows=True,
        stateless_seed=3407,
        group_by_video=False,
        pipeline=_pipeline(training=False, with_gt=True),
    ),
    test=dict(
        _delete_=True,
        type="DucaVideoGroupedThumosSlidingDataset",
        ann_file="data/thumos-14/annotations/thumos_14_anno.json",
        subset_name="validation",
        block_list=None,
        class_map="data/thumos-14/annotations/category_idx.txt",
        data_path="data/thumos-14/raw_data/video",
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=dense_window_size,
        window_overlap_ratio=0.5,
        ioa_thresh=0.75,
        include_background_windows=True,
        stateless_seed=3407,
        group_by_video=False,
        pipeline=_pipeline(training=False, with_gt=False),
    ),
)

model = dict(
    offline_window_table=True,
    freeze_frame_selector=True,
    profile_variable_k=True,
    backbone=dict(
        backbone=dict(with_cp=False),
        custom=dict(
            pre_processing_pipeline=[],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c (n t)",
                    reduction="mean",
                ),
                dict(type="Interpolate", keys=["feats"], size=detector_length),
            ],
        ),
    ),
    projection=dict(max_seq_len=detector_length),
)

solver = dict(
    train=dict(batch_size=2, num_workers=2, sampler_seed=3407),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=True,
    find_unused_parameters=False,
)

workflow = dict(
    training_profile="duca_semantic_budget_matched",
    derive_train_loader_contract=True,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=6000,
    checkpoint_interval=5,
    end_epoch=60,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
)

work_dir = f"exps/thumos/adatad/duca_semantic_budget_matched_{duca_budget_arm}"
