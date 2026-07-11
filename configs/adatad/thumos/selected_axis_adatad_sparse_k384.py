import os


_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = window_size * scale_factor // 16

data_root = os.environ.get("OPENTAD_THUMOS14_ROOT", "data/thumos-14")
annotation_path = os.environ.get(
    "OPENTAD_THUMOS14_ANNOTATION",
    os.path.join(data_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "OPENTAD_THUMOS14_CLASS_MAP",
    os.path.join(data_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get(
    "OPENTAD_THUMOS14_TRAIN_VIDEOS",
    os.path.join(data_root, "raw", "Validation Data", "validation"),
)
test_data_path = os.environ.get(
    "OPENTAD_THUMOS14_TEST_VIDEOS",
    os.path.join(data_root, "raw", "Test Data", "TH14_test_set_mp4"),
)

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
        sample_stride=1,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="random_trunc",
                keep_ratio=0.5,
                target_len=window_size,
                source_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=True,
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
        ],
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                source_len=dense_window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=True,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
        sample_stride=1,
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_fixed_subsample",
                method_base="sliding_window",
                keep_ratio=0.5,
                target_len=window_size,
                source_len=dense_window_size,
                scale_factor=scale_factor,
                remap_gt_to_selected_axis=True,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"]),
        ],
    ),
)

# The inherited dataset base resolves its own default annotation variable.
# Override the evaluator explicitly so runtime data roots remain end-to-end consistent.
evaluation = dict(ground_truth_filename=annotation_path)

solver = dict(
    fail_on_non_finite_grad=True,
    # Formal Phase 1 is single-GPU; FP16 DDP bucket compression has no
    # communication benefit and can overflow already-scaled AMP gradients.
    fp16_compress=False,
)

model = dict(
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
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
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(max_seq_len=window_size),
)

work_dir = "exps/thumos/adatad/selected_axis_adatad_sparse_k384"
