_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]

import os


def _required_path(name, default=""):
    value = os.environ.get(name, default)
    if not value:
        raise ValueError(f"{name} is required for frontend holdout evaluation")
    return value


yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
annotation_path = _required_path(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = _required_path(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = _required_path(
    "THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train")
)
holdout_block_list = _required_path("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST")


dataset = dict(
    val=dict(
        _delete_=True,
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=holdout_block_list,
        class_map=class_map,
        data_path=train_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=768,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
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
            ),
        ],
    ),
)
