"""Official AdaTAD/VideoMAE-S reference trained on all THUMOS14 training videos."""

import os

_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

_root = os.environ.get("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")
_annotation = f"{_root}/thumos14/annotations/thumos_14_anno.json"
_class_map = f"{_root}/thumos14/annotations/category_idx.txt"
_video_root = f"{_root}/thumos14/raw_data/video"

dataset = dict(
    train=dict(ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
    val=dict(ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
    test=dict(ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
)
model = dict(
    backbone=dict(
        custom=dict(
            pretrain=f"{_root}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
        )
    )
)
work_dir = f"{os.environ['ZOOMTOKEN_FULL_RUN_ROOT']}/DO"
