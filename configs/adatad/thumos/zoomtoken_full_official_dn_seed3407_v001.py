"""Matched GeoRoute-source dense control on the complete official THUMOS14 split."""

import os

_base_ = ["./georoute_adatad_development_base.py"]

_root = os.environ.get("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")
_annotation = f"{_root}/thumos14/annotations/thumos_14_anno.json"
_class_map = f"{_root}/thumos14/annotations/category_idx.txt"
_video_root = f"{_root}/thumos14/raw_data/video"

dataset = dict(
    train=dict(subset_name="training", ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
    val=dict(subset_name="validation", ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
    test=dict(subset_name="validation", ann_file=_annotation, class_map=_class_map, data_path=_video_root, block_list=None),
)
evaluation = dict(subset="validation", ground_truth_filename=_annotation)
model = dict(
    backbone=dict(
        custom=dict(
            pretrain=f"{_root}/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            georoute_route_mode="dense",
            georoute_policy_estimator="none",
            georoute_window_token_budget=24576,
            georoute_zero_carrier_mode="masked_zero",
            georoute_branch_calibration_mode="none",
            georoute_dynamic_roi_modifier_enabled=False,
            georoute_dynamic_residual_modifier_enabled=False,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=False,
            georoute_roi_relative_coordinates_enabled=False,
            georoute_geometry_projection_enabled=False,
            georoute_geometry_side_channel=False,
            georoute_random_seed=3407,
            georoute_max_batch_size=1,
        )
    )
)
solver = dict(
    train=dict(batch_size=2),
    val=dict(batch_size=2),
    test=dict(batch_size=2),
    fp16_compress=False,
)
scheduler = dict(warmup_epoch=5, max_epoch=100)
workflow = dict(
    checkpoint_interval=60,
    checkpoint_policy="final_only",
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=60,
    end_epoch=60,
    max_amp_retries_per_batch=0,
    fail_on_skipped_update=False,
    require_successful_update_hook=False,
    schedule_and_ema_on_success_only=False,
    fail_on_nonfinite_loss=True,
)
georoute_protocol = dict(
    schema_version="zoomtoken_full_official_v001",
    status="full_official_training_and_evaluation",
    arm="matched_dense",
    full_training_split=True,
    official_validation_evaluation=True,
)
work_dir = f"{os.environ['ZOOMTOKEN_FULL_RUN_ROOT']}/DN"
