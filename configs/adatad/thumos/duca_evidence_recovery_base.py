"""Base configuration for DUCA Evidence Recovery experiments on THUMOS-14."""
_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os

# Training protocol settings
total_epochs = 60
max_updates = 6000
seed = 3407

window_size = 768
selected_budget = 384
scale_factor = 1
chunk_num = selected_budget * scale_factor // 16  # 384 // 16 = 24 chunks

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")

annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))
block_list = None

duca_h65_train_ledger_path = os.environ.get(
    "DUCA_H65_TRAIN_LEDGER_PATH",
    os.environ.get("C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH", "REPLACE_WITH_DUCA_H65_TRAIN_LEDGER_PATH"),
)
duca_h65_val_ledger_path = os.environ.get(
    "DUCA_H65_VAL_LEDGER_PATH",
    os.environ.get("C3_ASFORMER_DELTA_VAL_LEDGER_PATH", "REPLACE_WITH_DUCA_H65_VAL_LEDGER_PATH"),
)
duca_h65_test_ledger_path = os.environ.get(
    "DUCA_H65_TEST_LEDGER_PATH",
    os.environ.get("C3_ASFORMER_DELTA_TEST_LEDGER_PATH", "REPLACE_WITH_DUCA_H65_TEST_LEDGER_PATH"),
)
duca_h65_ledger_source = os.environ.get(
    "DUCA_H65_LEDGER_SOURCE",
    os.environ.get("C3_ASFORMER_DELTA_LEDGER_SOURCE", "c3_official_asformer_delta_p_action"),
)
duca_h65_ledger_config_hash = os.environ.get(
    "DUCA_H65_LEDGER_CONFIG_HASH",
    os.environ.get("C3_ASFORMER_DELTA_LEDGER_CONFIG_HASH", "c3_official_asformer_delta_ledger_384_over_768_v1"),
)

duca_h65_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "bata_selected_dense_indices",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_dense_valid_len",
    "irregular_native_axis",
    "selected_dense_indices",
    "selected_valid_len",
    "duca_h65_selection_row",
]


def _attach_h65_positions(ledger_path):
    return dict(
        type="DucaH65PositionsFromLedger",
        ledger_path=ledger_path,
        target_len=selected_budget,
        dense_len=window_size,
        require_deployable=True,
        require_selected_count=True,
        allow_short_valid_ratio_count=True,
        source=duca_h65_ledger_source,
        config_hash=duca_h65_ledger_config_hash,
        allow_missing=True,
    )


dataset = dict(
    train=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="training",
        block_list=block_list,
        class_map=class_map,
        data_path=train_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            _attach_h65_positions(duca_h65_train_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=duca_h65_meta_keys),
        ],
    ),
    val=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.25,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            _attach_h65_positions(duca_h65_val_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=duca_h65_meta_keys),
        ],
    ),
    test=dict(
        type="ThumosSlidingDataset",
        ann_file=annotation_path,
        subset_name="validation",
        block_list=block_list,
        class_map=class_map,
        data_path=test_data_path,
        filter_gt=False,
        test_mode=True,
        feature_stride=4,
        sample_stride=1,
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=scale_factor),
            _attach_h65_positions(duca_h65_test_ledger_path),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=duca_h65_meta_keys),
        ],
    ),
)

evaluation = dict(
    type="mAP",
    subset="validation",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
    ground_truth_filename=annotation_path,
)

# Optimizer canonical OpenTAD schema
optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    backbone=dict(
        lr=0,
        weight_decay=0,
        custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)],
        exclude=["backbone"],
    ),
)

solver = dict(
    amp=True,
    clip_grad_norm=1.0,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=3,
    max_epoch=total_epochs,
)

# Workflow settings
workflow = dict(
    formal_protocol="duca_evidence_recovery_full_matrix_v1",
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=10,
    val_start_epoch=40,
    end_epoch=total_epochs,
    max_train_iters=max_updates,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=max_updates,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
)

# Post processing setting: ensure save_dict=True so result_path is generated for structured metrics
post_processing = dict(
    sliding_window=False,
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=True,
)


# Base model with DucaEvidenceRecoveryFrameSelector
model = dict(
    frame_selector=dict(
        type="DucaEvidenceRecoveryFrameSelector",
        budget=selected_budget,
        window_size=768,
        use_coverage=True,
        use_time_conditioning=True,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=True,
        use_h65_selection=False,
    ),
    backbone=dict(
        backbone=dict(
            total_frames=selected_budget * scale_factor,
            num_frames=16,
            tubelet_size=2,
            bounded_interval_adapter=dict(enabled=True),
            continuous_timestamp_conditioner=dict(enabled=True),
            temporal_token_merge=dict(enabled=True),
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=chunk_num,
                ),
            ],
        ),
    ),
    projection=dict(max_seq_len=selected_budget),
)

