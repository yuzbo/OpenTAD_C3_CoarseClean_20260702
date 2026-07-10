import os


_base_ = [
    "../../_base_/datasets/thumos-14/features_i3d_train_trunc_test_sw.py",
    "../../_base_/models/actionformer.py",
]

thumos_root = os.environ.get("PHYSTIME_THUMOS_ROOT", "data/thumos-14")
annotation_path = os.environ.get(
    "PHYSTIME_ANNOTATION_PATH", os.path.join(thumos_root, "annotations", "thumos_14_anno.json")
)
class_map = os.environ.get(
    "PHYSTIME_CLASS_MAP", os.path.join(thumos_root, "annotations", "category_idx.txt")
)
feature_path = os.environ.get(
    "PHYSTIME_FEATURE_PATH", os.path.join(thumos_root, "features", "i3d_actionformer_stride4_thumos")
)
block_list = os.environ.get("PHYSTIME_BLOCK_LIST", os.path.join(feature_path, "missing_files.txt"))
observation_count = int(os.environ.get("PHYSTIME_OBSERVATION_COUNT", "384"))
train_strategy = os.environ.get("PHYSTIME_TRAIN_VIEW1", "random")
eval_strategy = os.environ.get("PHYSTIME_EVAL_STRATEGY", "uniform")
worker_count = int(os.environ.get("PHYSTIME_NUM_WORKERS", "4"))
batch_size = int(os.environ.get("PHYSTIME_BATCH_SIZE", "2"))


def timestamp_pipeline(training, with_gt):
    pipeline = [dict(type="LoadFeats", feat_format="npy")]
    tensor_keys = ["feats"] + (["gt_segments", "gt_labels"] if with_gt else [])
    pipeline.append(dict(type="ConvertToTensor", keys=tensor_keys))
    if training:
        pipeline.append(dict(type="RandomTrunc", trunc_len=2304, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]))
    else:
        pipeline.append(dict(type="SlidingWindowTrunc", with_mask=True))
    pipeline.extend(
        [
            dict(
                type="SampleIrregularFeatureObservations",
                num_observations=observation_count,
                strategy=train_strategy if training else eval_strategy,
                stochastic=training,
            ),
            dict(type="BuildSelectedAxisFeatureBaseline", append_timestamp_channels=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(
                type="Collect",
                inputs="feats",
                keys=["masks"] + (["gt_segments", "gt_labels"] if with_gt else []),
            ),
        ]
    )
    return pipeline


dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=timestamp_pipeline(training=True, with_gt=True),
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=timestamp_pipeline(training=False, with_gt=True),
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=timestamp_pipeline(training=False, with_gt=False),
    ),
)

model = dict(
    projection=dict(
        in_channels=2052,
        max_seq_len=observation_count,
        attn_cfg=dict(n_head=4, n_mha_win_size=-1),
    ),
)
solver = dict(
    train=dict(batch_size=batch_size, num_workers=worker_count),
    val=dict(batch_size=batch_size, num_workers=worker_count),
    test=dict(batch_size=batch_size, num_workers=worker_count),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=False,
    ema=True,
)
optimizer = dict(type="AdamW", lr=1.0e-4, weight_decay=0.05, paramwise=True)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(
    nms=dict(
        use_soft_nms=True,
        sigma=0.7,
        max_seg_num=2000,
        multiclass=True,
        voting_thresh=0.7,
    ),
    save_dict=False,
)
workflow = dict(
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=20,
    end_epoch=60,
)
evaluation = dict(ground_truth_filename=annotation_path)
work_dir = os.environ.get(
    "PHYSTIME_WORK_DIR", "exps/thumos/actionformer/timestamp_selected_axis_i3d_k384"
)
