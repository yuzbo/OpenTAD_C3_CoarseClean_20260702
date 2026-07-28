import os


_base_ = ["../../_base_/datasets/thumos-14/features_i3d_train_trunc_test_sw.py"]

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
paired_train = bool(int(os.environ.get("PHYSTIME_PAIRED_TRAIN", "1")))
train_view1 = os.environ.get("PHYSTIME_TRAIN_VIEW1", "random")
train_view2 = os.environ.get("PHYSTIME_TRAIN_VIEW2", "bursty")
eval_strategy = os.environ.get("PHYSTIME_EVAL_STRATEGY", "uniform")
observation_measure = os.environ.get("PHYSTIME_OBSERVATION_MEASURE", "support_overlap")
discretization_weight = float(os.environ.get("PHYSTIME_DISCRETIZATION_WEIGHT", "0.1"))
worker_count = int(os.environ.get("PHYSTIME_NUM_WORKERS", "4"))
batch_size = int(os.environ.get("PHYSTIME_BATCH_SIZE", "2"))

if paired_train:
    train_pipeline = [
        dict(type="LoadFeats", feat_format="npy"),
        dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
        dict(type="RandomTrunc", trunc_len=2304, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]),
        dict(
            type="BuildPairedPhysTimeFeatureViews",
            first_view=dict(
                num_observations=observation_count,
                strategy=train_view1,
                stochastic=True,
            ),
            second_view=dict(
                num_observations=observation_count,
                strategy=train_view2,
                stochastic=True,
            ),
        ),
        dict(type="Rearrange", keys=["feats", "paired_feats"], ops="t c -> c t"),
        dict(
            type="Collect",
            inputs="feats",
            keys=["masks", "gt_segments", "gt_labels"],
            paired_inputs="paired_feats",
            paired_masks="paired_masks",
            paired_metas="paired_metas",
        ),
    ]
else:
    train_pipeline = [
        dict(type="LoadFeats", feat_format="npy"),
        dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
        dict(type="RandomTrunc", trunc_len=2304, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]),
        dict(
            type="SampleIrregularFeatureObservations",
            num_observations=observation_count,
            strategy=train_view1,
            stochastic=True,
        ),
        dict(type="BuildPhysTimeFeatureGeometry", convert_gt_to_seconds=True),
        dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
        dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
    ]

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=train_pipeline,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(
                type="SampleIrregularFeatureObservations",
                num_observations=observation_count,
                strategy=eval_strategy,
                stochastic=False,
            ),
            dict(type="BuildPhysTimeFeatureGeometry", convert_gt_to_seconds=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=feature_path,
        block_list=block_list,
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(
                type="SampleIrregularFeatureObservations",
                num_observations=observation_count,
                strategy=eval_strategy,
                stochastic=False,
            ),
            dict(type="BuildPhysTimeFeatureGeometry", convert_gt_to_seconds=False),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks"]),
        ],
    ),
)

model = dict(
    type="PhysTimeTAD",
    discretization_loss_weight=discretization_weight,
    projection=dict(
        type="PhysTimeMeasureProjection",
        in_channels=2048,
        out_channels=512,
        attention_channels=128,
        observation_measure=observation_measure,
        base_spacing_sec=0.5,
        num_levels=6,
        dropout=0.1,
    ),
    rpn_head=dict(
        type="PhysTimeHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        regression_ranges_sec=[
            (0.0, 2.0),
            (2.0, 4.0),
            (4.0, 8.0),
            (8.0, 16.0),
            (16.0, 32.0),
            (32.0, 100000000.0),
        ],
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample_radius=1.5,
        cls_prior_prob=0.01,
        endpoint_loss_weight=0.25,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
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
work_dir = os.environ.get("PHYSTIME_WORK_DIR", "exps/thumos/adatad/phystime_tad_i3d_feature_gate0b")
