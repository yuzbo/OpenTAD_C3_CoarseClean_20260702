_base_ = ["../../_base_/datasets/thumos-14/features_i3d_train_trunc_test_sw.py"]

observation_count = 384

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="RandomTrunc", trunc_len=2304, trunc_thresh=0.75, crop_ratio=[0.9, 1.0]),
            dict(
                type="BuildPairedPhysTimeFeatureViews",
                first_view=dict(
                    num_observations=observation_count,
                    strategy="random",
                    stochastic=True,
                ),
                second_view=dict(
                    num_observations=observation_count,
                    strategy="bursty",
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
        ],
    ),
    val=dict(
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats", "gt_segments", "gt_labels"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(
                type="SampleIrregularFeatureObservations",
                num_observations=observation_count,
                strategy="uniform",
                stochastic=False,
            ),
            dict(type="BuildPhysTimeFeatureGeometry", convert_gt_to_seconds=True),
            dict(type="Rearrange", keys=["feats"], ops="t c -> c t"),
            dict(type="Collect", inputs="feats", keys=["masks", "gt_segments", "gt_labels"]),
        ],
    ),
    test=dict(
        pipeline=[
            dict(type="LoadFeats", feat_format="npy"),
            dict(type="ConvertToTensor", keys=["feats"]),
            dict(type="SlidingWindowTrunc", with_mask=True),
            dict(
                type="SampleIrregularFeatureObservations",
                num_observations=observation_count,
                strategy="uniform",
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
    discretization_loss_weight=0.1,
    projection=dict(
        type="PhysTimeMeasureProjection",
        in_channels=2048,
        out_channels=512,
        attention_channels=128,
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
    train=dict(batch_size=2, num_workers=4),
    val=dict(batch_size=2, num_workers=4),
    test=dict(batch_size=2, num_workers=4),
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
work_dir = "exps/thumos/adatad/phystime_tad_i3d_feature_gate0b"
