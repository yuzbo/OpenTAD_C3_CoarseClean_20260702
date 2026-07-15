_base_ = ["./phystime_g1a_physical_metric_native_j192.py"]

raw_observation_count = 384
native_token_count = 192
chunk_num = 24
transformer_depth = 12
adapter_indices = list(range(transformer_depth))

model = dict(
    _delete_=True,
    type="PhysTimeTAD",
    discretization_loss_weight=0.0,
    native_temporal_geometry=dict(
        enabled=True,
        tubelet_size=2,
        expected_raw_count=raw_observation_count,
        expected_token_count=native_token_count,
        expected_transformer_depth=transformer_depth,
        expected_adapter_indices=adapter_indices,
        expected_adapter_kernel_size=3,
        expected_adapter_dilation=1,
    ),
    backbone=dict(
        type="mmaction.Recognizer3D",
        backbone=dict(
            type="VisionTransformerAdapter",
            img_size=224,
            patch_size=16,
            embed_dims=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4,
            qkv_bias=True,
            num_frames=16,
            drop_path_rate=0.1,
            norm_cfg=dict(type="LN", eps=1e-6),
            return_feat_map=True,
            with_cp=True,
            total_frames=raw_observation_count,
            adapter_index=adapter_indices,
        ),
        data_preprocessor=dict(
            type="mmaction.ActionDataPreprocessor",
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            format_shape="NCTHW",
        ),
        custom=dict(
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            strict_temporal_padding_mask=True,
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
            ],
            norm_eval=False,
            freeze_backbone=False,
        ),
    ),
    projection=dict(
        type="PhysTimeMeasureProjection",
        in_channels=384,
        out_channels=512,
        attention_channels=128,
        observation_measure="support_overlap",
        base_spacing_sec=0.5,
        num_levels=6,
        dropout=0.1,
        keep_uncovered_queries=True,
        use_null_evidence=True,
    ),
    rpn_head=dict(
        type="SupportDecoupledPhysicalQueryHead",
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
            (32.0, 1.0e8),
        ],
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample_radius=2.0,
        cls_prior_prob=0.01,
        endpoint_loss_weight=0.25,
        max_abs_delta_center=8.0,
        min_log_width=-6.0,
        max_log_width=6.0,
        diagnostics=True,
        loss=dict(cls_loss=dict(type="FocalLoss"), reg_loss=dict(type="DIOULoss")),
    ),
)

work_dir = "exps/thumos/adatad/phystime_g1b_sdpq_pool_native_j192"
