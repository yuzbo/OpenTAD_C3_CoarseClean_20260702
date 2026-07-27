_base_ = ["./duca_rime_uniform_fixed384_total60.py"]


model = dict(
    type="TriDet",
    projection=dict(
        _delete_=True,
        type="TriDetProj",
        in_channels=384,
        out_channels=512,
        sgp_mlp_dim=768,
        arch=(2, 2, 5),
        downsample_type="max",
        sgp_win_size=[1, 1, 1, 1, 1, 1],
        k=5,
        init_conv_vars=0,
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=512,
        input_noise=0.0,
    ),
    neck=dict(
        _delete_=True,
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        _delete_=True,
        type="TriDetHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4),
                (4, 8),
                (8, 16),
                (16, 32),
                (32, 64),
                (64, 10000),
            ],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        boundary_kernel_size=3,
        iou_weight_power=0.2,
        num_bins=16,
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract="duca_protected_e2e_physical_v1",
        ),
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            iou_rate=dict(type="GIOULoss"),
        ),
    ),
)

duca_rime_variant = dict(
    arm="U-fixed-TriDet",
    phase=4,
    detector_backend="TriDet",
    exact_uniform=True,
    cross_backend_control=True,
    empirically_supported=False,
    paper_ready=False,
)

work_dir = "exps/thumos/adatad/duca_rime_uniform_fixed_tridet_total60"
