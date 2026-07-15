_base_ = ["./phystime_tad_i3d_feature_gate0b.py"]

model = dict(
    projection=dict(
        keep_uncovered_queries=True,
        use_null_evidence=True,
    ),
    rpn_head=dict(
        _delete_=True,
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
            (32.0, 100000000.0),
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
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

work_dir = "exps/thumos/adatad/phystime_sdpq_i3d_feature_gate0b"
