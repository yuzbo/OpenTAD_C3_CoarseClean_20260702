_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


train_block_list = os.environ.get("DUCA_RIME_TRAIN_BLOCK_LIST", "").strip()
development_block_list = os.environ.get(
    "DUCA_RIME_DEVELOPMENT_BLOCK_LIST",
    "",
).strip()
if not train_block_list or not development_block_list:
    raise RuntimeError(
        "dense TriDet training requires the frozen train/development block lists"
    )

dataset = dict(
    train=dict(
        block_list=train_block_list,
    ),
    val=None,
    test=dict(
        subset_name="training",
        block_list=development_block_list,
        test_mode=True,
        window_size=768,
    ),
)

evaluation = dict(
    subset="training",
    blocked_videos=development_block_list,
)

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
        max_seq_len=768,
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
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            iou_rate=dict(type="GIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=60,
)

workflow = dict(
    formal_protocol="duca_rime_dense_tridet_cost_baseline_v1",
    logging_interval=50,
    checkpoint_interval=10,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
)

duca_rime_dense_contract = dict(
    role="dense_adatad_baseline",
    task="offline_temporal_action_detection",
    detector_backend="TriDet",
    backbone="VideoMAE-S-AdaTAD",
    dense_window_size=768,
    detector_projection_in_channels=384,
    selector=None,
    dynamic_budget=False,
    train_role="detector_selector_train",
    evaluation_role="certification_development",
    official_final_subset_consumed=False,
    claim_scope="trained_dense_tridet_cost_reference_not_candidate_method",
    empirically_supported=False,
    paper_ready=False,
)

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/duca_rime_dense_tridet_total60"

del train_block_list
del development_block_list
