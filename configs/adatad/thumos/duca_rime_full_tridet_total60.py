_base_ = ["./duca_rime_physical_total60_base.py"]

import os

_formal_target = float(os.environ.get("DUCA_RIME_TARGET_MEAN_COST", "384"))
_dynamic_panel = _formal_target > 192.0


# Phase 4 cross-detector backend.  The selector, RGB backbone, candidate-K
# grid, train-only targets, optimizer-update count, and cost ledger are
# inherited unchanged from RIME/ActionFormer; only the detector tail changes.
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
            contract="duca_rime_physical_dynamic_k_v1",
        ),
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
            iou_rate=dict(type="GIOULoss"),
        ),
    ),
)

duca_rime_variant = dict(
    arm="RIME-full-TriDet",
    phase=4,
    detector_backend="TriDet",
    dynamic_budget=_dynamic_panel,
    pair_risk_used_for_allocation=_dynamic_panel,
    allocation=(
        "frozen_per_video_dual"
        if _dynamic_panel
        else "fixed_floor_budget_position_only"
    ),
    budget_panel_semantics=(
        "content_conditioned_dynamic_budget_panel"
        if _dynamic_panel
        else "exact_k192_learned_position_stress_panel"
    ),
    cross_backend_replication=True,
    dense_physical_training_axis=True,
    q_to_t_before_nms=True,
    candidate_model=True,
    empirically_supported=False,
    paper_ready=False,
)

work_dir = "exps/thumos/adatad/duca_rime_full_tridet_total60"

del _formal_target, _dynamic_panel
