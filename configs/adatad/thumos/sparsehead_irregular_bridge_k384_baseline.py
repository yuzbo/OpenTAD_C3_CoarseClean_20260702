"""Canonical SparseHead K384 baseline.

This config is intentionally diagnostic-only. It consolidates the bounded
irregular-grid bridge from the archived SparseHead checkout without claiming
dense equivalence or authorizing a full training run.
"""

_base_ = ["./input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"]

route_label = "SPARSEHEAD_SINGLE_MAINLINE"
route_family = "SPARSEHEAD_IRREGULAR_GRID"
candidate_name = "sparsehead_irregular_bridge_k384_baseline"

window_size = 384
dense_window_size = 768
chunk_num = window_size // 16

protocol_flags = dict(
    precheck_only=True,
    changed_surface="irregular_projection_neck_head_assignment",
    selector_changed=False,
    dynamic_budget_changed=False,
    token_compression_changed=False,
    adapter_backbone_changed=False,
    detector_head_logic_changed=True,
    loss_assignment_changed=True,
    post_processing_changed=True,
    uses_p2_head=False,
    uses_raw_prediction_cache=False,
    uses_teacher=False,
    uses_test_gt=False,
    uses_offline_ledger=False,
    tools_test_allowed=False,
    tools_train_allowed=False,
    remote_sync_allowed=False,
    slurm_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)

model = dict(
    type="IrregularActionFormer",
    max_seq_len=window_size,
    backbone=dict(
        backbone=dict(total_frames=window_size),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(
        _delete_=True,
        type="IrregularConvTransformerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 2, 5),
        conv_cfg=dict(kernel_size=1, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(
            n_head=4,
            local_k=4,
            safe_geometry=True,
            geometry_fp32=True,
            rel_dt_clip=64.0,
            rel_span_clip=8.0,
        ),
        path_pdrop=0.1,
        use_abs_pe=False,
        max_seq_len=window_size,
        input_pdrop=0.2,
    ),
    neck=dict(
        _delete_=True,
        type="IrregularFPN",
        in_channels=512,
        out_channels=512,
        num_levels=6,
        attn_cfg=dict(
            n_head=4,
            local_k=4,
            safe_geometry=True,
            geometry_fp32=True,
            rel_dt_clip=64.0,
            rel_span_clip=8.0,
        ),
        path_pdrop=0.1,
    ),
    rpn_head=dict(
        _delete_=True,
        type="IrregularActionFormerBridgeHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        predictor_kernel_size=3,
        assignment_mode="hard",
        regression_mode="symmetric_linear",
        center_radius_scale="point_radius",
        reg_denom_mode="left_right_mean",
        allow_legacy_full_cell_span=False,
        allow_center_fallback_inside_gt=False,
        hard_min_points_per_gt=1,
        hard_min_points_per_level=0,
        hard_max_points_per_gt=0,
        route_contract=dict(
            route_label="SPARSEHEAD_SINGLE_MAINLINE_BASELINE",
            compatibility="irregular_geometry_diagnostic_candidate",
            dense_equivalent_claim_allowed=False,
            allow_legacy_full_cell_span=False,
            allow_center_fallback_inside_gt=False,
            gt_axis="native",
            proposal_axis="native",
            nms_axis="native",
            postprocess_axis="native",
            eval_axis="seconds",
            expected_axis_contract=dict(
                gt_axis="native",
                proposal_axis="native",
                nms_axis="native",
                postprocess_axis="native",
            ),
            diagnostic_only=True,
            primary_result_allowed=False,
        ),
        cls_prior_prob=0.01,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        prior_generator=dict(
            type="IrregularPointGeneratorV2",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[(0, 8), (2, 16), (4, 32), (8, 64), (16, 128), (32, 10000)],
            range_mode="absolute",
            decode_scale_mode="level_stride",
            radius_scale_mode="level_stride",
        ),
        debug_cfg=dict(enable=True),
    ),
)

post_processing = dict(save_dict=True)
work_dir = "exps/thumos/adatad/sparsehead_irregular_bridge_k384_baseline"
