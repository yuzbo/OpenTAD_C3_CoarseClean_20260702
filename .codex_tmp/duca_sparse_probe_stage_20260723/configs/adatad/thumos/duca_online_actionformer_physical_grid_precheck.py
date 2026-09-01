dense_window_size = 768
window_size = 384
toy_input_channels = 3
num_classes = 20

duca_online_metadata_keys = dict(
    selected_positions="duca_online_selected_positions",
    selected_positions_unit="duca_online_selected_positions_unit",
    selected_mask="duca_online_selected_mask",
    selected_count="duca_online_selected_count",
    remap="duca_online_selected_axis_remap",
    source="duca_online_actionness_source",
)

duca_online_precheck_contract = dict(
    route="DUCA_ONLINE_ACTIONFORMER_PHYSICAL_GRID_PRECHECK",
    stage="duca_online_actionformer_physical_grid_precheck",
    detector_stack="OpenTAD_SingleStageDetector_ActionFormerHead_PhysicalGrid",
    detector_head_type="ActionFormerHead",
    actionness_source="zero_shot_motion",
    no_ledger_decision=True,
    budget_max=window_size,
    dense_window_size=dense_window_size,
    coordinate_space="original_time",
    detector_output_coordinate_space="true_time_dense_index",
    selected_positions_unit="original_time_index",
    teacher_free_eval=True,
    teacher_train_loss_only=True,
    selected_axis_remap_required=False,
    physical_grid_actionformer_required=True,
    gt_required_in_train=True,
    raw_prediction_cache_forbidden=True,
    build_detector_required=True,
    standard_forward_required=True,
    real_head_required=True,
    toy_head_allowed=False,
    main_method_candidate=False,
    diagnostic_only=True,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    metadata_keys=duca_online_metadata_keys,
)

model = dict(
    type="ActionFormer",
    frame_selector=dict(
        type="DucaOnlineFrameSelector",
        in_channels=toy_input_channels,
        dense_window_size=dense_window_size,
        budget=window_size,
        max_radius=16,
        selector_hidden_channels=16,
        detector_gradient_mode="st_sparse_gather_soft_context",
        coordinate_space="original_time",
        detector_output_coordinate_space="true_time_dense_index",
        selected_positions_unit="original_time_index",
        loss_weights=dict(
            teacher=0.0,
            budget=0.0,
            boundary=0.0,
            hole=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
        ),
        no_ledger_decision=True,
        remap_gt_to_selected_axis=False,
        selected_axis_remap_required=True,
        forbid_ledger=True,
        forbid_raw_prediction_cache=True,
        metadata_keys=duca_online_metadata_keys,
        actionness_source_cfg=dict(
            type="ZeroShotMotionActionnessSource",
            source_name="zero_shot_motion_actionness",
            mode="motion",
            thumos_trained=False,
            uses_labels=False,
            uses_teacher=False,
            uses_gt=False,
            uses_prediction_cache=False,
            no_train_gt=True,
            no_teacher=True,
            no_oracle=True,
            no_raw_prediction_cache=True,
            no_gt_generation=True,
            calibration_split="none",
            checkpoint_hash="no_checkpoint_motion_energy",
        ),
    ),
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=toy_input_channels,
        out_channels=8,
        arch=(1, 0, 0),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=1, n_mha_win_size=1),
        path_pdrop=0.0,
        use_abs_pe=False,
        max_seq_len=window_size,
        input_pdrop=0.0,
    ),
    neck=None,
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=num_classes,
        in_channels=8,
        feat_channels=8,
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            requires_irregular_native_axis=True,
            eps=1.0e-6,
            coordinate_space="true_time_dense_index",
            selected_position_key="irregular_selected_positions",
            dense_valid_len_key="irregular_dense_valid_len",
        ),
        num_convs=0,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1],
            regression_range=[(0, 10000)],
        ),
        loss_normalizer=16,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

dataset = dict(
    train=dict(type="ToyDucaOnlineActionFormerPrecheckDataset", subset_name="training", window_size=dense_window_size),
    val=dict(type="ToyDucaOnlineActionFormerPrecheckDataset", subset_name="validation", window_size=dense_window_size),
    test=dict(
        type="ToyDucaOnlineActionFormerPrecheckDataset",
        subset_name="validation",
        test_mode=True,
        window_size=dense_window_size,
    ),
)

solver = dict(
    train=dict(batch_size=2, num_workers=0),
    val=dict(batch_size=1, num_workers=0),
    test=dict(batch_size=1, num_workers=0),
    amp=False,
    ema=False,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(sliding_window=True, nms=None, pre_nms_thresh=0.0, pre_nms_topk=16)
evaluation = dict(type="mAP", subset="validation")
workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=1,
    val_start_epoch=0,
    end_epoch=1,
    max_train_iters=1,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/adatad/duca_online_actionformer_physical_grid_precheck"
