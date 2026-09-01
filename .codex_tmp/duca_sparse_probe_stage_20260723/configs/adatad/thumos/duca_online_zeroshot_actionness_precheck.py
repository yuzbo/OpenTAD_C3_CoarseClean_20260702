dense_window_size = 768
window_size = 384
toy_input_channels = 8
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
    route="DUCA_ONLINE_OPENTAD_PRECHECK",
    stage="duca_online_zeroshot_actionness_precheck",
    detector_stack="OpenTAD_SingleStageDetector",
    actionness_source="zero_shot_motion",
    no_ledger_decision=True,
    budget_max=window_size,
    dense_window_size=dense_window_size,
    coordinate_space="original_time",
    selected_positions_unit="original_time_index",
    teacher_free_eval=True,
    teacher_train_loss_only=True,
    selected_axis_remap_required=True,
    gt_required_in_train=True,
    raw_prediction_cache_forbidden=True,
    build_detector_required=True,
    standard_forward_required=True,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    metadata_keys=duca_online_metadata_keys,
)

model = dict(
    type="SingleStageDetector",
    frame_selector=dict(
        type="DucaOnlineFrameSelector",
        in_channels=toy_input_channels,
        dense_window_size=dense_window_size,
        budget=window_size,
        coordinate_space="original_time",
        selected_positions_unit="original_time_index",
        no_ledger_decision=True,
        remap_gt_to_selected_axis=True,
        selected_axis_remap_required=True,
        forbid_ledger=True,
        forbid_raw_prediction_cache=True,
        metadata_keys=duca_online_metadata_keys,
        actionness_source_cfg=dict(
            type="ZeroShotMotionActionnessSource",
            source_name="zero_shot_motion_actionness",
            score_method="frame_difference_energy",
            normalize="per_video",
            no_train_gt=True,
            no_gt_generation=True,
            no_teacher=True,
            no_oracle=True,
            no_raw_prediction_cache=True,
        ),
    ),
    rpn_head=dict(
        type="DucaOnlinePrecheckHead",
        num_classes=num_classes,
        in_channels=toy_input_channels,
        dense_window_size=dense_window_size,
        budget=window_size,
        require_gt_in_train=True,
        require_selected_metas=True,
        require_original_time_positions=True,
        require_selected_axis_remap=True,
        forbid_raw_prediction_cache=True,
        metadata_keys=duca_online_metadata_keys,
        teacher_cfg=dict(
            enabled=True,
            train_only=True,
            loss_name="loss_duca_teacher_train_only",
            loss_weight=0.0,
            enabled_for_inference=False,
            forbid_inference=True,
        ),
    ),
)

dataset = dict(
    train=dict(type="ToyDucaOnlinePrecheckDataset", subset_name="training", window_size=dense_window_size),
    val=dict(type="ToyDucaOnlinePrecheckDataset", subset_name="validation", window_size=dense_window_size),
    test=dict(type="ToyDucaOnlinePrecheckDataset", subset_name="validation", test_mode=True, window_size=dense_window_size),
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

work_dir = "exps/thumos/adatad/duca_online_zeroshot_actionness_precheck"
