_base_ = ["./duca_online_official_adatad_backend_full_train.py"]


duca_end_epoch = 132
duca_schedule_steps_per_epoch = 100
duca_loss_schedule_total_steps = 13200
duca_loss_schedule_warmup_steps = 660
duca_loss_schedule_transition_steps = 1980


duca_online_main_contract = dict(
    stage="matched_13200_direct_boundary_baseline",
    main_method_candidate=False,
    diagnostic_only=False,
    baseline_for="shared_asformer_transition_only",
    loss_schedule_total_steps=duca_loss_schedule_total_steps,
    loss_schedule_steps_per_epoch=duca_schedule_steps_per_epoch,
    loss_schedule_warmup_steps=duca_loss_schedule_warmup_steps,
    loss_schedule_transition_steps=duca_loss_schedule_transition_steps,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
)


model = dict(
    backbone=dict(
        backbone=dict(with_cp=False),
    ),
    frame_selector=dict(
        selector_variant="direct_boundary",
        structured_temperature=0.7,
        actionness_weight=0.05,
        transition_weight=1.0,
        uncertainty_weight=0.25,
        utility_weight=0.50,
        boundary_weight=1.0,
        coarse_hidden_dim=96,
        max_unselected_hole=15,
        max_gap_loss_max_unselected_hole=15,
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        transition_scorer_lr=1.0e-4,
        loss_weight_schedule=dict(
            shape="cosine",
            warmup_steps=duca_loss_schedule_warmup_steps,
            transition_steps=duca_loss_schedule_transition_steps,
            actionness=dict(start=0.25, end=0.05),
            boundary=dict(start=0.50, end=1.0),
            start=dict(start=0.50, end=0.50),
            end=dict(start=0.50, end=0.50),
            context=dict(start=0.10, end=0.10),
            detector_gradient=dict(start=0.0, end=1.0),
            detector_utility=dict(start=0.0, end=0.10),
            max_gap_hole=dict(start=0.0, end=0.25),
            hole=dict(start=0.0, end=0.0),
            budget=dict(start=0.0, end=0.0),
            entropy=dict(start=0.0, end=0.0),
        ),
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name="online_c3_official_asformer_coarse_actionness",
            probe_model="official-action-seg",
            tcn_variant="official_asformer",
            spatial_size=64,
            tcn_hidden_dim=96,
            return_hidden_features=True,
            require_hidden_features=True,
            checkpoint_path="",
            require_checkpoint=False,
            frozen=False,
            trainable=True,
            mobilenet_pretrained=True,
            mobilenet_freeze_backbone=False,
            official_action_seg_backend="official_asformer",
            hidden_output_kind="pre_temporal_spatial_stem_hidden",
            thumos_trained=True,
            uses_labels=True,
            uses_gt=True,
            uses_teacher=False,
            uses_prediction_cache=False,
            trained_with_thumos_labels=True,
            trained_with_gt_segments=True,
            training_dataset="THUMOS14",
            training_supervision_scope="train_only",
            uses_labels_at_inference=False,
            uses_gt_at_inference=False,
            uses_teacher_at_inference=False,
            uses_prediction_cache_at_inference=False,
            calibration_split="none",
            calibration_artifact=None,
            calibration_artifact_sha256=None,
        ),
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=duca_end_epoch)

# Match the transition arms' batch-dependent parameter-use contract.
solver = dict(static_graph=False, find_unused_parameters=True)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=duca_end_epoch,
    training_profile="exposure132",
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_schedule_steps_per_epoch,
    expected_successful_optimizer_updates=duca_loss_schedule_total_steps,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=duca_end_epoch - 1,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_131_state_dict_ema",
)

work_dir = "exps/thumos/adatad/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train"
