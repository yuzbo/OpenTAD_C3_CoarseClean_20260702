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
    frame_selector=dict(
        selector_variant="direct_boundary",
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        transition_scorer_lr=1.0e-4,
        loss_weight_schedule=dict(
            warmup_steps=duca_loss_schedule_warmup_steps,
            transition_steps=duca_loss_schedule_transition_steps,
        ),
        actionness_source_cfg=dict(
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
            calibration_split="train_only",
        ),
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=duca_end_epoch)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_eval_interval_anchor_epoch=47,
    val_start_epoch=47,
    end_epoch=duca_end_epoch,
)

work_dir = "exps/thumos/adatad/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train"
