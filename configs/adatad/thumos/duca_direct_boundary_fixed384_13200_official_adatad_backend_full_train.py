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
        loss_weight_schedule=dict(
            warmup_steps=duca_loss_schedule_warmup_steps,
            transition_steps=duca_loss_schedule_transition_steps,
        ),
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=duca_end_epoch)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_eval_interval_anchor_epoch=5,
    val_start_epoch=4,
    end_epoch=duca_end_epoch,
)

work_dir = "exps/thumos/adatad/duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train"
