"""Small real-training diagnostic for sampling-rate mechanism figures.

This deliberately reuses the full sampling-rate model.  It runs only forty
optimizer updates so the same fixed train and validation windows can show how
the policy changes from the uniform warmup to a learned sampling rate.  It is
not an official-60 result and never reports mAP.
"""

_base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]


duca_mini_visual_contract = dict(
    task="offline_temporal_action_detection",
    route="DUCA_BUDGET_CALIBRATED_SAMPLING_RATE_MINI_VISUAL",
    purpose="trained_small_sample_mechanism_diagnostic_not_official_map",
    train_epochs=10,
    train_updates_per_epoch=4,
    expected_optimizer_updates=40,
    tracked_train_batch_index=0,
    tracked_validation_batches=2,
    checkpoint_epochs=[1, 5, 10],
    metric_claim_allowed=False,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        # Compress the same curriculum into forty real updates.  Epoch one is
        # still the near-uniform reference, epoch five is the handoff, and
        # epoch ten is the learned-rate state used by the diagnostic panels.
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=8,
                transition_steps=12,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=4,
                transition_steps=12,
            ),
            detector_contribution=dict(
                start=0.0,
                end=1.0,
                warmup_steps=4,
                transition_steps=12,
            ),
            asformer_adapt=dict(
                start=0.0,
                end=1.0,
                warmup_steps=4,
                transition_steps=12,
            ),
        ),
    )
)


scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=1,
    max_epoch=10,
)


solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
    find_unused_parameters=True,
)


workflow = dict(
    # The official60 selected-axis protocol is intentionally not used: this
    # is a bounded mechanism diagnostic with a real but tiny optimization run.
    formal_protocol="",
    training_profile="mini_visualization_only",
    formal_successful_update_contract=False,
    end_epoch=10,
    checkpoint_interval=1,
    logging_interval=1,
    max_train_iters=4,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    max_amp_retries_per_batch=2,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
)


work_dir = "exps/thumos/adatad/duca_sampling_rate_both_asformer_full_mini_visual"
