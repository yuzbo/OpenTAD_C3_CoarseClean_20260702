"""A3: NO_ROBUST counterfactual arm (disables support dropout, consistency, and distillation)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "A3"
arm_name = "NO_ROBUST"

model = dict(
    frame_selector=dict(
        use_coverage=True,
        use_time_conditioning=True,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=False,
        use_h65_selection=False,
    ),
    backbone=dict(
        backbone=dict(
            bounded_interval_adapter=dict(enabled=True),
            continuous_timestamp_conditioner=dict(enabled=True),
            temporal_token_merge=dict(enabled=True),
        )
    ),
)

# The no-robust counterfactual removes the selector-side stabilizing losses;
# keep its detector path in full precision so the ablation remains runnable.
solver = dict(amp=False, clip_grad_norm=1.0)

# AMP replay is undefined without a GradScaler; keep the non-finite-loss
# replay/finite guard from the base workflow but disable AMP-only retries.
workflow = dict(max_amp_retries_per_batch=0)

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a3_no_robust"
