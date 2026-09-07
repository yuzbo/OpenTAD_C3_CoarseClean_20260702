"""A6: H65_SELECTION counterfactual arm (replays H65 selection with new recovery/merge)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "A6"
arm_name = "H65_SELECTION"

model = dict(
    frame_selector=dict(
        use_coverage=True,
        use_time_conditioning=True,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=True,
        use_h65_selection=True,
    ),
    backbone=dict(
        backbone=dict(
            bounded_interval_adapter=dict(enabled=True),
            continuous_timestamp_conditioner=dict(enabled=True),
            temporal_token_merge=dict(enabled=True),
        )
    ),
)

# H65's irregular physical-time path is numerically unstable in fp16 even
# with bounded attention bias. Keep this selection counterfactual fail-closed
# in float32; the other arms retain the shared AMP protocol.
solver = dict(amp=False, clip_grad_norm=1.0)
workflow = dict(max_amp_retries_per_batch=0)

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a6_h65_selection"
