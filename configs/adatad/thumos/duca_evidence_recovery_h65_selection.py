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

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a6_h65_selection"
