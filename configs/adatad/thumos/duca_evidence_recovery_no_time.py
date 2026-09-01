"""A2: NO_TIME counterfactual arm (fixed g=1, no timestamp conditioning)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "A2"
arm_name = "NO_TIME"

model = dict(
    frame_selector=dict(
        use_coverage=True,
        use_time_conditioning=False,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=True,
        use_h65_selection=False,
    ),
    backbone=dict(
        backbone=dict(
            bounded_interval_adapter=dict(enabled=False),
            continuous_timestamp_conditioner=dict(enabled=False),
            temporal_token_merge=dict(enabled=True),
        )
    ),
)

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a2_no_time"
