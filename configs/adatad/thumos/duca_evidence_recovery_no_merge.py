"""A4: NO_MERGE counterfactual arm (disables VideoMAE internal temporal token merging)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "A4"
arm_name = "NO_MERGE"

model = dict(
    frame_selector=dict(
        use_coverage=True,
        use_time_conditioning=True,
        use_temporal_merge=False,
        use_dense_recovery=True,
        use_robust_training=True,
        use_h65_selection=False,
    ),
    backbone=dict(
        backbone=dict(
            bounded_interval_adapter=dict(enabled=True),
            continuous_timestamp_conditioner=dict(enabled=True),
            temporal_token_merge=dict(enabled=False),
        )
    ),
)

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a4_no_merge"
