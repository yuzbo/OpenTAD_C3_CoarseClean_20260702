"""A1: NO_COVERAGE counterfactual arm (pure exact semantic Top-K)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "A1"
arm_name = "NO_COVERAGE"

model = dict(
    frame_selector=dict(
        use_coverage=False,
        use_time_conditioning=True,
        use_temporal_merge=True,
        use_dense_recovery=True,
        use_robust_training=True,
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

work_dir = "exps/thumos/adatad/duca_evidence_recovery_a1_no_coverage"
