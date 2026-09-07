"""C0: MATCHED_H65_60 control arm (matched 60 epochs, 6000 updates, K=384)."""
_base_ = ["./duca_evidence_recovery_base.py"]

arm_id = "C0"
arm_name = "MATCHED_H65_60"

model = dict(
    frame_selector=dict(
        use_coverage=False,
        use_time_conditioning=False,
        use_temporal_merge=False,
        use_dense_recovery=False,
        use_robust_training=False,
        use_h65_selection=True,
    ),
    backbone=dict(
        backbone=dict(
            bounded_interval_adapter=dict(enabled=False),
            continuous_timestamp_conditioner=dict(enabled=False),
            temporal_token_merge=dict(enabled=False),
        )
    ),
)

work_dir = "exps/thumos/adatad/duca_evidence_recovery_c0_matched_h65_60"
