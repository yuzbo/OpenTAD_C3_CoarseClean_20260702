"""DUCA online acquisition plugin smoke config for AdaTAD integration checks.

This config is intentionally small: it validates the online wrapper contract
without introducing an offline ledger decision path.
"""

duca_online_plugin = dict(
    type="DucaOnlineSparseDetectorWrapper",
    route="online_acquire_then_sparse_detector_forward",
    detector_family="AdaTAD",
    budget=384,
    max_radius=16,
    budget_unit="detector_consumed_temporal_observation",
    coordinate="original_time",
    detector_consumes_selected_positions=True,
    uses_ledger_for_decision=False,
    adapter=dict(
        type="DucaAcquisitionAdapter",
        actionness_source="ZeroShotActionnessSource",
        decoder="budgeted_center_radius_decode",
        hard_forward=True,
        train_forward_hard_select=True,
        test_forward_hard_select=True,
    ),
    audit=dict(
        enabled=True,
        ledger_role="audit_only",
        fail_closed_no_leak=True,
        forbidden_inference_payload_keys=[
            "teacher_utility",
            "teacher_points",
            "dense_teacher",
            "gt_segments",
            "gt_labels",
            "oracle_boundary",
            "prediction_cache",
            "raw_prediction",
            "ledger",
            "ledger_path",
        ],
    ),
)
