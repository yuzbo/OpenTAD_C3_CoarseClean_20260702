_base_ = ["./duca_boundary_burst_g0_no_feedback_fixed384_official60.py"]


duca_t1_contract = dict(
    task="offline_temporal_action_detection",
    intervention="zero_initialized_true_time_feature_residual",
    descriptor_mode="actual",
    selected_positions_unchanged=True,
    detector_head_coordinate_system="selected_axis",
    physical_head_assignment_decode_unchanged=True,
    paper_claim_allowed=False,
)


model = dict(
    true_time_residual=dict(
        feature_dim=384,
        hidden_dim=64,
        descriptor_mode="actual",
    ),
)


work_dir = "exps/thumos/adatad/duca_t1_true_time_residual_g0_fixed384_official60"
