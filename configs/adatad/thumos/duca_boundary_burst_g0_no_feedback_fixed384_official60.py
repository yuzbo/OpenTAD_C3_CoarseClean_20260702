_base_ = ["./duca_global_curriculum_g0_no_feedback_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_G0_NO_FEEDBACK_FIXED384_OFFICIAL60",
    transition_objective="oracle_calibrated_boundary_burst",
    boundary_burst_radius=2,
    boundary_burst_quota=3,
    boundary_burst_budget_fraction=0.25,
    detector_gradient_bridge="none",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        transition_objective="boundary_burst",
        transition_target_radius=0,
        transition_boundary_radius=2,
        boundary_burst_quota=3.0,
        boundary_burst_budget_fraction=0.25,
        boundary_burst_context_weight=0.05,
        boundary_burst_center_temperature=0.7,
        boundary_burst_offset_temperature=1.0,
        boundary_burst_side_min_mass=1.0,
        boundary_burst_anchor_weight=1.0,
        boundary_burst_bilateral_weight=1.0,
        boundary_burst_quota_weight=1.0,
        boundary_burst_fairness_weight=0.5,
        boundary_burst_overfill_weight=0.25,
    ),
)


work_dir = "exps/thumos/adatad/duca_boundary_burst_g0_no_feedback_fixed384_official60"
