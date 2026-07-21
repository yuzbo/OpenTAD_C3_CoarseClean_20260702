_base_ = ["./duca_frontend_pretrain_fixed384_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_FRONTEND_PRETRAIN_FIXED384",
    transition_objective="oracle_calibrated_boundary_burst",
    oracle_reference="radius2_three_frame_bilateral_boundary_burst_then_global_fill",
    boundary_burst_radius=2,
    boundary_burst_quota=3,
    max_unselected_hole_dense_candidates=2,
    detector_executed=False,
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
        coarse_trunk_lr=5.0e-5,
        action_head_lr=1.0e-4,
        transition_scorer_lr=1.0e-4,
        loss_weights=dict(
            transition=0.10,
            transition_boundary=2.0,
        ),
        loss_weight_schedule=dict(
            transition=dict(end=0.10),
            transition_boundary=dict(end=2.0),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_boundary_burst_frontend_pretrain_fixed384"
