_base_ = ["./duca_boundary_burst_frontend_pretrain_fixed384.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_SOFT_ADAPTED_FRONTEND_PRETRAIN_FIXED384",
    hard_local_burst_support="center_plus_best_left_plus_best_right_when_feasible",
    hard_global_burst_support="none",
    local_bilateral_utility_relaxation=True,
    global_mandatory_group_decoder=False,
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        boundary_burst_require_bilateral_offsets=True,
        boundary_burst_require_global_mandatory_groups=False,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384"
)
