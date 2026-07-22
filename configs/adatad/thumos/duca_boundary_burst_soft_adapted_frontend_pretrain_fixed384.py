_base_ = ["./duca_boundary_burst_frontend_pretrain_fixed384.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_SOFT_ADAPTED_FRONTEND_PRETRAIN_FIXED384",
    hard_local_burst_support="soft_bilateral_objective_only",
    hard_global_burst_support="none",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(boundary_burst_require_bilateral_offsets=False),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384"
)
