_base_ = ["./duca_boundary_burst_g0_no_feedback_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_SOFT_G0_NO_FEEDBACK_FIXED384_OFFICIAL60",
    hard_local_burst_support="soft_bilateral_objective_only",
    hard_global_burst_support="none",
    paper_claim_allowed=False,
)


model = dict(
    frame_selector=dict(boundary_burst_require_bilateral_offsets=False),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_soft_g0_no_feedback_fixed384_official60"
)
