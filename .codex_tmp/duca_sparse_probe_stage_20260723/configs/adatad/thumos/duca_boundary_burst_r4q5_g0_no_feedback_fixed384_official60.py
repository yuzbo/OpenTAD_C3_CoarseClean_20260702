_base_ = ["./duca_boundary_burst_g0_no_feedback_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_R4Q5_G0_NO_FEEDBACK_FIXED384_OFFICIAL60",
    boundary_burst_radius=4,
    boundary_burst_quota=5,
)


model = dict(
    frame_selector=dict(
        transition_boundary_radius=4,
        boundary_burst_quota=5.0,
    ),
)


work_dir = (
    "exps/thumos/adatad/"
    "duca_boundary_burst_r4q5_g0_no_feedback_fixed384_official60"
)
