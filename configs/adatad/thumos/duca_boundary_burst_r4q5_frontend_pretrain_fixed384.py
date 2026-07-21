_base_ = ["./duca_boundary_burst_frontend_pretrain_fixed384.py"]


duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_BURST_R4Q5_FRONTEND_PRETRAIN_FIXED384",
    oracle_reference="radius4_five_frame_bilateral_boundary_burst_then_global_fill",
    boundary_burst_radius=4,
    boundary_burst_quota=5,
)


model = dict(
    frame_selector=dict(
        transition_boundary_radius=4,
        boundary_burst_quota=5.0,
    ),
)


work_dir = "exps/thumos/adatad/duca_boundary_burst_r4q5_frontend_pretrain_fixed384"
