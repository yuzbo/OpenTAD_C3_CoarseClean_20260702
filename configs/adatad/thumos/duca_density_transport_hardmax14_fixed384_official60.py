_base_ = ["./duca_density_transport_nomax_fixed384_official60.py"]


duca_density_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_HARDMAX14_FIXED384_OFFICIAL60",
    hard_max_gap_enabled=True,
    hard_max_unselected_hole=14,
    soft_max_gap_enabled=False,
)

duca_transition_only_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_HARDMAX14_FIXED384_OFFICIAL60",
    max_unselected_hole=14,
    soft_max_gap_loss_enabled=False,
)


model = dict(
    frame_selector=dict(
        max_unselected_hole=14,
        max_gap_loss_max_unselected_hole=14,
        soft_max_gap_loss_enabled=False,
        fail_on_infeasible_max_gap=True,
        loss_weights=dict(max_gap_hole=0.0),
    ),
)


work_dir = "exps/thumos/adatad/duca_density_transport_hardmax14_fixed384_official60"
