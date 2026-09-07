_base_ = ["./duca_density_transport_nomax_fixed384_official60.py"]


duca_density_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_SOFTMAX14_FIXED384_OFFICIAL60",
    hard_max_gap_enabled=False,
    soft_max_gap_enabled=True,
    soft_max_unselected_hole_target=14,
)

duca_transition_only_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_SOFTMAX14_FIXED384_OFFICIAL60",
    max_unselected_hole=None,
    soft_max_gap_loss_enabled=True,
)


model = dict(
    frame_selector=dict(
        max_unselected_hole=None,
        max_gap_loss_max_unselected_hole=14,
        soft_max_gap_loss_enabled=True,
        loss_weights=dict(max_gap_hole=0.05),
    ),
)


work_dir = "exps/thumos/adatad/duca_density_transport_softmax14_fixed384_official60"
