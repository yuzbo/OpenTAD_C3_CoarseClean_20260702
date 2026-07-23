_base_ = ["./duca_density_transport_nomax_fixed384_official60.py"]


duca_density_contract = dict(
    route="DUCA_BOUNDARY_UNCERTAINTY_CONTEXT_DENSITY_NOMAX_FIXED384_OFFICIAL60",
    density_model="boundary_uncertainty_context_mixture",
    density_components=("boundary", "uncertainty", "context"),
    hard_max_gap_enabled=False,
    soft_max_gap_enabled=False,
)

duca_transition_only_contract = dict(
    route="DUCA_BOUNDARY_UNCERTAINTY_CONTEXT_DENSITY_NOMAX_FIXED384_OFFICIAL60",
    acquisition_policy="continuous_mixture_density_transport",
    max_unselected_hole=None,
    soft_max_gap_loss_enabled=False,
)


model = dict(
    frame_selector=dict(
        acquisition_policy="continuous_mixture_density_transport",
    ),
)


work_dir = "exps/thumos/adatad/duca_mixture_density_transport_nomax_fixed384_official60"
