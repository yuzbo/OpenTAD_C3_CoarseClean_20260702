_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="transition_no_bridge",
    ),
)

duca_variant_contract = dict(
    variant="transition_no_bridge",
    coarse_probe_executed=True,
    learned_selector=True,
    detector_gradient_bridge=False,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_transition_no_bridge_fixed384_official60"
