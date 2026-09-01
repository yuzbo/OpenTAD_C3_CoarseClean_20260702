_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="protected_e2e",
    ),
)

duca_variant_contract = dict(
    variant="protected_e2e",
    coarse_probe_executed=True,
    learned_selector=True,
    detector_gradient_bridge=True,
    detector_gradient_updates="selector_only",
)

work_dir = "exps/thumos/adatad/duca_protected_physical_e2e_fixed384_official60"
