_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="protected_e2e_bridge025",
        detector_bridge_gradient_scale=0.25,
    ),
)

duca_variant_contract = dict(
    variant="protected_e2e_bridge025",
    coarse_probe_executed=True,
    learned_selector=True,
    detector_gradient_bridge=True,
    detector_gradient_updates="selector_only",
    detector_bridge_gradient_scale=0.25,
    uniform_companion_fraction=0.0,
    inference_uses_learned_policy_only=True,
)

work_dir = (
    "exps/thumos/adatad/duca_protected_physical_e2e_bridge025_fixed384_official60"
)
