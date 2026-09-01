_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="exact_uniform",
        actionness_source_cfg=None,
    ),
)

duca_variant_contract = dict(
    variant="exact_uniform",
    coarse_probe_executed=False,
    learned_selector=False,
    detector_gradient_bridge=False,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_exact_uniform_fixed384_official60"
