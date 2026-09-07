_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="protected_e2e_rho001",
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="asformer_last_encoder_layer",
        ),
    ),
)

duca_variant_contract = dict(
    variant="protected_e2e_rho001",
    coarse_probe_executed=True,
    learned_selector=True,
    detector_gradient_bridge=True,
    detector_gradient_updates="selector_plus_last_asformer_encoder_block",
    detector_gradient_scale_to_last_asformer_block=0.01,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_e2e_rho001_fixed384_official60"
