_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]


model = dict(
    frame_selector=dict(
        arm="protected_e2e_uni_companion",
        detector_bridge_gradient_scale=0.25,
        uniform_companion_fraction=0.50,
    ),
)

duca_variant_contract = dict(
    variant="protected_e2e_uni_companion",
    inspiration="Uni-AdaFocus_training_input_diversity",
    coarse_probe_executed=True,
    learned_selector=True,
    detector_gradient_bridge=True,
    detector_gradient_updates="selector_only_on_learned_rows",
    detector_bridge_gradient_scale=0.25,
    training_uniform_companion_fraction=0.50,
    detector_forward_passes_per_batch=1,
    inference_uses_learned_policy_only=True,
    inference_extra_cost=False,
)

work_dir = (
    "exps/thumos/adatad/duca_protected_physical_e2e_uni_companion_fixed384_official60"
)
