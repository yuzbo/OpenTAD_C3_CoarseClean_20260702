_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]

model = dict(
    frame_selector=dict(
        arm="protected_e2e_homotopy025",
        detector_bridge_gradient_scale=0.25,
        uniform_companion_fraction=0.0,
        homotopy_total_steps=6000,
    ),
)

duca_variant_contract = dict(
    variant="protected_e2e_homotopy025",
    predecessor_commit="4d84acda4d073fb6aac956c21386df8ed5d4d2f5",
    coarse_probe_executed=True,
    learned_selector=True,
    physical_policy="shared_exact_k_dag_hard_uniform_to_learned_log_potential",
    policy_alpha_zero_contract="hard_forward_exact_uniform",
    policy_alpha_warmup_fraction=0.05,
    policy_alpha_transition_fraction=0.30,
    policy_alpha_transition_shape="cosine",
    successful_optimizer_updates=6000,
    detector_gradient_bridge=True,
    detector_gradient_updates="selector_only",
    detector_bridge_gradient_scale=0.25,
    inference_policy_alpha=1.0,
    schedule_step_checkpointed=True,
)

work_dir = (
    "exps/thumos/adatad/"
    "duca_protected_physical_e2e_homotopy025_fixed384_official60"
)
