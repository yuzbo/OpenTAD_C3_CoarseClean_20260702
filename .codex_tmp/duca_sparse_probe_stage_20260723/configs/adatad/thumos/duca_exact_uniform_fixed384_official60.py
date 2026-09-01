_base_ = ["./duca_transition_no_bridge_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_EXACT_UNIFORM_FIXED384_OFFICIAL60",
    stage="preregistered_official60_control_after_p0_p3",
    main_method_candidate=False,
    ablation="exact_uniform_same_global_feasible_family",
    policy_homotopy="disabled_exact_uniform_reference",
    detector_gradient_bridge="none",
)


model = dict(
    frame_selector=dict(
        inference_policy_alpha=0.0,
        loss_weight_schedule=dict(
            policy_alpha=dict(
                _delete_=True,
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_exact_uniform_fixed384_official60"
