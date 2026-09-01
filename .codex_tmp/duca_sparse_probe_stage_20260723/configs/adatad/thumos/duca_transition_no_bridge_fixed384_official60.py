_base_ = ["./duca_protected_e2e_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_TRANSITION_NO_BRIDGE_FIXED384_OFFICIAL60",
    stage="preregistered_official60_ablation_after_p0_p3",
    main_method_candidate=False,
    ablation="transition_policy_without_direct_detector_gradient",
    detector_gradient_bridge="none",
    detector_gradient_is_direct=False,
    detector_gradient_final_weight=0.0,
    detector_gradient_updates="none",
)


model = dict(
    frame_selector=dict(
        detector_gradient_mode="none",
        policy_hidden_gradient_scale=0.0,
        loss_weight_schedule=dict(
            detector_gradient=dict(
                _delete_=True,
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=0,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_transition_no_bridge_fixed384_official60"
