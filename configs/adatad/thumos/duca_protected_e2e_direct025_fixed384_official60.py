_base_ = ["./duca_protected_e2e_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_PROTECTED_E2E_DIRECT025_FIXED384_OFFICIAL60",
    stage="selected_axis_three_arm_optimization",
    optimization_variant="direct_detector_gradient_without_policy_homotopy",
    detector_gradient_final_weight=0.25,
    training_uniform_companion_fraction=0.0,
)

model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.0,
        loss_weight_schedule=dict(
            policy_alpha=dict(
                _delete_=True,
                start=1.0,
                end=1.0,
                warmup_steps=0,
                transition_steps=0,
            ),
            detector_gradient=dict(
                _delete_=True,
                start=0.25,
                end=0.25,
                warmup_steps=0,
                transition_steps=0,
            ),
        ),
    ),
)

work_dir = "exps/thumos/adatad/duca_protected_e2e_direct025_fixed384_official60"
