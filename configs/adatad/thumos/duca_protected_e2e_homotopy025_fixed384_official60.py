_base_ = ["./duca_protected_e2e_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_PROTECTED_E2E_HOMOTOPY025_FIXED384_OFFICIAL60",
    stage="selected_axis_three_arm_optimization",
    optimization_variant="uniform_to_learned_homotopy",
    detector_gradient_final_weight=0.25,
    training_uniform_companion_fraction=0.0,
)

model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.0,
    ),
)

work_dir = "exps/thumos/adatad/duca_protected_e2e_homotopy025_fixed384_official60"
