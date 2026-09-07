_base_ = ["./duca_protected_e2e_homotopy025_fixed384_official60.py"]


duca_transition_only_contract = dict(
    route="DUCA_PROTECTED_E2E_HOMOTOPY_UNI_COMPANION025_FIXED384_OFFICIAL60",
    stage="selected_axis_three_arm_optimization",
    optimization_variant="homotopy_with_uniform_training_companion",
    inspiration="Uni-AdaFocus_random_and_policy_view_training_stability",
    detector_gradient_final_weight=0.25,
    detector_gradient_updates="transition_scorer_only_on_learned_rows",
    training_uniform_companion_fraction=0.50,
    detector_forward_passes_per_batch=1,
    inference_uses_learned_policy_only=True,
    inference_extra_companion_cost=False,
)

model = dict(
    frame_selector=dict(
        training_uniform_companion_fraction=0.50,
    ),
)

work_dir = (
    "exps/thumos/adatad/"
    "duca_protected_e2e_homotopy_uni_companion025_fixed384_official60"
)
