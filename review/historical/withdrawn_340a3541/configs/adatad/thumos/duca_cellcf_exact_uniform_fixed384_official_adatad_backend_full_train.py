_base_ = ["./duca_cellcf_fixed384_official_adatad_backend_full_train.py"]


duca_transition_only_contract = dict(
    stage="implemented_pending_focused_tests_and_real_cuda_gate",
    main_method_candidate=False,
    ablation="exact_uniform_anchor_control_in_same_local_cell_family",
    detector_utility_learning="disabled_exact_uniform_control",
)


model = dict(
    frame_selector=dict(
        local_cell_force_exact_uniform=True,
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
    ),
)


work_dir = "exps/thumos/adatad/duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train"
