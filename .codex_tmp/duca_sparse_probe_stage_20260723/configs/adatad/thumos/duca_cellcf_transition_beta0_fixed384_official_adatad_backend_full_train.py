_base_ = ["./duca_cellcf_fixed384_official_adatad_backend_full_train.py"]


duca_transition_only_contract = dict(
    stage="implemented_pending_focused_tests_and_real_cuda_gate",
    main_method_candidate=False,
    ablation="transition_only_without_detector_utility",
    detector_utility_learning="disabled_beta0",
)


model = dict(
    frame_selector=dict(
        local_cell_force_exact_uniform=False,
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
    ),
)


work_dir = "exps/thumos/adatad/duca_cellcf_transition_beta0_fixed384_official_adatad_backend_full_train"
