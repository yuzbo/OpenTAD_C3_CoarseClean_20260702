_base_ = ["./duca_transition_only_fixed384_official_adatad_backend_full_train.py"]


duca_transition_only_contract = dict(
    stage="p0_matched_baseline_candidate",
    main_method_candidate=False,
    ablation="exact_uniform_fixed384_same_feasible_family",
    policy_homotopy="disabled_exact_uniform_reference",
    detector_gradient_final_weight=0.0,
    coverage_and_detector_updates="disabled_by_uniform_policy_and_zero_detector_bridge",
    paper_claim_allowed=False,
    metric_claim_allowed=False,
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

work_dir = "exps/thumos/adatad/duca_exact_uniform_fixed384_official_adatad_backend_full_train"
