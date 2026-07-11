_base_ = ["./duca_transition_only_fixed384_official_adatad_backend_full_train.py"]


duca_transition_only_contract = dict(
    stage="p0_ablation_candidate",
    main_method_candidate=False,
    ablation="transition_only_beta_zero",
    detector_gradient_final_weight=0.0,
    coverage_and_detector_updates="coverage_updates_shared_scorer_detector_bridge_disabled",
    paper_claim_allowed=False,
    metric_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
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

work_dir = "exps/thumos/adatad/duca_transition_only_fixed384_no_detector_bridge"
