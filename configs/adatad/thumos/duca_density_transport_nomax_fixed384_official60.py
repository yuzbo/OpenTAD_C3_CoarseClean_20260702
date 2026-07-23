_base_ = ["./duca_transition_only_fixed384_official_adatad_backend_full_train.py"]


duca_density_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_NOMAX_FIXED384_OFFICIAL60",
    task="offline_temporal_action_detection",
    density_model="single_transition_density",
    acquisition="smooth_density_inverse_cdf",
    hard_forward="deterministic_exact_k_strict_integer_projection",
    backward="two_point_inverse_cdf_resample_straight_through",
    exact_budget=384,
    hard_max_gap_enabled=False,
    soft_max_gap_enabled=False,
    detector_gradient_updates="transition_density_scorer_only",
    action_head_detector_gradient=False,
    spatial_stem_detector_gradient=False,
    paper_claim_allowed=False,
)

duca_transition_only_contract = dict(
    route="DUCA_CONTINUOUS_DENSITY_NOMAX_FIXED384_OFFICIAL60",
    stage="joint_density_transport_official60",
    acquisition_policy="continuous_density_transport",
    max_unselected_hole=None,
    soft_max_gap_loss_enabled=False,
    detector_gradient_bridge="hard_forward_soft_inverse_cdf_resample",
    detector_gradient_final_weight=0.25,
    detector_utility_learning="direct_detector_loss_gradient_through_density",
    detector_utility_is_direct_gradient=True,
    counterfactual_teacher_producer_integrated=False,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
)


model = dict(
    frame_selector=dict(
        acquisition_policy="continuous_density_transport",
        density_temperature=0.7,
        density_coverage_floor=0.05,
        density_smoothing_kernel=5,
        transition_objective="gaussian_mass",
        max_unselected_hole=None,
        max_gap_loss_max_unselected_hole=None,
        soft_max_gap_loss_enabled=False,
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
        detector_gradient_mode="density_transport_st",
        policy_hidden_gradient_scale=0.0,
        auxiliary_hidden_gradient_scale=0.25,
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
        loss_weights=dict(
            max_gap_hole=0.0,
            transition=0.5,
            transition_boundary=0.5,
        ),
        loss_weight_schedule=dict(
            detector_gradient=dict(
                start=0.0,
                end=0.25,
                warmup_steps=2100,
                transition_steps=1500,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=300,
                transition_steps=1800,
            ),
        ),
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="none",
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_density_transport_nomax_fixed384_official60"
