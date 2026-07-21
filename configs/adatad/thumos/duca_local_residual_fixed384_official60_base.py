_base_ = ["./duca_two_stage_pretrained_frozen_fixed384_official60.py"]


duca_local_residual_warmup_steps = 1000
duca_local_residual_policy_steps = 1500
duca_local_residual_bridge_warmup_steps = 2500
duca_local_residual_bridge_steps = 1500


duca_transition_only_contract = dict(
    _delete_=True,
    route="DUCA_LOCAL_RESIDUAL_FIXED384_OFFICIAL60_BASE",
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    training_profile="official60",
    exact_budget=384,
    dense_window_size=768,
    acquisition_policy="local_cell_deformation",
    base_policy="detached_abs_delta_actionness",
    residual_policy="bounded_local_cell_residual",
    local_cell_detector_grid_mode="selected",
    hard_forward="actual_selected_rgb",
    soft_backward="local_cell_categorical_straight_through",
    coarse_probe_training="frozen_for_all_official60_updates",
    detector_gradient_updates="residual_scorer_only",
    action_head_detector_gradient=False,
    asformer_trunk_detector_gradient=False,
    detector_extra_updates_outside_official60=0,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
)


duca_complete_loss_weights = dict(
    _delete_=True,
    detector=1.0,
    actionness=0.0,
    budget=0.0,
    boundary=0.0,
    hole=0.0,
    max_gap_hole=0.0,
    redundancy=0.0,
    radius=0.0,
    entropy=0.0,
    teacher=0.0,
    detector_utility=0.0,
    start=0.0,
    end=0.0,
    context=0.0,
    lagrangian_budget=0.0,
    marginal_monotonic=0.0,
    hard_budget_cap=0.0,
    transition=0.0,
    transition_boundary=0.0,
)


model = dict(
    frame_selector=dict(
        acquisition_policy="local_cell_deformation",
        local_cell_force_exact_uniform=False,
        local_cell_base_policy="abs_delta_actionness",
        local_cell_residual_scale=0.25,
        local_cell_detector_grid_mode="selected",
        detector_gradient_mode="none",
        inference_policy_alpha=1.0,
        training_uniform_companion_fraction=0.0,
        allow_frozen_coarse_probe=True,
        policy_hidden_gradient_scale=0.0,
        auxiliary_hidden_gradient_scale=0.0,
        max_unselected_hole=None,
        max_gap_loss_max_unselected_hole=None,
        temporal_sampling_contract=None,
        hard_max_gap_repair=False,
        soft_max_gap_loss_enabled=False,
        fail_on_infeasible_max_gap=False,
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
        actionness_source_cfg=dict(
            frozen=True,
            trainable=False,
            policy_hidden_gradient_scope="none",
        ),
        loss_weights=duca_complete_loss_weights,
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="cosine",
            warmup_steps=duca_local_residual_warmup_steps,
            transition_steps=duca_local_residual_policy_steps,
            actionness=dict(start=0.0, end=0.0),
            transition=dict(
                start=0.0,
                end=0.02,
                warmup_steps=duca_local_residual_warmup_steps,
                transition_steps=duca_local_residual_policy_steps,
            ),
            transition_boundary=dict(
                start=0.0,
                end=0.25,
                warmup_steps=duca_local_residual_warmup_steps,
                transition_steps=duca_local_residual_policy_steps,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=duca_local_residual_warmup_steps,
                transition_steps=duca_local_residual_policy_steps,
            ),
            detector_gradient=dict(
                start=0.0,
                end=0.0,
                warmup_steps=duca_local_residual_bridge_warmup_steps,
                transition_steps=duca_local_residual_bridge_steps,
            ),
        ),
    ),
)


work_dir = "exps/thumos/adatad/duca_local_residual_fixed384_official60_base"
