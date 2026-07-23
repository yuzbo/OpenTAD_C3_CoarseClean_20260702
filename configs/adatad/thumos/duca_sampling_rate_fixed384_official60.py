_base_ = ["./duca_transition_only_fixed384_official_adatad_backend_full_train.py"]

from tools.bata.duca_cellcf_protocol import protocol_for_name


duca_training_protocol = protocol_for_name("official60")
duca_end_epoch = duca_training_protocol.end_epoch
duca_steps_per_epoch = duca_training_protocol.steps_per_epoch
duca_total_steps = duca_training_protocol.expected_successful_optimizer_updates


duca_sampling_rate_contract = dict(
    route="DUCA_BUDGET_CALIBRATED_SAMPLING_RATE_FIXED384_OFFICIAL60",
    task="offline_temporal_action_detection",
    pre_backbone_plugin=True,
    acquisition="bounded_per_frame_retention_rate_plus_deterministic_cumulative_sampling",
    hard_forward="exact_k_strictly_increasing_original_time_observations",
    backward="hard_anchored_local_cumulative_rate_slope",
    exact_budget=384,
    hard_max_gap_enabled=False,
    soft_max_gap_enabled=False,
    detector_contribution_teacher="train_only_uniform_selected_input_grad_times_input",
    detector_contribution_interpolation="uniform_selected_axis_to_dense_768_linear",
    inference_teacher_free=True,
    paper_claim_allowed=False,
)


duca_transition_only_contract = dict(
    route="DUCA_BUDGET_CALIBRATED_SAMPLING_RATE_FIXED384_OFFICIAL60",
    stage="uniform_warmup_rate_homotopy_contribution_distillation",
    acquisition_policy="budget_calibrated_sampling_rate",
    max_unselected_hole=None,
    soft_max_gap_loss_enabled=False,
    detector_gradient_bridge="hard_forward_hard_anchored_cumulative_rate_local_slope",
    detector_gradient_final_weight=0.25,
    detector_utility_learning="uniform_detector_cls_reg_contribution_distillation",
    detector_utility_is_direct_gradient=True,
    counterfactual_teacher_producer_integrated=False,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
    training_profile=duca_training_protocol.name,
    schedule_steps_per_epoch=duca_steps_per_epoch,
    schedule_expected_total_steps=duca_total_steps,
)


model = dict(
    frame_selector=dict(
        acquisition_policy="budget_calibrated_sampling_rate",
        density_temperature=0.7,
        density_coverage_floor=0.05,
        density_smoothing_kernel=5,
        sampling_rate_utility_components="none",
        transition_objective="gaussian_mass",
        max_unselected_hole=None,
        max_gap_loss_max_unselected_hole=None,
        soft_max_gap_loss_enabled=False,
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
        detector_gradient_mode="density_transport_st",
        detector_contribution_distillation_weight=0.0,
        detector_contribution_components="none",
        detector_contribution_temperature=0.7,
        training_uniform_companion_fraction=0.50,
        training_uniform_companion_normalize_learned_gradient=True,
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
                warmup_steps=1500,
                transition_steps=1800,
            ),
            detector_contribution=dict(
                start=0.0,
                end=1.0,
                warmup_steps=1500,
                transition_steps=900,
            ),
            asformer_adapt=dict(
                start=0.0,
                end=0.0,
                warmup_steps=0,
                transition_steps=1,
            ),
        ),
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="none",
        ),
    ),
)


scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=duca_end_epoch,
)

workflow = dict(
    training_profile=duca_training_protocol.name,
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    # Full validation at epochs 5, 10, ..., 60 records the learning curve.
    # The terminal epoch-59 EMA remains the only primary checkpoint, so these
    # diagnostics cannot select a checkpoint after seeing validation mAP.
    val_eval_interval=5,
    val_eval_interval_anchor_epoch=5,
    val_start_epoch=4,
    intermediate_validation_role="diagnostic_only_no_checkpoint_selection",
    intermediate_validation_selects_checkpoint=False,
    end_epoch=duca_end_epoch,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_steps_per_epoch,
    expected_successful_optimizer_updates=duca_total_steps,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=duca_training_protocol.terminal_epoch,
    primary_checkpoint_state_key=duca_training_protocol.terminal_state_key,
    checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
)


work_dir = "exps/thumos/adatad/duca_sampling_rate_fixed384_official60"
