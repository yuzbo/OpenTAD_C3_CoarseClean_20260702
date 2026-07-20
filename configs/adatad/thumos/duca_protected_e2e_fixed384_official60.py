_base_ = ["./duca_transition_only_fixed384_official_adatad_backend_full_train.py"]

from tools.bata.duca_cellcf_protocol import protocol_for_name


duca_training_protocol = protocol_for_name("official60")
duca_end_epoch = duca_training_protocol.end_epoch
duca_steps_per_epoch = duca_training_protocol.steps_per_epoch
duca_total_steps = duca_training_protocol.expected_successful_optimizer_updates
duca_policy_warmup_steps = round(duca_total_steps * 0.05)
duca_policy_transition_steps = round(duca_total_steps * 0.30)
duca_bridge_warmup_steps = round(duca_total_steps * 0.35)
duca_bridge_transition_steps = round(duca_total_steps * 0.25)

# THUMOS candidates are four source frames apart. G=2 therefore bounds the
# largest selected-center interval to three candidate steps, or 12 source
# frames, which is the strict representable value below the requested 15-frame
# physical cap.
duca_max_unselected_hole = 2
duca_temporal_sampling_contract = dict(
    hard_budget=384,
    dense_window_size=768,
    max_unselected_hole_dense_candidates=duca_max_unselected_hole,
    dataset_feature_stride_source_frames=4,
    dataset_sample_stride=1,
    requested_max_source_frame_interval=15,
    detector_axis="selected_axis_index",
    dense_axis_unit="dense_candidate_index",
    task="offline_temporal_action_detection",
)


duca_transition_only_contract = dict(
    route="DUCA_PROTECTED_E2E_FIXED384_OFFICIAL60",
    stage="p0_p3_gate_required_before_official60",
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    main_method_candidate=True,
    empirically_supported=False,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
    training_profile="official60",
    official_epoch_count=duca_end_epoch,
    expected_successful_optimizer_updates=duca_total_steps,
    primary_checkpoint="terminal_epoch_59_state_dict_ema",
    intermediate_validation_selects_checkpoint=False,
    exact_budget=384,
    dense_window_size=768,
    max_unselected_hole_dense_candidates=duca_max_unselected_hole,
    max_selected_interval_dense_steps=3,
    max_selected_interval_source_frames=12,
    requested_max_source_frame_interval=15,
    temporal_sampling_contract=duca_temporal_sampling_contract,
    acquisition_policy="global_structured_topk",
    hard_soft_feasible_family="shared_exact_k_max_gap_dynamic_program",
    detector_gradient_bridge="protected_structured_transport",
    detector_gradient_is_direct=True,
    detector_gradient_final_weight=0.25,
    detector_gradient_updates="transition_scorer_only",
    policy_hidden_gradient_scale=0.0,
    action_head_detector_gradient=False,
    asformer_trunk_detector_gradient=False,
    counterfactual_teacher_producer_integrated=False,
    inference_teacher_free=True,
    inference_gt_free=True,
    inference_cache_free=True,
    deployment_allowed=False,
)


model = dict(
    frame_selector=dict(
        max_unselected_hole=duca_max_unselected_hole,
        max_gap_loss_max_unselected_hole=duca_max_unselected_hole,
        temporal_sampling_contract=duca_temporal_sampling_contract,
        detector_gradient_mode="protected_structured_transport",
        policy_hidden_gradient_scale=0.0,
        actionness_source_cfg=dict(
            policy_hidden_gradient_scope="none",
        ),
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
        loss_weight_schedule=dict(
            policy_alpha=dict(
                _delete_=True,
                start=0.0,
                end=1.0,
                warmup_steps=duca_policy_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
            detector_gradient=dict(
                _delete_=True,
                start=0.0,
                end=0.25,
                warmup_steps=duca_bridge_warmup_steps,
                transition_steps=duca_bridge_transition_steps,
            ),
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
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
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

work_dir = "exps/thumos/adatad/duca_protected_e2e_fixed384_official60"

del duca_training_protocol
