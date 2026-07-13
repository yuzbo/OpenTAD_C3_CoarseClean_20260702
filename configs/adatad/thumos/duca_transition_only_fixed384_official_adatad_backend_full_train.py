_base_ = ["./duca_online_official_adatad_backend_full_train.py"]

import os


def _require_fixed_env(name, expected):
    value = os.environ.get(name)
    if value is not None and int(value) != int(expected):
        raise ValueError(f"{name} is fixed to {expected} for this audited configuration, got {value}")


for _name, _expected in (
    ("DUCA_ONLINE_BUDGET", 384),
    ("DUCA_OFFICIAL_ADATAD_BUDGET", 384),
    ("DUCA_ONLINE_DENSE_WINDOW_SIZE", 768),
    ("DUCA_VALIDATOR_MAX_BUDGET", 384),
    ("DUCA_BUDGET_CURVE_MODE", 0),
    ("DUCA_OFFICIAL_ADATAD_END_EPOCH", 132),
    ("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH", 100),
    ("DUCA_LOSS_SCHEDULE_TOTAL_STEPS", 13200),
):
    _require_fixed_env(_name, _expected)


dense_window_size = 768
window_size = 384
scale_factor = 1
chunk_num = window_size // 16
duca_end_epoch = 132
duca_schedule_steps_per_epoch = 100
duca_loss_schedule_total_steps = duca_end_epoch * duca_schedule_steps_per_epoch
duca_policy_warmup_steps = 660
duca_policy_transition_steps = 3960
duca_detector_bridge_warmup_steps = 4620
duca_detector_bridge_transition_steps = 3300
duca_max_unselected_hole = 15
duca_coarse_hidden_dim = 96


duca_transition_only_contract = dict(
    _delete_=True,
    route="DUCA_SHARED_ASFORMER_TRANSITION_ONLY_FIXED384",
    stage="counterfactual_teacher_integrated_pending_formal_cuda_gate",
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    full_window_selector=True,
    pre_backbone_plugin=True,
    main_method_candidate=True,
    empirically_supported=False,
    paper_claim_allowed=False,
    metric_claim_allowed=False,
    official_adatad_backend=True,
    official_backend_semantics="official_components_with_duca_wrapper_not_source_identical",
    official_upstream_repository="https://github.com/sming256/OpenTAD",
    official_upstream_commit="1aa8ca4ac5e846b1e8ff69298dd6607121a01589",
    official_base_config="./e2e_thumos_videomae_s_768x1_160_adapter.py",
    official_base_config_blob_sha1="e0dd2a0d4ed9c1ecbe2a2c042c0f748cda016266",
    official_base_config_byte_identical=True,
    official_detector_source_identical=False,
    detector_head_config_matches_official=True,
    detector_head_source_extended=True,
    detector_head_type="ActionFormerHead",
    detector_head_changed=False,
    detector_loss_changed=False,
    detector_nms_changed=False,
    detector_coordinate_adapter="selected_axis_with_posthoc_true_time_remap",
    selector_variant="transition_only",
    coarse_probe="official_asformer_binary_actionness",
    coarse_hidden_kind="official_asformer_encoder_hidden",
    official_asformer_source_normalized_lf_sha256="e075ee4825a201cfe324d5fbfb1332c0800f532e85b9d3809f6ca5180381c600",
    coarse_probe_joint_trainable=True,
    ranking_inputs=[
        "delta_actionness_logit",
        "absolute_delta_actionness_logit",
        "delta_binary_entropy",
        "absolute_delta_binary_entropy",
        "delta_asformer_encoder_hidden",
        "absolute_delta_asformer_encoder_hidden",
        "consecutive_hidden_cosine_change",
    ],
    ranking_uses_absolute_hidden=False,
    ranking_uses_raw_rgb_mean=False,
    legacy_direct_heads_enabled=False,
    actionness_role="binary_coarse_classification_only",
    transition_role="indirect_boundary_localization_and_frame_utility",
    acquisition_policy="global_structured_topk",
    hard_soft_feasible_family="shared_exact_k_max_gap_dynamic_program",
    exact_budget=384,
    dense_window_size=768,
    max_unselected_hole=duca_max_unselected_hole,
    posthoc_hard_repair=False,
    soft_max_gap_loss_enabled=False,
    policy_homotopy="continuous_uniform_to_learned",
    policy_homotopy_modulo_switching=False,
    policy_uniform_steps=duca_policy_warmup_steps,
    policy_ramp_steps=duca_policy_transition_steps,
    detector_bridge_delay_steps=duca_detector_bridge_warmup_steps,
    detector_bridge_ramp_steps=duca_detector_bridge_transition_steps,
    protected_gradient_routing=True,
    action_loss_updates="spatial_stem_asformer_encoder_decoder_action_head",
    transition_loss_updates="spatial_stem_asformer_encoder_shared_transition_scorer",
    coverage_and_detector_updates="shared_transition_scorer_only_through_selector_branch",
    detector_gradient_bridge="removed_failed_hard_swap_alignment_gate",
    detector_gradient_final_weight=0.0,
    detector_utility_learning="train_only_detached_hard_counterfactual_utility_distillation",
    detector_utility_is_direct_gradient=False,
    counterfactual_teacher_producer_integrated=True,
    counterfactual_teacher_detector_objective="official_actionformer_cls_plus_reg",
    counterfactual_teacher_max_candidates=4,
    counterfactual_teacher_inference_passes=0,
    deployment_allowed=False,
    loss_schedule_step_update="successful_optimizer_step",
    schedule_steps_per_epoch=duca_schedule_steps_per_epoch,
    schedule_expected_total_steps=duca_loss_schedule_total_steps,
    actual_successful_steps_must_be_logged=True,
    no_ledger_decision=True,
    forbid_external_actionness=True,
    forbid_raw_prediction_cache=True,
    teacher_free_eval=True,
)


model = dict(
    frame_selector=dict(
        _delete_=True,
        type="DucaOnlineFrameSelector",
        in_channels=3,
        dense_window_size=dense_window_size,
        budget=window_size,
        budget_mode="fixed",
        max_radius=0,
        acquisition_policy="global_structured_topk",
        structured_temperature=0.7,
        inference_policy_alpha=1.0,
        selector_variant="transition_only",
        selector_hidden_channels=64,
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        transition_scorer_lr=1.0e-4,
        coarse_hidden_dim=duca_coarse_hidden_dim,
        use_coarse_hidden_features=True,
        require_coarse_hidden_features=True,
        max_unselected_hole=duca_max_unselected_hole,
        max_gap_loss_max_unselected_hole=duca_max_unselected_hole,
        max_gap_loss_min_window_mass=1.0,
        soft_max_gap_loss_enabled=False,
        transition_target_sigma=2.0,
        transition_target_radius=4,
        transition_boundary_radius=4,
        transition_distribution_temperature=0.7,
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=True,
        actionness_weight=0.0,
        transition_weight=0.0,
        uncertainty_weight=0.0,
        utility_weight=0.0,
        boundary_weight=0.0,
        detector_gradient_mode="st_sparse_gather",
        counterfactual_utility_distillation_weight=0.25,
        counterfactual_utility_temperature=0.7,
        counterfactual_max_candidates=4,
        require_counterfactual_utility_teacher=True,
        coordinate_space="original_time",
        detector_output_coordinate_space="selected_axis_index",
        selected_positions_unit="original_time_index",
        loss_weights=dict(
            actionness=1.0,
            detector=1.0,
            transition=0.5,
            transition_boundary=0.25,
            max_gap_hole=0.0,
            teacher=0.0,
            detector_utility=0.0,
            start=0.0,
            end=0.0,
            context=0.0,
            boundary=0.0,
            hole=0.0,
            budget=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
        ),
        loss_weight_schedule=dict(
            type="progressive_joint",
            shape="cosine",
            warmup_steps=0,
            transition_steps=1,
            actionness=dict(start=1.0, end=1.0),
            transition=dict(start=0.5, end=0.5),
            transition_boundary=dict(start=0.0, end=0.25),
            detector_gradient=dict(
                start=0.0,
                end=0.0,
                warmup_steps=duca_detector_bridge_warmup_steps,
                transition_steps=duca_detector_bridge_transition_steps,
            ),
            policy_alpha=dict(
                start=0.0,
                end=1.0,
                warmup_steps=duca_policy_warmup_steps,
                transition_steps=duca_policy_transition_steps,
            ),
        ),
        no_ledger_decision=True,
        remap_gt_to_selected_axis=True,
        selected_axis_remap_required=True,
        forbid_ledger=True,
        forbid_raw_prediction_cache=True,
        forbid_external_actionness=True,
        require_external_actionness=False,
        profile_runtime=os.environ.get("DUCA_PROFILE_RUNTIME", "0") == "1",
        profile_sync_cuda=os.environ.get("DUCA_PROFILE_SYNC_CUDA", "1") != "0",
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name="shared_official_asformer_binary_actionness",
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            spatial_size=64,
            tcn_hidden_dim=duca_coarse_hidden_dim,
            official_num_layers=2,
            return_hidden_features=True,
            require_hidden_features=True,
            hidden_output_kind="official_asformer_encoder_hidden",
            checkpoint_path="",
            require_checkpoint=False,
            frozen=False,
            trainable=True,
            thumos_trained=True,
            uses_labels=True,
            uses_teacher=False,
            uses_gt=True,
            uses_prediction_cache=False,
            trained_with_thumos_labels=True,
            trained_with_gt_segments=True,
            training_dataset="THUMOS14",
            training_supervision_scope="train_only",
            uses_labels_at_inference=False,
            uses_gt_at_inference=False,
            uses_teacher_at_inference=False,
            uses_prediction_cache_at_inference=False,
            # No fitted train-only calibration artifact is shipped with this config.
            calibration_split="none",
            calibration_temperature=1.0,
            calibration_bias=0.0,
        ),
    ),
    backbone=dict(
        backbone=dict(total_frames=window_size * scale_factor),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=window_size),
            ],
        ),
    ),
    projection=dict(max_seq_len=window_size),
)


scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=duca_end_epoch)

# Short windows can have no feasible counterfactual swap while longer windows do.
# Keep the four matched arms on the same dynamic-graph DDP protocol.
solver = dict(static_graph=False, find_unused_parameters=False)

workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=5,
    val_eval_interval_anchor_epoch=47,
    val_start_epoch=47,
    end_epoch=duca_end_epoch,
)

work_dir = "exps/thumos/adatad/duca_transition_only_fixed384_official_adatad_backend_full_train"
