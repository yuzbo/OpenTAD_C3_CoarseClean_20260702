_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


def _env_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _env_bool(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name, default):
    value = os.environ.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


dense_window_size = _env_int("DUCA_MUST_DENSE_WINDOW_SIZE", 768)
budget_max = _env_int("DUCA_MUST_BUDGET_MAX", 384)
budget_min = _env_int("DUCA_MUST_BUDGET_MIN", 64)
budget_target = _env_int("DUCA_MUST_BUDGET_TARGET", 256)
budget_multiple = _env_int("DUCA_MUST_BUDGET_MULTIPLE", 16)
strict_budget_claim_max = _env_int("DUCA_STRICT_CLAIM_MAX_BUDGET", 384)
duca_end_epoch = _env_int("DUCA_MUST_END_EPOCH", 60)
duca_schedule_steps_per_epoch = _env_int("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH", 99)
duca_loss_schedule_total_steps = _env_int(
    "DUCA_LOSS_SCHEDULE_TOTAL_STEPS",
    duca_end_epoch * duca_schedule_steps_per_epoch,
)
duca_loss_schedule_warmup_fraction = _env_float("DUCA_LOSS_SCHEDULE_WARMUP_FRACTION", 0.08)
duca_loss_schedule_transition_fraction = _env_float("DUCA_LOSS_SCHEDULE_TRANSITION_FRACTION", 0.67)
duca_loss_schedule_warmup_steps = _env_int(
    "DUCA_LOSS_SCHEDULE_WARMUP_STEPS",
    int(round(duca_loss_schedule_total_steps * duca_loss_schedule_warmup_fraction)),
)
duca_loss_schedule_transition_steps = _env_int(
    "DUCA_LOSS_SCHEDULE_TRANSITION_STEPS",
    int(round(duca_loss_schedule_total_steps * duca_loss_schedule_transition_fraction)),
)
duca_loss_schedule_shape = os.environ.get("DUCA_LOSS_SCHEDULE_SHAPE", "cosine")
duca_profile_runtime = os.environ.get("DUCA_PROFILE_RUNTIME", "0") == "1"
duca_profile_sync_cuda = os.environ.get("DUCA_PROFILE_SYNC_CUDA", "1") != "0"
duca_coarse_probe_model = os.environ.get("DUCA_COARSE_PROBE_MODEL", "official-action-seg")
duca_coarse_tcn_variant = os.environ.get("DUCA_COARSE_TCN_VARIANT", "official_asformer")
duca_coarse_official_backend = os.environ.get("DUCA_COARSE_OFFICIAL_BACKEND", "official_asformer")
duca_coarse_spatial_size = _env_int("DUCA_COARSE_SPATIAL_SIZE", 64)
duca_coarse_hidden_dim = _env_int("DUCA_COARSE_HIDDEN_DIM", 96)
duca_coarse_checkpoint = os.environ.get("DUCA_COARSE_PROBE_CHECKPOINT", "")
duca_coarse_require_checkpoint = _env_bool("DUCA_COARSE_REQUIRE_CHECKPOINT", False)
duca_coarse_frozen = _env_bool("DUCA_COARSE_FROZEN", False)
duca_coarse_source_name = os.environ.get(
    "DUCA_COARSE_SOURCE_NAME",
    (
        f"online_c3_{duca_coarse_official_backend}_coarse_actionness"
        if duca_coarse_probe_model == "official-action-seg"
        else f"online_c3_{duca_coarse_probe_model}_{duca_coarse_tcn_variant}_coarse_actionness"
    ),
)
duca_selector_actionness_weight = _env_float("DUCA_SELECTOR_ACTIONNESS_WEIGHT", 0.05)
duca_selector_transition_weight = _env_float("DUCA_SELECTOR_TRANSITION_WEIGHT", 1.0)
duca_selector_uncertainty_weight = _env_float("DUCA_SELECTOR_UNCERTAINTY_WEIGHT", 0.25)
duca_selector_utility_weight = _env_float("DUCA_SELECTOR_UTILITY_WEIGHT", 0.50)
duca_selector_boundary_weight = _env_float("DUCA_SELECTOR_BOUNDARY_WEIGHT", 1.0)
duca_max_unselected_hole = _env_int("DUCA_MAX_UNSELECTED_HOLE", 15)
if duca_coarse_tcn_variant == "asformer_lite":
    raise ValueError("DUCA main method forbids asformer_lite; use official-action-seg with official_asformer")
if dense_window_size <= 0:
    raise ValueError("DUCA_MUST_DENSE_WINDOW_SIZE must be positive")
if budget_max <= 0:
    raise ValueError("DUCA_MUST_BUDGET_MAX must be positive")
if budget_max > dense_window_size:
    raise ValueError("DUCA_MUST_BUDGET_MAX must be <= DUCA_MUST_DENSE_WINDOW_SIZE")
if budget_min <= 0 or budget_min > budget_max:
    raise ValueError("DUCA_MUST_BUDGET_MIN must lie in (0, DUCA_MUST_BUDGET_MAX]")
if budget_multiple <= 0 or (budget_max - budget_min) % budget_multiple != 0:
    raise ValueError("DUCA_MUST_BUDGET_MULTIPLE must divide DUCA_MUST_BUDGET_MAX - DUCA_MUST_BUDGET_MIN")
if budget_target <= 0 or budget_target > budget_max:
    raise ValueError("DUCA_MUST_BUDGET_TARGET must lie in (0, DUCA_MUST_BUDGET_MAX]")
if budget_max % 16 != 0:
    raise ValueError("DUCA_MUST_BUDGET_MAX must be divisible by 16 for the VideoMAE tubelet rearrange")
if duca_loss_schedule_shape not in {"linear", "cosine"}:
    raise ValueError("DUCA_LOSS_SCHEDULE_SHAPE must be linear or cosine")
if duca_schedule_steps_per_epoch <= 0:
    raise ValueError("DUCA_LOSS_SCHEDULE_STEPS_PER_EPOCH must be positive")
if duca_loss_schedule_total_steps <= 0:
    raise ValueError("DUCA_LOSS_SCHEDULE_TOTAL_STEPS must be positive")
if not (0.0 <= duca_loss_schedule_warmup_fraction < 1.0):
    raise ValueError("DUCA_LOSS_SCHEDULE_WARMUP_FRACTION must be in [0, 1)")
if not (0.0 < duca_loss_schedule_transition_fraction <= 1.0):
    raise ValueError("DUCA_LOSS_SCHEDULE_TRANSITION_FRACTION must be in (0, 1]")
if duca_loss_schedule_warmup_steps < 0:
    raise ValueError("DUCA_LOSS_SCHEDULE_WARMUP_STEPS must be non-negative")
if duca_loss_schedule_transition_steps <= 0:
    raise ValueError("DUCA_LOSS_SCHEDULE_TRANSITION_STEPS must be positive")
if duca_max_unselected_hole <= 0:
    raise ValueError("DUCA_MAX_UNSELECTED_HOLE must be positive")
if not (
    0.0 <= duca_selector_actionness_weight < duca_selector_transition_weight
    and duca_selector_actionness_weight < duca_selector_boundary_weight
):
    raise ValueError("DUCA selector scoring must keep actionness as a small auxiliary term")

window_size = budget_max
scale_factor = 1
chunk_num = window_size * scale_factor // 16

yuzibo_root = os.environ.get("YUZIBO_ROOT", os.path.expanduser("~/run/yuzibo"))
thumos14_root = os.path.join(yuzibo_root, "thumos14")
annotation_path = os.environ.get(
    "THUMOS14_ANNOTATION_PATH",
    os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"),
)
class_map = os.environ.get(
    "THUMOS14_CLASS_MAP",
    os.path.join(thumos14_root, "annotations", "category_idx.txt"),
)
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))

duca_must_metadata_keys = dict(
    selected_positions="duca_must_selected_positions",
    selected_positions_unit="duca_must_selected_positions_unit",
    selected_mask="duca_must_selected_mask",
    selected_count="duca_must_selected_count",
    remap="duca_must_selected_axis_remap",
    source="duca_must_actionness_source",
)

duca_must_dynamic_contract = dict(
    route="DUCA_MUST_DYNAMIC_OFFICIAL_ADATAD_BACKEND",
    stage="duca_must_dynamic_official_adatad_backend_full_train",
    official_adatad_backend=True,
    official_base_config="./e2e_thumos_videomae_s_768x1_160_adapter.py",
    detector_stack="official_OpenTAD_AdaTAD_VideoMAE-S_ActionFormerHead_plus_DUCA_MUST_prebackbone_plugin_plus_runtime_C3_coarse_probe",
    detector_head_type="ActionFormerHead",
    main_method_candidate=False,
    diagnostic_only=True,
    diagnostic_reason="padded_cap_detector_does_not_realize_dynamic_compute_and_uses_legacy_center_radius_policy",
    dynamic_budget=True,
    actionness_source="runtime_trainable_c3_coarse_probe",
    coarse_probe_model=duca_coarse_probe_model,
    coarse_probe_tcn_variant=duca_coarse_tcn_variant,
    coarse_probe_official_backend=duca_coarse_official_backend,
    coarse_probe_joint_trainable=not duca_coarse_frozen,
    coarse_probe_checkpoint_is_initialization=bool(duca_coarse_checkpoint),
    budget_policy="prefix_marginal_utility_stop",
    dynamic_budget_dual_update_after_optimizer_step=True,
    dynamic_budget_dual_update_source="dynamic_must_expected_cost",
    dynamic_budget_dual_update_observation="expected_cost_mean",
    acquisition_policy="duca_center_radius_st_acquisition",
    loss_schedule_policy="progressive_joint",
    loss_schedule_step_update="optimizer_step",
    loss_schedule_shape=duca_loss_schedule_shape,
    loss_schedule_total_steps=duca_loss_schedule_total_steps,
    loss_schedule_steps_per_epoch=duca_schedule_steps_per_epoch,
    loss_schedule_warmup_fraction=duca_loss_schedule_warmup_fraction,
    loss_schedule_transition_fraction=duca_loss_schedule_transition_fraction,
    loss_schedule_warmup_steps=duca_loss_schedule_warmup_steps,
    loss_schedule_transition_steps=duca_loss_schedule_transition_steps,
    coarse_actionness_dominates_initial_training=False,
    state_transition_boundary_dominates_selection=True,
    actionness_role="auxiliary_calibration_not_coverage",
    actionness_score_role="small_auxiliary_score",
    selector_score_priority="transition_boundary_utility_first",
    selector_score_actionness_weight=duca_selector_actionness_weight,
    selector_score_transition_weight=duca_selector_transition_weight,
    selector_score_uncertainty_weight=duca_selector_uncertainty_weight,
    selector_score_utility_weight=duca_selector_utility_weight,
    selector_score_boundary_weight=duca_selector_boundary_weight,
    selection_supervision="state_transition_boundary_first",
    boundary_utility_proxy_target_kind="gt_boundary_utility_proxy",
    detector_utility_target_kind="deprecated_alias_to_gt_boundary_utility_proxy",
    detector_utility_target_is_true_detector_derived=False,
    max_unselected_hole=duca_max_unselected_hole,
    soft_max_gap_loss="temporal_max_gap_hole_loss",
    hard_max_gap_repair=True,
    detector_loss_always_trains_backend=True,
    detector_gradient_bridge_enabled_after_schedule_transition=True,
    budget_controller_enabled_after_schedule_transition=True,
    budget_max=budget_max,
    budget_min=budget_min,
    budget_target=budget_target,
    budget_multiple=budget_multiple,
    strict_budget_claim_max=strict_budget_claim_max,
    strict_budget_lte_384=budget_max <= 384,
    external_budget_override_allowed=False,
    forced_budget_curve=False,
    no_ledger_decision=True,
    online_acquisition=False,
    runtime_generated_selection=True,
    cache_free_selection=True,
    full_window_selector=True,
    streaming=False,
    pre_backbone_plugin=True,
    dense_window_size=dense_window_size,
    selected_positions_unit="original_time_index",
    coordinate_space="original_time",
    detector_output_coordinate_space="selected_axis_index",
    selected_axis_remap_required=True,
    physical_grid_actionformer_required=False,
    changes_detector_head=False,
    changes_loss_assignment=False,
    changes_detector_nms=False,
    changes_post_processing_only_for_coordinate_remap=True,
    actual_variable_length_detector=False,
    dynamic_compute_realized=False,
    runtime_flops_claim_allowed=False,
    runtime_profile_available=True,
    runtime_profile_default_enabled=duca_profile_runtime,
    teacher_free_eval=True,
    teacher_train_loss_only=False,
    gt_required_in_train=True,
    forbid_external_actionness=True,
    raw_prediction_cache_forbidden=True,
    uses_offline_deploy_selection_ledger=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    metadata_keys=duca_must_metadata_keys,
)

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=train_data_path,
    ),
    val=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=test_data_path,
    ),
)

evaluation = dict(
    ground_truth_filename=annotation_path,
)

model = dict(
    frame_selector=dict(
        type="DucaOnlineFrameSelector",
        in_channels=3,
        dense_window_size=dense_window_size,
        budget=None,
        budget_mode="dynamic_must",
        budget_min=budget_min,
        budget_max=budget_max,
        budget_multiple=budget_multiple,
        target_budget=budget_target,
        allow_external_budget_override=False,
        max_radius=16,
        selector_hidden_channels=64,
        coarse_hidden_dim=duca_coarse_hidden_dim,
        use_coarse_hidden_features=True,
        require_coarse_hidden_features=True,
        max_unselected_hole=duca_max_unselected_hole,
        max_gap_loss_max_unselected_hole=duca_max_unselected_hole,
        max_gap_loss_min_window_mass=1.0,
        hard_max_gap_repair=True,
        fail_on_infeasible_max_gap=True,
        actionness_weight=duca_selector_actionness_weight,
        transition_weight=duca_selector_transition_weight,
        uncertainty_weight=duca_selector_uncertainty_weight,
        utility_weight=duca_selector_utility_weight,
        boundary_weight=duca_selector_boundary_weight,
        detector_gradient_mode="soft_to_hard_resample",
        coordinate_space="original_time",
        detector_output_coordinate_space="selected_axis_index",
        selected_positions_unit="original_time_index",
        loss_weights=dict(
            actionness=0.10,
            detector=1.0,
            detector_utility=0.10,
            max_gap_hole=0.25,
            lagrangian_budget=1.0,
            marginal_monotonic=0.01,
            budget=0.0,
            boundary=1.0,
            hole=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
            teacher=0.0,
        ),
        loss_weight_schedule=dict(
            type="progressive_joint",
            shape=duca_loss_schedule_shape,
            warmup_steps=duca_loss_schedule_warmup_steps,
            transition_steps=duca_loss_schedule_transition_steps,
            actionness=dict(
                start=_env_float("DUCA_LOSS_ACTIONNESS_START", 0.25),
                end=_env_float("DUCA_LOSS_ACTIONNESS_END", 0.05),
            ),
            boundary=dict(start=_env_float("DUCA_LOSS_BOUNDARY_START", 0.50), end=_env_float("DUCA_LOSS_BOUNDARY_END", 1.0)),
            detector_gradient=dict(start=_env_float("DUCA_LOSS_DETECTOR_GRADIENT_START", 0.0), end=1.0),
            detector_utility=dict(start=0.0, end=_env_float("DUCA_LOSS_DETECTOR_UTILITY_END", 0.10)),
            max_gap_hole=dict(start=0.0, end=_env_float("DUCA_LOSS_MAX_GAP_HOLE_END", 0.25)),
            hole=dict(start=0.0, end=_env_float("DUCA_LOSS_HOLE_END", 0.0)),
            lagrangian_budget=dict(start=0.0, end=1.0),
            marginal_monotonic=dict(start=0.0, end=0.01),
            budget=dict(start=0.0, end=0.0),
            entropy=dict(start=0.0, end=0.0),
        ),
        no_ledger_decision=True,
        remap_gt_to_selected_axis=True,
        selected_axis_remap_required=True,
        forbid_ledger=True,
        forbid_raw_prediction_cache=True,
        forbid_external_actionness=True,
        profile_runtime=duca_profile_runtime,
        profile_sync_cuda=duca_profile_sync_cuda,
        metadata_keys=duca_must_metadata_keys,
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name=duca_coarse_source_name,
            probe_model=duca_coarse_probe_model,
            tcn_variant=duca_coarse_tcn_variant,
            spatial_size=duca_coarse_spatial_size,
            tcn_hidden_dim=duca_coarse_hidden_dim,
            return_hidden_features=True,
            require_hidden_features=True,
            checkpoint_path=duca_coarse_checkpoint,
            require_checkpoint=duca_coarse_require_checkpoint,
            frozen=duca_coarse_frozen,
            trainable=not duca_coarse_frozen,
            mobilenet_pretrained=True,
            mobilenet_freeze_backbone=False,
            official_action_seg_backend=duca_coarse_official_backend,
            thumos_trained=False,
            uses_labels=False,
            uses_teacher=False,
            uses_gt=False,
            uses_prediction_cache=False,
            calibration_split="none",
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

workflow = dict(
    logging_interval=50,
    checkpoint_interval=int(os.environ.get("DUCA_MUST_CHECKPOINT_INTERVAL", "5")),
    val_loss_interval=-1,
    val_eval_interval=int(os.environ.get("DUCA_MUST_VAL_INTERVAL", "5")),
    val_eval_interval_anchor_epoch=5,
    val_start_epoch=int(os.environ.get("DUCA_MUST_VAL_START_EPOCH", "4")),
    end_epoch=duca_end_epoch,
)

work_dir = "exps/thumos/adatad/duca_must_dynamic_official_adatad_backend_full_train"
