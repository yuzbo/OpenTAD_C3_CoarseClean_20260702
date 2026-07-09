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


dense_window_size = _env_int("DUCA_ONLINE_DENSE_WINDOW_SIZE", 768)
window_size = _env_int("DUCA_ONLINE_BUDGET", _env_int("DUCA_OFFICIAL_ADATAD_BUDGET", 384))
strict_budget_claim_max = _env_int("DUCA_STRICT_CLAIM_MAX_BUDGET", 384)
duca_end_epoch = _env_int("DUCA_OFFICIAL_ADATAD_END_EPOCH", 60)
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
if duca_coarse_tcn_variant == "asformer_lite":
    raise ValueError("DUCA main method forbids asformer_lite; use official-action-seg with official_asformer")
if dense_window_size <= 0:
    raise ValueError("DUCA_ONLINE_DENSE_WINDOW_SIZE must be positive")
if window_size <= 0:
    raise ValueError("DUCA_ONLINE_BUDGET must be positive")
if window_size > dense_window_size:
    raise ValueError("DUCA_ONLINE_BUDGET must be <= DUCA_ONLINE_DENSE_WINDOW_SIZE")
if window_size % 16 != 0:
    raise ValueError("DUCA_ONLINE_BUDGET must be divisible by 16 for the VideoMAE tubelet rearrange")
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

duca_online_metadata_keys = dict(
    selected_positions="duca_online_selected_positions",
    selected_positions_unit="duca_online_selected_positions_unit",
    selected_mask="duca_online_selected_mask",
    selected_count="duca_online_selected_count",
    remap="duca_online_selected_axis_remap",
    source="duca_online_actionness_source",
)

duca_online_main_contract = dict(
    route="DUCA_ONLINE_OFFICIAL_ADATAD_BACKEND",
    stage="duca_online_official_adatad_backend_full_train",
    official_adatad_backend=True,
    official_base_config="./e2e_thumos_videomae_s_768x1_160_adapter.py",
    detector_stack="official_OpenTAD_AdaTAD_VideoMAE-S_ActionFormerHead_plus_DUCA_prebackbone_plugin_plus_online_C3_coarse_probe",
    detector_head_type="ActionFormerHead",
    main_method_candidate=True,
    diagnostic_only=False,
    no_ledger_decision=True,
    online_acquisition=True,
    pre_backbone_plugin=True,
    acquisition_policy="duca_center_radius_st_acquisition",
    budget_policy="fixed_budget",
    loss_schedule_policy="progressive_joint",
    loss_schedule_step_update="optimizer_step",
    loss_schedule_shape=duca_loss_schedule_shape,
    loss_schedule_total_steps=duca_loss_schedule_total_steps,
    loss_schedule_steps_per_epoch=duca_schedule_steps_per_epoch,
    loss_schedule_warmup_fraction=duca_loss_schedule_warmup_fraction,
    loss_schedule_transition_fraction=duca_loss_schedule_transition_fraction,
    loss_schedule_warmup_steps=duca_loss_schedule_warmup_steps,
    loss_schedule_transition_steps=duca_loss_schedule_transition_steps,
    coarse_actionness_dominates_initial_training=True,
    detector_loss_always_trains_backend=True,
    detector_gradient_bridge_enabled_after_schedule_transition=True,
    actionness_source="online_trainable_c3_coarse_probe",
    coarse_probe_model=duca_coarse_probe_model,
    coarse_probe_tcn_variant=duca_coarse_tcn_variant,
    coarse_probe_official_backend=duca_coarse_official_backend,
    coarse_probe_joint_trainable=not duca_coarse_frozen,
    coarse_probe_checkpoint_is_initialization=bool(duca_coarse_checkpoint),
    budget_max=window_size,
    strict_budget_claim_max=strict_budget_claim_max,
    strict_budget_lte_384=window_size <= 384,
    budget_curve_mode=os.environ.get("DUCA_BUDGET_CURVE_MODE", "0") == "1",
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
    teacher_free_eval=True,
    teacher_train_loss_only=False,
    gt_required_in_train=True,
    forbid_external_actionness=True,
    raw_prediction_cache_forbidden=True,
    uses_offline_deploy_selection_ledger=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    runtime_profile_available=True,
    runtime_profile_default_enabled=duca_profile_runtime,
    metadata_keys=duca_online_metadata_keys,
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
        budget=window_size,
        budget_mode="fixed",
        max_radius=16,
        selector_hidden_channels=64,
        detector_gradient_mode="st_sparse_gather_soft_context",
        coordinate_space="original_time",
        detector_output_coordinate_space="selected_axis_index",
        selected_positions_unit="original_time_index",
        loss_weights=dict(
            actionness=0.5,
            detector=1.0,
            detector_utility=0.05,
            teacher=0.0,
            budget=0.0,
            boundary=0.0,
            hole=0.05,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
        ),
        loss_weight_schedule=dict(
            type="progressive_joint",
            shape=duca_loss_schedule_shape,
            warmup_steps=duca_loss_schedule_warmup_steps,
            transition_steps=duca_loss_schedule_transition_steps,
            actionness=dict(
                start=_env_float("DUCA_LOSS_ACTIONNESS_START", 1.0),
                end=_env_float("DUCA_LOSS_ACTIONNESS_END", 0.25),
            ),
            detector_gradient=dict(start=_env_float("DUCA_LOSS_DETECTOR_GRADIENT_START", 0.0), end=1.0),
            detector_utility=dict(start=0.0, end=_env_float("DUCA_LOSS_DETECTOR_UTILITY_END", 0.05)),
            hole=dict(start=0.0, end=_env_float("DUCA_LOSS_HOLE_END", 0.05)),
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
        metadata_keys=duca_online_metadata_keys,
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name=duca_coarse_source_name,
            probe_model=duca_coarse_probe_model,
            tcn_variant=duca_coarse_tcn_variant,
            spatial_size=duca_coarse_spatial_size,
            tcn_hidden_dim=duca_coarse_hidden_dim,
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
    checkpoint_interval=int(os.environ.get("DUCA_OFFICIAL_ADATAD_CHECKPOINT_INTERVAL", "5")),
    val_loss_interval=-1,
    val_eval_interval=int(os.environ.get("DUCA_OFFICIAL_ADATAD_VAL_INTERVAL", "5")),
    val_eval_interval_anchor_epoch=5,
    val_start_epoch=int(os.environ.get("DUCA_OFFICIAL_ADATAD_VAL_START_EPOCH", "4")),
    end_epoch=duca_end_epoch,
)

work_dir = "exps/thumos/adatad/duca_online_official_adatad_backend_full_train"
