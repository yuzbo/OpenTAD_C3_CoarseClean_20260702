_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


def _env_int(name, default):
    value = os.environ.get(name, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


dense_window_size = _env_int("DUCA_ONLINE_DENSE_WINDOW_SIZE", 768)
window_size = _env_int("DUCA_ONLINE_BUDGET", _env_int("DUCA_OFFICIAL_ADATAD_BUDGET", 384))
strict_budget_claim_max = _env_int("DUCA_STRICT_CLAIM_MAX_BUDGET", 384)
if dense_window_size <= 0:
    raise ValueError("DUCA_ONLINE_DENSE_WINDOW_SIZE must be positive")
if window_size <= 0:
    raise ValueError("DUCA_ONLINE_BUDGET must be positive")
if window_size > dense_window_size:
    raise ValueError("DUCA_ONLINE_BUDGET must be <= DUCA_ONLINE_DENSE_WINDOW_SIZE")
if window_size % 16 != 0:
    raise ValueError("DUCA_ONLINE_BUDGET must be divisible by 16 for the VideoMAE tubelet rearrange")
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
    detector_stack="official_OpenTAD_AdaTAD_VideoMAE-S_ActionFormerHead_plus_DUCA_prebackbone_plugin",
    detector_head_type="ActionFormerHead",
    main_method_candidate=True,
    diagnostic_only=False,
    no_ledger_decision=True,
    online_acquisition=True,
    pre_backbone_plugin=True,
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
    raw_prediction_cache_forbidden=True,
    uses_offline_deploy_selection_ledger=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
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
        max_radius=16,
        selector_hidden_channels=64,
        detector_gradient_mode="st_sparse_gather_soft_context",
        coordinate_space="original_time",
        detector_output_coordinate_space="selected_axis_index",
        selected_positions_unit="original_time_index",
        loss_weights=dict(
            teacher=0.0,
            budget=0.0,
            boundary=0.0,
            hole=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
        ),
        no_ledger_decision=True,
        remap_gt_to_selected_axis=True,
        selected_axis_remap_required=True,
        forbid_ledger=True,
        forbid_raw_prediction_cache=True,
        metadata_keys=duca_online_metadata_keys,
        actionness_source_cfg=dict(
            type="ZeroShotMotionActionnessSource",
            source_name="zero_shot_motion_actionness",
            mode="motion",
            thumos_trained=False,
            uses_labels=False,
            uses_teacher=False,
            uses_gt=False,
            uses_prediction_cache=False,
            no_train_gt=True,
            no_teacher=True,
            no_oracle=True,
            no_raw_prediction_cache=True,
            no_gt_generation=True,
            calibration_split="none",
            checkpoint_hash="no_checkpoint_motion_energy",
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
    end_epoch=int(os.environ.get("DUCA_OFFICIAL_ADATAD_END_EPOCH", "60")),
)

work_dir = "exps/thumos/adatad/duca_online_official_adatad_backend_full_train"
