_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os

value_mode = os.environ.get("DUCA_UVT_MODE", "geo_ema")
allowed_value_modes = ("off", "geo", "ema", "geo_ema", "geo_ema_portal")
if value_mode not in allowed_value_modes:
    raise ValueError(f"DUCA_UVT_MODE must be one of {allowed_value_modes}")
if value_mode == "geo_ema_portal" and os.environ.get("DUCA_UVT_PORTAL_GATE_PASSED", "0") != "1":
    raise ValueError("DUCA_UVT_MODE=geo_ema_portal requires a passed DUCA_UVT_PORTAL_GATE_PASSED=1 artifact")

dense_window_size = 768
target_len = 512
scale_factor = 1

yuzibo_root = os.environ.get("YUZIBO_ROOT", "/data/run01/sczc063/yuzibo")
thumos14_root = os.path.join(yuzibo_root, "thumos14")
annotation_path = os.environ.get("THUMOS14_ANNOTATION_PATH", os.path.join(thumos14_root, "annotations", "thumos_14_anno.json"))
class_map = os.environ.get("THUMOS14_CLASS_MAP", os.path.join(thumos14_root, "annotations", "category_idx.txt"))
train_data_path = os.environ.get("THUMOS14_TRAIN_DATA_PATH", os.path.join(thumos14_root, "train"))
test_data_path = os.environ.get("THUMOS14_TEST_DATA_PATH", os.path.join(thumos14_root, "test"))

value_alpha = float(os.environ.get("DUCA_UVT_ALPHA", "0.10"))
boundary_quota = int(os.environ.get("DUCA_UVT_BOUNDARY_QUOTA", "0"))
boundary_center_top_m = int(os.environ.get("DUCA_UVT_BOUNDARY_CENTER_TOP_M", "0"))
mmr_lambda = float(os.environ.get("DUCA_UVT_MMR_LAMBDA", "0.0"))

model = dict(
    frame_selector=dict(
        type="PCOTMRASPreBackboneFrameSelector",
        target_len=target_len,
        dense_window_size=dense_window_size,
        descriptor_dim=3 * 32 * 32,
        selection_unit=1,
        remap_gt_to_selected_axis=False,
        selection_strategy="dynamic_B",
        scout_feature_source="compressed_pixels",
        scout_spatial_size=32,
        straight_through_detector_loss=True,
        physical_dense_reconstruction=True,
        variable_length_output=True,
        variable_compute_multiple=16,
        frame_score_st_surrogate="local_softmax",
        frame_score_st_gradient_scale=1.0,
        dynamic_budget=dict(
            enabled=True,
            protocol="marginal_utility_v0",
            min_budget=256,
            target_budget=384,
            max_budget=512,
            average_budget=384,
            budget_step=16,
            score_midpoint=0.5,
            actionness_weight=1.0,
            boundary_weight=0.0,
            uncertainty_weight=0.0,
            redundancy_weight=0.0,
        ),
        aux_gt_acquisition_loss_weight=0.05,
        aux_frame_score_boundary_loss_weight=0.05,
        aux_risk_loss_weight=0.05,
        aux_uncertainty_loss_weight=0.025,
        aux_redundancy_loss_weight=0.01,
        reader_regularizer_loss_weight=0.01,
        reader=dict(
            type="PCOTMRASBoundaryDifficultyTemporalFrameScout",
            in_dim=3 * 32 * 32,
            hidden_dim=96,
            num_slots=target_len,
            temporal_layers=4,
            temporal_kernel_size=5,
            dilations=(1, 2, 4, 8),
            dropout=0.10,
            action_bias_weight=0.20,
            boundary_bias_weight=0.65,
            uncertainty_bias_weight=0.15,
            redundancy_bias_weight=0.10,
        ),
        value_mode=value_mode,
        value_hidden_dim=96,
        value_alpha=value_alpha,
        value_alpha_max=0.25,
        value_ema_enabled=value_mode in ("ema", "geo_ema", "geo_ema_portal"),
        value_ema_decay=0.999,
        value_geometry_weight=0.0 if value_mode in ("off", "ema") else 1.0,
        value_ema_loss_weight=0.0 if value_mode not in ("ema", "geo_ema", "geo_ema_portal") else 0.25,
        value_portal_enabled=False,
        value_boundary_radius=2,
        value_short_action_duration_sec=2.0,
        value_short_action_weight=0.5,
        boundary_quota=boundary_quota,
        boundary_center_top_m=boundary_center_top_m,
        boundary_radius_decode=2,
        boundary_pair_max_gap=8,
        mmr_lambda=mmr_lambda,
    ),
    backbone=dict(
        backbone=dict(allow_variable_total_frames=True),
        custom=dict(
            dynamic_sparse_temporal=dict(
                enabled=True,
                clip_len=16,
                tubelet_size=2,
                output_len=dense_window_size,
            ),
        ),
    ),
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

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
)

workflow = dict(
    disable_checkpoint=False,
    logging_interval=50,
    checkpoint_interval=2,
    val_loss_interval=-1,
    val_eval_interval=2,
    val_start_epoch=40,
    end_epoch=60,
)

work_dir = os.environ.get("DUCA_UVT_WORK_DIR", "exps/thumos/adatad/duca_uvt_value_portal")
