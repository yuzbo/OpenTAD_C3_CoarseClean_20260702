"""Development-only GeoRoute-AdaTAD source configuration.

This file intentionally contains no official-test route.  A Slurm stage runner
binds a frozen development annotation, fit/gate split and immutable work dir
before training or development evaluation.
"""

_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

window_size = 768

georoute_train_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(
        type="LoadFrames",
        num_clips=1,
        method="random_trunc",
        trunc_len=window_size,
        trunc_thresh=0.75,
        crop_ratio=[0.9, 1.0],
        scale_factor=1,
    ),
    dict(type="mmaction.DecordDecode"),
    dict(type="GeoRouteSourceViews", scout_size=96, output_key="georoute_inputs"),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="georoute_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

georoute_eval_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="GeoRouteSourceViews", scout_size=96, output_key="georoute_inputs"),
    dict(type="ConvertToTensor", keys=["gt_segments", "gt_labels"]),
    dict(
        type="Collect",
        inputs="georoute_inputs",
        keys=["masks", "gt_segments", "gt_labels"],
    ),
]

georoute_test_pipeline = [
    dict(type="PrepareVideoInfo", format="mp4"),
    dict(type="mmaction.DecordInit", num_threads=4),
    dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
    dict(type="mmaction.DecordDecode"),
    dict(type="GeoRouteSourceViews", scout_size=96, output_key="georoute_inputs"),
    dict(type="Collect", inputs="georoute_inputs", keys=["masks"]),
]

dataset = dict(
    train=dict(subset_name="training", pipeline=georoute_train_pipeline),
    val=dict(
        subset_name="training",
        window_size=window_size,
        window_overlap_ratio=0.5,
        pipeline=georoute_eval_pipeline,
    ),
    # This is rebound to the development gate split by the stage runner.  It
    # never names the THUMOS validation/test subset in this source config.
    test=dict(
        subset_name="training",
        window_size=window_size,
        window_overlap_ratio=0.5,
        test_mode=True,
        pipeline=georoute_test_pipeline,
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(with_cp=False, total_frames=window_size, adapter_index=list(range(12))),
        custom=dict(
            _delete_=True,
            wrapper_type="georoute_native_packed_v1",
            pretrain="pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth",
            norm_eval=False,
            freeze_backbone=False,
            georoute_source_key="source",
            georoute_scout_key="scout",
            georoute_window_size=window_size,
            georoute_output_length=window_size,
            georoute_scout_size=96,
            georoute_patch_size=16,
            georoute_tubelet_size=2,
            georoute_tokens_per_tubelet=64,
            georoute_context_tokens=8,
            georoute_roi_fraction=0.50,
            georoute_route_mode="hybrid",
            georoute_policy_estimator="straight_through",
            georoute_policy_temperature=0.5,
            # Per-tubelet normalization preserves the exact-K Plackett-Luce
            # policy-gradient direction without scaling it by T=384.
            georoute_score_function_temporal_reduction="mean",
            georoute_roi_temperature=0.25,
            georoute_geometry_stride_tubelets=1,
            georoute_absolute_position_enabled=True,
            georoute_absolute_coordinates_enabled=True,
            georoute_roi_relative_coordinates_enabled=True,
            georoute_geometry_projection_enabled=True,
            georoute_diagnostic_telemetry_enabled=False,
            georoute_pooling_mode="uniform_selected",
            georoute_adapter_mode="coordinate_lineage_packed",
            georoute_geometry_side_channel=False,
            georoute_min_roi_extent=0.20,
            georoute_max_roi_extent=1.00,
            georoute_geometry_smoothness_weight=0.002,
            georoute_area_prior_weight=0.001,
            georoute_area_prior=0.30,
            georoute_random_seed=3407,
            georoute_max_batch_size=1,
        ),
    ),
    projection=dict(in_channels=384, max_seq_len=window_size, attn_cfg=dict(n_mha_win_size=-1)),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    clip_grad_norm=1,
    amp=True,
    fp16_compress=True,
    static_graph=True,
    ema=True,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(
        lr=0,
        weight_decay=0,
        # Substring matching is order-sensitive: sparse_adapter must precede
        # adapter so its parameters cannot be swallowed by the adapter group.
        custom=[
            dict(name="sparse_adapter", lr=1e-4, weight_decay=0.05),
            dict(name="scout", lr=2e-4, weight_decay=0.05),
            dict(name="adapter", lr=2e-4, weight_decay=0.05),
        ],
        exclude=["backbone"],
    ),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=2, max_epoch=60)

workflow = dict(
    logging_interval=25,
    checkpoint_interval=1,
    checkpoint_policy="final_only",
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=60,
    end_epoch=60,
    require_successful_update_hook=True,
    schedule_and_ema_on_success_only=True,
    max_amp_retries_per_batch=8,
    fail_on_skipped_update=True,
)

inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
post_processing = dict(save_dict=False)

evaluation = dict(
    type="mAP",
    subset="training",
    tiou_thresholds=[0.3, 0.4, 0.5, 0.6, 0.7],
)

georoute_protocol = dict(
    schema_version="georoute_adatad_development_v3",
    route="native-token-select-first-conditional-geometry-routing",
    status="development_only",
    official_test_open_allowed=False,
    manual_roi_allowed=False,
    gt_for_route_allowed=False,
    teacher_for_route_allowed=False,
    raw_prediction_cache_allowed=False,
    one_heavy_videomae_forward_required=True,
    source_resizing_before_native_gather_allowed=False,
    local_crop_resize_allowed=False,
    detector_contract="[B,384,768]",
    detector_losses=["FocalLoss", "DIOULoss"],
    score_function_temporal_reduction="mean_per_tubelet",
    valid_native_support="floor_complete_patches_with_explicit_mask",
    pooling_mode="uniform_selected",
    adapter_mode="coordinate_lineage_packed",
    checkpoint_policy="final_only_atomic",
)

work_dir = "exps/thumos/adatad/georoute_development_unbound"
