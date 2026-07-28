_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]


dense_window_size = 768
selected_budget = 384
scale_factor = 1
chunk_num = selected_budget * scale_factor // 16
protected_contract = "duca_protected_e2e_physical_v1"

duca_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "avg_fps",
    "duration",
    "total_frames",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "frame_inds",
]

duca_protected_physical_contract = dict(
    route="DUCA_PROTECTED_PHYSICAL_FIXED384_OFFICIAL60",
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    pre_backbone_plugin=True,
    dense_window_size=dense_window_size,
    exact_budget=selected_budget,
    dynamic_budget=False,
    selector_input="low_resolution_rgb_full_offline_window",
    coarse_probe="official_asformer_binary_actionness",
    coarse_hidden_kind="official_asformer_encoder_hidden",
    transition_descriptor_dim=197,
    selector_adapter_dim=64,
    actionness_role="binary_coarse_classification_only",
    selection_role="indirect_state_transition_and_boundary_coverage",
    hard_policy="physical_exact_k_viterbi",
    soft_policy="same_graph_gibbs_slot_marginals",
    max_gap_cap="exact_uniform_reference_in_seconds",
    posthoc_repair=False,
    selected_axis_gt_remap=False,
    detector_axis="native_dense_physical_candidate_axis",
    detector_head="official_actionformer_head",
    detector_gradient="protected_hard_forward_soft_backward",
    backbone_tail_padding="none_exact_k_bucket",
    execution_quantum=16,
    main_detector_gradient_updates="selector_adapter_and_score_head_only",
    rho_detector_gradient_updates="selector_plus_last_asformer_encoder_block_scaled_0.01",
    test_selection="terminal_epoch_59_state_dict_ema_only",
    seed=3407,
    target_avg_map="strictly_greater_than_matched_uniform_approximately_65",
    empirically_supported=False,
    paper_ready=False,
)

dataset = dict(
    train=dict(
        type="DucaStatelessThumosPaddingDataset",
        stateless_seed=3407,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="random_trunc",
                trunc_len=dense_window_size,
                trunc_thresh=0.75,
                crop_ratio=[0.9, 1.0],
                scale_factor=scale_factor,
                emit_boundary_validity=True,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(
                type="ConvertToTensor",
                keys=[
                    "imgs",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                ],
            ),
            dict(
                type="Collect",
                inputs="imgs",
                keys=[
                    "masks",
                    "gt_segments",
                    "gt_labels",
                    "gt_boundary_validity",
                ],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
    val=None,
    test=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(
                type="LoadFrames",
                num_clips=1,
                method="sliding_window",
                scale_factor=scale_factor,
            ),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(
                type="Collect",
                inputs="imgs",
                keys=["masks"],
                meta_keys=duca_meta_keys,
            ),
        ],
    ),
)

model = dict(
    frame_selector=dict(
        type="DucaProtectedE2EFrameSelector",
        in_channels=3,
        dense_window_size=dense_window_size,
        budget=selected_budget,
        execution_quantum=16,
        coarse_hidden_dim=96,
        selector_hidden_dim=64,
        coverage_floor_weight=0.10,
        score_temperature=0.70,
        path_temperature=1.0,
        transition_target_sigma=2.0,
        transition_target_radius=4,
        transition_boundary_radius=4,
        transition_distribution_temperature=0.70,
        action_loss_weight=1.0,
        transition_loss_weight=0.50,
        transition_boundary_loss_weight=0.25,
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        selector_lr=1.0e-4,
        strict_physical_metadata=True,
        forbid_raw_prediction_cache=True,
        actionness_source_cfg=dict(
            type="C3CoarseProbeActionnessSource",
            source_name="protected_official_asformer_binary_actionness",
            probe_model="official-action-seg",
            official_action_seg_backend="official_asformer",
            spatial_size=64,
            tcn_hidden_dim=96,
            official_num_layers=2,
            dropout=0.0,
            return_hidden_features=True,
            require_hidden_features=True,
            hidden_output_kind="official_asformer_encoder_hidden",
            policy_hidden_gradient_scope="none",
            checkpoint_path="",
            require_checkpoint=False,
            frozen=False,
            trainable=True,
            train_split_supervised=True,
            calibration_split="none",
            calibration_temperature=1.0,
            calibration_bias=0.0,
            thumos_trained=True,
            uses_labels=True,
            uses_teacher=False,
            uses_gt=True,
            uses_prediction_cache=False,
            trained_with_thumos_labels=True,
            trained_with_gt_segments=True,
            training_dataset="THUMOS14",
            training_supervision_scope="train_only",
        ),
    ),
    backbone=dict(
        backbone=dict(
            total_frames=selected_budget * scale_factor,
            with_cp=False,
        ),
        custom=dict(
            dynamic_temporal_bucket=True,
            dynamic_temporal_clip_len=16,
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=selected_budget),
            ],
        ),
    ),
    projection=dict(max_seq_len=selected_budget),
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract=protected_contract,
        ),
    ),
)

optimizer = dict(
    lr=1.0e-4,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=60,
)

solver = dict(
    static_graph=False,
    find_unused_parameters=True,
)

workflow = dict(
    formal_protocol="duca_protected_physical_v1",
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    seal_eval_dataloaders_during_training=True,
    derive_train_loader_contract=True,
)

work_dir = "exps/thumos/adatad/duca_protected_physical_base_do_not_run"
