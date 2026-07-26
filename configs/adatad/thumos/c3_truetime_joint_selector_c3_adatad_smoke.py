_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


route_label = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
selected_window_size = 384
dense_window_size = 768
scale_factor = 1
chunk_num = selected_window_size * scale_factor // 16

stage2_offline_detector_utility_selector_dependency = False
truetime_selector_grad_proof_path = os.environ.get(
    "TRUETIME_SELECTOR_GRAD_PROOF_JSON",
    "REPLACE_WITH_TRUETIME_SELECTOR_GRAD_PROOF_JSON",
)

experiment_scope = dict(
    route="STAGE3_4_EXPERIMENTAL_TRUETIME_SPARSE_TAD",
    route_variant=route_label,
    stage="stage3_4_experimental_smoke",
    detector_stack="truetime_physical_grid_actionformer_frame_selector_slot",
    separate_from_stage2_detector_utility_offline_selector=True,
    minimum_useful_deliverable="true_time_roundtrip_and_selector_detector_loss_gradient_smoke",
    full_map_claim_required=False,
    changes_input_sampling=True,
    changes_detector_head=False,
    changes_neck=False,
    changes_loss_assignment=True,
    changes_post_processing=False,
    modifies_evaluator_map_semantics=False,
    uses_physical_grid_actionformer=True,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
)

truetime_joint_selector_gate = dict(
    route="STAGE3_4_EXPERIMENTAL_TRUETIME_SPARSE_TAD",
    route_variant=route_label,
    stage="stage3_4_experimental_smoke",
    smoke_only=True,
    max_epochs=1,
    max_train_iters=2,
    default_off=True,
    explicit_config_opt_in=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    requires_selector_grad_nonzero=True,
    requires_actionformer_detector_grad_nonzero=True,
    requires_geometry_roundtrip=True,
    requires_physical_grid_actionformer=True,
    selector_grad_proof_path=truetime_selector_grad_proof_path,
    end_to_end_claim_allowed=False,
    paper_claim_allowed=False,
    runtime_flops_claim_allowed=False,
    deploy_claim_allowed=False,
    metric_claim_allowed=False,
    allow_tools_train=True,
    allow_tools_test=False,
    allow_detector_map=False,
    allow_long_training=False,
    allow_slurm=True,
    allow_precheck_only=True,
    required_gpu="GPU1",
    allowed_entrypoints=("tools/train.py",),
    forbidden_entrypoints=("tools/test.py",),
    required_phases=[
        "dense_teacher",
        "selector_pretrain",
        "frozen_detector",
        "sparse_detector",
        "joint_finetune",
    ],
    eight_week_questions_answered_by=[
        "true_time_roundtrip_tests",
        "segment_inverse_map_tests",
        "selector_detector_loss_gradient_smoke",
        "actionformer_forward_train_selector_gradient_smoke",
        "fail_closed_curriculum_and_claim_gates",
    ],
    limitations=[
        "smoke route only; no mAP improvement claim",
        "straight-through hard gather uses a relaxed temporal surrogate for selector gradient proof",
        "teacher utility is disabled for val/test selection and not required by this route",
    ],
)

sparse_detector_distillation_gate = dict(
    enabled=False,
    fail_closed=True,
    required_before_full_detector_loss=True,
    proof_source="fail_closed_sparse_detector_distillation_adapter",
    teacher_targets_train_only=True,
    map_claim_allowed=False,
    paper_claim_allowed=False,
)

sparse_detector_distillation = dict(
    stage="stage3_sparse_detector_distillation_framework",
    enabled=False,
    loss_adapter=dict(
        type="SparseDetectorDistillationLossAdapter",
        fail_closed_without_teacher_targets=True,
        teacher_targets_train_only=True,
        allowed_targets=["proposal_logits", "boundary_distributions", "ranking_quality"],
        map_claim_allowed=False,
        paper_claim_allowed=False,
    ),
)

truetime_curriculum = dict(
    active_phase=os.environ.get("TRUETIME_CURRICULUM_PHASE", "joint_finetune"),
    dense_teacher=dict(enabled=False, allow_teacher_targets_train_only=False),
    selector_pretrain=dict(enabled=True, detector_frozen=True, uses_gt_selection=False, uses_teacher_selection=False),
    frozen_detector=dict(enabled=True, detector_frozen=True, selector_trainable=True),
    sparse_detector=dict(enabled=True, detector_frozen=False, selector_trainable=False),
    joint_finetune=dict(enabled=True, detector_frozen=False, selector_trainable=True, require_selector_grad_nonzero=True),
)

selection_contract = dict(
    train=dict(selection_uses_gt=False, selection_uses_teacher=False, allow_gt_for_detector_loss=True),
    val=dict(selection_uses_gt=False, selection_uses_teacher=False),
    test=dict(selection_uses_gt=False, selection_uses_teacher=False),
)

truetime_metrics_to_log = [
    "selector_grad_norm",
    "detector_loss_selector_grad_norm",
    "selected_input_selector_grad_norm",
    "selected_count_mean",
    "selected_count_std",
    "entropy",
    "loss_cls",
    "loss_reg",
    "actionformer_cls_loss",
        "actionformer_reg_loss",
        "actionformer_detector_loss_selector_grad_norm",
        "sparse_distill_loss",
        "geometry_roundtrip",
    "prediction_inverse_map",
    "claim_locks",
]

c3_truetime_meta_keys = [
    "video_name",
    "data_path",
    "fps",
    "duration",
    "snippet_stride",
    "window_start_frame",
    "resize_length",
    "window_size",
    "offset_frames",
    "truetime_selected_positions",
    "truetime_dense_len",
    "truetime_dense_valid_len",
    "truetime_selected_count",
    "detector_output_coordinate_space",
    "detector_prediction_inverse_map_required",
    "selected_axis_to_true_time_dense_index",
    "irregular_selected_positions",
    "irregular_selected_valid_len",
    "irregular_selected_count",
    "irregular_dense_valid_len",
    "irregular_native_axis",
]

dataset = dict(
    train=dict(
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="random_trunc", trunc_len=dense_window_size, scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 182)),
            dict(type="mmaction.RandomResizedCrop"),
            dict(type="mmaction.Resize", scale=(160, 160), keep_ratio=False),
            dict(type="mmaction.Flip", flip_ratio=0.5),
            dict(type="mmaction.ImgAug", transforms="default"),
            dict(type="mmaction.ColorJitter"),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_truetime_meta_keys),
        ],
    ),
    val=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs", "gt_segments", "gt_labels"]),
            dict(type="Collect", inputs="imgs", keys=["masks", "gt_segments", "gt_labels"], meta_keys=c3_truetime_meta_keys),
        ],
    ),
    test=dict(
        window_size=dense_window_size,
        pipeline=[
            dict(type="PrepareVideoInfo", format="mp4"),
            dict(type="mmaction.DecordInit", num_threads=4),
            dict(type="LoadFrames", num_clips=1, method="sliding_window", scale_factor=1),
            dict(type="mmaction.DecordDecode"),
            dict(type="mmaction.Resize", scale=(-1, 160)),
            dict(type="mmaction.CenterCrop", crop_size=160),
            dict(type="mmaction.FormatShape", input_format="NCTHW"),
            dict(type="ConvertToTensor", keys=["imgs"]),
            dict(type="Collect", inputs="imgs", keys=["masks"], meta_keys=c3_truetime_meta_keys),
        ],
    ),
)

model = dict(
    frame_selector=dict(
        type="TrueTimeRelaxedHardTopKSelector",
        in_channels=3,
        selected_count=selected_window_size,
        dense_len=dense_window_size,
        temperature=0.7,
        selector_hidden_channels=16,
        allow_gt_selection=False,
        allow_teacher_utility=False,
        coordinate_space="selected_axis_index",
        true_time_source_axis="true_time_dense_index",
        detector_gradient_mode="st_sparse_gather",
        slot_softmax_temperature=0.7,
        slot_distance_penalty=2.0,
    ),
    backbone=dict(
        backbone=dict(total_frames=selected_window_size * scale_factor),
        custom=dict(
            pre_processing_pipeline=[
                dict(type="Rearrange", keys=["frames"], ops="b n c (t1 t) h w -> (b t1) n c t h w", t1=chunk_num),
            ],
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=chunk_num),
                dict(type="Interpolate", keys=["feats"], size=selected_window_size),
            ],
        ),
    ),
    projection=dict(max_seq_len=selected_window_size),
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            requires_irregular_native_axis=True,
            eps=1.0e-6,
            coordinate_space="true_time_dense_index",
            selected_position_key="irregular_selected_positions",
            dense_valid_len_key="irregular_dense_valid_len",
        ),
    ),
)

truetime_detector_path_smoke_model = dict(
    type="TrueTimeJointSelectorSmokeDetector",
    in_channels=3,
    hidden_channels=8,
    frame_selector=dict(
        type="TrueTimeRelaxedHardTopKSelector",
        in_channels=3,
        selected_count=4,
        dense_len=8,
        temperature=0.7,
        selector_hidden_channels=8,
        allow_gt_selection=False,
        allow_teacher_utility=False,
        coordinate_space="selected_axis_index",
        true_time_source_axis="true_time_dense_index",
        detector_gradient_mode="st_sparse_gather",
        slot_softmax_temperature=0.7,
        slot_distance_penalty=2.0,
    ),
)

truetime_actionformer_path_smoke_model = dict(
    type="ActionFormer",
    frame_selector=dict(
        type="TrueTimeRelaxedHardTopKSelector",
        in_channels=3,
        selected_count=4,
        dense_len=8,
        temperature=0.7,
        selector_hidden_channels=8,
        allow_gt_selection=False,
        allow_teacher_utility=False,
        coordinate_space="selected_axis_index",
        true_time_source_axis="true_time_dense_index",
        detector_gradient_mode="st_sparse_gather",
        slot_softmax_temperature=0.7,
        slot_distance_penalty=2.0,
    ),
    projection=dict(
        type="Conv1DTransformerProj",
        in_channels=3,
        out_channels=8,
        arch=(1, 0, 0),
        conv_cfg=dict(kernel_size=3, proj_pdrop=0.0),
        norm_cfg=dict(type="LN"),
        attn_cfg=dict(n_head=1, n_mha_win_size=1),
        path_pdrop=0.0,
        use_abs_pe=False,
        max_seq_len=4,
        input_pdrop=0.0,
    ),
    neck=None,
    rpn_head=dict(
        type="ActionFormerHead",
        num_classes=1,
        in_channels=8,
        feat_channels=8,
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            requires_irregular_native_axis=True,
            eps=1.0e-6,
            coordinate_space="true_time_dense_index",
            selected_position_key="irregular_selected_positions",
            dense_valid_len_key="irregular_dense_valid_len",
        ),
        num_convs=1,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1],
            regression_range=[(0, 10000)],
        ),
        loss_normalizer=4,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0.0,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=1),
    val=dict(batch_size=1, num_workers=1),
    test=dict(batch_size=1, num_workers=1),
    clip_grad_norm=1,
    amp=False,
    fp16_compress=False,
    static_graph=False,
    ema=False,
)

optimizer = dict(
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
    backbone=dict(lr=0, weight_decay=0, custom=[dict(name="adapter", lr=2e-4, weight_decay=0.05)], exclude=["backbone"]),
)
scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=1, max_epoch=1)
inference = dict(load_from_raw_predictions=False, save_raw_prediction=False)
workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=99,
    end_epoch=1,
    max_train_iters=2,
    disable_checkpoint=True,
)

work_dir = "exps/thumos/adatad/c3_truetime_joint_selector_c3_adatad_smoke"
