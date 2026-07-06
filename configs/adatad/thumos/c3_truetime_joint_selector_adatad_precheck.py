_base_ = ["./c3_truetime_joint_selector_c3_adatad_smoke.py"]

import os


route_label = "DIVERGENT_INNOVATION_TRUETIME_JOINT_SELECTOR_DO_NOT_MERGE_WITH_C3"
selected_window_size = 384
dense_window_size = 768
precheck_only = os.environ.get("PRECHECK_ONLY", "1") != "0"
precheck_max_train_iters = int(os.environ.get("TRUETIME_PRECHECK_MAX_TRAIN_ITERS", "4" if precheck_only else "2000"))
precheck_end_epoch = int(os.environ.get("TRUETIME_PRECHECK_END_EPOCH", "1" if precheck_only else "12"))
truetime_selector_grad_proof_path = os.environ.get(
    "TRUETIME_SELECTOR_GRAD_PROOF_JSON",
    "REPLACE_WITH_TRUETIME_SELECTOR_GRAD_PROOF_JSON",
)


experiment_scope = dict(
    route="STAGE3_TRUE_TIME_E2E_ADATAD_SELECTOR",
    route_variant=route_label,
    stage="stage3_true_time_e2e_adatad_selector_precheck",
    detector_stack="truetime_physical_grid_actionformer_frame_selector_slot",
    separate_from_stage2_detector_utility_offline_selector=True,
    minimum_useful_deliverable="precheck_ready_true_adatad_selector_training_candidate",
    full_map_claim_required=True,
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
    end_to_end_claim_allowed=False,
)

truetime_joint_selector_gate = dict(
    route="STAGE3_TRUE_TIME_E2E_ADATAD_SELECTOR",
    route_variant=route_label,
    stage="stage3_true_time_e2e_adatad_selector_precheck",
    smoke_only=False,
    fulltrain_candidate=True,
    precheck_only_default=precheck_only,
    max_epochs=precheck_end_epoch,
    max_train_iters=precheck_max_train_iters,
    default_off=True,
    explicit_config_opt_in=True,
    requires_launch_gate=True,
    launch_gate_passed=False,
    reviewed_execution_config=False,
    requires_selector_grad_nonzero=True,
    requires_actionformer_detector_grad_nonzero=True,
    real_detector_gradient_proof_required=True,
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
    allow_long_training=not precheck_only,
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
        "real_actionformer_forward_train_selector_gradient_precheck",
        "selected_axis_prediction_inverse_map_postprocess_tests",
        "fail_closed_curriculum_and_claim_gates",
    ],
    limitations=[
        "precheck/fulltrain candidate only; no mAP, paper, runtime, or deploy claim",
        "default PRECHECK_ONLY keeps training bounded until explicitly unlocked",
        "straight-through hard gather uses a relaxed temporal surrogate for selector gradient evidence",
        "teacher utility is disabled for val/test selection and not required by this route",
    ],
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=precheck_end_epoch,
    max_train_iters=precheck_max_train_iters,
    disable_checkpoint=precheck_only,
)

truetime_actionformer_path_precheck_model = dict(
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

work_dir = "exps/thumos/adatad/c3_truetime_joint_selector_adatad_precheck"
