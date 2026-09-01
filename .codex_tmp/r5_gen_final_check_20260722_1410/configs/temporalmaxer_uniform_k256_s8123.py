_base_ = ['E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py']

window_size = 256
chunk_num = 16
duca_temporal_sampling_contract = dict(
    hard_budget=256,
    dense_window_size=768,
    max_unselected_hole_dense_candidates=2,
    dataset_feature_stride_source_frames=4,
    dataset_sample_stride=1,
    requested_max_source_frame_interval=15,
    detector_axis="selected_axis_index",
    dense_axis_unit="dense_candidate_index",
    task="offline_temporal_action_detection",
)

r5_cell = dict(
    backend='temporalmaxer',
    arm='uniform',
    budget=256,
    seed=8123,
    source_config='E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_two_stage_exact_uniform_fixed384_official60.py',
    live_duca_to_videomae=True,
    detector_type='TemporalMaxer',
    paper_claim_allowed=False,
)

duca_transition_only_contract = dict(
    exact_budget=256,
    detector_pretraining_policy="exact_uniform_k256",
    temporal_sampling_contract=duca_temporal_sampling_contract,
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        budget=256,
        temporal_sampling_contract=duca_temporal_sampling_contract,
    ),
    backbone=dict(
        backbone=dict(total_frames=256),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=16,
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
                    t1=16,
                ),
                dict(type="Interpolate", keys=["feats"], size=256),
            ],
        ),
    ),
    type="TemporalMaxer",
    selector_train_only=False,
    selector_train_only_skip_detector=False,
    projection=dict(
        _delete_=True,
        type="TemporalMaxerProj",
        in_channels=384,
        out_channels=512,
        arch=(2, 0, 5),
        conv_cfg=dict(kernel_size=3),
        norm_cfg=dict(type="LN"),
    ),
    neck=dict(
        _delete_=True,
        type="FPNIdentity",
        in_channels=512,
        out_channels=512,
        num_levels=6,
    ),
    rpn_head=dict(
        _delete_=True,
        type="TemporalMaxerHead",
        num_classes=20,
        in_channels=512,
        feat_channels=512,
        num_convs=2,
        cls_prior_prob=0.01,
        prior_generator=dict(
            type="PointGenerator",
            strides=[1, 2, 4, 8, 16, 32],
            regression_range=[
                (0, 4), (4, 8), (8, 16),
                (16, 32), (32, 64), (64, 10000),
            ],
        ),
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        loss=dict(
            cls_loss=dict(type="FocalLoss"),
            reg_loss=dict(type="DIOULoss"),
        ),
        assigner=dict(
            type="AnchorFreeSimOTAAssigner",
            iou_weight=2,
            cls_weight=1.0,
            center_radius=1.5,
            keep_percent=1.0,
            confuse_weight=0.0,
        ),
    ),
)

workflow = dict(
    formal_protocol="duca_r5_mechanism_matrix",
    formal_successful_update_contract=False,
    training_probe_json=None,
    require_training_probe_context=False,
    paper_claim_allowed=False,
)

work_dir = 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/temporalmaxer_uniform_k256_s8123'
