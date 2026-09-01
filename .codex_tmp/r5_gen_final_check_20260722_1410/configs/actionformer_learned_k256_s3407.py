_base_ = ['E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py']

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
    backend='actionformer',
    arm='learned',
    budget=256,
    seed=3407,
    source_config='E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/OpenTAD_DUCA_GlobalCurriculum_20260721/configs/adatad/thumos/duca_boundary_burst_g1_protected_fixed384_official60.py',
    live_duca_to_videomae=True,
    detector_type='ActionFormer',
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
    projection=dict(max_seq_len=256),
)

workflow = dict(
    formal_protocol="duca_r5_mechanism_matrix",
    formal_successful_update_contract=False,
    training_probe_json=None,
    require_training_probe_context=False,
    paper_claim_allowed=False,
)

work_dir = 'E:/DeskTop/TAD/OpenTAD_C3_CoarseClean_20260702/.codex_tmp/r5_gen_final_check_20260722_1410/runs/actionformer_learned_k256_s3407'
