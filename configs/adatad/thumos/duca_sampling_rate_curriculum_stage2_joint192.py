_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

# Matched 25% budget arm. Contribution distillation, detector gradients,
# coarse adaptation and every schedule endpoint remain identical to K384.
window_size = 192
chunk_num = 12

duca_sampling_rate_contract = dict(
    route="DUCA_RATE_CURRICULUM_STAGE2_JOINT192",
    exact_budget=192,
    stage1_initialization="full_uniform_k192_ema_model",
)

duca_transition_only_contract = dict(
    route="DUCA_BUDGET_CALIBRATED_SAMPLING_RATE_K192_CURRICULUM",
    exact_budget=192,
)

model = dict(
    frame_selector=dict(
        budget=192,
    ),
    backbone=dict(
        backbone=dict(
            total_frames=192,
        ),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=12,
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
                    t1=12,
                ),
                dict(type="Interpolate", keys=["feats"], size=192),
            ],
        ),
    ),
    projection=dict(max_seq_len=192),
)

workflow = dict(
    training_profile="duca_rate_curriculum_stage2_joint192",
)

work_dir = "exps/thumos/adatad/duca_sampling_rate_curriculum_stage2_joint192"
