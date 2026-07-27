_base_ = ["./duca_protected_physical_exact_uniform_fixed384_official60.py"]

import os


evaluation_block_list = os.environ.get(
    "DUCA_RIME_PHASE2_EVAL_BLOCK_LIST",
    "",
).strip()
if not evaluation_block_list:
    raise RuntimeError(
        "Phase-2 exact-uniform baseline requires its frozen train-role block list"
    )
fixed_budget = int(os.environ.get("DUCA_RIME_FIXED_BUDGET", "384"))
if fixed_budget not in {128, 192, 256, 384, 512} or fixed_budget % 16:
    raise RuntimeError("Phase-2 exact-uniform baseline uses the registered K panel")
fixed_chunk_num = fixed_budget // 16

dataset = dict(
    val=None,
    test=dict(
        subset_name="training",
        block_list=evaluation_block_list,
        test_mode=True,
        window_size=768,
    ),
)

evaluation = dict(subset="training")

model = dict(
    frame_selector=dict(
        budget=fixed_budget,
    ),
    backbone=dict(
        backbone=dict(total_frames=fixed_budget),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=fixed_chunk_num,
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
                    t1=fixed_chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=fixed_budget),
            ],
        ),
    ),
    projection=dict(max_seq_len=512),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
)

post_processing = dict(save_dict=True)

duca_rime_baseline_contract = dict(
    phase=2,
    variant="U-fixed",
    position_policy="exact_uniform",
    target_mean_cost=float(fixed_budget),
    detector_backend="ActionFormer",
    padded_to_kmax=False,
    uses_official_final=False,
    training_identity_required=False,
    checkpoint_compatibility_mode=(
        "historical_uniform_score_net_unused_exact_whitelist_v1"
    ),
    claim_scope="phase2_clean_baseline_measurement_only",
)

duca_rime_contract = dict(
    pad_to_kmax=False,
    target_mean_cost=float(fixed_budget),
    execution_quantum=16,
)

work_dir = (
    f"exps/thumos/adatad/duca_rime_uniform_phase2_baseline_k{fixed_budget}"
)

del evaluation_block_list
del fixed_budget
del fixed_chunk_num
