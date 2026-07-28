_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

import os


evaluation_block_list = os.environ.get(
    "DUCA_RIME_PHASE1_EVAL_BLOCK_LIST",
    "",
).strip()
dense_variant = os.environ.get(
    "DUCA_RIME_PHASE1_DENSE_VARIANT",
    "",
).strip()
if not evaluation_block_list:
    raise RuntimeError("Phase-1 dense control requires its frozen block list")
if dense_variant not in {"released_dense", "local_dense"}:
    raise RuntimeError("Phase-1 dense control variant is not registered")

dataset = dict(
    val=None,
    test=dict(
        subset_name="training",
        block_list=evaluation_block_list,
        test_mode=True,
        window_size=768,
    ),
)

evaluation = dict(
    subset="training",
    blocked_videos=evaluation_block_list,
)

model = dict(
    backbone=dict(
        backbone=dict(with_cp=False),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
)

post_processing = dict(save_dict=True)

workflow = dict(
    formal_protocol="duca_rime_phase1_dense_control_v1",
)

duca_rime_baseline_contract = dict(
    phase=1,
    variant=dense_variant,
    position_policy="dense_all",
    target_mean_cost=768.0,
    detector_backend="ActionFormer",
    padded_to_kmax=False,
    uses_official_final=False,
    training_identity_required=False,
    checkpoint_compatibility_mode="strict_exact_v1",
    claim_scope="phase1_dense_sanity_control_only",
)

work_dir = f"exps/thumos/adatad/duca_rime_phase1_{dense_variant}"

del evaluation_block_list
del dense_variant
