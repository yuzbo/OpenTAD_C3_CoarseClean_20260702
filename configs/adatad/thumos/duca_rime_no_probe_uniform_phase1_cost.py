_base_ = ["./duca_protected_physical_exact_uniform_fixed384_official60.py"]

import os


evaluation_block_list = os.environ.get(
    "DUCA_RIME_PHASE1_EVAL_BLOCK_LIST",
    "",
).strip()
if not evaluation_block_list:
    raise RuntimeError("Phase-1 no-probe cost control requires its frozen block list")

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
    test=dict(batch_size=1, num_workers=0),
    static_graph=False,
)
post_processing = dict(save_dict=False)

duca_rime_phase1_cost_contract = dict(
    contract="duca_rime_phase1_no_probe_uniform_cost_v1",
    coarse_probe_executed=False,
    selection_policy="exact_uniform",
    checkpoint_drop_prefixes=("frame_selector.score_net.",),
    accuracy_claim_allowed=False,
    uses_official_final=False,
)

work_dir = "exps/thumos/adatad/duca_rime_phase1_no_probe_uniform_cost"

del evaluation_block_list
