_base_ = ["./duca_rime_physical_total60_base.py"]

import os

if not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip():
    raise RuntimeError("AdapTok-TAD requires its immutable test-batch ILP replay")

model = dict(
    frame_selector=dict(
        rime_arm="adaptok_tad",
        require_frozen_protocol=False,
    ),
)

duca_rime_variant = dict(
    arm="AdapTok-TAD",
    direct_transfer_baseline=True,
    budget_source="total_loss_curve_test_batch_ilp",
    uses_test_batch_composition=True,
    deployment_candidate=False,
)

work_dir = "exps/thumos/adatad/duca_adaptok_tad_direct_total60"
