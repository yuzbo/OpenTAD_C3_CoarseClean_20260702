_base_ = ["./duca_rime_physical_total60_base.py"]

import os

if not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip():
    raise RuntimeError("D-shuffle requires DUCA_RIME_REPLAY_JSONL")

model = dict(
    frame_selector=dict(
        rime_arm="dynamic_shuffle",
        require_frozen_protocol=False,
    ),
)

duca_rime_variant = dict(
    arm="D-shuffle",
    dynamic_budget=False,
    replay_role="histogram_shuffled_budget_control",
    exact_k_histogram_matched=True,
)

work_dir = "exps/thumos/adatad/duca_rime_dynamic_shuffle_total60"
