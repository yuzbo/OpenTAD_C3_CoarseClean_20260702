_base_ = ["./duca_rime_physical_total60_base.py"]

import os

if not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip():
    raise RuntimeError("U-same-K requires the paired full-model K replay")

model = dict(
    frame_selector=dict(
        rime_arm="uniform_same_k",
        require_frozen_protocol=False,
    ),
)

duca_rime_variant = dict(
    arm="U-same-K",
    trainable=False,
    evaluation_only=True,
    positions="canonical_exact_uniform",
    per_window_k="identical_to_rime_full",
)

work_dir = "exps/thumos/adatad/duca_rime_uniform_same_k_eval"
