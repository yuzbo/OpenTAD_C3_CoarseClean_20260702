_base_ = ["./duca_rime_full_tridet_total60.py"]

import os

if not os.environ.get("DUCA_RIME_REPLAY_JSONL", "").strip():
    raise RuntimeError("U-same-K-TriDet requires the paired full-model K replay")

model = dict(
    frame_selector=dict(
        rime_arm="uniform_same_k",
        require_frozen_protocol=False,
    ),
)

duca_rime_variant = dict(
    arm="U-same-K-TriDet",
    trainable=False,
    evaluation_only=True,
    detector_backend="TriDet",
    positions="canonical_exact_uniform",
    per_window_k="identical_to_rime_full",
)

work_dir = "exps/thumos/adatad/duca_rime_uniform_same_k_tridet_eval"
