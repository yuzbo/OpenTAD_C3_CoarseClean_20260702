_base_ = ["./duca_boundary_burst_g0_no_feedback_fixed384_official60.py"]

import os


duca_sparse_probe_stride = int(os.environ.get("DUCA_SPARSE_PROBE_STRIDE", "1"))
if duca_sparse_probe_stride not in {1, 2, 3, 4}:
    raise ValueError("DUCA_SPARSE_PROBE_STRIDE must be one of 1, 2, 3, 4")

duca_sparse_probe_contract = dict(
    task="offline_temporal_action_detection",
    stage="official60_terminal_ema_map",
    candidate_stride_source_frames=4,
    sparse_probe_stride_dense_candidates=duca_sparse_probe_stride,
    sparse_probe_interval_source_frames=4 * duca_sparse_probe_stride,
    reconstruction="linear_interpolation_of_multidimensional_asformer_hidden",
    reconstructed_dense_length=768,
    selector_receives_anchor_mask=False,
    selector_receives_anchor_distance=False,
    hard_budget=384,
    max_unselected_hole_dense_candidates=2,
    detector_backend="official_derived_adatad_actionformer",
    paper_metric_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        actionness_source_cfg=dict(
            temporal_probe_stride=duca_sparse_probe_stride,
            temporal_interpolation_mode="hidden_linear",
        ),
    ),
)

work_dir = (
    "exps/thumos/adatad/duca_sparse_probe_hidden_linear_"
    f"d{duca_sparse_probe_stride}_g0_fixed384_official60"
)
