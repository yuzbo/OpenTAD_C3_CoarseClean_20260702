"""Concrete DUCA six-arm protocol; declaration only, never submitted here."""
_base_ = ["./input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"]
DATA = dict(video_root="/data/run01/sczc063/yuzibo/thumos14/raw_data/video", ann_file="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json", category_file="/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt")
HOLD_MANIFEST = None  # supplied by the operator; never fabricated by this config
SCOUT = dict(type="PCOTMRASCoarseActionnessFrameScout", in_dim=4, hidden_dim=32, temporal_layers=1, temporal_kernel_size=3)
SELECTOR = dict(type="PCOTMRASPreBackboneFrameSelector", target_len=384, dense_window_size=768, boundary_radius=1, reader=SCOUT)
COMMON = dict(
    data=DATA,
    model=dict(type="ActionFormer", base="./e2e_thumos_videomae_s_768x1_160_adapter.py"),
    detector="ActionFormer",
    loss="official-derived-loss", nms="official-derived-nms", evaluator="official-derived-evaluator",
    optimizer="official-derived", lr_schedule="official-derived", seeds=(0,1,2,3), update_unit="optimizer_step",
    workflow=dict(train_split="train", cal_split="validation", hold_split="explicit_manifest"),
    firewall=dict(FIT="train", CAL="validation", HOLD="explicit_manifest", selection_uses_holdout=False),
)
arms = {
 "shared_dense_reference": dict(mode="dense", target_len=768, transport="none", shared_base="official_derived", **COMMON),
 "native_uniform_fixed_k": dict(mode="uniform", target_len=384, transport="native_selector", coarse_scout_transport=False, **COMMON),
 "semantic_actionness_only_fixed_k_control": dict(mode="actionness", selector=dict(SELECTOR, selection_strategy="semantic_indirect", boundary_enabled=False), fixed_k=384, **COMMON),
 "semantic_actionness_boundary_fixed_k": dict(mode="actionness_boundary", selector=dict(SELECTOR, selection_strategy="semantic_indirect", boundary_enabled=True), fixed_k=384, **COMMON),
 "semantic_dynamic_outer_k_headline": dict(mode="dynamic_outer_k", selector=dict(SELECTOR, selection_strategy="semantic_indirect", dynamic_k_min=192, dynamic_k_max=384, dynamic_k_step=32, dynamic_k_threshold=0.5, boundary_enabled=True), **COMMON),
 "direct_selector_ablation": dict(mode="direct_selector", selector=dict(SELECTOR, selection_strategy="frame_score_topk"), fixed_k=384, **COMMON),
}
recovery = dict(interval_epochs=5, retention=3, resume_fields=("model","optimizer","scheduler","amp_scaler","epoch","update","python_rng","numpy_rng","torch_rng","cuda_rng","dataloader"), final_selection="final/final-EMA fixed selection")
PRE_RUN = dict(data=DATA, fit_cal_hold_firewall=True, hold_manifest=HOLD_MANIFEST, submit=False, recovery=recovery)
