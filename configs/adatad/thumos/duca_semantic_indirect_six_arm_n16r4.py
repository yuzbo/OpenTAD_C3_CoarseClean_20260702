"""Concrete DUCA six-arm protocol; declaration only, never submitted here."""
_base_ = ["./input_random_fixed_50pct_c3_physical_grid_actionformer_precheck.py"]
DATA = dict(video_root="/data/run01/sczc063/yuzibo/thumos14/raw_data/video", ann_file="/data/run01/sczc063/yuzibo/thumos14/annotations/thumos_14_anno.json", category_file="/data/run01/sczc063/yuzibo/thumos14/annotations/category_idx.txt")
SCOUT = dict(type="PCOTMRASCoarseActionnessFrameScout", in_dim=4, hidden_dim=32, temporal_layers=1, temporal_kernel_size=3)
SELECTOR = dict(type="PCOTMRASPreBackboneFrameSelector", target_len=384, dense_window_size=768, boundary_radius=1, reader=SCOUT)
COMMON = dict(data=DATA, detector="AdaTAD-derived-ActionFormer", loss="official-derived-loss", nms="official-derived-nms", evaluator="official-derived-evaluator", optimizer="official-derived", lr_schedule="official-derived", seeds=(0,1,2,3), update_unit="optimizer_step", firewall=dict(FIT="train", CAL="validation", HOLD="test", selection_uses_holdout=False))
arms = {
 "dense": dict(mode="dense", target_len=768, transport="none", **COMMON),
 "native_uniform_fixed_k": dict(mode="uniform", target_len=384, transport="native_selector", coarse_scout_transport=False, **COMMON),
 "actionness_only_fixed_k": dict(mode="actionness", selector=dict(SELECTOR, selection_strategy="semantic_indirect"), fixed_k=384, **COMMON),
 "actionness_boundary_fixed_k": dict(mode="actionness_boundary", selector=dict(SELECTOR, selection_strategy="semantic_indirect"), fixed_k=384, **COMMON),
 "dynamic_outer_k_headline": dict(mode="dynamic_outer_k", selector=dict(SELECTOR, selection_strategy="semantic_indirect", dynamic_k_min=192, dynamic_k_max=384, dynamic_k_step=32, dynamic_k_threshold=0.5), **COMMON),
 "direct_selector_ablation": dict(mode="direct_selector", selector=dict(SELECTOR, selection_strategy="frame_score_topk"), fixed_k=384, **COMMON),
}
recovery = dict(interval_epochs=5, retention=3, resume_fields=("model","optimizer","scheduler","amp_scaler","epoch","update","python_rng","numpy_rng","torch_rng","cuda_rng","dataloader"), final_selection="final/final-EMA fixed selection")
PRE_RUN = dict(data=DATA, fit_cal_hold_firewall=True, submit=False, recovery=recovery)
