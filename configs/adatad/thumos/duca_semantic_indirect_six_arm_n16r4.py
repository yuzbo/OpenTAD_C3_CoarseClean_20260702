"""DUCA semantic-indirect six-arm decision configuration (no execution)."""
selector = dict(type="PCOTMRASPreBackboneFrameSelector", selection_strategy="semantic_indirect", target_len=384, dense_window_size=768, dynamic_k_min=192, dynamic_k_max=384, dynamic_k_threshold=0.5, dynamic_k_step=32, boundary_radius=1, reader=dict(type="PCOTMRASBoundaryDifficultyTemporalFrameScout", in_dim=4, hidden_dim=32, temporal_layers=1, temporal_kernel_size=3))
shared = dict(detector="same_adatad_detector", loss="same_loss", nms="same_nms", evaluator="same_evaluator", split="same_split", updates="same_updates", seeds="same_seeds")
arms = {
 "dense_placeholder": dict(mode="dense_placeholder", **shared),
 "native_uniform_fixed_k": dict(mode="native_uniform_fixed_k", selector=dict(selector, selection_strategy="slot_transport"), **shared),
 "actionness_only_fixed_k_control": dict(mode="actionness_only_fixed_k_control", selector=dict(selector, dynamic_k_min=384, dynamic_k_max=384, semantic_acquisition="actionness_only"), **shared),
 "actionness_boundary_fixed_k": dict(mode="actionness_boundary_fixed_k", selector=dict(selector, dynamic_k_min=384, dynamic_k_max=384, semantic_acquisition="actionness_boundary"), **shared),
 "actionness_boundary_dynamic_k_headline": dict(mode="actionness_boundary_dynamic_k_headline", selector=selector, **shared),
 "direct_selector_ablation": dict(mode="direct_selector_ablation", selector=dict(selector, selection_strategy="frame_score_topk"), **shared),
}
recovery = dict(interval_contract="min 5 epochs unless unchanged official more frequent", retention=3, resume_fields=("model", "optimizer", "scheduler", "scaler", "epoch", "update", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"), final_selection="final/final-EMA fixed selection")
