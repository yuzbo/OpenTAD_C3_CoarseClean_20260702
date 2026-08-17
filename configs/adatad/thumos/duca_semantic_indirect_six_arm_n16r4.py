"""Static six-arm DUCA protocol; official dense receipt remains PENDING."""

arms = (
    "official_dense_pending_zoomtoken_receipt",
    "native_uniform_fixed_k",
    "indirect_actionness_only_fixed_k",
    "indirect_actionness_boundary_fixed_k",
    "indirect_actionness_boundary_dynamic_outer_k",
    "direct_selector_ablation",
)
shared = dict(detector="same_adatad_detector", loss="same_loss", nms="same_nms", evaluator="same_evaluator", split="same_split", updates="same_updates", seeds="same_seeds", full_stack_cost_fields=True)
recovery = dict(interval_epochs=5, retain_recovery=3, retain_milestone_and_final=True, resume_fields=("model", "optimizer", "scheduler", "scaler", "epoch", "update", "python_rng", "numpy_rng", "torch_rng", "cuda_rng"), preserve_final_selection=True)
