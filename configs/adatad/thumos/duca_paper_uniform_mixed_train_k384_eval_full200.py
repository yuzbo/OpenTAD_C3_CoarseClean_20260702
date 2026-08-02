_base_ = ["./duca_paper_rime_selected_axis_base.py"]


model = dict(
    frame_selector=dict(
        rime_arm="uniform_mixed_k",
        fixed_budget=384,
        target_mean_cost=384.0,
        require_frozen_protocol=False,
        mixed_k_schedule_counts=(8, 12, 16, 24),
        mixed_k_schedule_seed=3407,
        action_loss_weight=0.0,
        transition_loss_weight=0.0,
        transition_boundary_loss_weight=0.0,
        budget_utility_loss_weight=0.0,
        budget_risk_loss_weight=0.0,
        budget_uncertainty_loss_weight=0.0,
        rank_alignment_loss_weight=0.0,
        detector_bridge_gradient_scale=0.0,
        actionness_source_cfg=None,
    ),
)

duca_paper_cell = dict(
    arm="uniform_mixed_train_k384_eval",
    detector_backend="ActionFormer",
    train_video_count=200,
    evaluation_video_count=211,
    world_size=2,
    global_batch_size=2,
    successful_updates=6000,
    training_mean_heavy_k=384.0,
    evaluation_heavy_k=384,
    position_policy="exact_uniform",
    dynamic_budget_at_evaluation=False,
)

work_dir = (
    "exps/thumos/adatad/duca_paper_uniform_mixed_train_k384_eval_full200"
)
