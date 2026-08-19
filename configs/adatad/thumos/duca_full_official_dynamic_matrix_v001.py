import os


_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

arm = os.environ.get("DUCA_ARM", "").strip().lower()
allowed_arms = ("official_dense", "uniform_k384", "learned_k384", "dynamic_k", "dynamic_k_no_risk")
if arm not in allowed_arms:
    raise ValueError(f"DUCA_ARM must be one of {allowed_arms}, got {arm!r}")

seed = int(os.environ.get("DUCA_SEED", "3407"))
dense_window_size = 768

experiment_scope = dict(
    route="duca_hierarchical_dynamic_frame_acquisition",
    arm=arm,
    seed=seed,
    task="THUMOS14_temporal_action_detection",
    detector="official_AdaTAD_ActionFormer_VideoMAE_S",
    official_training_subset=True,
    official_validation_evaluator=True,
    dynamic_budget_is_main_claim=arm in ("dynamic_k", "dynamic_k_no_risk"),
    fixed_budget_is_baseline=arm in ("uniform_k384", "learned_k384"),
    arbitrary_frame_selection=arm in ("learned_k384", "dynamic_k", "dynamic_k_no_risk"),
    physical_time_reconstruction=arm != "official_dense",
)

if arm != "official_dense":
    is_dynamic = arm in ("dynamic_k", "dynamic_k_no_risk")
    no_risk = arm == "dynamic_k_no_risk"
    target_len = 512 if is_dynamic else 384
    selection_strategy = "uniform_exact_k" if arm == "uniform_k384" else ("dynamic_B" if is_dynamic else "frame_score_global_rank_st")
    # Frame-score fusion preset (fixed-K learned ranking and dynamic outer-K evidence).
    # Boundary-first contract: actionness is auxiliary, boundary dominates,
    # uncertainty is a small difficulty prior, redundancy is a light penalty.
    reader_action_weight = 0.70 if no_risk else 0.20
    reader_boundary_weight = 0.0 if no_risk else 0.65
    reader_uncertainty_weight = 0.0 if no_risk else 0.15
    reader_redundancy_weight = 0.15 if no_risk else 0.10
    dynamic_budget = None
    if is_dynamic:
        dynamic_budget = dict(
            enabled=True,
            protocol="marginal_utility_v0",
            min_budget=256,
            target_budget=384,
            max_budget=512,
            average_budget=384,
            budget_step=16,
            score_midpoint=0.5,
            # The already-fused frame_selection_logits is the single dynamic-K
            # evidence. Re-fusing action/boundary/uncertainty/redundancy here
            # double-counts boundary evidence and biases neutral evidence
            # below score_midpoint, so all secondary weights are zero.
            actionness_weight=1.0,
            boundary_weight=0.0,
            uncertainty_weight=0.0,
            redundancy_weight=0.0,
        )

    model = dict(
        frame_selector=dict(
            type="PCOTMRASPreBackboneFrameSelector",
            target_len=target_len,
            dense_window_size=dense_window_size,
            descriptor_dim=3 * 32 * 32,
            selection_unit=1,
            remap_gt_to_selected_axis=False,
            selection_strategy=selection_strategy,
            scout_feature_source="compressed_pixels",
            scout_spatial_size=32,
            straight_through_detector_loss=arm not in ("uniform_k384",),
            physical_dense_reconstruction=True,
            variable_length_output=is_dynamic,
            variable_compute_multiple=16,
            frame_score_st_surrogate="global_rank_topk" if arm == "learned_k384" else "local_softmax",
            frame_score_st_gradient_scale=1.0,
            dynamic_budget=dynamic_budget,
            aux_gt_acquisition_loss_weight=0.0 if arm == "uniform_k384" else 0.05,
            aux_frame_score_boundary_loss_weight=0.0 if arm == "uniform_k384" else (0.0 if no_risk else 0.05),
            aux_risk_loss_weight=0.0 if arm == "uniform_k384" else (0.0 if no_risk else 0.05),
            aux_uncertainty_loss_weight=0.0 if arm == "uniform_k384" else (0.0 if no_risk else 0.025),
            aux_redundancy_loss_weight=0.0 if arm == "uniform_k384" else 0.01,
            reader_regularizer_loss_weight=0.0 if arm == "uniform_k384" else 0.01,
            reader=dict(
                type="PCOTMRASBoundaryDifficultyTemporalFrameScout",
                in_dim=3 * 32 * 32,
                hidden_dim=96,
                num_slots=target_len,
                temporal_layers=4,
                temporal_kernel_size=5,
                dilations=(1, 2, 4, 8),
                dropout=0.10,
                action_bias_weight=reader_action_weight,
                boundary_bias_weight=reader_boundary_weight,
                uncertainty_bias_weight=reader_uncertainty_weight,
                redundancy_bias_weight=reader_redundancy_weight,
            ),
        ),
        backbone=dict(
            backbone=dict(allow_variable_total_frames=True),
            custom=dict(
                dynamic_sparse_temporal=dict(
                    enabled=True,
                    clip_len=16,
                    tubelet_size=2,
                    output_len=dense_window_size,
                ),
            ),
        ),
    )
    solver = dict(
        train=dict(batch_size=1, num_workers=2),
        val=dict(batch_size=1, num_workers=2),
        test=dict(batch_size=1, num_workers=2),
    )

work_dir = f"exps/thumos/adatad/duca_full_official/{arm}/seed_{seed}"
