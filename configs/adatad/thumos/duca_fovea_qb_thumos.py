import os


_base_ = ["./e2e_thumos_videomae_s_768x1_160_adapter.py"]

arm = os.environ.get("DUCA_ARM", "").strip().lower()
allowed_arms = (
    "baseline_fused",
    "query_only",
    "query_gt_mask",
    "query_cycle",
    "query_fovea",
    "query_fovea_dpp",
    "full",
)
if arm not in allowed_arms:
    raise ValueError(f"DUCA_ARM must be one of {allowed_arms}, got {arm!r}")

seed = int(os.environ.get("DUCA_SEED", "3407"))
dense_window_size = 768

# Single-variable ablation chain around the approved FoveaSampler/Query-Bridge
# contract.  The three manual score branches are always constructed; arms only
# change the score source / supervision / sampler policy.
is_query_only = arm == "query_only"
is_foveated = arm in ("query_fovea", "query_fovea_dpp", "full")
uses_gt_mask = arm in ("query_gt_mask", "full")
uses_cycle = arm in ("query_cycle", "full")
uses_coarse = arm == "full"
score_mode = "query_contribution" if is_query_only else "fused_three_branch"
cycle_warmup_iterations = 1500

# query_fovea_dpp uses the same greedy DPP-style MMR policy with a stronger
# diversity penalty; deterministic DPP MAP on the full kernel is deferred as a
# standalone follow-up (the selection API stays unchanged).
mmr_lambda = 0.20 if arm == "query_fovea_dpp" else (0.10 if is_foveated else 0.0)

experiment_scope = dict(
    route="duca_foveasampler_query_bridge",
    arm=arm,
    seed=seed,
    task="THUMOS14_temporal_action_detection",
    detector="official_AdaTAD_ActionFormer_VideoMAE_S",
    official_training_subset=True,
    official_validation_evaluator=True,
    score_source=score_mode,
    foveated_sampler=is_foveated,
    gt_geometry_mask_supervision=uses_gt_mask,
    postheavy_cycle_feedback=uses_cycle,
    coarse_proposal_supervision=uses_coarse,
    dynamic_budget=True,
    fixed_budget_is_control=arm in ("baseline_fused", "query_only", "query_gt_mask", "query_cycle"),
)

model = dict(
    frame_selector=dict(
        type="FoveaQueryBridgeFrameSelector",
        # scout / query bridge
        scout_in_dim=3 * 32 * 32,
        scout_hidden_dim=96,
        scout_temporal_layers=4,
        scout_kernel_size=5,
        scout_dilations=(1, 2, 4, 8),
        scout_dropout=0.10,
        scout_target_len=32,
        query_hidden_dim=96,
        num_queries=4,
        query_decoder_layers=2,
        query_num_heads=4,
        query_dropout=0.10,
        # foveated sampler
        target_k=384,
        min_k=256,
        max_k=512,
        budget_step=16,
        boundary_quota=64 if is_foveated else 0,
        boundary_center_top_m=8 if is_foveated else 0,
        boundary_radius=2,
        boundary_pair_max_gap=8,
        mmr_lambda=mmr_lambda,
        gumbel_tau=1.0,
        # score composition (three manual branches always built)
        score_mode=score_mode,
        # auxiliary losses
        loss_mask_weight=1.0 if uses_gt_mask else 0.0,
        loss_coarse_weight=1.0 if uses_coarse else 0.0,
        loss_cycle_weight=0.5 if uses_cycle else 0.0,
        loss_budget_weight=0.05,
        loss_diversity_weight=0.05,
        cycle_warmup_iterations=cycle_warmup_iterations,
        cycle_enabled=uses_cycle,
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

work_dir = f"exps/thumos/adatad/duca_fovea_qb/{arm}/seed_{seed}"
