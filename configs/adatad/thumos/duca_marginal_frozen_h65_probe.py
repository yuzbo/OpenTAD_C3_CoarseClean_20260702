"""Frozen-H65 counterfactual marginal-budget probe (no detector training)."""

_base_ = ["./duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]

seed = 3407

duca_marginal_probe = dict(
    method="DUCA-Marginal-v1",
    base_revision="04c35a3b76897e6c1569eeede41ed3aecaf7f854",
    h65_checkpoint_path=(
        "/data/run01/sczc063/yuzibo/duca_h65_90_stage2_off_04c35a3b_20260823/"
        "gpu1_id0/checkpoint/epoch_59.pth"
    ),
    h65_checkpoint_sha256="dafcfbd0b1e0a13c400789e73ee13a20cf69551813ef62fc8185fde609806a1c",
    detector_checkpoint_epoch=59,
    detector_checkpoint_state_key="state_dict_ema",
    detector_frozen=True,
    scout_frozen=True,
    budgets=[256, 384, 512],
    baseline_budget=384,
    observation_packet=16,
    contiguous_raw_clip=False,
    nested_h65_priority=True,
    reproduce_h65_at_k384=True,
    controller_fit_videos=160,
    controller_holdout_videos=40,
    controller_split_seed=3407,
    utility_head=dict(
        type="SignedTwoSidedMarginalUtilityHead",
        hidden_dim=128,
        feature_contract=(
            "masked_mean_frozen_scout_hidden_plus_mean_std_max_of_"
            "actionness_transition_uncertainty_plus_k384_gap_statistics"
        ),
        outputs=["downgrade_penalty", "upgrade_gain"],
        target=["loss_k256 - loss_k384", "loss_k384 - loss_k512"],
        detector_gradient=False,
        epochs=20,
        optimizer="AdamW",
        learning_rate=1.0e-3,
        weight_decay=1.0e-4,
        batch_size=256,
    ),
    allocation=dict(
        policy="video_level_exact_total_marginal_reallocation",
        target_actual_observations="384 * actual_window_count",
        max_changed_fraction=0.5,
        transfer_only_when_predicted_total_utility_positive=True,
        padding_to_max_budget_allowed=False,
    ),
    execution=dict(
        stage_order=["select_k384", "counterfactual_k256", "counterfactual_k512", "summarize"],
        group_windows_by_budget=True,
        one_real_heavy_shape_per_stage=True,
        detector_length=384,
        padding_to_upper_budget=False,
    ),
    gates=dict(
        oracle_holdout_delta_avg_map_pp=0.8,
        oracle_holdout_delta_map_07_pp=1.0,
        marginal_spearman=0.25,
        marginal_sign_accuracy=0.60,
        learned_oracle_gain_fraction=0.40,
        minimum_lower_budget_window_fraction=0.10,
        minimum_upper_budget_window_fraction=0.10,
        video_total_budget_error=0,
    ),
)

model = dict(
    backbone=dict(
        backbone=dict(with_cp=False),
    ),
)

workflow = dict(
    formal_protocol="frozen_h65_counterfactual_marginal_budget_probe",
    paper_claim_allowed=False,
    training_enabled=False,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="sealed_h65_terminal_epoch_59_state_dict_ema",
)

work_dir = "exps/thumos/adatad/duca_marginal_frozen_h65_probe"
