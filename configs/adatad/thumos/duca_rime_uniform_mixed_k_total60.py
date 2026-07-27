_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]

import os

from tools.bata.duca_cellcf_protocol import protocol_for_name


train_block_list = os.environ.get("DUCA_RIME_TRAIN_BLOCK_LIST", "").strip()
development_block_list = os.environ.get(
    "DUCA_RIME_DEVELOPMENT_BLOCK_LIST",
    "",
).strip()
if not train_block_list or not development_block_list:
    raise RuntimeError(
        "U-mixed-K requires the frozen detector-train/development block lists"
    )

protocol = protocol_for_name("official60")
candidate_budgets = (192, 256, 384, 512)
candidate_costs = tuple(float(value) for value in candidate_budgets)
schedule_counts = (8, 12, 16, 24)
target_mean_cost = 384.0
evaluation_budget = int(os.environ.get("DUCA_RIME_EVAL_FIXED_BUDGET", "384"))
if evaluation_budget not in candidate_budgets:
    raise RuntimeError("U-mixed-K evaluation K must belong to the registered panel")

dataset = dict(
    train=dict(
        block_list=train_block_list,
    ),
    val=None,
    test=dict(
        subset_name="training",
        block_list=development_block_list,
        test_mode=True,
        window_size=768,
    ),
)

evaluation = dict(
    subset="training",
    blocked_videos=development_block_list,
)

model = dict(
    frame_selector=dict(
        _delete_=True,
        type="DucaRimeFrameSelector",
        in_channels=3,
        rime_arm="uniform_mixed_k",
        candidate_budgets=candidate_budgets,
        candidate_costs=candidate_costs,
        fixed_budget=evaluation_budget,
        dense_window_size=768,
        target_mean_cost=target_mean_cost,
        decoder_family="independent",
        execution_quantum=16,
        require_frozen_protocol=False,
        mixed_k_schedule_counts=schedule_counts,
        mixed_k_schedule_seed=3407,
        coarse_hidden_dim=96,
        selector_hidden_dim=64,
        coverage_floor_weight=0.10,
        score_temperature=0.70,
        path_temperature=1.0,
        transition_target_sigma=2.0,
        transition_target_radius=4,
        transition_boundary_radius=4,
        transition_distribution_temperature=0.70,
        action_loss_weight=0.0,
        transition_loss_weight=0.0,
        transition_boundary_loss_weight=0.0,
        rank_alignment_loss_weight=0.0,
        coarse_trunk_lr=2.5e-5,
        action_head_lr=5.0e-5,
        selector_lr=1.0e-4,
        detector_bridge_gradient_scale=0.0,
        actionness_source_cfg=None,
        strict_physical_metadata=True,
        forbid_raw_prediction_cache=True,
    ),
    backbone=dict(
        backbone=dict(total_frames=candidate_budgets[-1], with_cp=False),
        custom=dict(
            dynamic_temporal_bucket=True,
            dynamic_temporal_clip_len=16,
        ),
    ),
    projection=dict(max_seq_len=candidate_budgets[-1]),
    rpn_head=dict(
        physical_grid_actionformer=dict(
            enabled=True,
            required=True,
            strict=True,
            contract="duca_rime_physical_dynamic_k_v1",
        ),
    ),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
    find_unused_parameters=True,
)

scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=5,
    max_epoch=protocol.end_epoch,
)

workflow = dict(
    formal_protocol="duca_rime_phase2_mixed_k_baseline_v1",
    training_profile=protocol.name,
    logging_interval=50,
    checkpoint_interval=protocol.checkpoint_interval,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=protocol.end_epoch,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=protocol.steps_per_epoch,
    expected_successful_optimizer_updates=(
        protocol.expected_successful_optimizer_updates
    ),
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=protocol.terminal_epoch,
    primary_checkpoint_state_key=protocol.terminal_state_key,
    checkpoint_criterion=protocol.checkpoint_criterion,
    seal_eval_dataloaders_during_training=True,
    derive_train_loader_contract=True,
)

duca_rime_variant = dict(
    arm="U-mixed-K",
    position_policy="exact_uniform",
    candidate_budgets=candidate_budgets,
    training_schedule_counts=schedule_counts,
    training_target_mean_cost=target_mean_cost,
    evaluation_budget=evaluation_budget,
    detector_backend="ActionFormer",
    detector_training_exposure="mixed_k_registered_panel",
    coarse_probe_executed=False,
    uses_gt_for_budget_or_position_decision=False,
    expected_successful_updates=protocol.expected_successful_optimizer_updates,
)

duca_rime_contract = dict(
    task="offline_temporal_action_detection",
    online_tad=False,
    streaming=False,
    pre_backbone_plugin=True,
    candidate_budgets=candidate_budgets,
    target_mean_cost=target_mean_cost,
    dynamic_heavy_compute=True,
    pad_to_kmax=False,
    execution_quantum=16,
    batch_size=1,
    exact_successful_updates=protocol.expected_successful_optimizer_updates,
    development_split="train_only_certification_development",
    official_final_subset_consumed=False,
    empirically_supported=False,
    paper_ready=False,
)

post_processing = dict(save_dict=True)

work_dir = "exps/thumos/adatad/duca_rime_uniform_mixed_k_total60"

del protocol
del candidate_costs
del evaluation_budget
