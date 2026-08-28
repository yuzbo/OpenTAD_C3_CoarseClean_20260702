"""Matched native-tubelet control for the fixed-budget coreset attribution."""

_base_ = ["./duca_sampling_rate_curriculum_stage1_uniform384.py"]

import os


def _required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the native-tubelet experiment")
    return value


stage1_checkpoint = _required("DUCA_STAGE1_CHECKPOINT")
stage1_checkpoint_sha256 = _required("DUCA_STAGE1_CHECKPOINT_SHA256")
stage1_checkpoint_epoch = int(os.environ.get("DUCA_STAGE1_CHECKPOINT_EPOCH", "29"))
if stage1_checkpoint_epoch != 29:
    raise ValueError("native-tubelet experiments require the Stage-1 epoch-29 EMA scout")

native_tubelet_contract = dict(
    task="offline_temporal_action_detection",
    attribution="fixed_budget_native_tubelet_temporal_coreset",
    dynamic_budget_role="required_next_stage_after_positive_fixed_budget_evidence",
    fixed_k_role="attribution_control_not_final_method",
    dense_frames=768,
    native_tubelets=384,
    selected_tubelets=192,
    heavy_frames=384,
    max_unselected_hole_tubelets=7,
    selection_policy="native_tubelet_uniform",
    detector_grid="physical_tubelet_grid_384",
    paper_claim_allowed=False,
)

model = dict(
    frame_selector=dict(
        acquisition_policy="native_tubelet_uniform",
        native_tubelet_selected_count=192,
        structured_temperature=0.7,
        max_unselected_hole=7,
        max_gap_loss_max_unselected_hole=7,
        soft_max_gap_loss_enabled=False,
        hard_max_gap_repair=False,
        detector_gradient_mode="none",
        detector_contribution_distillation_weight=0.0,
        detector_contribution_components="none",
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
        training_uniform_companion_fraction=0.0,
        training_uniform_companion_normalize_learned_gradient=False,
        policy_hidden_gradient_scale=0.0,
        auxiliary_hidden_gradient_scale=0.0,
        allow_frozen_coarse_probe=True,
        remap_gt_to_selected_axis=False,
        selected_axis_remap_required=False,
        detector_output_coordinate_space="true_time_dense_index",
        actionness_source_cfg=dict(
            frozen=True,
            trainable=False,
            policy_hidden_gradient_scope="none",
        ),
    ),
    backbone=dict(
        custom=dict(
            post_processing_pipeline=[
                dict(type="Reduce", keys=["feats"], ops="b n c t h w -> b c t", reduction="mean"),
                dict(type="Rearrange", keys=["feats"], ops="(b t1) c t -> b c (t1 t)", t1=24),
            ],
        ),
    ),
    token_compressor=dict(
        type="PhysicalTimeCoresetReconstructor",
        target_len=384,
        feature_dim=384,
        scout_hidden_dim=96,
        context_temperature=0.7,
        time_hidden_dim=64,
    ),
    projection=dict(max_seq_len=384),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)

workflow = dict(
    training_profile="duca_native_tubelet_coreset_official60",
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    intermediate_validation_role="disabled",
    intermediate_validation_selects_checkpoint=False,
    end_epoch=60,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=6000,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    selector_initialization=dict(
        enabled=True,
        checkpoint_path=stage1_checkpoint,
        checkpoint_sha256=stage1_checkpoint_sha256,
        state_key="state_dict_ema",
        expected_checkpoint_epoch=29,
        reset_state_keys=[],
    ),
)

solver = dict(static_graph=False, find_unused_parameters=True)
seed = 3407
total_epochs = 60
max_updates = 6000
checkpoint_interval_epochs = 5
checkpoint_policy = dict(
    resumable=True,
    keep_latest=3,
    milestones=True,
    final=True,
    final_ema=True,
)
paper_claim_allowed = False
work_dir = "exps/thumos/adatad/duca_native_tubelet_fixed384_official60"
