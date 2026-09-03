_base_ = ["../duca_sampling_rate_both_asformer_full_adapt_fixed384_official60.py"]

from tools.bata.duca_cellcf_protocol import protocol_for_name

duca_training_protocol = protocol_for_name("official60")
seed = 3407
total_epochs = 60
max_updates = 6000
dense_window_size = 768
window_size = 384
scale_factor = 2
chunk_num = window_size * scale_factor // 16
h65_pro_base_commit = "04c35a3b76897e6c1569eeede41ed3aecaf7f854"
h65_pro_branch = "codex/h65-pro-fullmatrix-strict60-20260902"
h65_pro_budget_contract = dict(
    dataset="THUMOS14",
    dense_frames=768,
    selected_frames=384,
    epochs=60,
    successful_optimizer_updates=6000,
    pretrain="public VideoMAE-S Kinetics-400",
    detector="AdaTAD ActionFormer",
    primary_checkpoint="epoch_59.pth/state_dict_ema",
    secondary_checkpoint="epoch_59.pth/state_dict",
)

model = dict(
    frame_selector=dict(
        budget=window_size,
        dense_window_size=dense_window_size,
        acquisition_policy="budget_calibrated_sampling_rate",
        semantic_phase_sigma=2.0,
        semantic_phase_scaffold_budget=128,
        semantic_phase_onset_budget=64,
        semantic_phase_offset_budget=64,
        semantic_phase_core_budget=128,
        detector_gradient_mode="density_transport_st",
        sampling_rate_utility_components="both",
        detector_contribution_distillation_weight=1.0,
        detector_contribution_components="both",
        detector_contribution_mode="abs_grad_times_input",
        training_uniform_companion_fraction=0.50,
        training_uniform_companion_normalize_learned_gradient=True,
        hard_max_gap_repair=False,
        fail_on_infeasible_max_gap=False,
        loss_weights=dict(
            actionness=1.0,
            detector=1.0,
            transition=0.5,
            transition_boundary=0.25,
            max_gap_hole=0.0,
            teacher=0.0,
            detector_utility=0.0,
            start=0.0,
            end=0.0,
            context=0.0,
            boundary=0.0,
            hole=0.0,
            budget=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
        ),
    ),
    backbone=dict(
        backbone=dict(
            total_frames=dense_window_size,
            num_frames=16,
            tubelet_size=2,
            relative_physical_time_residual=True,
            tubelet_packed_runtime_route=dict(enabled=False),
            amod_config=dict(enabled=False),
        ),
        custom=dict(
            global_rank_selection=True,
            canonical_selection="exact_uniform_positions_once_over_dense_window",
        ),
    ),
    projection=dict(max_seq_len=window_size),
    rpn_head=dict(conv_cfg=None),
)

scheduler = dict(type="LinearWarmupCosineAnnealingLR", warmup_epoch=5, max_epoch=60)
solver = dict(static_graph=False, find_unused_parameters=True)
workflow = dict(
    formal_protocol="duca_selected_axis_optimization_v1",
    training_profile=duca_training_protocol.name,
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    intermediate_validation_role="disabled",
    intermediate_validation_selects_checkpoint=False,
    end_epoch=60,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=duca_training_protocol.steps_per_epoch,
    expected_successful_optimizer_updates=duca_training_protocol.expected_successful_optimizer_updates,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion=duca_training_protocol.checkpoint_criterion,
)

work_dir = "exps/thumos/adatad/h65_pro_fullmatrix_20260902/base"
