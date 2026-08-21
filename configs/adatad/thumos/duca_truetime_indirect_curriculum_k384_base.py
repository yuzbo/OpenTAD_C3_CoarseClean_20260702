"""Paired online indirect-selector curriculum (K=384, 20/20/20 epochs).

Both arms use the same train-only ASFormer actionness/boundary scout, exact-K
decoder, physical ActionFormer assignment/regression, optimizer and evaluator.
They differ only in the temporal interpretation inside the VideoMAE heavy path.
"""

_base_ = ["./duca_protected_physical_fixed384_official60_base.py"]

window_size = 384
dense_window_size = 768
expected_successful_updates_per_epoch = 100
expected_successful_optimizer_updates = 6000
shared_root = "/data/run01/sczc063/yuzibo"
annotation_path = f"{shared_root}/thumos14/annotations/thumos_14_anno.json"
class_map = f"{shared_root}/thumos14/annotations/category_idx.txt"
video_path = f"{shared_root}/thumos14/raw_data/video"
videomae_pretrain = (
    f"{shared_root}/pretrained/"
    "vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
)

experiment_scope = dict(
    route="DUCA_TRUE_TIME_INDIRECT_CURRICULUM",
    parent_selector_revision="42dba3f90b37243e7965d18b6707e88e81bf7109",
    requested_k=window_size,
    effective_k=window_size,
    executed_k=window_size,
    selector="train_only_ASFormer_actionness_boundary_indirect_exact_k",
    selector_online_in_training=True,
    direct_index_prediction=False,
    dynamic_outer_k=False,
    repeats_dense_uniform_random=False,
    split_and_evaluator_unchanged=True,
    nms_unchanged=True,
    head_coordinate_contract="shared_dense_physical_for_paired_isolation",
    claim_status="implemented_pending_pre_run",
)

duca_curriculum = dict(
    total_epochs=60,
    phase_boundaries=(20, 40, 60),
    phase_successful_update_boundaries=(2000, 4000, 6000),
    phase_names=("semantic_warmup", "cosine_homotopy", "joint_training"),
    warmup_detector_sampling="exact_uniform_k384",
    warmup_selector_controls_acquisition=False,
    warmup_selector_detector_bridge=False,
    homotopy_rate="(1-alpha)*uniform_rate + alpha*semantic_rate",
    homotopy_alpha="0.5*(1-cos(pi*p))",
    homotopy_decoder="existing_deterministic_sorted_unique_exact_k",
    joint_selector_supervision=True,
    joint_bounded_detector_bridge=True,
    checkpoint_interval=5,
    checkpoint_selection="terminal_epoch_59_state_dict_ema",
    retain_latest_at_least=3,
    retain_all_five_epoch_milestones=True,
    resume_state=(
        "model",
        "optimizer",
        "scheduler",
        "amp_scaler",
        "epoch",
        "successful_update",
        "rng",
        "selector_schedule_step",
    ),
)

duca_variant_contract = dict(
    policy_alpha_warmup_fraction=1.0 / 3.0,
    policy_alpha_transition_fraction=1.0 / 3.0,
    policy_alpha_transition_shape="cosine",
    policy_alpha_zero_contract="hard_forward_exact_uniform",
    inference_policy_alpha=1.0,
    successful_optimizer_updates=expected_successful_optimizer_updates,
    schedule_step_checkpointed=True,
)

model = dict(
    frame_selector=dict(
        arm="protected_e2e_homotopy025",
        detector_bridge_gradient_scale=0.25,
        uniform_companion_fraction=0.0,
        homotopy_total_steps=expected_successful_optimizer_updates,
        homotopy_warmup_steps=2000,
        homotopy_transition_steps=2000,
    ),
    backbone=dict(custom=dict(pretrain=videomae_pretrain)),
)

dataset = dict(
    train=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=video_path,
    ),
    test=dict(
        ann_file=annotation_path,
        class_map=class_map,
        data_path=video_path,
    ),
)
evaluation = dict(ground_truth_filename=annotation_path)

workflow = dict(
    # Reuse the proven resumable/EMA/checkpoint engine. The paired route is
    # distinguished by immutable config and launch manifests, not by weakening
    # the existing formal training semantics.
    formal_protocol="duca_protected_physical_v1",
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=60,
    max_train_iters=None,
    disable_checkpoint=False,
    expected_successful_optimizer_updates=expected_successful_optimizer_updates,
    expected_successful_updates_per_epoch=expected_successful_updates_per_epoch,
    primary_checkpoint_epoch=59,
    primary_checkpoint_state_key="state_dict_ema",
    checkpoint_criterion="terminal_epoch_59_state_dict_ema",
    seal_eval_dataloaders_during_training=True,
    derive_train_loader_contract=True,
)
