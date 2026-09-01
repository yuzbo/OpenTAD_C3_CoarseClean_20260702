_base_ = ["./duca_protected_e2e_fixed384_official60.py"]

import os


def _required_path(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for train-only frontend pretraining")
    return value


duca_frontend_end_epoch = 20
duca_frontend_transition_ramp_steps = 100
duca_frontend_train_block_list = _required_path(
    "DUCA_FRONTEND_TRAIN_BLOCK_LIST"
)


duca_transition_only_contract = dict(
    _delete_=True,
    route="DUCA_FRONTEND_PRETRAIN_FIXED384",
    stage="train_only_frontend_weight_and_duration_selection",
    task="offline_temporal_action_detection",
    detector_executed=False,
    detector_trained=False,
    test_subset_consumed=False,
    selector_supervision=[
        "binary_actionness",
        "state_transition_distribution",
        "transition_boundary_coverage",
    ],
    exact_budget=384,
    dense_window_size=768,
    max_unselected_hole_dense_candidates=2,
    checkpoint_interval=5,
    maximum_epoch_count=duca_frontend_end_epoch,
    paper_metric_claim_allowed=False,
    strict_explicit_loss_inventory=True,
    actionness_reduction="class_balanced_positive_negative_means",
    transition_supervision_updates_coarse_representation=False,
    spatial_stem_normalization="groupnorm_padding_invariant",
    official_asformer_source_normalized_lf_sha256=(
        "e075ee4825a201cfe324d5fbfb1332c0800f532e85b9d3809f6ca5180381c600"
    ),
)


model = dict(
    selector_train_only=True,
    selector_train_only_skip_detector=True,
    frame_selector=dict(
        detector_gradient_mode="none",
        counterfactual_utility_distillation_weight=0.0,
        require_counterfactual_utility_teacher=False,
        training_uniform_companion_fraction=0.0,
        inference_policy_alpha=1.0,
        auxiliary_hidden_gradient_scale=0.0,
        actionness_loss_mode="class_balanced_mean",
        strict_loss_contract=True,
        actionness_source_cfg=dict(spatial_norm="groupnorm"),
        loss_weights=dict(
            _delete_=True,
            detector=0.0,
            actionness=1.0,
            budget=0.0,
            boundary=0.0,
            hole=0.0,
            max_gap_hole=0.0,
            redundancy=0.0,
            radius=0.0,
            entropy=0.0,
            teacher=0.0,
            detector_utility=0.0,
            start=0.0,
            end=0.0,
            context=0.0,
            lagrangian_budget=0.0,
            marginal_monotonic=0.0,
            hard_budget_cap=0.0,
            transition=0.10,
            transition_boundary=16.0,
        ),
        loss_weight_schedule=dict(
            _delete_=True,
            type="progressive_joint",
            shape="linear",
            warmup_steps=0,
            transition_steps=duca_frontend_transition_ramp_steps,
            actionness=dict(start=1.0, end=1.0),
            transition=dict(
                start=0.0,
                end=0.10,
                warmup_steps=0,
                transition_steps=duca_frontend_transition_ramp_steps,
            ),
            transition_boundary=dict(
                start=0.0,
                end=16.0,
                warmup_steps=0,
                transition_steps=duca_frontend_transition_ramp_steps,
            ),
            policy_alpha=dict(start=1.0, end=1.0),
            detector_gradient=dict(start=0.0, end=0.0),
        ),
    ),
)


dataset = dict(
    train=dict(block_list=duca_frontend_train_block_list),
    val=None,
    test=None,
)


scheduler = dict(
    type="LinearWarmupCosineAnnealingLR",
    warmup_epoch=2,
    max_epoch=duca_frontend_end_epoch,
)


solver = dict(
    static_graph=False,
    find_unused_parameters=True,
    clip_grad_norm=-1.0,
)


# Frontend-only training freezes and skips the complete AdaTAD backbone.  Drop
# the inherited backbone optimizer group so it cannot leak into torch AdamW.
optimizer = dict(
    _delete_=True,
    type="AdamW",
    lr=1e-4,
    weight_decay=0.05,
    paramwise=True,
)


workflow = dict(
    _delete_=True,
    formal_protocol="",
    training_profile="frontend_pretrain20_train_only_holdout",
    logging_interval=20,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=duca_frontend_end_epoch,
    seal_eval_dataloaders_during_training=True,
    max_amp_retries_per_batch=8,
    fail_on_amp_replay_exhaustion=True,
    require_finite_train_loss=True,
)


work_dir = "exps/thumos/adatad/duca_frontend_pretrain_fixed384_base"
