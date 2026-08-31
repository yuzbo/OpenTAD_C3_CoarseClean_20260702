_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]

import os


def _required(name):
    value = os.environ.get(name, "")
    if not value:
        raise ValueError(f"{name} is required for the H65 multi-budget matched experiment")
    return value


seed = int(_required("DUCA_EXPERIMENT_SEED"))
if seed not in {3407, 3408, 3409}:
    raise ValueError("DUCA_EXPERIMENT_SEED must be 3407, 3408, or 3409")
duca_videomae_pretrain = _required("DUCA_VIDEOMAE_PRETRAIN")


duca_multibudget_exposure_contract = dict(
    experiment="h65_system_multibudget_exposure_v1",
    arm="control_k384_exposure",
    scientific_variable="stage2_budget_exposure_distribution",
    training_videos=200,
    held_out_videos=211,
    successful_optimizer_updates=6000,
    terminal_state_key="state_dict_ema",
    terminal_update=6000,
    held_out_checkpoint_selection=False,
    detector_loss_nms_evaluator_unchanged=True,
    prediction_sealing_required=True,
)


workflow = dict(
    training_profile="duca_h65_system_multibudget_exposure_control",
    # This Stage-2 experiment is not one of the historical P0 variants.  Keep
    # the inherited fail-closed update audit without routing through the P0
    # provenance binder.
    formal_successful_update_contract=False,
    seal_eval_dataloaders_during_training=True,
    expected_train_batches_per_epoch=100,
    expected_successful_optimizer_updates=6000,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    val_eval_interval_anchor_epoch=9999,
    intermediate_validation_role="learning_curve_only",
    intermediate_validation_selects_checkpoint=False,
    training_update_audit_json=os.environ.get("DUCA_STAGE2_UPDATE_AUDIT_JSON", ""),
)


# The official 211-video held-out population is opened only after all six
# training units and all nine prediction views have been sealed.  The test
# split remains available to the separate post-training inference entrypoint,
# but tools/train.py must not construct either held-out loader during training.
dataset = dict(val=None)


# Bind the canonical artifact explicitly so a remote launch never depends on
# the submit directory resolving the inherited relative pretrain path.
model = dict(
    backbone=dict(
        custom=dict(pretrain=duca_videomae_pretrain),
    ),
)


work_dir = f"exps/thumos/adatad/duca_h65_system_multibudget_exposure_control_seed{seed}"
