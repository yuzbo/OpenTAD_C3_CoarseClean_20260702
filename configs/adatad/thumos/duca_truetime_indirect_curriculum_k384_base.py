"""Frozen paired indirect-selector curriculum (K=384, 20/20/20 epochs).

This base deliberately reuses the historical ASFormer value-transport ledger
loader and the official AdaTAD detector/evaluator.  Arm configs only change
the temporal coordinate contract at the heavy path.
"""
_base_ = ["./c3_official_asformer_delta_ledger_original_adatad_full_train.py"]

import os

experiment_scope = dict(
    route="DUCA_TRUE_TIME_INDIRECT_CURRICULUM",
    parent_selector_revision="42dba3f90b37243e7965d18b6707e88e81bf7109",
    requested_k=384,
    effective_k=384,
    executed_k=384,
    selector="ASFormer actionness_boundary_indirect_exact_k",
    dynamic_outer_k=False,
    repeats_dense_uniform_random=False,
    split_and_evaluator_unchanged=True,
    nms_unchanged=True,
    claim_status="designed",
)

duca_curriculum = dict(
    total_epochs=60,
    phase_boundaries=(20, 40, 60),
    phase_names=("semantic_warmup", "cosine_homotopy", "joint_training"),
    warmup_detector_sampling="uniform",
    warmup_selector_controls_acquisition=False,
    warmup_selector_detector_bridge=False,
    homotopy_rate="(1-alpha)*uniform_rate + alpha*semantic_rate",
    homotopy_alpha="0.5*(1-cos(pi*p))",
    homotopy_p="(epoch-20)/20",
    homotopy_decoder="existing_deterministic_sorted_unique_exact_k",
    joint_selector_supervision=True,
    joint_bounded_detector_bridge=True,
    checkpoint_interval=5,
    retain_latest=3,
    retain_milestones=True,
    resume_state=("model", "optimizer", "scheduler", "amp_scaler", "epoch", "update", "rng", "dataloader", "duca_curriculum"),
)

window_size = 384
dense_window_size = 768
workflow = dict(
    logging_interval=50,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=999,
    end_epoch=60,
    max_train_iters=6000,
    disable_checkpoint=False,
)

solver = dict(train=dict(batch_size=2, num_workers=2), val=dict(batch_size=2, num_workers=2), test=dict(batch_size=2, num_workers=2), amp=True, ema=True)

