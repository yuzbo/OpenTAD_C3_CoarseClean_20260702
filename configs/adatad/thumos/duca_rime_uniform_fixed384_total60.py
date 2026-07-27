_base_ = ["./duca_protected_physical_exact_uniform_fixed384_official60.py"]

import os

from tools.bata.duca_cellcf_protocol import protocol_for_name


train_block_list = os.environ.get("DUCA_RIME_TRAIN_BLOCK_LIST", "").strip()
development_block_list = os.environ.get(
    "DUCA_RIME_DEVELOPMENT_BLOCK_LIST",
    "",
).strip()
if not train_block_list or not development_block_list:
    raise RuntimeError("U-fixed requires the frozen RIME train/development block lists")
protocol = protocol_for_name("official60")
fixed_budget = int(os.environ.get("DUCA_RIME_FIXED_BUDGET", "384"))
if fixed_budget not in {192, 384} or fixed_budget % 16:
    raise RuntimeError("U-fixed requires a registered 192/384 budget divisible by 16")
fixed_chunk_num = fixed_budget // 16

dataset = dict(
    train=dict(
        block_list=train_block_list,
    ),
    val=None,
    test=dict(
        subset_name="training",
        block_list=development_block_list,
        test_mode=True,
    ),
)

evaluation = dict(
    subset="training",
    blocked_videos=development_block_list,
)

# Keep the detector/projection capacity identical to the dynamic RIME arms.
# The RGB backbone still executes exactly K=384 frames; only the detector tail
# sees the common Kmax=512 canvas used by every Phase-3 arm.
model = dict(
    frame_selector=dict(
        budget=fixed_budget,
    ),
    backbone=dict(
        backbone=dict(total_frames=fixed_budget),
        custom=dict(
            pre_processing_pipeline=[
                dict(
                    type="Rearrange",
                    keys=["frames"],
                    ops="b n c (t1 t) h w -> (b t1) n c t h w",
                    t1=fixed_chunk_num,
                ),
            ],
            post_processing_pipeline=[
                dict(
                    type="Reduce",
                    keys=["feats"],
                    ops="b n c t h w -> b c t",
                    reduction="mean",
                ),
                dict(
                    type="Rearrange",
                    keys=["feats"],
                    ops="(b t1) c t -> b c (t1 t)",
                    t1=fixed_chunk_num,
                ),
                dict(type="Interpolate", keys=["feats"], size=fixed_budget),
            ],
        ),
    ),
    projection=dict(max_seq_len=512),
)

solver = dict(
    train=dict(batch_size=1, num_workers=2),
    val=dict(batch_size=1, num_workers=2),
    test=dict(batch_size=1, num_workers=2),
    static_graph=False,
    find_unused_parameters=True,
)

post_processing = dict(save_dict=True)

workflow = dict(
    formal_protocol="duca_rime_uniform_control_v1",
    training_profile=protocol.name,
    checkpoint_interval=protocol.checkpoint_interval,
    checkpoint_retention=1,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_eval_interval_anchor_epoch=9999,
    val_start_epoch=9999,
    end_epoch=protocol.end_epoch,
    formal_successful_update_contract=True,
    expected_train_batches_per_epoch=protocol.steps_per_epoch,
    expected_successful_optimizer_updates=protocol.expected_successful_optimizer_updates,
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
    arm="U-fixed",
    exact_uniform=True,
    exact_budget=fixed_budget,
    training_video_count=100,
    batch_size=1,
    expected_successful_updates=protocol.expected_successful_optimizer_updates,
)

work_dir = "exps/thumos/adatad/duca_rime_uniform_fixed384_total60"

del protocol
del fixed_budget
del fixed_chunk_num
