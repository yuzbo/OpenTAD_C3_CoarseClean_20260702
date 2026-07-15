_base_ = ["./duca_direct_boundary_fixed384_13200_official_adatad_backend_full_train.py"]

import os


model = dict(
    frame_selector=dict(
        loss_weight_schedule=dict(
            detector_gradient=dict(warmup_steps=2, transition_steps=4),
        ),
    ),
)

workflow = dict(
    logging_interval=1,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=1,
    max_train_iters=10,
    disable_checkpoint=True,
    training_probe_json=os.environ.get(
        "DUCA_TRAINING_PROBE_JSON", "logs/duca_direct_p0_ddp_pilot.json"
    ),
)
