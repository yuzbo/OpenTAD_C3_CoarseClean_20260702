_base_ = ["./duca_cellcf_exact_uniform_fixed384_official_adatad_backend_full_train.py"]

import os


workflow = dict(
    formal_protocol="duca_cellcf_pilot_v1",
    formal_successful_update_contract=False,
    logging_interval=1,
    checkpoint_interval=5,
    val_loss_interval=-1,
    val_eval_interval=-1,
    val_start_epoch=9999,
    end_epoch=1,
    max_train_iters=10,
    force_amp_overflow_attempts=1,
    disable_checkpoint=True,
    require_training_probe_context=True,
    training_probe_json=os.environ.get(
        "DUCA_TRAINING_PROBE_JSON", "logs/duca_cellcf_uniform_p0_ddp_pilot.json"
    ),
)
