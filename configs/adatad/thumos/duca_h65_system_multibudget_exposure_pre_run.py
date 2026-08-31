_base_ = ["./duca_h65_system_multibudget_exposure_candidate.py"]

import os


probe_path = os.environ.get("DUCA_PRE_RUN_PROBE_JSON", "")
if not probe_path:
    raise ValueError("DUCA_PRE_RUN_PROBE_JSON is required")


# This is a four-update execution smoke, not a third scientific arm.  The
# successful-update course begins K384, K256, K384, K512 for seed 3407, so one
# short epoch reaches every authorized heavy-execution bucket without opening
# the held-out loader.
workflow = dict(
    formal_successful_update_contract=False,
    training_profile="duca_h65_system_multibudget_exposure_pre_run",
    end_epoch=1,
    max_train_iters=4,
    checkpoint_interval=1,
    training_probe_json=probe_path,
    training_update_audit_json=os.environ.get("DUCA_PRE_RUN_UPDATE_AUDIT_JSON", ""),
)


work_dir = os.environ.get(
    "DUCA_PRE_RUN_WORK_DIR",
    "exps/thumos/adatad/duca_h65_system_multibudget_exposure_pre_run",
)
