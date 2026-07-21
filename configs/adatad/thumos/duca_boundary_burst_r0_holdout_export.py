_base_ = ["./duca_allocation_ceiling_training_windows.py"]

import os


holdout_block_list = os.environ.get("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST", "")
if not holdout_block_list:
    raise ValueError("DUCA_FRONTEND_HOLDOUT_BLOCK_LIST is required for R0")


allocation_ceiling_training_window_contract = dict(
    purpose="exhaustive_train_holdout_selected_axis_oracle_map",
    action_intersecting_windows_only=False,
    includes_background_windows=True,
    validation_consumed=False,
    test_subset_consumed=False,
)


dataset = dict(
    train=dict(
        block_list=holdout_block_list,
        include_background_windows=True,
        ioa_thresh=1.0e-8,
    ),
)


work_dir = "exps/thumos/adatad/duca_boundary_burst_r0_holdout_export"
