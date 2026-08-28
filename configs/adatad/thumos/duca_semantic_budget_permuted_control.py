_base_ = ["./duca_semantic_budget_matched_common.py"]

import os

if os.environ.get("DUCA_WINDOW_BUDGET_ARM") != "permuted_control":
    raise ValueError(
        "duca_semantic_budget_permuted_control.py requires "
        "DUCA_WINDOW_BUDGET_ARM=permuted_control"
    )

work_dir = "exps/thumos/adatad/duca_semantic_budget_matched_permuted_control"

