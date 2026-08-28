_base_ = ["./duca_semantic_budget_matched_common.py"]

import os

if os.environ.get("DUCA_WINDOW_BUDGET_ARM") != "fixed384":
    raise ValueError("duca_semantic_budget_fixed384.py requires DUCA_WINDOW_BUDGET_ARM=fixed384")

work_dir = "exps/thumos/adatad/duca_semantic_budget_matched_fixed384"

