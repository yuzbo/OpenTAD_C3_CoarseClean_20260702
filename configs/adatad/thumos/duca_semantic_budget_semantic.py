_base_ = ["./duca_semantic_budget_matched_common.py"]

import os

if os.environ.get("DUCA_WINDOW_BUDGET_ARM") != "semantic":
    raise ValueError("duca_semantic_budget_semantic.py requires DUCA_WINDOW_BUDGET_ARM=semantic")

work_dir = "exps/thumos/adatad/duca_semantic_budget_matched_semantic"

