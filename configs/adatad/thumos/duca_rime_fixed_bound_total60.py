_base_ = ["./duca_rime_physical_total60_base.py"]

model = dict(
    frame_selector=dict(
        rime_arm="fixed_bound",
        fixed_budget=384,
        require_frozen_protocol=False,
    ),
)

duca_rime_variant = dict(
    arm="F-bound",
    dynamic_budget=False,
    fixed_budget=384,
    hard_utility_rank=True,
    physical_exact_k=True,
)

work_dir = "exps/thumos/adatad/duca_rime_fixed_bound_total60"
