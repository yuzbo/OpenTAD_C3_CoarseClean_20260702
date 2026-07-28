_base_ = ["./duca_rime_uniform_phase2_baseline.py"]

import os


fixed_budget = int(os.environ.get("DUCA_RIME_FIXED_BUDGET", "384"))
if fixed_budget not in {192, 384}:
    raise RuntimeError("Phase-1 exact-uniform control requires K=192 or K=384")
variant = f"uniform_k{fixed_budget}"

duca_rime_baseline_contract = dict(
    phase=1,
    variant=variant,
    position_policy="exact_uniform",
    target_mean_cost=float(fixed_budget),
    detector_backend="ActionFormer",
    padded_to_kmax=False,
    uses_official_final=False,
    training_identity_required=False,
    checkpoint_compatibility_mode=(
        "historical_uniform_score_net_unused_exact_whitelist_v1"
    ),
    claim_scope="phase1_exact_uniform_execution_and_localization_control_only",
)

work_dir = (
    f"exps/thumos/adatad/duca_rime_uniform_phase1_control_k{fixed_budget}"
)

del fixed_budget
del variant
