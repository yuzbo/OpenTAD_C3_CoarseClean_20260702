"""H65-60 attribution: mature Stage-1 plus truncated historical cosine."""

_base_ = ["./duca_h65_60_stage2_am_rpch25.py"]


duca_sampling_rate_contract = dict(
    route="DUCA_H65_60_STAGE2_LONGCOSINE_H6000",
    stage="mature_stage1_then_truncated_historical_cosine_horizon",
)

scheduler = dict(
    mode="longcosine_h6000",
)

workflow = dict(
    training_profile="duca_h65_60_stage2_longcosine_h6000",
)

work_dir = "exps/thumos/adatad/duca_h65_60_stage2_longcosine_h6000"

