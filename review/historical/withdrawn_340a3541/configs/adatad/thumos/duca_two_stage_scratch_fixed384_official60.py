_base_ = ["./duca_two_stage_joint_fixed384_official60_base.py"]


duca_transition_only_contract = dict(
    route="DUCA_TWO_STAGE_SCRATCH_FIXED384_OFFICIAL60",
    frontend_initialization="none",
    coarse_probe_training="joint_from_random_initialization",
)

work_dir = "exps/thumos/adatad/duca_two_stage_scratch_fixed384_official60"
