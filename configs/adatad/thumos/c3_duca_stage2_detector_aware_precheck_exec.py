_base_ = ["./c3_duca_stage2_detector_aware_precheck.py"]


c3_detector_aware_full_train_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="DUCA Stage2 runner requires validate_duca_stage23_precheck.py PASS before PRECHECK_ONLY=0",
)
