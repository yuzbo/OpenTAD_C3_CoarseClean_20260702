_base_ = ["./c3_detector_aware_ledger_adatad_full_train.py"]


c3_detector_aware_full_train_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="Stage-2 detector-aware offline selector strict ledger precheck plus explicit full-train unlock",
)
