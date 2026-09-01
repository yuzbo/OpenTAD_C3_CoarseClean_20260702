_base_ = ["./c3_gas_vt_ledger_adatad_full_train.py"]


c3_gas_vt_ledger_full_train_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="strict GAS-VT ledger path precheck plus explicit full-train unlock",
)

