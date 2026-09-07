_base_ = ["./c3_uniform_sparse_384_ledger_adatad_full_train.py"]


c3_uniform_sparse_384_full_train_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="strict exact uniform sparse 384 ledger baseline precheck plus explicit full-train unlock",
)
