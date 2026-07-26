_base_ = ["./c3_truetime_joint_selector_c3_adatad_smoke.py"]


truetime_joint_selector_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="PRECHECK_ONLY validator plus selector_grad_norm/geometry proof required before claims",
)
