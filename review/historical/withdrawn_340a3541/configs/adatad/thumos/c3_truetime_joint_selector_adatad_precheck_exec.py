_base_ = ["./c3_truetime_joint_selector_adatad_precheck.py"]


truetime_joint_selector_gate = dict(
    launch_gate_passed=True,
    reviewed_execution_config=True,
    reviewed_execution_reason="DUCA Stage3 precheck runner requires real ActionFormer selector-gradient proof before PRECHECK_ONLY=0",
)
