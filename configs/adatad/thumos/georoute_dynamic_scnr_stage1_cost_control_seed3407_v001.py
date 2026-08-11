"""ZoomToken steady-cost control: seed-3407 SCNR without residual centering."""

_base_ = ["./georoute_dynamic_scnr_stage1_base.py"]

model = dict(
    backbone=dict(
        custom=dict(
            georoute_branch_calibration_mode="none",
            georoute_diagnostic_telemetry_enabled=False,
            georoute_role_calibration_telemetry_enabled=False,
        )
    )
)

georoute_protocol = dict(branch_calibration="none")

zoomtoken_scnr_steady_cost = dict(
    schema_version="zoomtoken_scnr_steady_cost_config_v001",
    study_id="ZT_SCNR_STEADY_COST_V001",
    arm="control",
    training_seed=3407,
    calibration_mode="none",
    exact_window_budget=24576,
    physical_windows=136,
    split="Gate/development",
    treatment_from_cli_allowed=False,
    training_or_resume_allowed=False,
    metric_evaluation_allowed=False,
    held_out_test_allowed=False,
)
