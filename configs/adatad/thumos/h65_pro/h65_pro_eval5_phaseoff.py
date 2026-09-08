_base_ = ["./h65_pro_eval5_phaseon.py"]

h65_pro_experiment_id = "TEST-PHASEOFF"
h65_pro_factor_policy = dict(phase=False)
model = dict(frame_selector=dict(acquisition_policy="budget_calibrated_sampling_rate"))
work_dir = "exps/thumos/h65_tia_test_guided/phaseoff"
