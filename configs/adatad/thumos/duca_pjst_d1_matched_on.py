_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
pjst_derivative_only = True
model = dict(backbone=dict(custom=dict(pjst_derivative_only=True)))
single_clock_admission = False
work_dir = "exps/thumos/adatad/DUCA_PJST_DERIVATIVE_CAUSAL_FREEZE-v002/pjst_d1_on"
