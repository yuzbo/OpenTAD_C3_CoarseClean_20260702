_base_ = ["./duca_sampling_rate_curriculum_stage2_joint384.py"]
model = dict(frame_selector=dict(loss_weight_schedule=dict(policy_alpha=dict(start=0.0, end=0.0), detector_gradient=dict(start=0.0, end=0.0), detector_contribution=dict(start=0.0, end=0.0), asformer_adapt=dict(start=0.0, end=0.0))), backbone=dict(custom=dict(pjst_derivative_only=False)))
work_dir = "exps/thumos/adatad/duca_pjst_d1_matched_off"
