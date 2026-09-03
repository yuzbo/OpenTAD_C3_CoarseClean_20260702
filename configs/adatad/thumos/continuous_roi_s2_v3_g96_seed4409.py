_base_ = ["./continuous_roi_s2_v3_g96_seed4407.py"]

seed = 4409
continuous_roi_s2_v3_full200_compute = dict(seed=seed)
continuous_roi_d2s_v3_full200_compute = dict(seed=seed)
continuous_roi_patad_v3_full200_compute = dict(seed=seed)
work_dir = f"exps/thumos/adatad/continuous_roi_s2_v3_g96_seed{seed}"
