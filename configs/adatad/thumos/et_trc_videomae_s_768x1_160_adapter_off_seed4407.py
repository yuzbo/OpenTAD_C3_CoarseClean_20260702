_base_ = ["./et_trc_videomae_s_768x1_160_adapter_seed4407.py"]
model = dict(backbone=dict(backbone=dict(enable_taylor=False)))
work_dir = "exps/thumos/adatad/et_trc_videomae_s_768x1_160_adapter_off_seed4407"
