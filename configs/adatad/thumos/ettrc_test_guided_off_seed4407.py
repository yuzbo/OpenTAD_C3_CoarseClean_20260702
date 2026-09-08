_base_ = ["./ettrc_test_guided_on_seed4407.py"]
model = dict(backbone=dict(backbone=dict(enable_taylor=False)))
work_dir = "exps/thumos/adatad/ettrc_test_guided_off_seed4407"
