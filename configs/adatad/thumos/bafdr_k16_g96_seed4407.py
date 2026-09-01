_base_ = ["./continuous_roi_s2_v3_g96_seed4407.py"]

seed = 4407
bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="G96",
    seed=4407,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_g96_seed4407"
