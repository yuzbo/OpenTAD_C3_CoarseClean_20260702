_base_ = ["./continuous_roi_s2_v3_g96_seed4409.py"]

seed = 4409
bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="G96",
    seed=4409,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_g96_seed4409"
