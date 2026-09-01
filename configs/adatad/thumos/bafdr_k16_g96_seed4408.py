_base_ = ["./continuous_roi_s2_v3_g96_seed4408.py"]

seed = 4408
bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="G96",
    seed=4408,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_g96_seed4408"
