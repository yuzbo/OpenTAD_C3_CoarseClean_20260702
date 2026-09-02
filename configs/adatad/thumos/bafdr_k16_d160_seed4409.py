_base_ = ["./continuous_roi_s2_v3_d160_seed4409.py"]

seed = 4409
solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
)

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="D160",
    seed=4409,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_d160_seed4409"
