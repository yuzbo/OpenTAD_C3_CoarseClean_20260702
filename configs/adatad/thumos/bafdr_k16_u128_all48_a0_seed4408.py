_base_ = ["./continuous_roi_s2_v3_u128_a0_seed4408.py"]

seed = 4408
solver = dict(
    train=dict(batch_size=2, num_workers=2),
    val=dict(batch_size=2, num_workers=2),
    test=dict(batch_size=2, num_workers=2),
)

bafdr_protocol = dict(
    protocol="ZOOMTOKEN-BA-FDR-K16-FULLMATRIX-v001",
    arm="U128-ALL48-A0",
    seed=4408,
)
work_dir = f"exps/thumos/adatad/bafdr_k16_u128_all48_a0_seed4408"
