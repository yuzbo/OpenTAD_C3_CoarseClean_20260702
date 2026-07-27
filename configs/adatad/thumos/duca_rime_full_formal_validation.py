_base_ = ["./duca_rime_full_total60.py"]

dataset = dict(
    test=dict(
        subset_name="validation",
        block_list=None,
        test_mode=True,
    ),
)

evaluation = dict(
    subset="validation",
    blocked_videos=None,
)

duca_rime_formal_scope = dict(
    phase=4,
    split="official_final_evaluation",
    method_and_thresholds_frozen=True,
    development_seed_excluded=True,
)

work_dir = "exps/thumos/adatad/duca_rime_full_formal_validation"
