_base_ = ["./c3_detector_aware_ledger_adatad_full_train.py"]


duca_stage23_runner_precheck = dict(
    route="DUCA_STAGE2_DETECTOR_UTILITY_PRECHECK",
    stage="Stage2 dense AdaTAD teacher utility",
    validates=[
        "dense_teacher_checkpoint_fail_fast",
        "train_only_utility_export_provenance",
        "signed_detector_utility_v1",
        "train_global_dynamic_gain_calibration",
        "no_val_test_teacher_gt_cache_leakage",
        "strict_value_transport_ledgers",
    ],
    full_run_gate="DUCA_STAGE2_PRECHECK_PASS required before PRECHECK_ONLY=0",
    edits_model_core=False,
)
