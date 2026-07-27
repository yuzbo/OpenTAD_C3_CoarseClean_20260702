from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_phase1_launcher_requires_all_clean_controls_and_hashes():
    text = (ROOT / "scripts" / "run_duca_rime_phase1_gate.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_CODE_GATE_RECEIPT_SHA256" in text
    for control in (
        "released_dense",
        "local_dense",
        "uniform_k384",
        "uniform_k192",
        "wrapper_parity",
        "q_to_t_before_nms",
        "no_probe_uniform_cost",
        "probe_uniform_cost",
    ):
        assert control in text
    assert "duca_rime_stage_contract.py phase1" in text


def test_phase2_launcher_is_slurm_hash_bound_and_seals_o1_o4():
    text = (ROOT / "scripts" / "run_duca_rime_phase2_gates.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "sha256sum -c" in text
    assert "commit=${DUCA_RIME_EXPECTED_COMMIT}" in text
    for gate in (" o1 ", " o2 ", " o3 ", " o4 ", " freeze "):
        assert gate in text
    assert "duca_rime_stage_contract.py phase2" in text
    assert "official_final_subset_consumed" in text


def test_phase4_is_not_submitted_from_phase2_launcher():
    text = (ROOT / "scripts" / "run_duca_rime_phase2_gates.sh").read_text(
        encoding="utf-8"
    )
    assert "sbatch" not in text
    assert "authorize-phase4" not in text


def test_phase3_training_launcher_enforces_exact_git_and_6000_updates():
    text = (ROOT / "scripts" / "run_duca_rime_phase3_train_arm.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ -d "${DUCA_RIME_REPO_ROOT}/.git" ]]' in text
    assert "formal training requires a complete exact Git worktree" in text
    assert "phase3_training_authorized" in text
    assert "expected_successful_optimizer_updates" in text
    assert "!= 6000" in text
    assert "successful_optimizer_updates" in text
    assert "U-same-K is evaluation-only" in text
    assert "--rdzv-backend=c10d --rdzv-endpoint=localhost:0" in text
    assert '--rdzv-id="${SLURM_JOB_ID}"' in text


def test_phase4_training_launcher_is_authorization_and_cell_bound():
    text = (ROOT / "scripts" / "run_duca_rime_phase4_train_cell.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_PHASE4_AUTHORIZATION_SHA256" in text
    assert "authorization does not cover this cell" in text
    assert "U-fixed-TriDet" in text
    assert "RIME-full-TriDet" in text
    assert "successful_optimizer_updates" in text
    assert "compact_duca_rime_checkpoint.py" in text
    assert "--remove-source" in text


def test_rime_evaluation_launcher_separates_development_and_official_final():
    text = (ROOT / "scripts" / "run_duca_rime_evaluate_arm.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ "${DUCA_RIME_EVAL_PHASE}" == 3' in text
    assert 'expected_subset = "training" if phase == 3 else "validation"' in text
    assert "Phase-3 evaluation lacks the development block list" in text
    assert "Phase-4 evaluation is not the full official validation set" in text
    assert "finalize_duca_rime_inference_ledger.py" in text
    assert "--expected-checkpoint-epoch 59" in text
    assert "--checkpoint-state-key state_dict_ema" in text
    assert "detector_selector_train" in text
    assert "certification_development" in text
    assert "DUCA_RIME_SPLIT_MANIFEST_SHA256" in text


def test_rime_cost_launcher_profiles_paired_full_stack_on_one_slurm_allocation():
    text = (ROOT / "scripts" / "run_duca_rime_cost_cell.sh").read_text(
        encoding="utf-8"
    )
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "profile_duca_full_stack_cost.py" in text
    assert text.count("profile_rime") >= 3
    assert "--profile-session-id" in text
    assert "--profile-pair-id" in text
    assert "--rime-training-receipt-sha256" in text
    assert "finalize_duca_rime_cost.py" in text


def test_phase3_and_phase4_sealers_keep_official_and_auxiliary_bootstrap_distinct():
    phase3 = (ROOT / "scripts" / "run_duca_rime_phase3_seal.sh").read_text(
        encoding="utf-8"
    )
    phase4 = (ROOT / "scripts" / "run_duca_rime_phase4_seal_cell.sh").read_text(
        encoding="utf-8"
    )
    matrix = (ROOT / "scripts" / "run_duca_rime_phase4_seal_matrix.sh").read_text(
        encoding="utf-8"
    )
    assert "finalize_duca_rime_phase3_arm.py" in phase3
    assert "authorize-phase4" in phase3
    assert "bootstrap_duca_rime_phase4.py" in phase4
    assert "finalize_duca_rime_phase4_cell.py" in phase4
    assert "duca_rime_stage_contract.py phase4" in matrix


def test_phase0_and_phase2_record_launchers_only_build_from_supplied_sources():
    phase0 = (ROOT / "scripts" / "run_duca_rime_phase0_measurements.sh").read_text(
        encoding="utf-8"
    )
    phase2 = (ROOT / "scripts" / "run_duca_rime_phase2_record_build.sh").read_text(
        encoding="utf-8"
    )
    assert "DUCA_RIME_PHASE0_SOURCE_MANIFEST" in phase0
    assert "build_duca_rime_gate_records.py phase0" in phase0
    assert "DUCA_RIME_O1_SOURCE_MANIFEST" in phase2
    assert "DUCA_RIME_O3_SOURCE_JSONL" in phase2
    assert "DUCA_RIME_PRICE_SOURCE_JSONL" in phase2


def test_phase2_baseline_launcher_is_train_role_and_checkpoint_bound():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase2_baseline_eval.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_PHASE2_SPLIT_ROLE" in text
    assert "DUCA_RIME_PHASE2_BASELINE_CHECKPOINT_SHA256" in text
    assert "DUCA_RIME_PHASE2_EVAL_BLOCK_LIST" in text
    assert "duca_rime_uniform_phase2_baseline" not in text
    assert "--phase 2" in text
    assert "--split-role" in text
    assert "--rdzv-endpoint=localhost:0" in text


def test_phase2_mixed_k_training_is_phase1_bound_and_target_free():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase2_mixed_k_train.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_PHASE1_RECEIPT_SHA256" in text
    assert "duca_rime_phase2_mixed_k_training_exposure_v1" in text
    assert "U-mixed-K" in text
    assert "(192, 256, 384, 512)" in text
    assert "(8, 12, 16, 24)" in text
    assert "expected_successful_optimizer_updates" in text
    assert "DUCA_RIME_TARGETS_JSONL" in text
    assert "unset" in text
    assert "--rdzv-backend=c10d --rdzv-endpoint=localhost:0" in text
    assert '--rdzv-id="${SLURM_JOB_ID}"' in text
    assert "compact_duca_rime_checkpoint.py" in text


def test_phase2_mixed_k_evaluation_is_checkpoint_and_exact_k_ledger_bound():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase2_mixed_k_eval.sh"
    ).read_text(encoding="utf-8")
    assert "duca_rime_phase2_mixed_k_training_receipt_v1" in text
    assert "detector_training_exposure" in text
    assert "mixed_k_registered_panel" in text
    assert "--expected-arm uniform_mixed_k" in text
    assert "--phase 2" in text
    assert "--split-role" in text
    assert "--rdzv-endpoint=localhost:0" in text
    assert "detector_selector_train" in text


def test_rime_development_configs_bind_dataset_and_official_evaluator_to_same_role():
    for name in (
        "duca_rime_uniform_phase2_baseline.py",
        "duca_rime_uniform_mixed_k_total60.py",
        "duca_rime_physical_total60_base.py",
        "duca_rime_uniform_fixed384_total60.py",
    ):
        text = (
            ROOT / "configs" / "adatad" / "thumos" / name
        ).read_text(encoding="utf-8")
        assert "blocked_videos=" in text


def test_rime_official_final_configs_explicitly_clear_development_block_list():
    for name in (
        "duca_rime_full_formal_validation.py",
        "duca_rime_full_tridet_formal_validation.py",
        "duca_rime_uniform_fixed_formal_validation.py",
        "duca_rime_uniform_fixed_tridet_formal_validation.py",
        "duca_rime_uniform_same_k_formal_validation.py",
        "duca_rime_uniform_same_k_tridet_formal_validation.py",
    ):
        text = (
            ROOT / "configs" / "adatad" / "thumos" / name
        ).read_text(encoding="utf-8")
        assert "blocked_videos=None" in text
