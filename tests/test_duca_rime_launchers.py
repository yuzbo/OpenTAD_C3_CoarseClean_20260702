from pathlib import Path
import subprocess
import sys


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


def test_phase1_cost_controls_require_one_shared_checkpoint_identity():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase1_cost_controls.sh"
    ).read_text(encoding="utf-8")
    assert "paired cost controls must use the same checkpoint bytes" in text
    assert "paired cost controls must use the same trained commit" in text
    assert "shared_checkpoint_sha256=" in text
    assert "shared_trained_commit=" in text


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
    assert "python -m tools.bata.compact_duca_rime_checkpoint" in text
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
    assert "U-same-K" in text
    assert "DUCA_RIME_REPLAY_JSONL" in text
    assert "without_replay" in text
    assert "with_replay" in text
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
    assert "python -m tools.bata.compact_duca_rime_checkpoint" in text


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


def test_phase2_evidence_pipeline_builds_all_four_gates_from_one_checkpoint():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase2_evidence_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "192 256 384 512" in text
    assert "run_duca_rime_phase2_mixed_k_eval.sh" in text
    assert "run_duca_rime_phase2_counterfactual_measurements.sh" in text
    assert "run_duca_rime_phase2_crossfit_producer.sh" in text
    assert "run_duca_rime_phase2_o2_panel.sh" in text
    assert "run_duca_rime_phase2_gates.sh" in text
    assert "pipeline_receipt.json" in text


def test_phase3_and_phase4_controllers_preserve_fail_closed_dependencies():
    phase3 = (
        ROOT / "scripts" / "run_duca_rime_phase3_submit_controller.sh"
    ).read_text(encoding="utf-8")
    phase4 = (
        ROOT / "scripts" / "run_duca_rime_phase4_submit_controller.sh"
    ).read_text(encoding="utf-8")
    assert "run_duca_rime_phase3_asset_producer.sh" in phase3
    assert 'dependency="afterok:${phase3_seal_job}"' in phase3
    assert "run_duca_rime_phase4_submit_controller.sh" in phase3
    assert 'phase4_submission_enabled="${DUCA_RIME_ENABLE_PHASE4:-0}"' in phase3
    assert 'if [[ "${phase4_submission_enabled}" == 1 ]]' in phase3
    assert '"phase4_submission_enabled": sys.argv[6] == "1"' in phase3
    assert "phase4_authorization.json" in phase4
    assert "submit_duca_rime_phase4_matrix.sh" in phase4
    assert "DUCA_RIME_SUBMIT_CONTROLLER=1" in phase3
    assert "DUCA_RIME_SUBMIT_CONTROLLER=1" in phase4
    assert (
        'sha_var="DUCA_RIME_DENSE_CHECKPOINT_EVIDENCE_SHA256_${backend}"'
        in phase3
    )
    assert '${evidence_var}_SHA256' not in phase3


def test_phase1_evidence_pipeline_uses_real_controls_before_sealing():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase1_evidence_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert text.count("run_duca_rime_phase1_dense_eval.sh") >= 2
    assert "for budget in 384 192" in text
    assert "run_duca_protected_physical_full_model_gate_gpu1.sh" in text
    assert "run_duca_rime_phase1_cost_controls.sh" in text
    assert "run_duca_rime_phase1_seal.sh" in text
    assert "bash scripts/run_duca_protected_physical_full_model_gate_gpu1.sh" in text
    assert "phase1_receipt_sha256" in text


def test_phase2_pipeline_trains_mixed_k_before_building_evidence():
    text = (
        ROOT
        / "scripts"
        / "run_duca_rime_phase2_train_and_evidence_pipeline.sh"
    ).read_text(encoding="utf-8")
    train = text.index("run_duca_rime_phase2_mixed_k_train.sh")
    evidence = text.index("run_duca_rime_phase2_evidence_pipeline.sh")
    assert train < evidence
    assert "create_duca_rime_training_exposure.py" in text
    assert "phase2_authorized" in text
    assert "terminal_ema.pth" in text


def test_phase1_uniform_launcher_requires_protocol_bound_explicit_budget_truth():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase1_uniform_eval.sh"
    ).read_text(encoding="utf-8")
    assert "DUCA_PROTECTED_PROTOCOL_MANIFEST_JSON" in text
    assert "DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256" in text
    assert "python -m tools.bata.finalize_duca_rime_inference_ledger" in text
    assert "--expected-protocol-sha256" in text
    assert "--require-explicit-budget-truth" in text
    pipeline = (
        ROOT / "scripts" / "run_duca_rime_phase1_evidence_pipeline.sh"
    ).read_text(encoding="utf-8")
    protocol_export = pipeline.index(
        'export DUCA_PROTECTED_PROTOCOL_MANIFEST_SHA256='
    )
    uniform_loop = pipeline.index("for budget in 384 192")
    assert protocol_export < uniform_loop


def test_four_phase_submitter_records_and_releases_a_fail_closed_dag():
    text = (
        ROOT / "scripts" / "submit_duca_rime_four_phase_dag.sh"
    ).read_text(encoding="utf-8")
    assert text.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
    assert "--hold" in text
    assert 'dependency_args=(--dependency="afterok:${dependency}")' in text
    assert '--wrap="exec /bin/bash -lc' in text
    assert "dense_actionformer_job" in text
    assert "dense_tridet_job" in text
    assert "phase3_dependency" in text
    assert "submission_manifest.json" in text
    assert "scontrol release" in text
    assert 'DUCA_RIME_DENSE_RECOVERY_MODE:-fresh_train' in text
    assert "run_duca_rime_dense_salvage.sh" in text
    assert '"source_jobs_remain_failed"' in text
    assert '"failed_transaction_mutated": False' in text
    assert '"phase4": None' in text
    assert '"phase4_submission_enabled": False' in text
    assert '"official_final_sealed": True' in text
    assert "this recovery DAG keeps official-final Phase 4 sealed" in text
    assert 'DUCA_RIME_DECODER_FAMILY}" == "weak_overlap"' in text
    assert 'DUCA_RIME_O4_MAX_BRIER}" == "0.25"' in text
    assert '"frozen_protocol_inputs"' in text


def test_phase2_crossfit_producer_rejects_surrogate_and_builds_all_targets():
    text = (
        ROOT / "scripts" / "run_duca_rime_phase2_crossfit_producer.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_COUNTERFACTUAL_MEASUREMENTS_SHA256" in text
    assert "produce_duca_rime_crossfit_records.py" in text
    assert "build_duca_rime_gate_records.py" in text
    assert "build_duca_rime_training_targets.py" in text


def test_phase2_counterfactual_and_o2_launchers_use_actual_runtime_detector_loss():
    counterfactual = (
        ROOT
        / "scripts"
        / "run_duca_rime_phase2_counterfactual_measurements.sh"
    ).read_text(encoding="utf-8")
    o2 = (
        ROOT / "scripts" / "run_duca_rime_phase2_o2_panel.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in counterfactual
    assert "produce_duca_rime_counterfactual_measurements.py" in counterfactual
    assert "all_train_empty_block_list.txt" in counterfactual
    assert "U-mixed-K" in counterfactual
    assert "produce_duca_rime_o2_panel.py" in o2
    assert "decode_rime_panel" in o2
    assert "counterfactual_negative_detector_loss" in o2
    assert "build_duca_rime_source_manifest.py o2" in o2
    assert "build_duca_rime_gate_records.py o2" in o2


def test_dense_tridet_launcher_is_slurm_checkpoint_evidence_bound():
    text = (
        ROOT / "scripts" / "run_duca_rime_dense_tridet_train.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "duca_rime_dense_tridet_cost_baseline_v1" in text
    assert "python -m tools.bata.compact_duca_rime_checkpoint" in text
    assert "build_trained_checkpoint_binding" in text
    assert "state_dict_ema" in text
    assert "uses_official_final" in text
    assert "seal_eval_dataloaders_during_training" in text
    assert "cfg.model.backbone.backbone.with_cp" in text


def test_dense_actionformer_reference_reuses_the_evidence_bound_backend_runner():
    text = (
        ROOT / "scripts" / "run_duca_rime_dense_actionformer_train.sh"
    ).read_text(encoding="utf-8")
    assert "DUCA_RIME_DENSE_BACKEND=ActionFormer" in text
    assert "run_duca_rime_dense_tridet_train.sh" in text
    cfg = (
        ROOT
        / "configs"
        / "adatad"
        / "thumos"
        / "duca_rime_dense_actionformer_total60.py"
    ).read_text(encoding="utf-8")
    assert "duca_rime_dense_actionformer_cost_baseline_v1" in cfg
    assert 'detector_backend="ActionFormer"' in cfg
    assert "seal_eval_dataloaders_during_training=True" in cfg


def test_checkpoint_compactor_module_invocation_resolves_from_clean_repo_cwd():
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.bata.compact_duca_rime_checkpoint",
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Compact a completed DUCA-RIME checkpoint" in result.stdout


def test_dense_salvage_is_hash_bound_non_mutating_and_evaluated():
    text = (
        ROOT / "scripts" / "run_duca_rime_dense_salvage.sh"
    ).read_text(encoding="utf-8")
    assert '[[ -n "${SLURM_JOB_ID:-}" ]]' in text
    assert "DUCA_RIME_DENSE_SALVAGE_MANIFEST_SHA256" in text
    assert "python -m tools.bata.salvage_duca_rime_dense_checkpoint" in text
    assert "--precheck-only" in text
    assert "source_training_git_commit" in text
    assert '"source_job_state": "FAILED"' in text
    assert '"original_job_reclassified_as_success": False' in text
    assert "evaluation_evidence.json" in text
    assert "build_trained_checkpoint_binding" in text
    assert "recovery_receipt.json" in text


def test_phase3_submission_has_six_train_jobs_and_no_same_k_training():
    submit = (ROOT / "scripts" / "submit_duca_rime_phase3.sh").read_text(
        encoding="utf-8"
    )
    assert submit.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
    pipeline = (
        ROOT / "scripts" / "run_duca_rime_phase3_arm_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "arms=(RIME-full U-fixed F-bound D-no-risk AdapTok-TAD D-shuffle)" in submit
    assert 'dependency_args=(--dependency="afterok:${arm_jobs[RIME-full]}")' in submit
    assert "training_job_count" in submit
    assert "u_same_k_training_job_count" in submit
    assert "scontrol release" in submit
    assert "--hold" in submit
    assert "submission_manifest.json.receipt.json" in submit
    assert "os.replace" in submit
    assert 'export "${name}"' in submit
    assert "U-same-K is evaluation-only" in pipeline
    assert "duca_rime_uniform_same_k_eval.py" in pipeline
    assert "a fresh shared U-same-K evaluation root is required" in pipeline
    assert "--mode paired" in pipeline
    assert "--mode shuffle" in pipeline
    assert "--mode merge" in pipeline


def test_phase4_submission_is_exactly_twelve_transactional_cells():
    submit = (
        ROOT / "scripts" / "submit_duca_rime_phase4_matrix.sh"
    ).read_text(encoding="utf-8")
    assert submit.startswith("#!/usr/bin/env bash\nset -Eeuo pipefail\n")
    pipeline = (
        ROOT / "scripts" / "run_duca_rime_phase4_cell_pipeline.sh"
    ).read_text(encoding="utf-8")
    assert "for backend in ActionFormer TriDet" in submit
    assert "for target in 384 192" in submit
    assert "for seed in 5801 8123 12011" in submit
    assert '"${#job_ids[@]}" == 12' in submit
    assert "--hold" in submit
    assert "scontrol release" in submit
    assert "--gres=gpu:1" in submit
    assert "--mem" not in submit
    assert "submission_manifest.json.receipt.json" in submit
    assert "os.replace" in submit
    assert 'export "${name}"' in submit
    assert "run_duca_rime_phase4_train_cell.sh" in pipeline
    assert pipeline.count("run_duca_rime_evaluate_arm.sh") == 1
    assert "run_duca_rime_phase4_seal_cell.sh" in pipeline
    assert "stale Phase-4 sibling output is forbidden" in pipeline
    assert "--mode paired" in pipeline
    assert ".same_k_replay" in pipeline


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
