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
    assert "torchrun --standalone --nproc_per_node=1" in text
