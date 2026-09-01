from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_official60_wrappers_select_profile_before_common_infrastructure() -> None:
    suite = (ROOT / "scripts/prepare_duca_cellcf_official60_suite.sh").read_text(
        encoding="utf-8"
    )
    pilot = (
        ROOT / "scripts/prepare_duca_cellcf_official60_ddp_pilot.sh"
    ).read_text(encoding="utf-8")

    assert "export DUCA_CELLCF_TRAINING_PROFILE=official60" in suite
    assert "prepare_duca_cellcf_suite.sh" in suite
    assert "export DUCA_CELLCF_TRAINING_PROFILE=official60" in pilot
    assert "prepare_duca_cellcf_ddp_pilot.sh" in pilot


def test_official60_gate_is_slurm_bound_and_uses_shared_gate_implementation() -> None:
    source = (
        ROOT / "scripts/run_duca_cellcf_official60_real_loader_gate.sh"
    ).read_text(encoding="utf-8")

    assert "SLURM_JOB_ID" in source
    assert "CUDA_VISIBLE_DEVICES" in source
    assert "export DUCA_CELLCF_TRAINING_PROFILE=official60" in source
    assert "tools.bata.run_duca_cellcf_real_loader_cuda_gate" in source
    assert "--device cuda:0" in source
    assert "duca_cellcf_require_external_path" in source
    assert "refusing to overwrite real-loader evidence" in source


def test_official60_synthetic_gate_is_clean_commit_and_profile_bound() -> None:
    source = (
        ROOT / "scripts/run_duca_cellcf_official60_synthetic_gate.sh"
    ).read_text(encoding="utf-8")

    assert "DUCA_CELLCF_TRAINING_PROFILE=official60" in source
    assert "git status --porcelain --untracked-files=normal" in source
    assert "duca_cellcf_require_external_path" in source
    assert "run_duca_cellcf_synthetic_gate" in source


def test_common_suite_binds_profile_into_every_generated_job() -> None:
    source = (ROOT / "scripts/prepare_duca_cellcf_suite.sh").read_text(
        encoding="utf-8"
    )

    assert source.count(
        "export DUCA_CELLCF_TRAINING_PROFILE='${DUCA_CELLCF_TRAINING_PROFILE}'"
    ) == 4
    assert source.count("source scripts/duca_cellcf_canonical_env.sh") == 2
    assert "epoch_${DUCA_CELLCF_TERMINAL_EPOCH}.pth" in source
    assert "require_single_quoted_heredoc_safe" in source
    assert "DUCA_CELLCF_TRAINING_PROFILE}_seed" in source
    assert "duca_cellcf_require_external_path" in source


def test_training_cost_summary_bootstraps_profile_from_hash_bound_aggregate() -> None:
    source = (
        ROOT / "scripts/summarize_duca_cellcf_training_cost.sh"
    ).read_text(encoding="utf-8")

    aggregate_profile = source.index("AGGREGATE_PROFILE=")
    canonical_env = source.index(
        'source "${REPO_ROOT}/scripts/duca_cellcf_canonical_env.sh"'
    )
    assert aggregate_profile < canonical_env
    assert "aggregate suite evidence hash mismatch" in source
    assert 'export DUCA_CELLCF_TRAINING_PROFILE="${AGGREGATE_PROFILE}"' in source
    assert "LEGACY_EXPOSURE132_COMMITS" in source


def test_canonical_environment_blocks_external_python_shadowing() -> None:
    source = (ROOT / "scripts/duca_cellcf_canonical_env.sh").read_text(
        encoding="utf-8"
    )

    assert "unset PYTHONPATH" in source
    assert "unset PYTHONHOME" in source
    assert "export PYTHONNOUSERSITE=1" in source
    assert "git ls-files --others --ignored --exclude-standard" in source
    assert "ignored Python source files could shadow the exact commit" in source
