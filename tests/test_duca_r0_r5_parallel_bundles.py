from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUBMITTER = ROOT / "scripts/submit_duca_r0_r5_parallel_bundles.sh"
WORKER = ROOT / "scripts/run_duca_r0_r5_parallel_bundle_gpu1.sh"


def _text(path: Path) -> str:
    assert path.is_file(), path
    return path.read_text(encoding="utf-8")


def test_submitter_requires_no_preexisting_trainable_duca_artifacts():
    text = _text(SUBMITTER)
    for forbidden in (
        "DUCA_PARALLEL_FRONTEND_DECISION_JSON",
        "DUCA_PARALLEL_GATE_SUITE_JSON",
        "DUCA_PARALLEL_TERMINAL_SUITE_JSON",
        "DUCA_PARALLEL_ALIGNMENT_JSON",
    ):
        assert forbidden not in text
    assert '"forbidden_external_trainable_artifacts"' in text
    assert '"frontend_decision", "gate_suite", "terminal_u_g0_suite", "alignment"' in text
    assert '"R0 checkpoint", "AdaTAD pretrain", "THUMOS split"' in text


def test_submitter_has_five_dependency_null_bundles_and_only_r5_aggregate_afterok():
    text = _text(SUBMITTER)
    for role in (
        "r0_r1",
        "r2_r3_core",
        "r2_r3_adapted",
        "r4_r2q3",
        "r5_all",
    ):
        assert role in text
    assert '"dependency_null_bundle_count": 5' in text
    assert '"aggregate_is_only_afterok_job": True' in text
    assert text.count("--dependency=") == 1
    assert 'aggregate_dependency="afterok:${job_ids[r5_all]}"' in text
    assert "printf '%s\\t%s\\tnone\\t%s\\t%s\\n'" in text


def test_submitter_requests_enough_slurm_gpus_without_overriding_device_ids():
    text = _text(SUBMITTER)
    assert "[r0_r1]=1" in text
    assert "[r2_r3_core]=3" in text
    assert "[r2_r3_adapted]=3" in text
    assert "[r4_r2q3]=2" in text
    assert "[r5_all]=4" in text
    assert "#SBATCH --gpus=${gpus}" in text
    assert "CUDA_VISIBLE_DEVICES=" not in text


def test_worker_rebuilds_current_commit_bootstrap_inside_r4_and_r5():
    text = _text(WORKER)
    assert "run_current_commit_bootstrap" in text
    assert text.count("DUCA_PARALLEL_BUNDLE_ROLE=current_commit_bootstrap") == 2
    assert "run_duca_boundary_burst_r0_holdout_map_gpu1.sh" in text
    assert "run_duca_boundary_burst_p0_gpu1.sh" in text
    assert 'cp -- "${DUCA_FRONTEND_TRAIN_BLOCK_LIST}"' in text
    assert 'cp -- "${DUCA_FRONTEND_HOLDOUT_BLOCK_LIST}"' in text
    assert "copied frontend train split drifted" in text
    assert "copied frontend holdout split drifted" in text
    assert "run_duca_boundary_burst_gate_gpu1.sh" in text
    assert "aggregate_duca_boundary_burst_results" in text
    assert "run_duca_boundary_burst_hard_swap_alignment_gpu1.sh" in text
    assert "current-commit P0, U/G0, terminal, alignment" in text
    assert "resolve_preregistered_route" in text
    assert "DUCA_PREREGISTERED_PROJECTED_FAMILY" in text
    assert 'PREREGISTERED_FAMILY=R2Q3_privileged_boundary_burst' in text
    assert '"schema": "duca_current_commit_bootstrap_v2"' in text
    assert "was not selected by current-commit R0" not in text
    assert "current-commit R0 selected another family" not in text


def test_r4_r5_continue_only_after_official60_g0_beats_exact_uniform():
    worker = _text(WORKER)
    alignment = _text(
        ROOT / "tools/bata/duca_boundary_burst_hard_swap_alignment.py"
    )
    assert "aggregate_duca_boundary_burst_results" in worker
    assert "run_duca_boundary_burst_hard_swap_alignment_gpu1.sh" in worker
    assert '_require(selected_map > uniform_map, "G0 must beat U before R4")' in alignment


def test_r2_r3_factorial_arms_keep_p0_gate_official60_inside_each_child():
    text = _text(WORKER)
    independent = _text(ROOT / "scripts/run_duca_independent_official60_gpu1.sh")
    assert "two_stage_exact_uniform" in text
    assert "boundary_burst_r2q3_soft_detached_g0" in text
    assert "boundary_burst_r2q3_hard_detached_g0" in text
    assert "boundary_burst_r2q3_soft_adapted_g0" in text
    assert "boundary_burst_r2q3_g0" in text
    assert "boundary_burst_r4q5_g0" in text
    assert "run_duca_independent_official60_gpu1.sh" in text
    assert "each learned arm ran P0, gate and official-60" in text
    for config in (
        "duca_boundary_burst_soft_detached_frontend_pretrain_fixed384.py",
        "duca_boundary_burst_hard_detached_frontend_pretrain_fixed384.py",
        "duca_boundary_burst_soft_adapted_frontend_pretrain_fixed384.py",
        "duca_boundary_burst_soft_g0_no_feedback_fixed384_official60.py",
    ):
        assert config in independent


def test_r5_derives_requested_groups_and_same_backend_costs():
    submitter = _text(SUBMITTER)
    worker = _text(WORKER)
    assert '"r5_cell_count": len(rows)' in submitter
    assert '"r5_budgets": summary["budgets"]' in submitter
    assert '"r5_seeds": summary["seeds"]' in submitter
    for group in (
        "actionformer:uniform",
        "actionformer:learned",
        "temporalmaxer:uniform",
        "temporalmaxer:learned",
    ):
        assert group in worker
    assert "expected_count" in worker
    assert '[[ "${expected_count}" -gt 0 && "${count}" == "${expected_count}" ]]' in worker
    assert 'if [[ "${seed}" == 3407 && "${backend}" == actionformer ]]' in worker
    assert "cost_dense_adatad_k768" not in worker
    assert "all requested R5 cells and paired ActionFormer candidate/dense cost profiles completed" in worker


def test_parallel_sources_bind_final_head_bilateral_and_hidden_gradient_contract():
    submitter = _text(SUBMITTER)
    p0 = _text(
        ROOT / "configs/adatad/thumos/duca_boundary_burst_frontend_pretrain_fixed384.py"
    )
    g0 = _text(
        ROOT / "configs/adatad/thumos/duca_boundary_burst_g0_no_feedback_fixed384_official60.py"
    )
    assert "duca_boundary_burst_g1_protected_fixed384_official60.py" in submitter
    assert "boundary_burst_require_bilateral_offsets=True" in p0
    assert "boundary_burst_require_global_mandatory_groups=True" in p0
    assert 'hard_global_burst_support="mandatory_group_constrained_exact_k_max_hole"' in p0
    assert "auxiliary_hidden_gradient_scale=0.25" in p0
    assert "boundary_burst_require_bilateral_offsets=True" in g0
    assert "boundary_burst_require_global_mandatory_groups=True" in g0
    assert 'hard_global_burst_support="mandatory_group_constrained_exact_k_max_hole"' in g0


def test_parallel_scripts_do_not_add_model_routes_or_git_side_effects():
    joined = _text(SUBMITTER) + _text(WORKER)
    assert "cat > configs/" not in joined
    assert "opentad/models" not in joined
    assert "local-cell" not in joined
    assert "X3D" not in joined
    assert "MUST" not in joined
    assert "git commit" not in joined
    assert "git worktree" not in joined
    assert "git push" not in joined
