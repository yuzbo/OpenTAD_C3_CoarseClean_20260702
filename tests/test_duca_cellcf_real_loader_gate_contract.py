from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "bata" / "run_duca_cellcf_real_loader_cuda_gate.py"
SOURCE = SCRIPT.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE, filename=str(SCRIPT))


def _constant(name: str):
    for node in TREE.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"module constant {name!r} is missing")


def _argument_contracts() -> dict[str, dict[str, object]]:
    arguments: dict[str, dict[str, object]] = {}
    for node in ast.walk(TREE):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        try:
            flag = ast.literal_eval(node.args[0])
        except (ValueError, TypeError):
            continue
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        values: dict[str, object] = {}
        for keyword in node.keywords:
            if keyword.arg is None:
                continue
            try:
                values[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                values[keyword.arg] = "<dynamic>"
        arguments[flag] = values
    return arguments


def test_gate_schema_scope_and_audited_surfaces_are_frozen() -> None:
    assert _constant("SCHEMA") == "duca_cellcf_real_loader_cuda_gate_v1"
    assert _constant("SYNTHETIC_GATE_SCHEMA") == "duca_cellcf_synthetic_gate_v1"
    assert _constant("CONFIG_DEFAULT") == (
        "configs/adatad/thumos/"
        "duca_cellcf_fixed384_official_adatad_backend_full_train.py"
    )
    audited = set(_constant("AUDITED_PATHS"))
    assert "tools/bata/run_duca_cellcf_synthetic_gate.py" in audited
    assert "tools/bata/run_duca_cellcf_real_loader_cuda_gate.py" in audited
    assert "opentad/cores/train_engine.py" in audited
    assert "opentad/datasets/transforms/end_to_end.py" in audited
    assert "opentad/models/detectors/actionformer.py" in audited
    assert "tests/test_duca_cellcf_real_loader_gate_contract.py" in audited


def test_cli_requires_exact_commit_synthetic_gate_and_external_asset_hashes() -> None:
    arguments = _argument_contracts()
    for flag in (
        "--expected-commit",
        "--synthetic-gate-json",
        "--videomae-checkpoint",
        "--expected-videomae-sha256",
        "--official-repos-root",
    ):
        assert arguments[flag].get("required") is True
    assert arguments["--device"].get("default") == "cuda:0"
    assert "--allow-dirty" not in arguments
    assert "--allow-cpu" not in arguments
    assert "--skip-synthetic-gate" not in arguments
    assert "--force-pass" not in arguments


def test_gate_is_clean_commit_and_slurm_logical_cuda_fail_closed() -> None:
    assert '"status", "--porcelain", "--untracked-files=normal"' in SOURCE
    assert "the gate requires a clean exact-commit checkout" in SOURCE
    assert "expected_commit_exact_match" in SOURCE
    assert "_verify_final_clean_binding" in SOURCE
    assert "git_tree_clean_after_gate" in SOURCE
    assert "audited_hashes_unchanged" in SOURCE
    assert "--output-json must be outside the Git worktree" in SOURCE
    assert 'str(device) == "cuda:0"' in SOURCE
    assert 'os.environ.get("SLURM_JOB_ID")' in SOURCE
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES")' in SOURCE
    assert "torch.cuda.device_count() == 1" in SOURCE
    assert "physical_gpu_index_assumed" in SOURCE
    assert '"cuda:1"' not in SOURCE
    assert "allow_dirty" not in SOURCE


def test_synthetic_evidence_binding_is_same_commit_schema_and_sha_bound() -> None:
    assert 'payload.get("schema") == SYNTHETIC_GATE_SCHEMA' in SOURCE
    assert 'payload.get("ok") is True' in SOURCE
    assert 'payload.get("git_tree_clean") is True' in SOURCE
    assert 'payload.get("git_commit") == git_commit' in SOURCE
    assert 'payload.get("real_dataset_loader_executed") is False' in SOURCE
    assert 'audited.get(synthetic_script) == _sha256(ROOT / synthetic_script)' in SOURCE
    assert '"synthetic_gate_sha256"' in SOURCE
    assert '"synthetic_gate_schema"' in SOURCE


def test_real_loader_model_and_training_engine_are_executed_without_substitutes() -> None:
    assert "build_dataset(copy.deepcopy(cfg.dataset.train)" in SOURCE
    assert "full_train_loader = build_dataloader(" in SOURCE
    assert "subset = Subset(dataset, indices)" in SOURCE
    assert "batch = next(iter(loader))" in SOURCE
    assert "model = build_detector(copy.deepcopy(cfg.model))" in SOURCE
    assert 'model.__class__.__name__ == "ActionFormer"' in SOURCE
    assert 'model.rpn_head.__class__.__name__ == "ActionFormerHead"' in SOURCE
    assert 'videomae.__class__.__name__ == "VisionTransformerAdapter"' in SOURCE
    assert "DistributedDataParallel(" in SOURCE
    assert "comm_hooks.fp16_compress_hook" in SOURCE
    assert "train_one_epoch(" in SOURCE
    assert "ModelEma(ddp_model)" in SOURCE
    assert "_ProofTemporalMeanBackbone" not in SOURCE
    assert "synthetic_inputs_used" in SOURCE


def test_official_sources_and_frozen_training_semantics_are_checked() -> None:
    assert "--expected-videomae-sha256 must be a 64-character SHA-256" in SOURCE
    assert "loaded_through_backbone_custom_pretrain" in SOURCE
    assert 'repos_root / "ASFormer" / "model.py"' in SOURCE
    assert "official_asformer_source_normalized_lf_sha256" in SOURCE
    assert 'int(cfg.workflow.end_epoch) == 132' in SOURCE
    assert 'int(cfg.scheduler.max_epoch) == 132' in SOURCE
    assert 'int(cfg.workflow.checkpoint_interval) == 5' in SOURCE
    assert 'int(selector.budget) == 384' in SOURCE
    assert 'int(selector.dense_window_size) == 768' in SOURCE
    assert 'cfg.model.type == "ActionFormer"' in SOURCE
    assert 'cfg.model.rpn_head.type == "ActionFormerHead"' in SOURCE
    assert "resolved ActionFormerHead config drifted from the official AdaTAD base" in SOURCE
    assert "resolved ActionFormer NMS config drifted from the official AdaTAD base" in SOURCE
    assert 'for key, expected in required_contract.items()' in SOURCE
    assert '"task": "offline_temporal_action_detection"' in SOURCE
    assert '"online_tad": False' in SOURCE


def test_validity_gt_coordinate_and_counterfactual_coverage_is_honest() -> None:
    for pattern in ("full", "mixed", "all_short"):
        assert f'"{pattern}"' in SOURCE
    assert "not_obtainable_from_real_train_annotation_index" in SOURCE
    assert "all_obtainable_patterns_executed" in SOURCE
    assert "_classify_validity_pattern" in SOURCE
    assert "real GT remap does not follow the fixed detector anchor grid" in SOURCE
    assert 'meta.get("duca_acquisition_positions")' in SOURCE
    assert 'meta.get("duca_detector_grid_positions")' in SOURCE
    assert 'meta.get("selected_axis_to_true_time_dense_index")' in SOURCE
    assert '"distinct_local_cell_weighted_signed_logistic"' in SOURCE
    assert '"detached_distinct_local_cell_hard_flip_official_actionformer_cls_plus_reg"' in SOURCE
    assert "candidate_count > 0" in SOURCE
    assert "any(value != 0.0 for value in utilities)" in SOURCE


def test_forced_amp_replay_and_one_successful_update_are_proven() -> None:
    assert "force_amp_overflow_attempts=1" in SOURCE
    assert '"optimizer_attempts": 2' in SOURCE
    assert '"successful_optimizer_updates": 1' in SOURCE
    assert '"amp_skipped_attempts": 1' in SOURCE
    assert '"replayed_batches": 1' in SOURCE
    assert '"replay_state_restorations": 1' in SOURCE
    assert '"scheduler_updates": 1' in SOURCE
    assert '"ema_updates": 1' in SOURCE
    assert '"duca_schedule_updates": 1' in SOURCE
    assert "parameter_delta > 0.0" in SOURCE
    assert "ema_delta > 0.0" in SOURCE
    assert "schedule_after == schedule_before + 1" in SOURCE
    assert "scheduler_after == scheduler_before + 1" in SOURCE
    assert '"one_successful_optimizer_ema_schedule_update_verified": True' in SOURCE


def test_failure_payload_never_grants_a_claim() -> None:
    assert '"schema": SCHEMA' in SOURCE
    assert '"ok": False' in SOURCE
    assert '"fail_closed": True' in SOURCE
    assert '"real_loader_cuda_gate_passed": False' in SOURCE
    assert '"metric_claim_allowed": False' in SOURCE
    assert '"paper_ready": False' in SOURCE
