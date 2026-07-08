import importlib.util
import json
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_official_asformer_delta_ledger_original_adatad_full_train.py"
EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_official_asformer_delta_ledger_original_adatad_full_train_exec.py"
VALIDATOR = ROOT / "tools" / "bata" / "validate_c3_asformer_delta_ledger_full_train.py"
LAUNCHER = ROOT / "scripts" / "run_c3_asformer_delta_ledger_adatad_full_train_gpu1.sh"
PACTION_LAUNCHER = ROOT / "scripts" / "run_c3_paction_learned_policy_adatad_full_train_gpu1.sh"
PACTION_LATTICE_LAUNCHER = ROOT / "scripts" / "run_c3_paction_lattice_replacement_adatad_full_train_gpu1.sh"
PACTION_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_paction_learned_ledger_adatad_full_train.py"
PACTION_EXEC_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_paction_learned_ledger_adatad_full_train_exec.py"
PACTION_VALIDATOR = ROOT / "tools" / "bata" / "validate_c3_paction_learned_adatad_full_train.py"
TRAIN = ROOT / "tools" / "train.py"
SCHEDULE = ROOT / "opentad" / "utils" / "train_schedule.py"
GUARD = ROOT / "opentad" / "utils" / "training_guard.py"


def _load_validator():
    spec = importlib.util.spec_from_file_location("validate_c3_asformer_delta_ledger_full_train_test", VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_paction_validator():
    spec = importlib.util.spec_from_file_location("validate_c3_paction_learned_adatad_full_train_test", PACTION_VALIDATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loadframes(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "LoadFrames"]
    assert len(matches) == 1
    return matches[0]


def _collect(split_cfg):
    matches = [step for step in split_cfg.pipeline if isinstance(step, dict) and step.get("type") == "Collect"]
    assert len(matches) == 1
    return matches[0]


def test_asformer_delta_ledger_full_train_config_is_original_adatad_selected_axis():
    cfg = Config.fromfile(str(CONFIG))

    assert cfg.experiment_scope.route == "C3_MAINLINE_OPTIMIZATION"
    assert cfg.experiment_scope.route_variant == "C3_ORIGINAL_OPTIMIZATION_ROUTE"
    assert cfg.experiment_scope.detector_stack == "original_adatad_actionformer_adapter"
    assert cfg.experiment_scope.changes_detector_head is False
    assert cfg.experiment_scope.changes_post_processing is False
    assert int(cfg.window_size) == 384
    assert int(cfg.dense_window_size) == 768
    assert int(cfg.model.backbone.backbone.total_frames) == 384
    assert int(cfg.model.projection.max_seq_len) == 384
    assert "frame_selector" not in repr(cfg.model)
    assert "pc_ot_mras_reader" not in repr(cfg.model)
    assert cfg.inference.load_from_raw_predictions is False
    assert cfg.inference.save_raw_prediction is False

    assert cfg.dataset.train.type == "ThumosSlidingDataset"
    assert cfg.dataset.train.subset_name == "training"
    assert cfg.dataset.val.subset_name == "validation"
    assert cfg.dataset.test.subset_name == "validation"
    for split in ("train", "val", "test"):
        loader = _loadframes(cfg.dataset[split])
        assert loader.method == "bata_value_transport_ledger_subsample"
        assert loader.method_base == "sliding_window"
        assert int(loader.target_len) == 384
        assert int(loader.bata_value_transport_require_selected_count) == 384
        assert loader.bata_value_transport_require_deployable is True
        assert loader.bata_value_transport_allow_missing_fallback is False
        assert loader.bata_value_transport_allow_short_valid_ratio_count is True
        assert loader.remap_gt_to_selected_axis is True

    assert "gt_segments" not in _collect(cfg.dataset.test).get("keys", [])
    assert "gt_labels" not in _collect(cfg.dataset.test).get("keys", [])
    assert cfg.solver.ema is True
    assert "val_eval_epochs" not in cfg.workflow
    assert int(cfg.workflow.val_eval_interval) == 10
    assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
    assert int(cfg.workflow.val_start_epoch) == 9
    assert cfg.workflow.max_train_iters is None


def test_asformer_delta_ledger_full_train_validator_passes_without_local_files(monkeypatch):
    monkeypatch.setenv("C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH", "/tmp/c3_asformer_delta_train.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_VAL_LEDGER_PATH", "/tmp/c3_asformer_delta_val.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_TEST_LEDGER_PATH", "/tmp/c3_asformer_delta_test.jsonl")
    validator = _load_validator()
    cfg = validator.validate_config(str(CONFIG), require_ledger_files=False)
    assert cfg.c3_asformer_delta_ledger_full_train_gate.requires_launch_gate is True
    assert cfg.c3_asformer_delta_ledger_full_train_gate.launch_gate_passed is False


def test_asformer_delta_ledger_exec_config_passes_training_guard(monkeypatch):
    monkeypatch.setenv("C3_ASFORMER_DELTA_TRAIN_LEDGER_PATH", "/tmp/c3_asformer_delta_train.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_VAL_LEDGER_PATH", "/tmp/c3_asformer_delta_val.jsonl")
    monkeypatch.setenv("C3_ASFORMER_DELTA_TEST_LEDGER_PATH", "/tmp/c3_asformer_delta_test.jsonl")
    validator = _load_validator()
    cfg = validator.validate_config(str(EXEC_CONFIG), require_ledger_files=False, allow_launch_unlocked=True)
    guard_spec = importlib.util.spec_from_file_location("training_guard_c3_asformer_delta_test", GUARD)
    guard = importlib.util.module_from_spec(guard_spec)
    guard_spec.loader.exec_module(guard)
    guard.assert_detector_training_allowed(cfg, entrypoint="tools/train.py")
    assert cfg.c3_asformer_delta_ledger_full_train_gate.launch_gate_passed is True
    assert cfg.c3_asformer_delta_ledger_full_train_gate.reviewed_execution_config is True


def test_training_eval_schedule_keeps_legacy_and_runs_zero_based_epoch9_to59():
    schedule_spec = importlib.util.spec_from_file_location("train_schedule_c3_asformer_delta_test", SCHEDULE)
    schedule_module = importlib.util.module_from_spec(schedule_spec)
    schedule_spec.loader.exec_module(schedule_module)

    cfg = Config.fromfile(str(CONFIG))
    eval_epochs = [
        epoch + 1
        for epoch in range(60)
        if epoch >= cfg.workflow.val_start_epoch and schedule_module.should_eval_epoch(epoch, cfg.workflow)
    ]
    assert eval_epochs == [10, 20, 30, 40, 50, 60]

    class LegacyWorkflow:
        val_eval_interval = 5

        def __contains__(self, key):
            return False

        def get(self, key, default=None):
            return default

    assert schedule_module.should_eval_epoch(4, LegacyWorkflow()) is True
    assert schedule_module.should_eval_epoch(3, LegacyWorkflow()) is False


def test_asformer_delta_ledger_launcher_is_gpu1_precheck_first_and_fail_closed():
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN="${ALLOW_C3_ASFORMER_DELTA_LEDGER_FULLTRAIN:-0}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert "EXEC_CONFIG" in text
    assert "--allow-launch-unlocked" in text
    assert "--require-ledger-files" in text
    assert "validate_c3_asformer_delta_ledger_full_train.py" in text
    assert "tests/test_c3_asformer_delta_ledger_full_train.py" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text


def test_paction_learned_policy_adatad_launcher_validates_ledgers_before_full_train():
    text = PACTION_LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'if [[ "${CUDA_VISIBLE_DEVICES}" != "1" ]]' in text
    assert "validate_paction_learned_policy_ledger.py" in text
    assert "validate_c3_paction_learned_adatad_full_train.py" in text
    assert "--metric-sample-jsonl" in text
    assert "--strategy" in text
    assert "learned_paction_gap_loss_value" in text
    assert "learned_paction_gap_loss_dynamic_budget" in text
    assert "--expected-target-len \"${target_len}\"" in text
    assert "--require-selected-count 384" in text
    assert "--require-deployable" in text
    assert "--require-policy-source learned_paction_gap_loss_policy_checkpoint" in text
    assert "--require-checkpoint-path" in text
    assert "--require-checkpoint-sha256" in text
    assert "--require-paction-provenance" in text
    assert "--max-unselected-hole" in text
    assert "--max-p95-unselected-hole" in text
    assert "--max-uniform-similarity" in text
    assert "PACTION_POLICY_CHECKPOINT" in text
    assert "PACTION_POLICY_CHECKPOINT_SHA256" in text
    assert "C3_PACTION_LEDGER_SOURCE" in text
    assert 'C3_PACTION_LEDGER_SOURCE="learned_paction_gap_loss_policy_checkpoint"' in text
    assert 'C3_PACTION_LEDGER_CONFIG_HASH="${PACTION_POLICY_CHECKPOINT_SHA256}"' in text
    assert "C3_PACTION_TRAIN_LEDGER_PATH" in text
    assert "C3_PACTION_VAL_LEDGER_PATH" in text
    assert "C3_PACTION_TEST_LEDGER_PATH" in text
    assert "C3_PACTION_SOURCE_ROOT" in text
    assert "train/samples.jsonl" in text
    assert "val/samples.jsonl" in text
    assert "test/samples.jsonl" in text
    assert "source.canonical_unique.jsonl" in text
    assert "ADATAD_PRETRAIN_PATH" in text
    assert "C3_PACTION_ADATAD_PRETRAIN_PATH" in text
    assert "model.backbone.custom.pretrain" in text
    assert "required file missing: ${ADATAD_PRETRAIN_PATH}" in text
    assert 'learned_dynamic) echo "${LEDGER_ROOT}/${split}/samples.learned_dynamic.jsonl"' in text
    assert '--val-jsonl "${C3_PACTION_VAL_SOURCE_JSONL}"' not in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert "tools/train.py" in text


def test_paction_learned_policy_adatad_launcher_builds_policy_ledgers_and_runs_all_variants():
    text = PACTION_LAUNCHER.read_text(encoding="utf-8")

    assert "train_paction_acquisition_policy.py" in text
    assert "--expected-split training" in text
    assert "PACTION_BOUNDARY_MISS_LOSS_WEIGHT" in text
    assert "PACTION_LARGE_GAP_LOSS_WEIGHT" in text
    assert "PACTION_TEMPORAL_HOLE_LOSS_WEIGHT" in text
    assert "--boundary-miss-loss-weight" in text
    assert "--large-gap-loss-weight" in text
    assert "--temporal-hole-loss-weight" in text
    assert "materialize_split_source_jsonl" in text
    assert "C3_PACTION_TRAIN_SOURCE_JSONL_ORIGINAL" in text
    assert "row['split'] = split_value" in text
    assert "row.pop('uses_gt_for_diagnostics', None)" in text
    assert "row.pop('diagnostic_only', None)" in text
    assert "provenance['no_gt_generation'] = True" in text
    assert "row['paction_positive_provenance'] = provenance" in text
    assert "run_paction_learned_policy_ledger_pipeline.py" in text
    assert "validate_c3_paction_learned_adatad_full_train.py" in text
    assert "PACTION_ADATAD_VARIANTS" in text
    assert "learned_fixed_384 learned_fixed_768 learned_dynamic" in text
    assert "C3_PACTION_LEDGER_VARIANT" in text
    assert "C3_PACTION_TRAIN_LEDGER_PATH" in text
    assert "C3_PACTION_VAL_LEDGER_PATH" in text
    assert "C3_PACTION_TEST_LEDGER_PATH" in text
    assert "c3_paction_learned_ledger_adatad_full_train_exec.py" in text
    assert "--strategy" in text
    assert "learned_paction_gap_loss_dynamic_budget" in text
    assert "--require-selected-count" in text
    assert "--require-nonconstant-selected-count" in text
    assert 'learned_dynamic" && "${split}" != "test" && "${REQUIRE_DYNAMIC_NONCONSTANT}"' not in text
    assert '[[ "${REQUIRE_DYNAMIC_NONCONSTANT}" == "1" ]] && args+=(--require-dynamic-nonconstant-count)' in text
    assert '[[ "${variant}" == "learned_dynamic" && "${REQUIRE_DYNAMIC_NONCONSTANT}" == "1" ]]' in text
    assert "ALLOW_C3_PACTION_LEARNED_ADATAD_FULLTRAIN" in text


def test_paction_lattice_replacement_adatad_launcher_reuses_checkpoint_and_same_adatad_gate():
    text = PACTION_LATTICE_LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert 'ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN="${ALLOW_C3_PACTION_LATTICE_ADATAD_FULLTRAIN:-0}"' in text
    assert 'export CUDA_VISIBLE_DEVICES="0"' in text
    assert 'export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"' in text
    assert 'SLURM_STEP_GPUS' in text
    assert 'must see one Slurm-bound GPU as logical 0/1' in text
    assert 'must use physical GPU1 outside Slurm remapping' in text
    assert "CUDA is unavailable for PACTION_LATTICE_DEVICE=cuda" in text
    assert 'PACTION_POLICY_CHECKPOINT must point to a trained learned p_action policy checkpoint' in text
    assert "run_paction_lattice_replacement_ledger_pipeline.py" in text
    assert "validate_paction_lattice_replacement_ledger.py" in text
    assert "validate_c3_paction_learned_adatad_full_train.py" in text
    assert "paction_lattice_radius_score_only_move25" in text
    assert "paction_lattice_replace_score_only_move50 paction_lattice_replace_score_only_move75" not in text
    assert "--variants ${PACTION_LATTICE_ADATAD_VARIANTS}" in text
    assert "--fixed-budget \"${PACTION_LATTICE_FIXED_BUDGET}\"" in text
    assert "--deploy-selection-ledger" in text
    assert "samples.paction_lattice_replacement.jsonl" in text
    assert "value_transport_ledger_${variant}.jsonl" in text
    assert 'C3_PACTION_LEDGER_SOURCE="learned_paction_gap_loss_policy_checkpoint"' in text
    assert 'C3_PACTION_LEDGER_CONFIG_HASH="${PACTION_POLICY_CHECKPOINT_SHA256}"' in text
    assert 'MASTER_PORT_BASE="${MASTER_PORT_BASE:-}"' in text
    assert "pick_master_port()" in text
    assert '--master_port="${master_port}"' in text
    assert "master_port=${master_port}" in text
    assert 'MASTER_PORT_BASE="${MASTER_PORT_BASE:-30410}"' not in text
    assert "c3_paction_learned_ledger_adatad_full_train_exec.py" in text
    assert 'PACTION_LATTICE_DISABLE_CHECKPOINT="${PACTION_LATTICE_DISABLE_CHECKPOINT:-0}"' in text
    assert 'PACTION_LATTICE_CHECKPOINT_INTERVAL="${PACTION_LATTICE_CHECKPOINT_INTERVAL:-2}"' in text
    assert 'PACTION_LATTICE_VAL_EVAL_INTERVAL="${PACTION_LATTICE_VAL_EVAL_INTERVAL:-5}"' in text
    assert 'PACTION_LATTICE_MIN_FREE_MB="${PACTION_LATTICE_MIN_FREE_MB:-2048}"' in text
    assert "insufficient free space for full train" in text
    assert '"workflow.disable_checkpoint=${C3_PACTION_ADATAD_DISABLE_CHECKPOINT}"' in text
    assert '"workflow.checkpoint_interval=${C3_PACTION_ADATAD_CHECKPOINT_INTERVAL}"' in text
    assert '"workflow.val_eval_interval=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL}"' in text
    assert '"workflow.val_eval_interval_anchor_epoch=${C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH}"' in text
    assert '"workflow.val_start_epoch=${C3_PACTION_ADATAD_VAL_START_EPOCH}"' in text
    assert "formal full train must run inside a Slurm allocation/step" in text
    assert "tools/train.py" in text
    assert "tools/test.py" not in text


def test_paction_learned_policy_adatad_config_supports_fixed384_fixed768_and_dynamic(monkeypatch):
    monkeypatch.delenv("C3_PACTION_ADATAD_DISABLE_CHECKPOINT", raising=False)
    monkeypatch.delenv("C3_PACTION_ADATAD_CHECKPOINT_INTERVAL", raising=False)
    expected = {
        "learned_fixed_384": (384, 384, "learned_paction_gap_loss_value", "C3_PACTION_LEARNED_STRICT_LEDGER"),
        "learned_fixed_768": (768, 768, "learned_paction_gap_loss_value", "C3_PACTION_LEARNED_STRICT_LEDGER"),
        "learned_dynamic": (768, None, "learned_paction_gap_loss_dynamic_budget", "C3_PACTION_LEARNED_STRICT_LEDGER"),
        "paction_lattice_radius_score_only_move25": (
            384,
            384,
            "paction_lattice_radius_score_only_move25",
            "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT_ADAPTIVE_RADIUS",
        ),
        "paction_lattice_replace_score_only_move25": (
            384,
            384,
            "paction_lattice_replace_score_only_move25",
            "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
        ),
        "paction_lattice_replace_score_only_move50": (
            384,
            384,
            "paction_lattice_replace_score_only_move50",
            "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
        ),
        "paction_lattice_replace_score_only_move75": (
            384,
            384,
            "paction_lattice_replace_score_only_move75",
            "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
        ),
        "paction_lattice_replace_score_only_no_protect": (
            384,
            384,
            "paction_lattice_replace_score_only_no_protect",
            "C3_PACTION_SCORE_ONLY_LATTICE_REPLACEMENT",
        ),
    }
    for variant, (target_len, required_count, strategy, route_variant) in expected.items():
        monkeypatch.setenv("C3_PACTION_LEDGER_VARIANT", variant)
        monkeypatch.setenv("C3_PACTION_TRAIN_LEDGER_PATH", f"/tmp/{variant}.train.jsonl")
        monkeypatch.setenv("C3_PACTION_VAL_LEDGER_PATH", f"/tmp/{variant}.val.jsonl")
        monkeypatch.setenv("C3_PACTION_TEST_LEDGER_PATH", f"/tmp/{variant}.test.jsonl")

        cfg = Config.fromfile(str(PACTION_CONFIG))

        assert cfg.experiment_scope.stage == "paction_learned_ledger_original_adatad_full_train"
        assert cfg.experiment_scope.route_variant == route_variant
        assert cfg.c3_paction_learned_ledger_full_train_gate.route_variant == route_variant
        assert cfg.experiment_scope.selection_strategy == strategy
        assert cfg.paction_ledger_variant == variant
        assert int(cfg.window_size) == target_len
        assert int(cfg.dense_window_size) == 768
        assert int(cfg.model.backbone.backbone.total_frames) == target_len
        assert int(cfg.model.projection.max_seq_len) == target_len
        assert cfg.evaluation.ground_truth_filename == cfg.annotation_path
        assert cfg.solver.ema is True
        assert "val_eval_epochs" not in cfg.workflow
        assert int(cfg.workflow.val_eval_interval) == 10
        assert int(cfg.workflow.val_eval_interval_anchor_epoch) == 10
        assert int(cfg.workflow.val_start_epoch) == 9
        assert int(cfg.workflow.checkpoint_interval) == 10
        assert cfg.workflow.disable_checkpoint is False
        assert "frame_selector" not in repr(cfg.model)
        for split in ("train", "val", "test"):
            loader = _loadframes(cfg.dataset[split])
            assert loader.method == "bata_value_transport_ledger_subsample"
            assert int(loader.target_len) == target_len
            assert loader.bata_value_transport_require_selected_count == required_count
            assert loader.bata_value_transport_require_deployable is True
            assert loader.bata_value_transport_allow_missing_fallback is False
            assert loader.bata_value_transport_allow_short_valid_ratio_count is True
            assert loader.remap_gt_to_selected_axis is True


def test_paction_lattice_radius_move25_config_allows_fast_eval_and_checkpoint_schedule(monkeypatch):
    monkeypatch.setenv("C3_PACTION_LEDGER_VARIANT", "paction_lattice_radius_score_only_move25")
    monkeypatch.setenv("C3_PACTION_TRAIN_LEDGER_PATH", "/tmp/radius_move25.train.jsonl")
    monkeypatch.setenv("C3_PACTION_VAL_LEDGER_PATH", "/tmp/radius_move25.val.jsonl")
    monkeypatch.setenv("C3_PACTION_TEST_LEDGER_PATH", "/tmp/radius_move25.test.jsonl")
    monkeypatch.setenv("C3_PACTION_ADATAD_DISABLE_CHECKPOINT", "0")
    monkeypatch.setenv("C3_PACTION_ADATAD_CHECKPOINT_INTERVAL", "2")
    monkeypatch.setenv("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL", "5")
    monkeypatch.setenv("C3_PACTION_ADATAD_VAL_EVAL_INTERVAL_ANCHOR_EPOCH", "5")
    monkeypatch.setenv("C3_PACTION_ADATAD_VAL_START_EPOCH", "4")

    cfg = Config.fromfile(str(PACTION_CONFIG))
    validator = _load_paction_validator()
    validated = validator.validate_config(str(PACTION_CONFIG), require_ledger_files=False)

    assert cfg.paction_ledger_variant == "paction_lattice_radius_score_only_move25"
    assert int(validated.workflow.checkpoint_interval) == 2
    assert int(validated.workflow.val_eval_interval) == 5
    assert int(validated.workflow.val_eval_interval_anchor_epoch) == 5
    assert int(validated.workflow.val_start_epoch) == 4
    assert validated.workflow.disable_checkpoint is False


def test_paction_learned_policy_adatad_config_uses_reviewed_absolute_pretrain(monkeypatch):
    pretrain_path = "/data/run01/sczc063/yuzibo/pretrained/vit-small-p16_videomae-k400-pre_16x4x1_kinetics-400_my.pth"
    monkeypatch.setenv("C3_PACTION_LEDGER_VARIANT", "learned_fixed_384")
    monkeypatch.setenv("C3_PACTION_TRAIN_LEDGER_PATH", "/tmp/c3_paction_train.jsonl")
    monkeypatch.setenv("C3_PACTION_VAL_LEDGER_PATH", "/tmp/c3_paction_val.jsonl")
    monkeypatch.setenv("C3_PACTION_TEST_LEDGER_PATH", "/tmp/c3_paction_test.jsonl")
    monkeypatch.setenv("C3_PACTION_ADATAD_PRETRAIN_PATH", pretrain_path)

    cfg = Config.fromfile(str(PACTION_CONFIG))

    assert cfg.model.backbone.custom.pretrain == pretrain_path
    assert cfg.c3_paction_adatad_pretrain_path == pretrain_path


def test_paction_learned_policy_adatad_validator_passes_locked_and_exec_configs(monkeypatch):
    monkeypatch.setenv("C3_PACTION_LEDGER_VARIANT", "learned_dynamic")
    monkeypatch.setenv("C3_PACTION_TRAIN_LEDGER_PATH", "/tmp/c3_paction_dynamic_train.jsonl")
    monkeypatch.setenv("C3_PACTION_VAL_LEDGER_PATH", "/tmp/c3_paction_dynamic_val.jsonl")
    monkeypatch.setenv("C3_PACTION_TEST_LEDGER_PATH", "/tmp/c3_paction_dynamic_test.jsonl")
    validator = _load_paction_validator()

    cfg = validator.validate_config(str(PACTION_CONFIG), require_ledger_files=False)
    assert cfg.c3_paction_learned_ledger_full_train_gate.launch_gate_passed is False
    assert cfg.paction_ledger_strategy == "learned_paction_gap_loss_dynamic_budget"

    exec_cfg = validator.validate_config(
        str(PACTION_EXEC_CONFIG),
        require_ledger_files=False,
        allow_launch_unlocked=True,
    )
    assert exec_cfg.c3_paction_learned_ledger_full_train_gate.launch_gate_passed is True
    assert exec_cfg.c3_paction_learned_ledger_full_train_gate.reviewed_execution_config is True


def test_paction_learned_policy_adatad_full_train_gate_requires_paction_provenance(tmp_path, monkeypatch):
    checkpoint_sha = "a" * 64
    ledger = tmp_path / "value_transport_ledger_learned_fixed_384.jsonl"
    row = {
        "schema_version": "pc_ot_mras_frontend_value_transport_ledger_v0",
        "sample_id": "video_test_0001|0",
        "selected_positions_unit": "local_dense_index",
        "selected_positions": list(range(384)),
        "target_len": 384,
        "selected_count": 384,
        "valid_len": 768,
        "dense_len": 768,
        "deploy_selection_ledger": True,
        "diagnostic_only": False,
        "policy_source": "learned_paction_gap_loss_policy_checkpoint",
        "policy_checkpoint_path": str(tmp_path / "policy.pth"),
        "policy_checkpoint_sha256": checkpoint_sha,
        "uses_gt": False,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "uses_checkpoint": False,
        "prediction_uses_gt": False,
        "training_only": False,
        "diagnostics": {
            "uniform_visible_fill_count": 0,
            "source_strategy": "learned_paction_gap_loss_value",
            "policy_source": "learned_paction_gap_loss_policy_checkpoint",
            "policy_checkpoint_sha256": checkpoint_sha,
        },
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    monkeypatch.setenv("C3_PACTION_LEDGER_VARIANT", "learned_fixed_384")
    monkeypatch.setenv("C3_PACTION_TRAIN_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_PACTION_VAL_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_PACTION_TEST_LEDGER_PATH", str(ledger))
    monkeypatch.setenv("C3_PACTION_LEDGER_CONFIG_HASH", checkpoint_sha)
    validator = _load_paction_validator()
    cfg = validator.validate_config(str(PACTION_CONFIG), require_ledger_files=False)

    try:
        validator._validate_ledger_file(ledger, cfg=cfg, require_exists=True)
    except AssertionError as exc:
        assert "p_action provenance" in str(exc)
    else:
        raise AssertionError("missing p_action provenance should fail the full-train ledger gate")

    row["diagnostics"]["p_action_provenance"] = {
        "p_action_source": "lowres_action_probe",
        "probe_model": "mobilenetv3_64px",
        "no_gt_generation": True,
        "uses_teacher": False,
        "uses_oracle": False,
        "uses_cache": False,
        "uses_prediction_cache": False,
        "uses_raw_prediction": False,
        "prediction_uses_gt": False,
    }
    ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    validator._validate_ledger_file(ledger, cfg=cfg, require_exists=True)
