# Copyright (c) OpenTAD. All rights reserved.
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "configs" / "adatad" / "thumos"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_bafdr_configs_use_supported_collect_contract():
    config_paths = sorted(CONFIG_DIR.glob("bafdr_k16_*.py"))
    assert len(config_paths) == 21
    wrapper_paths = [
        path
        for path in config_paths
        if any(token in path.name for token in ("u16_uniform", "late", "nokd", "full"))
    ]
    assert len(wrapper_paths) == 12
    for path in wrapper_paths:
        text = read(path)
        assert 'type="BAFDRSourceViews"' in text
        assert 'type="Collect", inputs="bafdr_inputs"' in text
        assert '"gt_segments", "gt_labels"' in text
        assert '"bafdr_geometry"' in text
        assert "extra_keys" not in text
        # LoadFrames takes trunc_len for random windows; sliding-window length
        # belongs to the dataset config, not the transform constructor.
        assert 'method="sliding_window", window_size=' not in text
        assert 'method="sliding_window", window_overlap_ratio=' not in text


def test_bafdr_generator_cannot_reintroduce_collect_extra_keys():
    text = read(ROOT / "tools" / "bata" / "generate_bafdr_configs.py")
    assert "extra_keys" not in text
    assert 'type="Collect", inputs="bafdr_inputs"' in text
    assert 'method="sliding_window", window_size=' not in text
    assert 'method="sliding_window", window_overlap_ratio=' not in text


def test_bafdr_configs_bind_world2_global_batch_contract():
    config_paths = sorted(CONFIG_DIR.glob("bafdr_k16_*.py"))
    assert len(config_paths) == 21
    for path in config_paths:
        cfg = Config.fromfile(str(path))
        assert cfg.solver.train.batch_size == 2
        assert cfg.solver.val.batch_size == 2
        assert cfg.solver.test.batch_size == 2


def test_bafdr_wrapper_exposes_mask_and_keeps_source_uint8_contract():
    text = read(ROOT / "opentad" / "models" / "backbones" / "bafdr_wrapper.py")
    assert "source_tensor.dtype != torch.uint8" in text
    assert "local_tensor = torch.stack(local_chunk_list, dim=0).to(selected_indices.device" in text
    assert "selected_mask_384" in text
    assert "selected_mask_768" in text
    assert '"selected_mask_768": selected_mask_768' in text
    assert "allow_synthetic_source_fallback" in text
    assert "elif self.allow_synthetic_source_fallback" in text
    assert "compute_contract" in text


def test_actionformer_passes_masks_to_backbone_and_captures_kd_features():
    text = read(ROOT / "opentad" / "models" / "detectors" / "actionformer.py")
    assert "self.backbone(inputs, masks=masks)" in text
    assert "self._latest_backbone_output = x" in text
    assert "self._latest_projection_output = x" in text


def test_anchor_free_head_captures_pre_nms_logits_for_distillation():
    text = read(ROOT / "opentad" / "models" / "dense_heads" / "anchor_free_head.py")
    assert "self.latest_kd_outputs" in text
    assert '"cls_pred": tuple(cls_pred)' in text
    assert '"reg_pred": tuple(reg_pred)' in text
    assert '"mask_list": tuple(mask_list)' in text


def test_train_driver_is_fail_closed_and_ddp_aware():
    text = read(ROOT / "tools" / "bata" / "bafdr_k16_fullmatrix_train.py")
    assert "DistributedDataParallel" in text
    assert "init_process_group" in text
    assert "EXPECTED_WORLD_SIZE = 2" in text
    assert "EXPECTED_GLOBAL_BATCH_SIZE = 2" in text
    assert "validate_loader_batch_contract" in text
    assert "static_graph=bool(getattr(cfg.solver, \"static_graph\", False))" in text
    assert "--prediction-only" in text
    assert "--open-metrics" in text
    assert 'receipt_name="train_receipt.json"' in text
    assert 'phase="training"' in text
    assert "strict=True" in text
    assert "strict=False" not in text
    assert "inputs[\"global\"]" not in text
    assert "make_teacher_d160_inputs" in text
    assert "latest_kd_outputs" in text
    assert "FileNotFoundError" in text
    assert "logger.warning" not in text
    assert "BAFDR_PRETRAIN" in text
    assert "BAFDR pretrained checkpoint" in text
    optimizer_text = read(ROOT / "opentad" / "cores" / "optimizer.py")
    assert "target_model = model.module if hasattr(model, \"module\") else model" in optimizer_text
    validator_text = read(ROOT / "tools" / "bata" / "bafdr_k16_fullmatrix.py")
    assert "dataset.window_size must be 768" in validator_text


def test_matrix_orchestrator_validates_semantic_contracts():
    text = read(ROOT / "tools" / "bata" / "bafdr_k16_fullmatrix.py")
    assert "validate_bafdr_pipeline" in text
    assert "validate_bafdr_model" in text
    assert "validate_distillation_contract" in text
    assert "BAFDRSourceViews" in text
    assert "optimizer custom groups missing" in text
    assert "--require-complete" in text
    assert "--seal-predictions" in text
    assert "PREDICTIONS_SEALED_NO_METRICS_OPENED" in text


def test_slurm_scripts_use_world2_arm_batch_dag():
    run_text = read(ROOT / "scripts" / "run_zoomtoken_bafdr_k16_fullmatrix_n16r4.sh")
    submit_text = read(ROOT / "scripts" / "submit_zoomtoken_bafdr_k16_fullmatrix_n16r4.sbatch")
    assert "torchrun --standalone --nproc_per_node" in run_text
    assert "PRECHECK_ONLY" in run_text
    assert "--prediction-only" in run_text
    assert "--seal-predictions" in run_text
    assert "--open-metrics" in run_text
    assert 'if [[ -n "${PROJECT_DIR:-}" ]]' in run_text
    assert 'SLURM_SUBMIT_DIR' in run_text
    assert "BAFDR_PRETRAIN" in run_text
    assert "pretrained checkpoint is not readable" in run_text
    assert "#SBATCH --gpus=2" in submit_text
    assert "arm_list=(D160 G96 U128-ALL48-A0 U16-UNIFORM-A0 BAFDR-K16-LATE BAFDR-K16-NOKD BAFDR-K16-FULL)" in submit_text
    assert "run_zoomtoken_bafdr_k16_arm_batch.sh" in submit_text
    assert "--array=0-20" not in submit_text
    assert 'if [[ -n "${PROJECT_DIR:-}" ]]' in submit_text
    assert 'SLURM_SUBMIT_DIR' in submit_text
    assert "afterok:${joined_train_deps}" in submit_text
    assert '--gpus="${gpus}"' in submit_text
    assert '"168:00:00"' in submit_text
    assert "zt-bafdr-metrics" in submit_text
    batch_task_text = read(ROOT / "scripts" / "run_zoomtoken_bafdr_k16_arm_batch.sh")
    assert "BAFDR-K16-FULL" in batch_task_text
    assert "SEEDS=(4407 4408 4409)" in batch_task_text


def test_raw_prediction_cache_is_window_level_for_sliding_window_eval():
    text = read(ROOT / "opentad" / "models" / "utils" / "post_processing" / "utils.py")
    assert "def _prediction_filename" in text
    assert "window_start_frame" in text
    assert "__window_" in text


def test_screen_submission_does_not_prematurely_pass_gate():
    submit = read(ROOT / "scripts" / "submit_zoomtoken_bafdr_k16_screen_n16r4.sh")
    finalizer = read(ROOT / "tools" / "bata" / "finalize_bafdr_screen_gate.py")
    assert '"status": "SUBMITTED"' in submit
    assert 'promoted["status"] = "PASS"' in finalizer
    assert "total_successful_updates" in finalizer
    assert "teacher_identity" in finalizer
    assert "teacher_checkpoint_sha256" in finalizer
    assert "BAFDR screen receipt is required" not in submit
