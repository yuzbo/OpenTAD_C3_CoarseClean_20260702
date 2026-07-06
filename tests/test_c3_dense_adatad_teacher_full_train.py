from __future__ import annotations

import importlib.util
from pathlib import Path

from mmengine.config import Config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "adatad" / "thumos" / "c3_dense_adatad_teacher_full_train.py"
BASE_CONFIG = ROOT / "configs" / "adatad" / "thumos" / "e2e_thumos_videomae_s_768x1_160_adapter.py"
LAUNCHER = ROOT / "scripts" / "run_c3_dense_adatad_teacher_full_train_gpu.sh"
SCHEDULE = ROOT / "opentad" / "utils" / "train_schedule.py"


def _load_schedule_module():
    spec = importlib.util.spec_from_file_location("train_schedule_test", SCHEDULE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_dense_teacher_config_inherits_official_adatad_and_uses_requested_schedule() -> None:
    cfg = Config.fromfile(str(CONFIG))
    base = Config.fromfile(str(BASE_CONFIG))
    schedule = _load_schedule_module()

    assert cfg.teacher_route.purpose == "train full dense AdaTAD teacher for train-only detector utility export"
    assert "frame_selector" not in repr(cfg.model)
    assert cfg.model.backbone.type == base.model.backbone.type
    assert cfg.model.backbone.backbone.type == base.model.backbone.backbone.type
    assert cfg.model.backbone.backbone.total_frames == base.model.backbone.backbone.total_frames == 768
    assert cfg.model.projection.max_seq_len == base.model.projection.max_seq_len == 768
    assert cfg.dataset.train.pipeline[2].trunc_len == base.dataset.train.pipeline[2].trunc_len == 768
    assert cfg.dataset.val.window_size == base.dataset.val.window_size == 768
    assert cfg.workflow.checkpoint_interval == 10
    assert cfg.workflow.val_start_epoch == 9
    assert cfg.workflow.val_eval_interval == 10
    assert cfg.workflow.val_eval_interval_anchor_epoch == 10
    assert cfg.workflow.end_epoch == 60

    eval_epochs = [
        epoch
        for epoch in range(cfg.workflow.end_epoch)
        if epoch >= cfg.workflow.val_start_epoch and schedule.should_eval_epoch(epoch, cfg.workflow)
    ]
    assert eval_epochs == [9, 19, 29, 39, 49, 59]


def test_dense_teacher_launcher_is_precheck_default_and_slurm_gated() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")

    assert 'PRECHECK_ONLY="${PRECHECK_ONLY:-1}"' in text
    assert "ALLOW_C3_DENSE_TEACHER_FULLTRAIN" in text
    assert "full dense teacher training must run inside a Slurm allocation/step" in text
    assert "C3_DENSE_ADATAD_TEACHER_PRECHECK_PASS" in text
    assert "${BASE}/retrained/${ADATAD_PRETRAIN_FILENAME}" in text
    assert "${BASE}/pretrained/${ADATAD_PRETRAIN_FILENAME}" in text
    assert "C3_DENSE_TEACHER_ADATAD_PRETRAIN_PATH" in text
    assert "CUDA_VISIBLE_DEVICES" in text
    assert "model.backbone.custom.pretrain=${ADATAD_PRETRAIN_PATH}" in text
