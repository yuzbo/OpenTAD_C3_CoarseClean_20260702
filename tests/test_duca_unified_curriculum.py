import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from mmengine.config import Config

sys.modules.setdefault("tqdm", types.SimpleNamespace(tqdm=lambda iterable, **_: iterable))

from opentad.models.selectors.duca_online_frame_selector import DucaOnlineFrameSelector
from tools.bata.aggregate_duca_unified_fullmatrix import build_summary
from tools.bata.duca_runtime_contract import resolve_effective_seed


ROOT = Path(__file__).resolve().parents[1]
MATRIX = ROOT / "scripts" / "duca_unified_fullmatrix" / "matrix.json"
TRAIN_ENGINE = ROOT / "opentad" / "cores" / "train_engine.py"


def _load_train_engine():
    spec = importlib.util.spec_from_file_location("duca_unified_train_engine_test", TRAIN_ENGINE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_matrix_has_17_development_and_24_confirmation_tasks():
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    rows = payload["rows"]

    assert payload["matrix_id"] == "DUCA-UNIFIED-FULLMATRIX-v001-20260902"
    assert payload["base_revision"] == "95ca6eb4a7e0ba8259c5afd976cc30d0fea58865"
    assert len(rows) == 41
    assert sum(row["phase"] == "development" for row in rows) == 17
    assert sum(row["phase"] == "confirmation" for row in rows) == 24
    assert {row["seed"] for row in rows if row["phase"] == "development"} == {3407}
    assert {row["seed"] for row in rows if row["phase"] == "confirmation"} == {4407, 5407, 6407}
    assert {row["arm_id"] for row in rows if row["confirmation"]} == {
        "U0",
        "H0",
        "A10",
        "A11",
        "C11",
        "D1",
        "E01",
        "F11",
    }
    for row in rows:
        assert (ROOT / row["config_path"]).is_file()


def test_generated_primary_and_mod_configs_preserve_successful_update_contract():
    payload = json.loads(MATRIX.read_text(encoding="utf-8"))
    by_task = {row["task_id"]: row for row in payload["rows"]}

    primary = Config.fromfile(str(ROOT / by_task["development_a11_seed3407"]["config_path"]))
    assert primary.seed == 3407
    assert primary.model.frame_selector.acquisition_policy == "robust_phase"
    assert primary.model.frame_selector.phase_quota_mode == "adaptive"
    assert primary.model.frame_selector.actionness_source_cfg.type == "C3CoarseProbeActionnessSource"
    assert primary.model.frame_selector.actionness_source_cfg.probe_model == "official-action-seg"
    assert primary.model.frame_selector.phase_use_curvature is False
    assert primary.workflow.end_epoch == 60
    assert primary.workflow.max_train_iters == 100
    assert primary.workflow.max_amp_retries_per_batch == 8
    assert primary.max_successful_updates == 6000
    assert primary.workflow.val_eval_interval == 0
    assert primary.terminal_state_key == "state_dict_ema"
    assert primary.inference.load_from_raw_predictions is False
    assert primary.inference.save_raw_prediction is True

    mod = Config.fromfile(str(ROOT / by_task["development_e01_seed3407"]["config_path"]))
    assert mod.model.backbone.backbone.amod_config.enabled is True
    assert mod.model.backbone.backbone.amod_config.final_capacity if "final_capacity" in mod.model.backbone.backbone.amod_config else True
    assert mod.model.backbone.backbone.amod_config.capacity == 0.65
    assert mod.model.backbone.backbone.amod_config.dense_companion_period_successful_steps == 8


def test_after_optimizer_step_helper_unwraps_ddp_like_module():
    class Runtime(nn.Module):
        def __init__(self):
            super().__init__()
            self.count = 0

        def after_optimizer_step(self):
            self.count += 1
            return {"count": self.count}

    class Wrapped:
        def __init__(self, module):
            self.module = module

    runtime = Runtime()
    train_engine = _load_train_engine()
    assert train_engine._call_after_optimizer_step(Wrapped(runtime)) == {"count": 1}
    assert runtime.count == 1


def test_train_script_resolves_seed_from_config_and_rejects_cli_conflict():
    cfg = Config(dict(seed=3407))

    assert resolve_effective_seed(cfg, None) == 3407
    assert resolve_effective_seed(cfg, 3407) == 3407
    with pytest.raises(ValueError, match="conflicts with config seed"):
        resolve_effective_seed(cfg, 42)


def test_train_one_epoch_stops_on_successful_update_target():
    class Runtime(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(1.0))
            self.step_count = 0

        def forward(self, **kwargs):
            return {"cost": self.weight * 1.0}

        def after_optimizer_step(self):
            self.step_count += 1

    class Wrapped(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, **kwargs):
            return self.module(**kwargs)

    class Logger:
        def info(self, *args, **kwargs):
            pass

    runtime = Runtime()
    model = Wrapped(runtime)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    train_engine = _load_train_engine()
    train_engine.reduce_loss = lambda losses: losses

    updates = train_engine.train_one_epoch(
        [{} for _ in range(5)],
        model,
        optimizer,
        scheduler,
        curr_epoch=0,
        logger=Logger(),
        logging_interval=1000,
        max_train_iters=5,
        max_successful_updates=2,
        successful_updates_start=0,
    )

    assert updates == 2
    assert runtime.step_count == 2


def test_aggregate_requires_cost_and_bootstrap_before_complete(tmp_path):
    project = tmp_path / "project"
    matrix_path = project / "scripts" / "duca_unified_fullmatrix" / "matrix.json"
    matrix_path.parent.mkdir(parents=True)
    matrix_path.write_text(
        json.dumps(
            {
                "schema_version": "duca_unified_fullmatrix_tasks_v1",
                "matrix_id": "DUCA-UNIFIED-FULLMATRIX-v001-20260902",
                "base_revision": "base",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    summary = build_summary(matrix_path, run_root=tmp_path / "run")
    cost = build_summary(matrix_path, run_root=tmp_path / "run", cost_index=0)

    assert summary["status"] == "INCOMPLETE"
    assert summary["bootstrap"]["complete"] is False
    assert summary["cost"]["complete"] is False
    assert cost["status"] == "INCOMPLETE"


def test_selector_loss_schedule_advances_only_after_pending_successful_step():
    selector = DucaOnlineFrameSelector(
        in_channels=3,
        budget=4,
        dense_window_size=8,
        selector_hidden_channels=0,
        acquisition_policy="exact_uniform",
        actionness_source_cfg={
            "type": "ZeroShotMotionActionnessSource",
            "mode": "motion",
            "source_name": "unit_motion",
            "thumos_trained": False,
            "uses_labels": False,
            "uses_teacher": False,
            "uses_gt": False,
            "uses_prediction_cache": False,
            "calibration_split": "none",
        },
        loss_weight_schedule={
            "type": "progressive_joint",
            "warmup_steps": 2,
            "transition_steps": 2,
            "shape": "linear",
            "detector_utility": {"start": 0.0, "end": 0.25},
        },
    )

    assert selector.after_optimizer_step()["updated"] is False
    assert int(selector._loss_weight_schedule_step.item()) == 0

    selector.train()
    selector._record_pending_loss_schedule_step()
    assert selector.after_optimizer_step()["updated"] is True
    assert int(selector._loss_weight_schedule_step.item()) == 1
