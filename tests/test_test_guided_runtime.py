import json
import random
from types import SimpleNamespace

import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("Torch unavailable on this host", allow_module_level=True)

import numpy as np
from mmengine.config import Config
from opentad.utils.test_guided_eval import TestGuidedEvaluation, preserve_training_state


def test_evaluation_restores_rng_and_each_module_mode():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout())
    model[0].eval()
    random.seed(31)
    np.random.seed(31)
    torch.manual_seed(31)
    before = torch.get_rng_state().clone()
    py_before, np_before = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError):
        with preserve_training_state(model):
            model.eval()
            torch.rand(3)
            random.random()
            np.random.rand()
            raise RuntimeError("evaluation failure")
    assert torch.equal(before, torch.get_rng_state())
    assert random.getstate() == py_before
    np.testing.assert_equal(np.random.get_state(), np_before)
    assert model.training and not model[0].training and model[1].training


def test_real_best_ema_is_saved_and_terminal_is_not_overwritten(tmp_path, monkeypatch):
    import opentad.utils.test_guided_eval as module
    monkeypatch.setattr(module.subprocess, "check_output", lambda command, **kw: "" if "status" in command else "a" * 40)
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"database": {"a": {"subset": "validation"}, "b": {"subset": "validation"}}}))
    cfg = Config(dict(work_dir=str(tmp_path), workflow=dict(val_eval_interval=5, val_start_epoch=4, val_loss_interval=-1),
                      solver=dict(ema=True), inference=dict(load_from_raw_predictions=False), post_processing={},
                      evaluation=dict(subset="validation", ground_truth_filename=str(annotation))))
    class Dataset:
        test_mode = True
        data_list = [("a",), ("b",)]
        def __len__(self):
            return len(self.data_list)
    loader = SimpleNamespace(dataset=Dataset(), drop_last=False)
    args = SimpleNamespace(not_eval=False, rank=0, world_size=1, seed=4407)
    evaluator = TestGuidedEvaluation(cfg, args, loader)
    model = torch.nn.Linear(2, 2)
    ema = SimpleNamespace(module=torch.nn.Linear(2, 2))
    logger = SimpleNamespace(info=lambda *args: None)
    def evaluate(*args, **kwargs):
        assert not kwargs["not_eval"]
        assert "max_batches" not in kwargs
        return {"metrics": {"average_mAP": 0.6, "mAP@0.7": 0.4}}
    evaluator.run(4, 500, model, ema, logger, False, evaluate)
    path = tmp_path / "test_guided/best_test.pth"
    saved = torch.load(path, map_location="cpu", weights_only=False)
    assert saved["epoch"] == 4
    torch.testing.assert_close(saved["state_dict_ema"]["weight"], ema.module.weight)
    with torch.no_grad():
        ema.module.weight.add_(5)
    evaluator.run(9, 1000, model, ema, logger, False, evaluate)
    assert torch.load(path, map_location="cpu", weights_only=False)["epoch"] == 4
    record = json.loads((tmp_path / "test_guided/epoch_09/metrics.json").read_text())
    assert record["model_selection_uses_test"] and not record["unseen_test_claim_allowed"]
    assert not (tmp_path / "checkpoint/epoch_59.pth").exists()


def test_partial_test_population_is_rejected(tmp_path):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps({"database": {"a": {"subset": "validation"}, "b": {"subset": "validation"}}}))
    cfg = Config(dict(work_dir=str(tmp_path), workflow=dict(val_eval_interval=5, val_start_epoch=4), solver=dict(ema=True),
                      inference={}, evaluation=dict(subset="validation", ground_truth_filename=str(annotation))))
    args = SimpleNamespace(not_eval=False, rank=0)
    loader = SimpleNamespace(dataset=SimpleNamespace(data_list=[("a",)], test_mode=True), drop_last=False)
    with pytest.raises(ValueError, match="complete official"):
        TestGuidedEvaluation(cfg, args, loader)
