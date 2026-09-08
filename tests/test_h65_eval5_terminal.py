from copy import deepcopy
import json
from pathlib import Path

from mmengine.config import Config
import pytest

from tools.bata import h65_eval5_terminal as terminal

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def binding(tmp_path, monkeypatch):
    source, runs = tmp_path / "source", tmp_path / "runs"
    monkeypatch.setattr(terminal, "TRAINING_SOURCE", source)
    monkeypatch.setattr(terminal, "TRAINING_ROOT", runs)
    monkeypatch.setattr(terminal, "_git", lambda root, *args: (
        terminal.TRAINING_COMMIT if args == ("rev-parse", "HEAD") else ""
    ))
    protocol = dict(
        source_sha=terminal.TRAINING_COMMIT, seed=3407, protocol=terminal.PROTOCOL,
        model_selection_uses_test=True, unseen_test_claim_allowed=False,
        evaluation_epochs=list(range(5, 61, 5)),
        test_video_ids=[f"video_{i}" for i in range(211)], test_window_count=792,
    )
    def make(arm):
        cfg = Config.fromfile(str(ROOT / f"configs/adatad/thumos/h65_pro/h65_pro_eval5_{arm}.py"))
        run = runs / f"{cfg.h65_pro_experiment_id}_seed3407/gpu1_id0"
        p = run / "test_guided/protocol.json"
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps(dict(protocol, receipt_sha256=terminal.canonical_sha256(protocol))))
        args = dict(
            config_path=source / f"configs/adatad/thumos/h65_pro/h65_pro_eval5_{arm}.py",
            checkpoint_path=run / "checkpoint/epoch_59.pth", evaluator_root=tmp_path / "evaluator",
            seed=3407, world_size=1, variant=f"h65_pro_eval5_{arm}",
        )
        return cfg, args, p
    return make


@pytest.mark.parametrize("arm", ["uniform", "phaseoff", "phaseon"])
def test_real_configs_bind_frozen_arm_and_disclose_test_reuse(binding, arm):
    cfg, args, _ = binding(arm)
    result = terminal.evaluation_binding(cfg, **args)
    assert result["protocol"] == terminal.PROTOCOL
    assert result["evaluation_role"] == "INDEPENDENT_TERMINAL_EMA"
    assert result["model_selection_uses_test"] and not result["unseen_test_claim_allowed"]
    assert len(result["evaluated_video_ids"]) == 211


@pytest.mark.parametrize("key,value", [
    ("seed", 4407), ("world_size", 2), ("variant", "h65_pro_f09"),
    ("checkpoint_path", "/old/epoch_59.pth"), ("config_path", "/evaluator/config.py"),
])
def test_rejects_wrong_run_identity(binding, key, value):
    cfg, args, _ = binding("phaseon")
    args[key] = value
    with pytest.raises(ValueError):
        terminal.evaluation_binding(cfg, **args)


@pytest.mark.parametrize("change", ["source", "disclosure", "hash", "population"])
def test_rejects_wrong_protocol(binding, change):
    cfg, args, p = binding("phaseoff")
    doc = json.loads(p.read_text())
    doc.pop("receipt_sha256")
    if change == "source":
        doc["source_sha"] = "old"
    elif change == "disclosure":
        doc["model_selection_uses_test"] = False
    elif change == "population":
        doc["test_video_ids"].pop()
    doc["receipt_sha256"] = "wrong" if change == "hash" else terminal.canonical_sha256(doc)
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        terminal.evaluation_binding(cfg, **args)


def test_rejects_model_drift(binding, monkeypatch):
    cfg, args, _ = binding("uniform")
    monkeypatch.setattr(terminal, "_git", lambda root, *a: (
        terminal.TRAINING_COMMIT if a == ("rev-parse", "HEAD") else
        "opentad/models/backbones/vit_adapter.py" if a[0] == "diff" else ""
    ))
    with pytest.raises(ValueError, match="model or configuration"):
        terminal.evaluation_binding(cfg, **args)


def test_dataset_population_and_label_free_mode(binding):
    cfg, args, _ = binding("uniform")
    result = terminal.evaluation_binding(cfg, **args)
    class Dataset:
        test_mode = True
        data_list = [(x,) for x in result["evaluated_video_ids"]]
        def __len__(self):
            return 792
    dataset = Dataset()
    terminal.validate_test_population(dataset, result)
    dataset.test_mode = False
    with pytest.raises(ValueError, match="label-free"):
        terminal.validate_test_population(dataset, result)


def test_global_successful_count_is_not_conditional_parameter_participation():
    checkpoint = dict(epoch=59, state_dict_ema={"weight": 1}, successful_optimizer_updates=6000,
                      scheduler={"last_epoch": 6000}, optimizer={"state": {0: {"step": 6000}, 1: {"step": 5044}}})
    counts = terminal.terminal_counts(checkpoint)
    assert counts["successful_optimizer_updates"] == 6000
    assert counts["optimizer_steps_distribution"] == {6000: 1, 5044: 1}
    for key, value in (("successful_optimizer_updates", 5999), ("epoch", 49)):
        bad = deepcopy(checkpoint)
        bad[key] = value
        with pytest.raises(ValueError):
            terminal.terminal_counts(bad)
    bad = deepcopy(checkpoint)
    bad["scheduler"]["last_epoch"] = 5999
    with pytest.raises(ValueError):
        terminal.terminal_counts(bad)


def test_precheck_cannot_write_formal_metrics_and_uses_existing_evaluator():
    source = (ROOT / "tools/test.py").read_text()
    assert "if args.rank == 0 and args.metrics_json and not args.precheck_only:" in source
    assert "not_eval=args.not_eval or args.precheck_only" in source
    assert "yield next(iter(test_loader))" in source
    assert "h65_eval5_terminal.TRAINING_COMMIT if h65_eval5" in source
    assert "h65_eval5_identity.update(h65_eval5_terminal.terminal_counts(checkpoint))" in source
