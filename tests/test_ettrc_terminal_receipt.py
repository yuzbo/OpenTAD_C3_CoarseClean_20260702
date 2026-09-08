import copy
import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata import ettrc_terminal_receipt as receipt
from tools.bata.ettrc_terminal_receipt import canonical_hash, seal_payload, terminal_counts, validate_request, write_receipt


ROOT = Path(__file__).resolve().parents[1]


def checkpoint():
    return dict(epoch=59, state_dict_ema={"weight": [1]},
                optimizer={"state": {0: {"step": 6000}, 1: {"step": 6000}}},
                scheduler={"last_epoch": 6000})


def test_terminal_counts_require_actual_optimizer_steps():
    payload = checkpoint()
    assert terminal_counts(payload)["successful_optimizer_updates"] == 6000
    payload["optimizer"]["state"][0]["step"] = 5999
    with pytest.raises(ValueError, match="optimizer"):
        terminal_counts(payload)


@pytest.mark.parametrize("key,value", [("epoch", 58), ("state_dict_ema", {}), ("scheduler", {"last_epoch": 5999})])
def test_terminal_counts_reject_incomplete_checkpoint(key, value):
    payload = checkpoint()
    payload[key] = value
    with pytest.raises(ValueError):
        terminal_counts(payload)


@pytest.mark.parametrize("arm", ["on", "off"])
def test_off_on_receipt_contract(arm):
    cfg = Config.fromfile(str(ROOT / f"configs/adatad/thumos/ettrc_test_guided_{arm}_seed4407.py"))
    cfg.post_processing.save_dict = True
    validate_request(cfg, seed=4407, world_size=2, not_eval=False, max_batches=None)
    with pytest.raises(ValueError, match="complete fresh"):
        validate_request(cfg, seed=4407, world_size=2, not_eval=True, max_batches=1)
    cfg.test_guided_exploratory = False
    with pytest.raises(ValueError, match="eval5"):
        validate_request(cfg, seed=4407, world_size=2, not_eval=False, max_batches=None)


@pytest.fixture
def training_run(tmp_path, monkeypatch):
    monkeypatch.setattr(receipt, "TRAINING_ROOT", tmp_path)
    checkpoint_path = tmp_path / "on/gpu2_id0/checkpoint/epoch_59.pth"
    protocol_path = checkpoint_path.parents[1] / "test_guided/protocol.json"
    protocol_path.parent.mkdir(parents=True)
    protocol = dict(source_sha=receipt.TRAINING_COMMIT, seed=4407, protocol=receipt.PROTOCOL,
                    model_selection_uses_test=True, unseen_test_claim_allowed=False,
                    evaluation_epochs=list(range(5, 61, 5)),
                    test_video_ids=[f"video_test_{i:07d}" for i in range(211)], test_window_count=792)
    protocol_path.write_text(json.dumps(seal_payload(protocol)), encoding="utf-8")
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/ettrc_test_guided_on_seed4407.py"))
    return cfg, checkpoint_path, protocol_path, protocol


def test_training_binding_rejects_wrong_arm_and_old_namespace(training_run):
    cfg, checkpoint_path, _, _ = training_run
    binding = receipt.training_binding(cfg, checkpoint_path, seed=4407)
    assert len(binding["evaluated_video_ids"]) == 211
    for wrong_path in (checkpoint_path.parents[4] / "old/epoch_59.pth",
                       receipt.TRAINING_ROOT / "off/gpu2_id0/checkpoint/epoch_59.pth"):
        with pytest.raises(ValueError, match="frozen training arm"):
            receipt.training_binding(cfg, wrong_path, seed=4407)


@pytest.mark.parametrize("key,value", [("source_sha", "74473c27"), ("seed", 42),
                                      ("model_selection_uses_test", False)])
def test_training_binding_rejects_incompatible_protocol(training_run, key, value):
    cfg, checkpoint_path, protocol_path, protocol = training_run
    protocol[key] = value
    protocol_path.write_text(json.dumps(seal_payload(protocol)), encoding="utf-8")
    with pytest.raises(ValueError, match="protocol"):
        receipt.training_binding(cfg, checkpoint_path, seed=4407)


def test_training_binding_rejects_changed_receipt(training_run):
    cfg, checkpoint_path, protocol_path, _ = training_run
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    payload["seed"] = 42
    protocol_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        receipt.training_binding(cfg, checkpoint_path, seed=4407)


def test_population_binding_requires_all_windows_and_label_free_inference(training_run):
    cfg, checkpoint_path, _, protocol = training_run
    binding = receipt.training_binding(cfg, checkpoint_path, seed=4407)

    class Dataset:
        test_mode = True
        data_list = [(protocol["test_video_ids"][i % 211],) for i in range(792)]

        def __len__(self):
            return len(self.data_list)

    dataset = Dataset()
    receipt.validate_test_population(dataset, binding)
    dataset.test_mode = False
    with pytest.raises(ValueError, match="label-free"):
        receipt.validate_test_population(dataset, binding)
    dataset.test_mode = True
    dataset.data_list = dataset.data_list[:-1]
    with pytest.raises(ValueError, match="complete"):
        receipt.validate_test_population(dataset, binding)


def test_self_hash_detects_changed_metric_without_mutating_input():
    payload = {"metrics": {"average_mAP": 0.5}, "training_git_commit": "original"}
    before = copy.deepcopy(payload)
    sealed = seal_payload(payload)
    digest = sealed.pop("receipt_sha256")
    assert digest == canonical_hash(sealed)
    assert payload == before
    sealed["metrics"] = {"average_mAP": 0.6}
    assert digest != canonical_hash(sealed)


def test_writer_serializes_complete_real_config(tmp_path):
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/ettrc_test_guided_on_seed4407.py"))
    artifact = tmp_path / "input.json"
    artifact.write_text("{}")
    cfg.work_dir = str(tmp_path)
    cfg.evaluation.ground_truth_filename = str(artifact)
    cfg.dataset.test.class_map = str(artifact)
    (tmp_path / "result_detection.json").write_text("{}")
    output = tmp_path / "metrics.json"
    metrics = {key: 0.5 for key in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7")}
    write_receipt(output, cfg=cfg, config_path=artifact, checkpoint_path=artifact,
                  checkpoint=checkpoint(), metrics=metrics, identity={"clean_tree": True},
                  seed=4407, world_size=2)
    sealed = json.loads(output.read_text())
    digest = sealed.pop("receipt_sha256")
    assert digest == canonical_hash(sealed)
    assert sealed["metrics"] == metrics
    assert sealed["checkpoint_state_key"] == "state_dict_ema"
    assert sealed["protocol"] == receipt.PROTOCOL
    assert sealed["evaluation_role"] == "INDEPENDENT_TERMINAL_EMA"
    assert sealed["model_selection_uses_test"] is True
    assert sealed["unseen_test_claim_allowed"] is False
    assert sealed["ema_update_count_stored"] is False
