import copy
import json
from pathlib import Path

import pytest
from mmengine.config import Config

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


@pytest.mark.parametrize("suffix", ["", "off_"])
def test_off_on_receipt_contract(suffix):
    cfg = Config.fromfile(str(ROOT / f"configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_{suffix}seed4407.py"))
    cfg.post_processing.save_dict = True
    validate_request(cfg, seed=4407, world_size=2, not_eval=False, max_batches=None)
    with pytest.raises(ValueError, match="complete fresh"):
        validate_request(cfg, seed=4407, world_size=2, not_eval=True, max_batches=1)


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
    cfg = Config.fromfile(str(ROOT / "configs/adatad/thumos/et_trc_videomae_s_768x1_160_adapter_seed4407.py"))
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
