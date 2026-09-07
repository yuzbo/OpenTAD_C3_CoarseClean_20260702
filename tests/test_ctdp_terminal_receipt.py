import copy
import json
from pathlib import Path

import pytest
from mmengine.config import Config

from tools.bata.ctdp_terminal_receipt import (
    TRAINING_COMMIT, checkpoint_counts, seal_payload, validate_request, write_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/adatad/thumos/duca_ctdp_geometry_g0.py"


def checkpoint(precheck=False):
    epoch, count, batches = (1, 6, 3) if precheck else (59, 6000, 100)
    return dict(epoch=epoch, state_dict_ema={"weight": 1},
                optimizer={"state": {0: {"step": count}, 1: {"step": count}}},
                scheduler={"last_epoch": count}, experiment_metadata=dict(
                    route="CT-DP", git_commit=TRAINING_COMMIT, clean_tree=True,
                    config_path=str(CONFIG), seed=3407, epoch=epoch,
                    checkpoint_state_key="state_dict_ema",
                    contract=dict(formal=not precheck, epochs=epoch + 1, batches_per_epoch=batches,
                                  expected_successful_optimizer_updates=count),
                    update_audit=dict(successful_optimizer_updates=count,
                                      scheduler_updates=count, ema_updates=count)))


@pytest.mark.parametrize("arm", range(4))
def test_frozen_configs_preserve_evaluation_recipe(arm):
    cfg = Config.fromfile(str(CONFIG).replace("g0.py", f"g{arm}.py"))
    validate_request(cfg, seed=3407, world_size=1, not_eval=False, max_batches=None,
                     precheck=False, cfg_options={"work_dir": "new"})
    for options in [{"post_processing.nms.sigma": 0.1}, {"solver.amp": False}]:
        with pytest.raises(ValueError, match="scientific config"):
            validate_request(cfg, seed=3407, world_size=1, not_eval=False, max_batches=None,
                             precheck=False, cfg_options=options)


def test_terminal_and_precheck_are_distinct():
    assert checkpoint_counts(checkpoint(), CONFIG)["successful_optimizer_updates"] == 6000
    assert checkpoint_counts(checkpoint(True), CONFIG, precheck=True)["successful_optimizer_updates"] == 6
    with pytest.raises(ValueError, match="epoch/EMA"):
        checkpoint_counts(checkpoint(True), CONFIG)


@pytest.mark.parametrize("field,value", [("git_commit", "old"), ("seed", 4407),
                                          ("config_path", "duca_ctdp_geometry_g1.py"),
                                          ("clean_tree", False)])
def test_checkpoint_identity_is_not_transferable(field, value):
    saved = checkpoint()
    saved["experiment_metadata"][field] = value
    with pytest.raises(ValueError, match="identity"):
        checkpoint_counts(saved, CONFIG)


def test_actual_optimizer_deficit_cannot_hide_behind_metadata():
    saved = checkpoint()
    saved["optimizer"]["state"][0]["step"] = 5997
    with pytest.raises(ValueError, match="actual optimizer"):
        checkpoint_counts(saved, CONFIG)
    saved = checkpoint()
    saved["experiment_metadata"]["update_audit"]["ema_updates"] = 5997
    with pytest.raises(RuntimeError, match="update mismatch"):
        checkpoint_counts(saved, CONFIG)


def test_partial_inference_cannot_create_terminal_metrics():
    cfg = Config.fromfile(str(CONFIG))
    with pytest.raises(ValueError, match="complete official"):
        validate_request(cfg, seed=3407, world_size=1, not_eval=True, max_batches=1,
                         precheck=False, cfg_options=None)
    validate_request(cfg, seed=3407, world_size=1, not_eval=True, max_batches=1,
                     precheck=True, cfg_options=None)


def test_receipt_uses_production_evaluator_file_and_seals_exact_values(tmp_path):
    cfg = Config.fromfile(str(CONFIG))
    cfg.work_dir = str(tmp_path)
    cfg.evaluation.ground_truth_filename = str(tmp_path / "annotation.json")
    cfg.dataset.test.class_map = str(tmp_path / "classes.txt")
    for name in ["annotation.json", "classes.txt", "checkpoint.pth", "result_detection.json"]:
        (tmp_path / name).write_text("fixture", encoding="utf-8")
    metrics = {"average_mAP": .5, **{f"mAP@{iou}": .5 for iou in [.3, .4, .5, .6, .7]},
               "evaluation_epoch": 59}
    (tmp_path / "evaluation_metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    output = tmp_path / "receipt.json"
    sealed = write_receipt(output, cfg=cfg, config_path=CONFIG,
                           checkpoint_path=tmp_path / "checkpoint.pth", checkpoint=checkpoint(),
                           identity=dict(training_git_commit=TRAINING_COMMIT), video_ids=["video_test_fixture"])
    on_disk = json.loads(output.read_text())
    assert json.loads(json.dumps(sealed)) == on_disk == seal_payload(on_disk)
    assert sealed["metrics"]["average_mAP"] == .5
    changed = copy.deepcopy(sealed)
    changed["metrics"]["average_mAP"] = .6
    assert seal_payload(changed)["receipt_sha256"] != sealed["receipt_sha256"]


def test_entrypoint_keeps_production_model_and_evaluator_unchanged():
    source = (ROOT / "tools/test.py").read_text(encoding="utf-8")
    assert "checkpoint_counts(checkpoint, args.config" in source
    assert "eval_one_epoch(" in source
    assert "CTDP_TERMINAL_RECEIPT_OK" in source
    launcher = (ROOT / "scripts/run_ctdp_terminal_eval_n16r4.sbatch").read_text(encoding="utf-8")
    assert "--ctdp-eval-precheck --not_eval --max-batches 1" in launcher
    assert "checkpoint/epoch_59.pth" in launcher
