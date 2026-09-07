import copy
import json
import random

import numpy as np
import pytest
import torch
from mmengine.config import ConfigDict

from geosparse_ext.runtime import ModelEMA
from geosparse_ext.training_validation import ema_evaluation, validate_dataset, update_best_checkpoint, periodic_validation


class Probe(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(2, 2)
        self.frozen = torch.nn.Linear(2, 2).requires_grad_(False)
        self.register_buffer("counter", torch.tensor(3))
        self.register_buffer("cache", torch.ones(2), persistent=False)
        self.geosparse = torch.nn.Module()
        self.geosparse.encoder = None
        self.epoch, self.minibatch = 4, 17
        self.latest_route_plan = "training plan"
        self.latest_acquisition = None
        self.pending_cost = []
        self.inference_seed, self.inference_video_index = 99, {"old": 7}

    def forward_test(self, **batch):
        assert not self.training
        self.latest_route_plan = "validation plan"
        self.geosparse.execution_rng = torch.get_rng_state()
        return None

    def post_processing(self, predictions, metas, post, class_map):
        return {meta["video_name"]: ([dict(label="a", segment=[0., 1.], score=.9)]
                                    if meta["video_name"] == "v1" else []) for meta in metas}


@pytest.mark.parametrize("fail", [False, True])
def test_ema_validation_restores_training_state_and_rng_even_on_failure(fail):
    model = Probe().train()
    model.frozen.eval()
    model.head.weight.grad = torch.ones_like(model.head.weight)
    ema = ModelEMA(model)
    ema.shadow["head.weight"].add_(2.)
    before = copy.deepcopy(model)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected = (random.random(), np.random.rand(), torch.rand(3))
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    try:
        with ema_evaluation(model, ema, 2, ["v2", "v1"]):
            torch.testing.assert_close(model.head.weight, ema.shadow["head.weight"])
            assert model.inference_video_index == {"v1": 0, "v2": 1}
            assert not model.training and not model.frozen.training
            model.head.weight.add_(10.)
            model.counter.add_(4)
            model.cache.zero_()
            model.latest_route_plan = "changed"
            model.geosparse.execution_rng = "changed"
            random.random(), np.random.rand(), torch.rand(5)
            if fail:
                raise RuntimeError("injected evaluator failure")
    except RuntimeError as exc:
        assert fail and str(exc) == "injected evaluator failure"
    assert model.training and not model.frozen.training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before.state_dict()[name], atol=0, rtol=0)
    torch.testing.assert_close(model.cache, before.cache, atol=0, rtol=0)
    assert torch.equal(model.head.weight.grad, torch.ones_like(model.head.weight))
    assert model.latest_route_plan == "training plan" and model.minibatch == 17
    assert model.inference_seed == 99 and model.inference_video_index == {"old": 7}
    assert not hasattr(model.geosparse, "execution_rng")
    assert random.random() == expected[0] and np.random.rand() == expected[1]
    torch.testing.assert_close(torch.rand(3), expected[2], atol=0, rtol=0)


def validation_fixture(tmp_path, monkeypatch, omit_second_video=False):
    annotation = tmp_path / "annotations.json"
    annotation.write_text(json.dumps(dict(database={
        "v1": dict(subset="validation", annotations=[dict(label="a", segment=[0., 1.])]),
        "v2": dict(subset="validation", annotations=[dict(label="a", segment=[2., 3.])]),
        "dev": dict(subset="internal_dev", annotations=[dict(label="a", segment=[0., 1.])]),
    })))
    cfg = ConfigDict(dict(dataset=dict(train=dict(data_path="train-videos"),
        test=dict(ann_file=str(annotation), data_path="all-test-videos", subset_name="validation")),
        solver=dict(amp=False), post_processing=dict(nms=dict(use_soft_nms=True, sigma=.7, multiclass=True)),
        evaluation=dict(type="mAP", ground_truth_filename=str(annotation), subset="validation",
                        tiou_thresholds=[.3, .4, .5, .6, .7])))
    names = ["v1"] if omit_second_video else ["v1", "v2"]

    class Dataset:
        class_map = ["a"]

        def __len__(self):
            return len(names)

    class Data(list):
        dataset = Dataset()

    def fake_loader(config, batch_size, workers, seed):
        assert config.subset_name == "validation" and config.data_path == "all-test-videos"
        return Data([dict(metas=[dict(video_name=name)]) for name in names])

    monkeypatch.setattr("geosparse_ext.runtime.loader", fake_loader)
    return cfg, dict(evaluation_batch=1, num_workers=0), dict(job_id="unit-test-only", seed=1)


def test_full_validation_counts_missed_test_video_and_restores_training(tmp_path, monkeypatch):
    cfg, runtime, job = validation_fixture(tmp_path, monkeypatch)
    model = Probe().train()
    row = validate_dataset(model, ModelEMA(model), cfg, runtime, job, {}, tmp_path / "measured", 5)
    assert row["status"] == "completed_validation" and row["subset"] == "validation"
    assert row["videos"] == ["v1", "v2"] and row["windows"] == 2
    assert row["metrics"]["average_mAP"] == pytest.approx(.5)
    assert row["metrics"]["mAP@0.7"] == pytest.approx(.5)
    assert model.training and model.latest_route_plan == "training plan"
    assert row["is_final_result"] is False


def test_partial_test_coverage_records_failure_without_changing_model(tmp_path, monkeypatch):
    cfg, runtime, job = validation_fixture(tmp_path, monkeypatch, omit_second_video=True)
    model = Probe().train()
    row = validate_dataset(model, ModelEMA(model), cfg, runtime, job, {}, tmp_path / "measured", 5)
    assert row["status"] == "failed_validation" and "exact split" in row["reason"]
    assert "metrics" not in row and model.training


def test_best_checkpoint_uses_peak_average_map_and_keeps_earlier_tie(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    last = checkpoint / "last.pth"
    for epoch, score in [(5, 0.), (10, .4), (15, .3), (20, .4)]:
        last.write_bytes(f"checkpoint-{epoch}".encode())
        row = dict(status="completed_validation", subset="validation", completed_epochs=epoch,
                   metrics=dict(average_mAP=score), provenance={"source_commit": "unit-test"},
                   source_train_id="unit-test-only", seed=0)
        update_best_checkpoint(row, last, tmp_path)
    best = json.loads((tmp_path / "best.json").read_text())
    assert best["completed_epochs"] == 10 and best["checkpoint_epoch"] == 9
    assert best["score"] == .4 and (checkpoint / "best.pth").read_bytes() == b"checkpoint-10"
    # A later training checkpoint remains available and is not reported as best.
    assert last.read_bytes() == b"checkpoint-20"


def test_periodic_validation_resumes_pending_measurement_at_saved_epoch(tmp_path, monkeypatch):
    calls = []

    def measure(*args):
        destination, completed = args[-2:]
        calls.append(completed)
        destination.mkdir(parents=True)
        row = dict(status="completed_validation", completed_epochs=completed, provenance={})
        (destination / "metrics.json").write_text(json.dumps(row))
        return row

    monkeypatch.setattr("geosparse_ext.training_validation.validate_dataset", measure)
    for completed in [0, 4, 5, 5, 6, 10, 60]:
        periodic_validation(None, None, None, None, None, {}, tmp_path, completed)
    assert calls == [5, 10, 60]


def test_final_model_loader_uses_selected_best_epoch_and_ema(tmp_path, monkeypatch):
    from geosparse_ext.runtime import load_trained_model

    model = Probe()
    ema = ModelEMA(model)
    ema.shadow["head.weight"].add_(2.)
    provenance = {key: "unit-test" for key in ("source_commit", "resolved_config_sha256", "split_sha256", "weights_sha256")}
    checkpoint = tmp_path / "best.pth"
    torch.save(dict(format="geosparse_full_state_v2", epoch=9, provenance=provenance,
                    state_dict_ema=ema.state_dict()), checkpoint)
    receipt = dict(status="completed", is_mock=False, completed_epochs=60, selected_checkpoint_epoch=9,
                   checkpoint_path=str(checkpoint))
    (tmp_path / "result.json").write_text(json.dumps(receipt))
    monkeypatch.setattr("geosparse_ext.runtime.build_detector", lambda cfg: copy.deepcopy(model))
    monkeypatch.setattr(torch.nn.Module, "cuda", lambda self: self)
    job = dict(depends_on=["own-training"], dependency_outputs={"own-training": str(tmp_path)}, seed=1)
    selected = load_trained_model(job, ConfigDict(dict(model={})), tmp_path, provenance)
    assert selected.epoch == 9 and selected.inference_seed == 1 and not selected.training
    torch.testing.assert_close(selected.head.weight, ema.shadow["head.weight"])
    receipt["selected_checkpoint_epoch"] = 59
    (tmp_path / "result.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="selected checkpoint epoch"):
        load_trained_model(job, ConfigDict(dict(model={})), tmp_path, provenance)


def test_evaluation_capability_requires_real_pipeline_receipt_for_same_configuration(tmp_path):
    from geosparse_ext.matrix import compile_all
    from geosparse_ext.capabilities import certify

    jobs, _ = compile_all()
    checked = next(job for job in jobs if job["kind"] == "train" and job["route"] == "B"
                   and job["family"] == "F01" and job["dataset"] == "thumos14" and job["seed"] == 0)
    precheck = dict(is_mock=False, completed_epochs=0, source_commit="unit-test", routes={"B": dict(
        status="PASS", job_id=checked["job_id"], optimizer_updates=1, ema_update="PASS")})
    path = tmp_path / "precheck.json"
    path.write_text(json.dumps(precheck))
    bindings = dict(source_commit="unit-test", capability_dir=str(tmp_path / "caps"))
    assert "evaluation" not in certify(jobs, path, bindings)
    precheck["routes"]["B"]["evaluation_pipeline_check"] = dict(status="PASS")
    path.write_text(json.dumps(precheck))
    assert certify(jobs, path, bindings)["evaluation"] == 3
    cap = json.loads((tmp_path / "caps/evaluation.json").read_text())
    selected = [job for job in jobs if job["job_id"] in cap["supported_variants"]]
    assert {job["seed"] for job in selected} == {0, 1, 2}
    assert all(job["model"] == checked["model"] and job["dataset"] == checked["dataset"] for job in selected)
