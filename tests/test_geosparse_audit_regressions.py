"""Regression evidence for reachable audit failures; no scientific result output."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from geosparse_ext.geometry import detector_validity, native_layout
from geosparse_ext.contracts import VideoBatch
from geosparse_ext.matrix import base, compile_all, matrix_summary, validate
from geosparse_ext.protocol import implementation_blockers
from geosparse_ext.training_validation import ema_evaluation, missing_validations, repair_missing_validations
from geosparse_ext.runtime import ModelEMA
from geosparse_ext.records import checkpoint_identity, require_same_checkpoint
from test_geosparse_detector import detector
from test_geosparse_training_validation import Probe


def video(length=768, valid=767, batch=1):
    clock = torch.arange(length, dtype=torch.float64)[None].expand(batch, -1) / 25
    return VideoBatch(torch.randn(batch, 3, length, 32, 32), clock, clock,
                      torch.arange(length)[None].expand(batch, -1) < valid,
                      torch.arange(length)[None].expand(batch, -1), torch.stack((clock, clock + .04), -1),
                      [f"fixture-{i}" for i in range(batch)], [f"fixture-{i}:0" for i in range(batch)],
                      torch.eye(3)[None].expand(batch, -1, -1))


@pytest.mark.parametrize("route", ["DENSE", "A", "B", "C"])
def test_odd_tail_reaches_real_detector_with_original_frame_validity(route):
    model = detector(route, estimator="none", input_positions=768, query_length=768).eval()
    model.geosparse.config.update(selector="none", budget=1.)
    sample = video()
    state = model.geosparse(sample)[0]
    assert state.valid.sum().item() == 767
    assert torch.equal(state.valid, sample.valid_frames)
    assert torch.count_nonzero(state.features[..., -1]) == 0
    captured = []
    hook = model.projection.register_forward_pre_hook(lambda module, args: captured.append(args[1].clone()))
    losses = model._task_losses(state, sample.valid_frames, [{}], [torch.tensor([[2., 10.]])], [torch.tensor([0])], 768)
    losses["cost"].backward()
    hook.remove()
    assert torch.equal(captured[0], sample.valid_frames)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.rpn_head.parameters())
    reference = copy.deepcopy(model).eval()
    from opentad.models.detectors.actionformer import ActionFormer
    expected = ActionFormer.forward_train(reference, state.features.detach(), sample.valid_frames, [{}],
                                         [torch.tensor([[2., 10.]])], [torch.tensor([0])])
    torch.testing.assert_close(losses["cost"].detach(), expected["cost"])


def test_ragged_masks_and_pair_grid_have_distinct_definitions():
    masks = torch.arange(768)[None] < torch.tensor([767, 763, 0])[:, None]
    assert torch.equal(detector_validity(masks, 768), masks)
    assert detector_validity(masks, 384).sum(-1).tolist() == [384, 382, 0]
    assert native_layout(video()).valid[:, -1].all()  # Heavy eligibility remains valid.


def test_b_tia_crosses_native_parent_boundary_before_upsampling():
    model = detector("B", estimator="none", input_positions=768, query_length=768).eval()
    observed = []
    tia = model.geosparse.regular_tia
    hook = tia.register_forward_pre_hook(lambda module, args: observed.append((args[0].shape, module.temporal_size)))
    with torch.no_grad():
        model.geosparse(video())
    hook.remove()
    assert observed == [(torch.Size([1, 384, 256]), 384)]
    # A nonzero adapter pulse must cross native 7/8, but never batch boundaries.
    with torch.no_grad():
        tia.down_proj.weight.zero_(); tia.down_proj.weight[0, 0] = 1; tia.down_proj.bias.zero_()
        tia.dwconv.weight.fill_(1); tia.dwconv.bias.zero_()
        tia.conv.weight.zero_(); tia.conv.weight[0, 0, 0] = 1; tia.conv.bias.zero_()
        tia.up_proj.weight.zero_(); tia.up_proj.weight[0, 0] = 1; tia.up_proj.bias.zero_()
        x = torch.zeros(2, 384, 256); x[0, 7, 0] = 1
        result = tia(x, 1, 1)
    assert result[0, 8, 0] > 0 and result[1].count_nonzero() == 0


@pytest.mark.parametrize("fail", [False, True])
def test_natural_frozen_ema_roundoff_is_fully_restored(fail):
    torch.manual_seed(19)
    model = Probe().train()
    model.register_parameter("frozen_many", torch.nn.Parameter(torch.randn(10000), requires_grad=False))
    ema = ModelEMA(model)
    for _ in range(500):
        ema.update(model)
    assert not torch.equal(model.frozen_many, ema.shadow["frozen_many"])
    before = {n: p.clone() for n, p in model.state_dict().items()}
    try:
        with ema_evaluation(model, ema, 0, ["fixture"]):
            assert torch.equal(model.frozen_many, ema.shadow["frozen_many"])
            if fail:
                raise RuntimeError("injected")
    except RuntimeError:
        assert fail
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("resolution", [160, 224])
def test_full_cost_uses_its_actual_spatial_reference(resolution):
    from geosparse_ext.analysis_capture import SelectionObserver
    side = resolution // 16
    observer = object.__new__(SelectionObserver)
    observer.evidence = None
    observer.model = SimpleNamespace(geosparse=SimpleNamespace(encoder=SimpleNamespace(
        source=SimpleNamespace(embed_dims=768, blocks=[None] * 12),
        trace=[dict(parent=p, layer=l, qkv_tokens=8 * side * side) for p in range(48) for l in range(12)])))
    observer.payload = dict(selected_native=np.ones((48, 8 * side * side), bool), source_frame_id=np.arange(768),
        valid_frames=np.ones(768, bool), regular_s=np.arange(768.), intervals_s=np.stack((np.arange(768.), np.arange(1, 769.)), -1),
        video_id="fixture", window_id="fixture:0", requested_budget=1., spatial_transform=np.eye(3))
    row, _ = observer.record(dict(route="A", model=dict(backbone="videomae_b")),
        dict(macs=1, components={}, scope="fixture", complete_for_conv_linear_matmul=False, unsupported={}), 0)
    assert row["heavy_mac_ratio"] == 1
    assert row["heavy_reference"]["resolution"] == resolution


@pytest.mark.parametrize("transforms", ["default", [dict(type="ShearX", shear=13)],
    [dict(type="TranslateX", percent=.18)], [dict(type="Rotate", rotate=27)]])
def test_spatial_recording_preserves_official_pixels_and_next_rng(transforms):
    import imgaug as ia
    from mmaction.datasets.transforms import ImgAug
    from geosparse_ext.data import GeoSparseImgAug
    # imgaug lazily seeds its global RNG from NumPy once. Initialize it before
    # resetting both trials, so the first trial does not pay an asymmetric draw.
    ia.random.get_global_rng()
    images = [np.arange(32 * 32 * 3, dtype=np.uint8).reshape(32, 32, 3)] * 2
    def run(cls):
        random.seed(11); np.random.seed(11); ia.seed(11)
        augmenter = cls(transforms=transforms)
        rows = []
        for _ in range(12):
            rows.append(augmenter(dict(imgs=copy.deepcopy(images), modality="RGB", img_shape=(32, 32))))
        return rows, (random.random(), np.random.rand(), ia.random.get_global_rng().random(size=None))
    expected, rng1 = run(ImgAug)
    recorded, rng2 = run(GeoSparseImgAug)
    assert rng1 == rng2
    for a, b in zip(expected, recorded):
        assert all(np.array_equal(x, y) for x, y in zip(a["imgs"], b["imgs"]))
        assert b["geosparse_imgaug_transform"].shape == (3, 3)
    if transforms != "default":
        assert not np.allclose(recorded[0]["geosparse_imgaug_transform"], np.eye(3))


def test_roi_inverse_uses_all_four_corners_and_keeps_polygon():
    from geosparse_ext.evidence import spatial_slots
    sample = video(length=16, valid=16)
    # Shear yields off-diagonal extrema that a diagonal-only box misses.
    sample.spatial_transform = torch.tensor([[[1., .7, -.3], [0, 1, 0], [0, 0, 1]]])
    layout = native_layout(sample)
    evidence = spatial_slots(torch.ones(1, 4, 8, 2, 2), torch.ones(1, 8, 2, 2, dtype=torch.bool),
                             layout, slots=4, spatial_transform=sample.spatial_transform)
    expected = torch.tensor([[.3, 0], [.8, 0], [.45, .5], [-.05, .5]])
    torch.testing.assert_close(evidence.roi_polygon[0, 0], expected)
    torch.testing.assert_close(evidence.roi_xyxy[0, 0], torch.tensor([0., 0., .8, .5]))


def test_full_diagnostic_uses_training_root_with_test_pipeline():
    from geosparse_ext.prediction_export import evaluation_dataset
    from mmengine.config import ConfigDict
    cfg = ConfigDict(dataset=dict(train=dict(data_path="train", pipeline=["random crop"]),
                                  test=dict(data_path="test", subset_name="validation", pipeline=["center crop"])))
    selected, subset = evaluation_dataset(cfg, "internal_diagnostic")
    assert subset == selected.subset_name == "training" and selected.data_path == "train"
    assert selected.pipeline == ["center crop"] and cfg.dataset.test.data_path == "test"


def test_missing_validation_can_be_repaired_without_training(tmp_path, monkeypatch):
    from geosparse_ext.records import save_json
    model = Probe(); ema = ModelEMA(model); provenance = {"source_commit": "unit"}
    for completed in range(5, 61, 5):
        if completed != 20:
            save_json(tmp_path / "intermediate_eval" / f"epoch_{completed:03d}" / "metrics.json",
                      dict(status="completed_validation", completed_epochs=completed, subset="validation", provenance=provenance))
    (tmp_path / "checkpoint").mkdir()
    torch.save(dict(format="geosparse_full_state_v2", epoch=19, provenance=provenance,
                    state_dict=model.state_dict(), state_dict_ema=ema.state_dict()), tmp_path / "checkpoint/epoch_19.pth")
    assert missing_validations(tmp_path, provenance) == [20]
    calls = []
    def measure(model, ema, cfg, runtime, job, provenance, output, completed):
        calls.append(completed)
        row = dict(status="completed_validation", completed_epochs=completed, subset="validation", provenance=provenance,
                   metrics=dict(average_mAP=.1), source_train_id="fixture", seed=0)
        save_json(output / "metrics.json", row)
        return row
    monkeypatch.setattr("geosparse_ext.training_validation.validate_dataset", measure)
    assert repair_missing_validations(model, ema, None, None, {}, provenance, tmp_path) == []
    assert calls == [20]


def test_matrix_contracts_and_two_invalid_arms():
    jobs, refs = compile_all(); validate(jobs)
    summary = matrix_summary(jobs, refs)
    assert summary["total_jobs"] == 1545 and summary["counts"] == dict(train=612, evaluate=612, benchmark=204, diagnostic=117)
    for job in jobs:
        if job["kind"] == "train":
            assert job["checkpoints"] == list(range(4, 60, 5))
    assert not implementation_blockers(base("A", spatial_group=2, source_resolution=160))
    assert implementation_blockers(base("A", spatial_group=7, source_resolution=160))
    assert not implementation_blockers(base("A", spatial_group=7, source_resolution=224))
    assert implementation_blockers(base("B", fusion_variant="feature_l2"))


def test_best_weight_mismatch_rejected_even_with_same_training_id():
    provenance = {k: "fixture" for k in ("source_commit", "resolved_config_sha256", "split_sha256", "weights_sha256")}
    first = checkpoint_identity("fixture", 4, provenance, "weight-A")
    second = checkpoint_identity("fixture", 9, provenance, "weight-B")
    with pytest.raises(ValueError, match="checkpoint identity"):
        require_same_checkpoint(first, second)
    require_same_checkpoint(first, copy.deepcopy(first))


def test_selection_failure_cannot_become_queue_done(tmp_path, monkeypatch):
    from geosparse_ext.slurm_queue import reconcile
    from geosparse_ext.records import save_json
    monkeypatch.setattr("geosparse_ext.slurm_queue.subprocess.run", lambda *a, **k: SimpleNamespace(stdout="", returncode=0))
    output = tmp_path / "runs/fixture"
    provenance = dict(source_commit="unit")
    save_json(output / "source_commits.json", provenance)
    receipt = dict(job_id="fixture", status="selection_pending", training_complete=True, completed_epochs=60,
                   best_checkpoint_selection_complete=False, selection_complete=False, is_mock=False, **provenance)
    save_json(output / "result.json", receipt)
    state = dict(status="SUBMITTED", slurm_id="unit-123")
    reconcile(dict(job_id="fixture", kind="train"), state, tmp_path)
    assert state["status"] == "SELECTION_PENDING"
    receipt["status"] = "completed"
    save_json(output / "result.json", receipt)
    state = dict(status="SUBMITTED", slurm_id="unit-123")
    reconcile(dict(job_id="fixture", kind="train"), state, tmp_path)
    assert state["status"] == "FAILED"


def test_pareto_refuses_old_latency_for_a_new_best(tmp_path):
    from geosparse_ext.figures import Report, plot_pareto
    from geosparse_ext.records import save_json
    provenance = {k: "fixture" for k in ("source_commit", "resolved_config_sha256", "split_sha256", "weights_sha256")}
    current = checkpoint_identity("fixture", 9, provenance, "new")
    previous = checkpoint_identity("fixture", 4, provenance, "old")
    job = dict(job_id="fixture", route="A", families=["F01"], dataset="fixture", model=dict(axis="ST", budget=.5))
    training = dict(source_commit="fixture", checkpoint_identity=current)
    metrics = dict(selected_checkpoint_epoch=9, checkpoint_identity=current, official=dict(average_mAP=.1))
    export = dict(source_train_id="fixture", is_final_checkpoint=True, full_split=True,
                  selected_checkpoint_epoch=9, model_source_commit="fixture", checkpoint_identity=current)
    windows = [dict(heavy_macs=1, model_macs_counted=2, mac_count_complete=True)]
    save_json(tmp_path / "hardware.json", dict(checkpoint_identity=previous, cases=[]))
    runs = {"benchmark": (tmp_path, dict(kind="benchmark", source_train_id="fixture"), dict(is_mock=False))}
    with pytest.raises(ValueError, match="checkpoint identity"):
        plot_pareto(Report(tmp_path / "figures"), runs, [(tmp_path, dict(seed=0), job, training, metrics)],
                    [(tmp_path, export, windows)], single_seed=True)


def test_common_epoch_supplement_reuses_only_50_and_60(tmp_path):
    from geosparse_ext.evaluation import epoch_selection_supplements
    from geosparse_ext.records import save_json
    provenance = dict(source_commit="fixture")
    for epoch, score in [(10, .9), (50, .4), (60, .3)]:
        save_json(tmp_path / "intermediate_eval" / f"epoch_{epoch:03d}" / "metrics.json",
            dict(status="completed_validation", completed_epochs=epoch, subset="validation", weights="ema",
                 provenance=provenance, metrics=dict(average_mAP=score), predictions_path=f"epoch-{epoch}.json"))
    report = epoch_selection_supplements(tmp_path, provenance)
    assert report["fixed_completed_60"]["checkpoint_epoch"] == 59
    assert report["common_completed_50_60_best"]["checkpoint_epoch"] == 49
