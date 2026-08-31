from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.config import Config

from opentad.models.backbones.backbone_wrapper import BackboneWrapper
from opentad.models.duca.acquisition import (
    SparseTemporalGrid,
    interpolate_h65_positions_to_detector_grid,
    nested_h65_budget_positions,
)
from opentad.models.selectors.duca_online_frame_selector import (
    DucaOnlineFrameSelector,
    _DEFAULT_METADATA_KEYS,
)
from opentad.cores.test_engine import summarize_duca_execution_cost
from opentad.evaluations.mAP import compute_average_precision_detection
from tools.bata.evaluate_duca_h65_system_multibudget_exposure import (
    OfficialAPBootstrapArm,
    TIOU_THRESHOLDS,
)
from tools.bata.prepare_duca_h65_system_multibudget_exposure import (
    budget_for_update,
    fixed_mixed_manifest,
    held_out_inference_annotation,
    held_out_windows,
    quantized_update_counts,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


def _baseline_positions(valid_counts: torch.Tensor) -> torch.Tensor:
    rows = []
    for count in valid_counts.tolist():
        active_count = min(384, int(count))
        if int(count) > 384:
            active = torch.linspace(0, int(count) - 1, active_count).round().long()
        else:
            active = torch.arange(active_count)
        row = torch.full((384,), -1, dtype=torch.long)
        row[:active_count] = active
        rows.append(row)
    return torch.stack(rows)


def _valid_mask(valid_counts: torch.Tensor, temporal_len: int = 768) -> torch.Tensor:
    return torch.arange(temporal_len)[None, :] < valid_counts[:, None]


def test_nested_h65_sets_preserve_k384_and_short_window_collapse() -> None:
    valid_counts = torch.tensor([768, 300, 200])
    positions = _baseline_positions(valid_counts)
    priority = torch.arange(768, dtype=torch.float32)[None, :].expand(3, -1)
    nested = nested_h65_budget_positions(
        positions,
        priority,
        _valid_mask(valid_counts),
    )

    assert torch.equal(nested.positions_by_budget[384], positions)
    assert nested.actual_count_by_budget[256].tolist() == [256, 256, 200]
    assert nested.actual_count_by_budget[512].tolist() == [512, 300, 200]
    assert nested.effective_budget_by_requested[256].tolist() == [256, 256, 384]
    assert nested.effective_budget_by_requested[512].tolist() == [512, 384, 384]
    assert nested.execution_slots_by_budget[256].tolist() == [256, 256, 384]
    assert nested.execution_slots_by_budget[512].tolist() == [512, 384, 384]
    for row in range(3):
        low = set(nested.positions_by_budget[256][row].tolist()) - {-1}
        middle = set(nested.positions_by_budget[384][row].tolist()) - {-1}
        high = set(nested.positions_by_budget[512][row].tolist()) - {-1}
        assert low < middle or low == middle
        assert middle < high or middle == high


def test_detector_grid_is_exact_at_k384_and_float_monotone_otherwise() -> None:
    valid_counts = torch.tensor([768])
    positions = _baseline_positions(valid_counts)
    exact = interpolate_h65_positions_to_detector_grid(
        positions, torch.tensor([384]), detector_length=384
    )
    assert torch.equal(exact, positions.float())

    smaller = positions[:, :256]
    mapped = interpolate_h65_positions_to_detector_grid(
        smaller, torch.tensor([256]), detector_length=384
    )
    assert mapped.shape == (1, 384)
    assert torch.all(mapped[:, 1:] > mapped[:, :-1])
    assert not torch.equal(mapped, mapped.long().float())


def _selector_shell(probabilities=None) -> DucaOnlineFrameSelector:
    selector = DucaOnlineFrameSelector.__new__(DucaOnlineFrameSelector)
    nn.Module.__init__(selector)
    selector.multi_budget_exposure = selector._normalize_multi_budget_exposure(
        dict(
            probabilities=probabilities or {256: 0.25, 384: 0.5, 512: 0.25},
            total_updates=6000,
            seed=3407,
            detector_length=384,
            packet_size=16,
            evaluation_budget=384,
        )
    )
    selector.register_buffer(
        "_loss_weight_schedule_step", torch.zeros((), dtype=torch.long), persistent=True
    )
    selector.metadata_keys = dict(_DEFAULT_METADATA_KEYS)
    selector.selected_positions_unit = "original_time_index"
    selector.detector_output_coordinate_space = "selected_axis_index"
    selector.temporal_sampling_contract = None
    selector.remap_gt_to_selected_axis = True
    return selector


def test_successful_update_budget_clock_is_exact_replay_safe_and_rng_free() -> None:
    selector = _selector_shell()
    selector.train()
    before = torch.random.get_rng_state().clone()
    counts = {256: 0, 384: 0, 512: 0}
    for step in range(6000):
        selector._loss_weight_schedule_step.fill_(step)
        first = selector._requested_multi_budget_exposure(
            batch_size=2, device=torch.device("cpu"), metas=None
        )
        second = selector._requested_multi_budget_exposure(
            batch_size=2, device=torch.device("cpu"), metas=None
        )
        assert torch.equal(first, second)
        assert int(torch.unique(first).numel()) == 1
        counts[int(first[0].item())] += 1
    assert counts == {256: 1500, 384: 3000, 512: 1500}
    assert torch.equal(before, torch.random.get_rng_state())


def test_asymmetric_calibrated_budget_clock_realizes_exact_6000_update_counts() -> None:
    probabilities = {
        256: 0.24235161911751213,
        384: 0.5,
        512: 0.25764838088248787,
    }
    expected = quantized_update_counts(probabilities)
    assert expected == {256: 1454, 384: 3000, 512: 1546}
    for seed in (3407, 3408, 3409):
        actual = {256: 0, 384: 0, 512: 0}
        for step in range(6000):
            actual[budget_for_update(step, seed, expected)] += 1
        assert actual == expected


def test_held_out_window_manifest_matches_official_sliding_dataset_geometry() -> None:
    database = {
        "v384": {"subset": "validation", "frame": 384 * 4},
        "v768": {"subset": "validation", "frame": 768 * 4},
        "v1152": {"subset": "validation", "frame": 1152 * 4},
        "v1536": {"subset": "validation", "frame": 1536 * 4},
        "train": {"subset": "training", "frame": 768 * 4},
    }
    rows = held_out_windows(database)
    keys = [row["key"] for row in rows]
    assert keys == [
        "v384|0",
        "v768|0",
        "v1152|0",
        "v1152|1536",
        "v1536|0",
        "v1536|1536",
        "v1536|3072",
    ]
    assert [row["valid_observations"] for row in rows] == [
        384,
        768,
        768,
        768,
        768,
        768,
        768,
    ]
    assert [row["multiplicity"] for row in rows] == [1, 2, 1, 2, 1, 1, 2]
    assert sum(row["multiplicity"] for row in rows) == 10


def test_fixed_mixed_manifest_is_complete_deterministic_and_within_cost() -> None:
    rows = [
        {
            "key": f"video_{index:03d}|0",
            "video_name": f"video_{index:03d}",
            "window_start_frame": 0,
            "valid_observations": 128 + (index % 7) * 96,
        }
        for index in range(32)
    ]
    probabilities = {256: 0.24, 384: 0.5, 512: 0.26}
    manifest, summary = fixed_mixed_manifest(rows, probabilities)
    repeated, repeated_summary = fixed_mixed_manifest(rows, probabilities)
    assert manifest == repeated
    assert summary == repeated_summary
    assert set(manifest) == {row["key"] for row in rows}
    assert sum(summary["unique_key_budget_counts"].values()) == len(rows)
    assert sum(summary["executed_window_budget_counts"].values()) == len(rows)
    assert summary["mixed_actual_observations"] <= summary["control_actual_observations"]


def test_held_out_inference_annotation_contains_geometry_without_semantics() -> None:
    database = {
        "held_out": {
            "subset": "validation",
            "frame": 3072,
            "duration": 12.5,
            "annotations": [{"label": "x", "segment": [1.0, 2.0]}],
        },
        "train": {
            "subset": "training",
            "frame": 1536,
            "duration": 6.25,
            "annotations": [{"label": "x", "segment": [1.0, 2.0]}],
        },
    }
    payload = held_out_inference_annotation(database)
    assert payload == {
        "database": {
            "held_out": {
                "subset": "validation",
                "frame": 3072,
                "duration": 12.5,
            }
        }
    }


def _explicit_video_bootstrap_ap(ground_truth, prediction, video_ids, counts):
    gt_rows = []
    prediction_rows = []
    for video_index, video_id in enumerate(video_ids):
        for copy_index in range(int(counts[video_index])):
            copied_video_id = f"{video_id}__copy_{copy_index}"
            copied_gt = ground_truth[ground_truth["video-id"] == video_id].copy()
            copied_gt["video-id"] = copied_video_id
            gt_rows.append(copied_gt)
            copied_prediction = prediction[prediction["video-id"] == video_id].copy()
            copied_prediction["video-id"] = copied_video_id
            prediction_rows.append(copied_prediction)
    replicated_gt = pd.concat(gt_rows, ignore_index=True)
    replicated_prediction = pd.concat(prediction_rows, ignore_index=True)
    return 100.0 * compute_average_precision_detection(
        replicated_gt,
        replicated_prediction,
        tiou_thresholds=TIOU_THRESHOLDS,
    )


def test_bootstrap_accelerator_matches_explicit_repeated_video_evaluation() -> None:
    video_ids = ["video_a", "video_b", "video_c"]
    ground_truth = pd.DataFrame(
        {
            "video-id": video_ids,
            "t-start": [0.0, 0.0, 0.0],
            "t-end": [1.0, 1.0, 1.0],
            "label": [0, 0, 0],
        }
    )
    prediction = pd.DataFrame(
        {
            "video-id": video_ids + video_ids,
            "t-start": [0.0, 0.0, 0.0, 2.0, 2.0, 2.0],
            "t-end": [1.0, 0.8, 0.6, 3.0, 3.0, 3.0],
            "label": [0, 0, 0, 0, 0, 0],
            "score": [0.91, 0.82, 0.73, 0.44, 0.35, 0.26],
        }
    )
    evaluator = SimpleNamespace(
        activity_index={"action": 0},
        ground_truth=ground_truth,
        prediction=prediction,
    )
    arm = OfficialAPBootstrapArm(evaluator, video_ids)
    counts = np.asarray([[2, 1, 0], [0, 1, 2]], dtype=np.int16)
    accelerated = arm.metrics_for_counts(counts)
    explicit = np.stack(
        [
            _explicit_video_bootstrap_ap(
                ground_truth, prediction, video_ids, replicate_counts
            )
            for replicate_counts in counts
        ]
    )
    assert np.allclose(accelerated, explicit, rtol=0.0, atol=1.0e-12)


def test_exposure_transform_aligns_positions_assignments_masks_and_packets() -> None:
    valid_counts = torch.tensor([768, 300])
    valid = _valid_mask(valid_counts)
    baseline = _baseline_positions(valid_counts)
    dense_mask = torch.zeros_like(valid)
    for row in range(2):
        dense_mask[row, baseline[row][baseline[row] >= 0]] = True
    grid = SparseTemporalGrid(
        selected_positions=baseline,
        selected_mask=dense_mask,
        original_length=768,
        valid_len=valid_counts,
        budget=384,
        requested_budget=torch.tensor([384, 384]),
        effective_budget=torch.minimum(valid_counts, torch.tensor(384)),
        detector_input_length=dense_mask.long().sum(dim=1),
    ).validate()
    assignment = F.one_hot(baseline.clamp_min(0), num_classes=768).float()
    assignment *= (baseline >= 0)[:, :, None]
    scores = {
        "sampling_rates": torch.arange(768, dtype=torch.float32)[None, :].expand(2, -1),
        "structured_soft_slot_assignment": assignment,
        "detector_grid_positions": baseline,
    }
    selector = _selector_shell()
    exposure = selector._apply_h65_multi_budget_exposure(
        grid=grid,
        scores=scores,
        valid_mask=valid,
        requested_budget=torch.tensor([256, 512]),
    )

    assert exposure["positions"].shape == (2, 384)
    assert exposure["actual_observations"].tolist() == [256, 300]
    assert exposure["effective_budget"].tolist() == [256, 384]
    assert exposure["execution_slots"].tolist() == [256, 384]
    assert exposure["detector_mask"][0].sum().item() == 384
    assert exposure["detector_mask"][1].sum().item() == 300
    active = exposure["slot_mask"]
    mass = scores["structured_soft_slot_assignment"].sum(dim=-1)
    assert torch.allclose(mass[active], torch.ones_like(mass[active]))
    assert torch.count_nonzero(mass[~active]) == 0


def test_forced_k384_exposure_preserves_h65_grid_assignment_and_mask_exactly() -> None:
    valid_counts = torch.tensor([768, 300])
    valid = _valid_mask(valid_counts)
    baseline = _baseline_positions(valid_counts)
    dense_mask = torch.zeros_like(valid)
    for row in range(2):
        dense_mask[row, baseline[row][baseline[row] >= 0]] = True
    grid = SparseTemporalGrid(
        selected_positions=baseline,
        selected_mask=dense_mask,
        original_length=768,
        valid_len=valid_counts,
        budget=384,
        requested_budget=torch.tensor([384, 384]),
        effective_budget=torch.minimum(valid_counts, torch.tensor(384)),
        detector_input_length=dense_mask.long().sum(dim=1),
    ).validate()
    assignment = F.one_hot(baseline.clamp_min(0), num_classes=768).float()
    assignment *= (baseline >= 0)[:, :, None]
    scores = {
        "sampling_rates": torch.arange(768, dtype=torch.float32)[None, :].expand(2, -1),
        "structured_soft_slot_assignment": assignment.clone(),
        "detector_grid_positions": baseline.clone(),
    }
    selector = _selector_shell()
    exposure = selector._apply_h65_multi_budget_exposure(
        grid=grid,
        scores=scores,
        valid_mask=valid,
        requested_budget=torch.tensor([384, 384]),
    )
    assert torch.equal(exposure["positions"], baseline)
    assert torch.equal(exposure["detector_grid_positions"], baseline.float())
    assert torch.equal(exposure["detector_mask"], baseline >= 0)
    assert torch.equal(scores["structured_soft_slot_assignment"], assignment)


def test_short_window_collapsed_budgets_publish_only_active_true_time_positions() -> None:
    valid_counts = torch.tensor([300, 200])
    valid = _valid_mask(valid_counts)
    baseline = _baseline_positions(valid_counts)
    dense_mask = torch.zeros_like(valid)
    for row in range(2):
        dense_mask[row, baseline[row][baseline[row] >= 0]] = True
    grid = SparseTemporalGrid(
        selected_positions=baseline,
        selected_mask=dense_mask,
        original_length=768,
        valid_len=valid_counts,
        budget=384,
        requested_budget=torch.tensor([384, 384]),
        effective_budget=torch.minimum(valid_counts, torch.tensor(384)),
        detector_input_length=dense_mask.long().sum(dim=1),
    ).validate()
    assignment = F.one_hot(baseline.clamp_min(0), num_classes=768).float()
    assignment *= (baseline >= 0)[:, :, None]
    scores = {
        "sampling_rates": torch.arange(768, dtype=torch.float32)[None, :].expand(2, -1),
        "structured_soft_slot_assignment": assignment,
        "detector_grid_positions": baseline,
    }
    selector = _selector_shell()
    exposure = selector._apply_h65_multi_budget_exposure(
        grid=grid,
        scores=scores,
        valid_mask=valid,
        requested_budget=torch.tensor([384, 512]),
    )
    metas = selector._write_metas(
        [{}, {}],
        grid,
        acquisition_positions=exposure["positions"],
        detector_grid_positions=exposure["detector_grid_positions"],
        exposure=exposure,
        actionness_source_name="synthetic",
    )

    for row, valid_len in enumerate(valid_counts.tolist()):
        active_positions = metas[row]["selected_axis_to_true_time_dense_index"]
        assert len(active_positions) == valid_len
        assert active_positions == [float(index) for index in range(valid_len)]
        assert all(0.0 <= position < valid_len for position in active_positions)
        assert metas[row]["irregular_selected_count"] == valid_len
        assert metas[row]["duca_budget_collapsed_to_k384"] is (row == 1)

    segments = [torch.tensor([[10.0, 100.0]]), torch.tensor([[10.0, 100.0]])]
    remapped, _, remapped_metas = selector._remap_train_targets_to_selected_axis(
        segments,
        [torch.tensor([0]), torch.tensor([0])],
        metas,
    )
    assert all(torch.isfinite(item).all() for item in remapped)
    assert all(meta["gt_remapped_to_selected_axis"] for meta in remapped_metas)


class _FakeDataPreprocessor:
    def preprocess(self, tensors, data_samples=None, training=False):
        del data_samples, training
        return torch.stack(tensors), None


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, frames, **kwargs):
        del kwargs
        self.calls.append(tuple(frames.shape))
        return frames.mean(dim=(1, 3, 4), keepdim=False)[:, None, :, None, None]


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.data_preprocessor = _FakeDataPreprocessor()
        self.backbone = _FakeBackbone()


class _LegacyPreprocess:
    def __call__(self, payload):
        frames = payload["frames"]
        batch, num_segs, channels, _, height, width = frames.shape
        frames = (
            frames.reshape(batch, num_segs, channels, 24, 16, height, width)
            .permute(0, 3, 1, 2, 4, 5, 6)
            .reshape(batch * 24, num_segs, channels, 16, height, width)
        )
        return {"frames": frames}


class _LegacyPostprocess:
    def __call__(self, payload):
        features = payload["feats"].mean(dim=(1, 4, 5))
        batch_packets, channels, tubelets = features.shape
        batch = batch_packets // 24
        features = (
            features.reshape(batch, 24, channels, tubelets)
            .permute(0, 2, 1, 3)
            .reshape(batch, channels, 24 * tubelets)
        )
        return {"feats": F.interpolate(features, size=384, mode="linear", align_corners=False)}


def _backbone_wrapper_shell() -> BackboneWrapper:
    wrapper = BackboneWrapper.__new__(BackboneWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = _FakeModel()
    wrapper.pre_processing_pipeline = _LegacyPreprocess()
    wrapper.post_processing_pipeline = _LegacyPostprocess()
    wrapper.norm_eval = False
    wrapper.freeze_backbone = False
    wrapper.use_temporal_checkpointing = False
    return wrapper


def test_backbone_executes_real_packet_groups_and_keeps_forced_k384_legacy_path() -> None:
    wrapper = _backbone_wrapper_shell()
    frames = torch.zeros(2, 1, 1, 384, 1, 1)
    frames[0, :, :, :256] = 1.0
    frames[1, :, :, :200] = 2.0
    masks = torch.ones(2, 384, dtype=torch.bool)
    context = {
        "requested_budget": torch.tensor([256, 256]),
        "effective_budget": torch.tensor([256, 384]),
        "actual_observations": torch.tensor([256, 200]),
        "execution_slots": torch.tensor([256, 384]),
        "collapsed_to_baseline": torch.tensor([False, True]),
        "detector_length": 384,
        "packet_size": 16,
    }
    features = wrapper(
        frames,
        masks=masks,
        duca_multibudget_context=context,
    )
    assert features.shape == (2, 1, 384)
    assert sorted(shape[0] for shape in wrapper.model.backbone.calls) == [16, 24]
    assert wrapper.last_execution_profile["total_actual_observations"] == 456
    assert wrapper.last_execution_profile["total_execution_slots"] == 640
    assert wrapper.last_execution_profile["used_legacy_k384_path"] is False

    legacy_wrapper = _backbone_wrapper_shell()
    legacy_frames = torch.randn(2, 1, 1, 384, 1, 1)
    legacy_masks = torch.ones(2, 384, dtype=torch.bool)
    expected = legacy_wrapper(legacy_frames, masks=legacy_masks)
    legacy_context = {
        "requested_budget": torch.tensor([384, 384]),
        "effective_budget": torch.tensor([384, 384]),
        "actual_observations": torch.tensor([384, 384]),
        "execution_slots": torch.tensor([384, 384]),
        "collapsed_to_baseline": torch.tensor([False, False]),
        "detector_length": 384,
        "packet_size": 16,
    }
    actual = legacy_wrapper(
        legacy_frames,
        masks=legacy_masks,
        duca_multibudget_context=legacy_context,
    )
    assert torch.equal(actual, expected)
    assert legacy_wrapper.last_execution_profile["used_legacy_k384_path"] is True


def test_k384_row_remains_exact_when_a_mixed_batch_uses_variable_execution() -> None:
    frames = torch.randn(2, 1, 1, 384, 1, 1)
    frames[1, :, :, 256:] = 0.0
    masks = torch.ones(2, 384, dtype=torch.bool)
    mixed_context = {
        "requested_budget": torch.tensor([384, 256]),
        "effective_budget": torch.tensor([384, 256]),
        "actual_observations": torch.tensor([384, 256]),
        "execution_slots": torch.tensor([384, 256]),
        "collapsed_to_baseline": torch.tensor([False, False]),
        "detector_length": 384,
        "packet_size": 16,
    }
    mixed_wrapper = _backbone_wrapper_shell()
    mixed = mixed_wrapper(
        frames,
        masks=masks,
        duca_multibudget_context=mixed_context,
    )
    legacy_wrapper = _backbone_wrapper_shell()
    expected_k384 = legacy_wrapper(frames[:1], masks=masks[:1])
    assert torch.equal(mixed[:1], expected_k384)
    assert mixed_wrapper.last_execution_profile["used_legacy_k384_path"] is False


def test_nonbaseline_final_packet_padding_is_not_interpolated_into_detector_grid() -> None:
    frames = torch.ones(1, 1, 1, 512, 1, 1)
    frames[:, :, :, 500:] = 99.0
    masks = torch.ones(1, 384, dtype=torch.bool)
    context = {
        "requested_budget": torch.tensor([512]),
        "effective_budget": torch.tensor([512]),
        "actual_observations": torch.tensor([500]),
        "execution_slots": torch.tensor([512]),
        "collapsed_to_baseline": torch.tensor([False]),
        "detector_length": 384,
        "packet_size": 16,
    }
    wrapper = _backbone_wrapper_shell()
    features = wrapper(frames, masks=masks, duca_multibudget_context=context)
    assert torch.equal(features, torch.ones_like(features))


def test_matched_configs_differ_only_by_budget_exposure(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "/tmp/stage1.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "dummy")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    monkeypatch.setenv("DUCA_VIDEOMAE_PRETRAIN", "/tmp/videomae.pth")
    monkeypatch.setenv("DUCA_EXPERIMENT_SEED", "3408")
    monkeypatch.setenv("DUCA_MB_P256", "0.25")
    monkeypatch.setenv("DUCA_MB_P512", "0.25")
    control = Config.fromfile(
        str(CONFIG_ROOT / "duca_h65_system_multibudget_exposure_control.py")
    )
    candidate = Config.fromfile(
        str(CONFIG_ROOT / "duca_h65_system_multibudget_exposure_candidate.py")
    )
    assert control.seed == candidate.seed == 3408
    assert control.workflow.formal_successful_update_contract is False
    assert control.workflow.seal_eval_dataloaders_during_training is True
    assert control.workflow.expected_successful_optimizer_updates == 6000
    assert control.workflow.val_eval_interval == -1
    assert control.workflow.val_start_epoch == 9999
    assert control.dataset.val is None
    assert control.model.backbone.custom.pretrain == "/tmp/videomae.pth"
    assert candidate.workflow == {
        **control.workflow,
        "training_profile": "duca_h65_system_multibudget_exposure_candidate",
    }
    assert control.model.frame_selector.get("multi_budget_exposure") is None
    exposure = candidate.model.frame_selector.multi_budget_exposure
    assert exposure.probabilities == {256: 0.25, 384: 0.5, 512: 0.25}
    for key in ("optimizer", "scheduler", "solver", "dataset"):
        assert control[key] == candidate[key]
    control_model = control.model.to_dict()
    candidate_model = candidate.model.to_dict()
    candidate_model["frame_selector"].pop("multi_budget_exposure")
    assert candidate_model == control_model


def test_pre_run_is_four_update_candidate_smoke_with_held_out_loaders_sealed(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "/tmp/stage1.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "dummy")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    monkeypatch.setenv("DUCA_VIDEOMAE_PRETRAIN", "/tmp/videomae.pth")
    monkeypatch.setenv("DUCA_EXPERIMENT_SEED", "3407")
    monkeypatch.setenv("DUCA_MB_P256", "0.24235161911751213")
    monkeypatch.setenv("DUCA_MB_P512", "0.25764838088248787")
    monkeypatch.setenv("DUCA_PRE_RUN_PROBE_JSON", "/tmp/pre_run_probe.json")
    cfg = Config.fromfile(
        str(CONFIG_ROOT / "duca_h65_system_multibudget_exposure_pre_run.py")
    )
    assert cfg.workflow.formal_successful_update_contract is False
    assert cfg.workflow.end_epoch == 1
    assert cfg.workflow.max_train_iters == 4
    assert cfg.workflow.training_probe_json == "/tmp/pre_run_probe.json"
    assert cfg.workflow.seal_eval_dataloaders_during_training is True
    assert cfg.dataset.val is None


def test_execution_cost_summary_aggregates_windows_without_metrics() -> None:
    rows = [
        {
            "video_name": "video_a",
            "window_start_frame": 0,
            "requested_budget": 256,
            "effective_budget": 256,
            "actual_observations": 250,
            "execution_slots": 256,
            "data_wait_ms": 2.0,
            "model_and_window_postprocess_ms": 8.0,
            "selector_ms": 1.0,
            "videomae_ms": 7.0,
            "detector_ms": 2.0,
        },
        {
            "video_name": "video_a",
            "window_start_frame": 1536,
            "requested_budget": 384,
            "effective_budget": 384,
            "actual_observations": 384,
            "execution_slots": 384,
            "data_wait_ms": 4.0,
            "model_and_window_postprocess_ms": 16.0,
            "selector_ms": 2.0,
            "videomae_ms": 15.0,
            "detector_ms": 3.0,
        },
        {
            "video_name": "video_b",
            "window_start_frame": 0,
            "requested_budget": 512,
            "effective_budget": 512,
            "actual_observations": 500,
            "execution_slots": 512,
            "data_wait_ms": 5.0,
            "model_and_window_postprocess_ms": 25.0,
            "selector_ms": 3.0,
            "videomae_ms": 23.0,
            "detector_ms": 4.0,
        },
    ]
    summary = summarize_duca_execution_cost(
        rows,
        peak_cuda_memory_mb=123.0,
        final_postprocess_ms_by_video={"video_a": 3.0, "video_b": 5.0},
        full_population_wall_ms=75.0,
    )
    assert summary["schema_version"] == "duca_h65_system_multibudget_execution_cost_v2"
    assert summary["held_out_semantics_read"] is False
    assert summary["window_count"] == 3
    assert summary["video_count"] == 2
    assert summary["total_actual_observations"] == 1134
    assert summary["total_execution_slots"] == 1152
    assert summary["requested_budget_window_counts"] == {"256": 1, "384": 1, "512": 1}
    assert summary["per_video"]["video_a"]["data_wait_ms"] == 6.0
    assert summary["per_video"]["video_a"]["model_and_window_postprocess_ms"] == 24.0
    assert summary["per_video"]["video_a"]["final_video_postprocess_ms"] == 3.0
    assert summary["per_video"]["video_a"]["wall_ms"] == 33.0
    assert summary["per_video_wall_ms_p50"] == 34.0
    assert np.isclose(summary["per_video_wall_ms_p95"], 34.9)
    assert summary["per_video_component_ms"]["videomae_ms"]["p50"] == 22.5
    assert np.isclose(
        summary["per_video_component_ms"]["videomae_ms"]["p95"], 22.95
    )
    assert summary["attributed_per_video_wall_ms_sum"] == 68.0
    assert summary["full_population_wall_ms"] == 75.0
    assert summary["unattributed_framework_wall_ms"] == 7.0
