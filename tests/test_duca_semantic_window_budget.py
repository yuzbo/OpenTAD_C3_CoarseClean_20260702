import json

import numpy as np
import pytest
import torch
import torch.nn as nn

from opentad.datasets.thumos import DucaVideoGroupedThumosSlidingDataset
from opentad.datasets.transforms.loading import LoadDucaWindowBudgetFrames
from opentad.models.detectors.actionformer import ActionFormer
from opentad.models.utils.truetime_geometry import TrueTimeMap
from tools.bata.build_duca_semantic_window_budget_table import (
    percentile_rank_with_median_ties,
    semantic_and_permuted_budgets,
)


def _rows(count, reverse=False):
    values = list(range(count))
    if reverse:
        values.reverse()
    return [
        {
            "split": "training",
            "video_name": "video_validation_0001",
            "window_index": index,
            "window_start_frame": index * 1536,
            "boundary_evidence": float(values[index]),
            "uncertainty_evidence": float(values[index]),
        }
        for index in range(count)
    ]


@pytest.mark.parametrize("count", [1, 2, 3, 5])
def test_budget_assignment_preserves_video_mean_and_control_multiset(count):
    assigned = semantic_and_permuted_budgets(_rows(count))
    semantic = [row["semantic_budget"] for row in assigned]
    control = [row["permuted_control_budget"] for row in assigned]
    assert sum(semantic) == 384 * count
    assert sorted(semantic) == sorted(control)
    assert set(semantic).issubset({256, 384, 512})
    if count == 1:
        assert semantic == [384]
    elif count == 2:
        assert sorted(semantic) == [256, 512]
    else:
        assert 256 in semantic and 512 in semantic


def test_permuted_control_is_content_independent():
    first = semantic_and_permuted_budgets(_rows(5))
    second = semantic_and_permuted_budgets(_rows(5, reverse=True))
    assert [row["permuted_control_budget"] for row in first] == [
        row["permuted_control_budget"] for row in second
    ]


def test_percentile_rank_uses_median_ties():
    ranks = percentile_rank_with_median_ties([1.0, 1.0, 3.0])
    assert np.allclose(ranks, [0.25, 0.25, 1.0])


def test_semantic_ranking_is_per_video_not_cross_video():
    first = semantic_and_permuted_budgets(_rows(3))
    second_rows = _rows(3)
    for row in second_rows:
        row["video_name"] = "video_validation_0002"
        row["boundary_evidence"] += 100.0
        row["uncertainty_evidence"] += 100.0
    second = semantic_and_permuted_budgets(second_rows)
    assert [row["semantic_budget"] for row in first] == [
        row["semantic_budget"] for row in second
    ]


def test_canonical_windows_include_right_aligned_tail():
    dataset = object.__new__(DucaVideoGroupedThumosSlidingDataset)
    dataset.fps = -1
    dataset.snippet_stride = 4
    dataset.window_size = 768
    dataset.window_stride = 384
    dataset.ioa_thresh = 0.75
    records = dataset.split_video_to_windows(
        "video_validation_0001",
        {"frame": 4000, "duration": 160.0},
        {},
    )
    starts = [int(record[3][0] // 4) for record in records]
    assert starts == [0, 232]
    assert records[-1][4:] == [1, 2]


def _table_row():
    positions = {
        str(k): np.linspace(0, 767, num=k).round().astype(np.int64).tolist()
        for k in (256, 384, 512)
    }
    for key, values in positions.items():
        assert len(values) == len(set(values)), key
    return {
        "split": "training",
        "video_name": "video_validation_0001",
        "window_index": 0,
        "window_count": 1,
        "window_start_frame": 0,
        "window_end_frame": 3068,
        "semantic_budget": 512,
        "permuted_control_budget": 256,
        "positions_by_budget": positions,
    }


def test_table_loader_executes_exact_k_and_physical_roundtrip(tmp_path):
    table = tmp_path / "table.jsonl"
    table.write_text(json.dumps(_table_row()) + "\n", encoding="utf-8")
    transform = LoadDucaWindowBudgetFrames(table, "semantic")
    results = transform(
        {
            "duca_split": "training",
            "video_name": "video_validation_0001",
            "duca_window_index": 0,
            "duca_window_count": 1,
            "window_start_frame": 0,
            "window_end_frame": 3068,
            "feature_start_idx": 0,
            "feature_end_idx": 767,
            "snippet_stride": 4,
            "window_size": 768,
            "total_frames": 4000,
            "gt_segments": np.asarray([[100.0, 200.0]], dtype=np.float32),
        }
    )
    assert results["frame_inds"].shape == (512,)
    assert results["num_clips"] == 32
    assert results["duca_actual_backbone_input_k"] == 512
    assert results["duca_dynamic_compute_realized"] is False
    assert results["masks"].shape == (384,)
    time_map = TrueTimeMap(
        results["selected_axis_to_true_time_dense_index"],
        dense_len=768,
        valid_len=768,
    )
    original = torch.tensor([[100.0, 200.0]])
    roundtrip = time_map.selected_to_true(time_map.true_to_selected(original))
    assert torch.max(torch.abs(roundtrip - original)).item() <= 1.0


class _FakeVariableBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs):
        actual_k = int(inputs.shape[1] * inputs.shape[3])
        self.calls.append(actual_k)
        value = inputs.mean(dim=(1, 2, 3, 4, 5), keepdim=False)
        return value[:, None, None].expand(-1, 2, 384)


class _OfflineHarness(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _FakeVariableBackbone()
        self.max_seq_len = 384
        self.profile_variable_k = False
        self.last_variable_k_work = None


def test_variable_k_backbone_buckets_without_global_padding_and_restores_order():
    harness = _OfflineHarness()
    ks = [512, 256, 384]
    inputs = [
        torch.full((k // 16, 3, 16, 2, 2), float(index + 1), requires_grad=True)
        for index, k in enumerate(ks)
    ]
    masks = [torch.ones(384, dtype=torch.bool) for _ in ks]
    metas = [{"duca_actual_backbone_input_k": k} for k in ks]
    output, output_masks = ActionFormer._offline_window_table_backbone(
        harness, inputs, masks, metas
    )
    assert harness.backbone.calls == [256, 384, 512]
    reference = torch.cat([harness.backbone(sample.unsqueeze(0)) for sample in inputs], dim=0)
    assert output.shape == (3, 2, 384)
    assert torch.allclose(output, reference)
    assert output_masks.shape == (3, 384)
    assert torch.allclose(output[:, 0, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert harness.last_variable_k_work["padded_to_global_max_k"] is False
    assert all(meta["duca_dynamic_compute_realized"] is True for meta in metas)
    output.sum().backward()
    assert all(sample.grad is not None for sample in inputs)
