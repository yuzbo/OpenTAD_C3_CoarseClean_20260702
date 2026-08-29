from __future__ import annotations

from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models.duca.tubelet_coreset import (
    aggregate_frame_signals_to_tubelets,
    assign_dynamic_native_tubelet_clip_budgets,
    build_native_tubelet_candidates,
    gather_native_tubelet_rgb,
    select_native_tubelet_coreset,
    task_state_tubelet_scores,
)
from opentad.models.detectors.actionformer import ActionFormer


ROOT = Path(__file__).resolve().parents[1]


def test_native_tubelet_candidates_preserve_two_frame_atoms() -> None:
    valid = torch.tensor([[True] * 8 + [False] * 4])
    pairs, tubelet_valid = build_native_tubelet_candidates(valid)
    assert pairs.tolist() == [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]]
    assert tubelet_valid.tolist() == [[True, True, True, True, False, False]]

    inputs = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12, 1, 1)
    positions = torch.tensor([[0, 2, 3, -1]])
    slot_mask = positions >= 0
    gathered, frame_positions = gather_native_tubelet_rgb(inputs, positions, slot_mask)
    assert frame_positions.tolist() == [[0, 1, 4, 5, 6, 7, 0, 1]]
    assert gathered.flatten().tolist() == [0.0, 1.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]


def test_native_tubelet_gather_preserves_ncthw_layout_for_videomae_clips() -> None:
    inputs = torch.arange(36, dtype=torch.float32).reshape(1, 1, 3, 12, 1, 1)
    positions = torch.tensor([[0, 2, 3, 5]])
    slot_mask = torch.ones_like(positions, dtype=torch.bool)
    gathered, frame_positions = gather_native_tubelet_rgb(inputs, positions, slot_mask)
    assert gathered.shape == (1, 1, 3, 8, 1, 1)
    assert frame_positions.tolist() == [[0, 1, 4, 5, 6, 7, 10, 11]]
    # The formal K=192 path analogously returns T=384, which the inherited
    # t1=24 preprocessing partitions into 24 physical-input clips of 16 frames.
    clips = gathered.reshape(1, 1, 3, 2, 4, 1, 1).permute(0, 3, 1, 2, 4, 5, 6)
    assert clips.shape == (1, 2, 1, 3, 4, 1, 1)


def test_task_state_coreset_is_exact_k_covered_and_endpoint_anchored() -> None:
    frames = 32
    actionness = torch.linspace(0.0, 1.0, frames).reshape(1, frames)
    boundary = torch.zeros_like(actionness)
    boundary[:, 14:18] = 1.0
    hidden = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1).repeat(1, 1, 4)
    valid = torch.ones((1, frames), dtype=torch.bool)
    aggregated = aggregate_frame_signals_to_tubelets(actionness, boundary, hidden, valid)
    scored = task_state_tubelet_scores(
        aggregated["actionness"],
        aggregated["boundary"],
        aggregated["hidden"],
        aggregated["valid_mask"],
    )
    selection = select_native_tubelet_coreset(
        scored["score"],
        aggregated["valid_mask"],
        policy="native_tubelet_coreset",
        selected_tubelets=8,
        max_unselected_hole=2,
    )
    positions = selection["selected_positions"][0].tolist()
    assert len(positions) == 8
    assert positions == sorted(set(positions))
    assert positions[0] == 0 and positions[-1] == 15
    padded = [-1] + positions + [16]
    assert max(right - left - 1 for left, right in zip(padded, padded[1:])) <= 2
    assert int(selection["selected_mask"].sum().item()) == 8


def test_uniform_and_coreset_configs_have_one_scientific_difference(monkeypatch) -> None:
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT", "/tmp/stage1_epoch29.pth")
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_SHA256", "b" * 64)
    monkeypatch.setenv("DUCA_STAGE1_CHECKPOINT_EPOCH", "29")
    uniform = Config.fromfile(
        ROOT
        / "configs/adatad/thumos/duca_native_tubelet_uniform_reconstruct_fixed384_official60.py"
    )
    coreset = Config.fromfile(
        ROOT
        / "configs/adatad/thumos/duca_native_tubelet_coreset_fixed384_official60.py"
    )
    assert uniform.model.frame_selector.acquisition_policy == "native_tubelet_uniform"
    assert coreset.model.frame_selector.acquisition_policy == "native_tubelet_coreset"
    uniform.model.frame_selector.acquisition_policy = "matched"
    coreset.model.frame_selector.acquisition_policy = "matched"
    uniform.native_tubelet_contract.selection_policy = "matched"
    coreset.native_tubelet_contract.selection_policy = "matched"
    assert uniform.to_dict() == coreset.to_dict()

    assert uniform.workflow.formal_successful_update_contract is False
    assert uniform.workflow.preserve_resume_state is True
    assert uniform.workflow.intermediate_validation_role == "disabled"
    assert uniform.workflow.primary_checkpoint_epoch == 59
    assert uniform.workflow.primary_checkpoint_state_key == "state_dict_ema"


def test_dynamic_window_budgets_are_deterministic_and_average_twenty() -> None:
    rows = [
        {
            "video_name": "v1",
            "window_start_frame": start,
            "mean_actionness": value,
            "p90_boundary": value,
            "p90_novelty": value,
        }
        for start, value in [(0, 0.1), (384, 0.2), (768, 0.3), (1152, 0.4), (1536, 0.5)]
    ]
    assigned = assign_dynamic_native_tubelet_clip_budgets(rows)
    assert [row["clip_budget"] for row in assigned] == [16, 16, 20, 24, 24]
    assert sum(row["clip_budget"] for row in assigned) / len(assigned) == 20
    assert assigned == assign_dynamic_native_tubelet_clip_budgets(rows)


def test_dynamic_uniform_selects_real_per_row_tubelet_counts() -> None:
    scores = torch.zeros((3, 384), dtype=torch.float32)
    valid = torch.ones_like(scores, dtype=torch.bool)
    selected = select_native_tubelet_coreset(
        scores,
        valid,
        policy="native_tubelet_dynamic_uniform",
        selected_tubelets=[128, 160, 192],
    )
    assert selected["slot_mask"].sum(dim=1).tolist() == [128, 160, 192]
    assert selected["selected_positions"][:, 0].tolist() == [0, 0, 0]
    assert [
        selected["selected_positions"][row, count - 1].item()
        for row, count in enumerate((128, 160, 192))
    ] == [383, 383, 383]


class _RecordingVariableBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, variable_num_clips=None):
        self.calls.append((int(variable_num_clips), int(inputs.shape[3])))
        value = inputs.float().mean(dim=(1, 2, 3, 4, 5))[:, None, None]
        return value.expand(-1, 4, int(variable_num_clips) * 8)


def test_actionformer_dynamic_backbone_executes_three_real_buckets_and_restores_order() -> None:
    detector = ActionFormer.__new__(ActionFormer)
    torch.nn.Module.__init__(detector)
    detector.backbone = _RecordingVariableBackbone()
    inputs = torch.stack(
        [torch.full((1, 3, 384, 1, 1), float(value)) for value in (3, 1, 2)]
    ).requires_grad_(True)
    masks = torch.zeros((3, 192), dtype=torch.bool)
    clips = [24, 16, 20]
    metas = []
    for index, count in enumerate(clips):
        masks[index, : count * 8] = True
        metas.append(
            {
                "duca_native_tubelet_policy": "native_tubelet_dynamic_uniform",
                "duca_native_tubelet_actual_clips": count,
            }
        )
    output = detector._forward_heavy_backbone(inputs, masks, metas)
    assert detector.backbone.calls == [(16, 256), (20, 320), (24, 384)]
    assert output.shape == (3, 4, 192)
    assert output[:, 0, 0].tolist() == [3.0, 1.0, 2.0]
    assert detector.last_variable_compute_summary["padded_before_heavy_backbone"] is False
    output.sum().backward()
    assert inputs.grad is not None
