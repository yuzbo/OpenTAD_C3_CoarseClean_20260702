from __future__ import annotations

from pathlib import Path

import torch
from mmengine.config import Config

from opentad.models.duca.tubelet_coreset import (
    aggregate_frame_signals_to_tubelets,
    build_native_tubelet_candidates,
    gather_native_tubelet_rgb,
    select_native_tubelet_coreset,
    task_state_tubelet_scores,
)


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
