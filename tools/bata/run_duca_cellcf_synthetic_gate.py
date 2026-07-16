from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from opentad.models.duca.counterfactual_utility import (
    build_local_cell_hard_flip_candidates,
    build_swap_incidence_matrix,
    local_cell_signed_logistic_loss,
)
from opentad.models.duca.structured_selection import (
    exact_uniform_cell_bounds,
    exact_uniform_positions,
    local_cell_deformation,
)
from opentad.models.duca.transition_only import DucaTransitionUtilityScorer
from tools.bata.validate_duca_cellcf_fixed384 import VARIANTS, validate_config


AUDITED_PATHS = (
    "opentad/models/detectors/actionformer.py",
    "opentad/models/duca/acquisition.py",
    "opentad/models/duca/counterfactual_utility.py",
    "opentad/models/duca/structured_selection.py",
    "opentad/models/duca/transition_only.py",
    "opentad/models/selectors/duca_online_frame_selector.py",
    "tools/bata/validate_duca_cellcf_fixed384.py",
    "tools/bata/run_duca_cellcf_synthetic_gate.py",
    *VARIANTS.values(),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True, encoding="utf-8").strip()


def _observed_hole(positions: torch.Tensor, temporal_len: int) -> int:
    sentinels = torch.cat((positions.new_tensor([-1]), positions, positions.new_tensor([temporal_len])))
    return int((sentinels[1:] - sentinels[:-1] - 1).max().item())


def _exhaustive_geometry(device: torch.device) -> dict[str, Any]:
    max_theoretical = 0
    max_observed_anchor = 0
    short_all_selected = True
    for temporal_len in range(1, 769):
        budget = min(384, temporal_len)
        logits = torch.zeros((1, temporal_len), device=device)
        output = local_cell_deformation(logits, k=budget, training=False)
        expected = exact_uniform_positions(temporal_len, budget, device=device)
        _require(torch.equal(output.selected_positions[0], expected), f"anchor tie failed at L={temporal_len}")
        _require(int(output.hard_occupancy.sum().item()) == budget, f"exact K failed at L={temporal_len}")
        _require(torch.all(output.selected_positions[0] >= output.cell_starts).item(), "selection below cell")
        _require(torch.all(output.selected_positions[0] < output.cell_ends).item(), "selection above cell")
        _require(int(output.cell_starts[0].item()) == 0, "cells do not start at zero")
        _require(int(output.cell_ends[-1].item()) == temporal_len, "cells do not cover the suffix")
        if budget > 1:
            _require(torch.equal(output.cell_ends[:-1], output.cell_starts[1:]), "cells have a gap/overlap")
        if temporal_len <= 384:
            short_all_selected &= torch.equal(expected, torch.arange(temporal_len, device=device))
        max_theoretical = max(max_theoretical, int(output.max_unselected_hole))
        max_observed_anchor = max(max_observed_anchor, _observed_hole(expected, temporal_len))
    anchors, starts, ends = exact_uniform_cell_bounds(768, 384, device=device)
    widths = ends - starts
    width_histogram = {
        str(width): int((widths == width).sum().item())
        for width in torch.unique(widths).detach().cpu().tolist()
    }
    _require(width_histogram == {"1": 1, "2": 382, "3": 1}, "768/384 cell widths drifted")
    _require(max_theoretical == 3, "local-cell theoretical maximum hole must be three")
    _require(short_all_selected, "L<=K must select every valid observation")
    return {
        "lengths_checked": 768,
        "max_theoretical_unselected_hole": max_theoretical,
        "max_anchor_unselected_hole": max_observed_anchor,
        "width_histogram_768_384": width_histogram,
        "short_windows_select_all": bool(short_all_selected),
        "anchor_count": int(anchors.numel()),
    }


def _soft_family_and_step_zero(device: torch.device) -> dict[str, Any]:
    scorer = DucaTransitionUtilityScorer(8, 12, zero_init_output=True).to(device)
    first = torch.randn((2, 19, scorer.input_dim), device=device)
    second = torch.randn((2, 19, scorer.input_dim), device=device)
    first_scores = scorer(first)
    second_scores = scorer(second)
    _require(torch.equal(first_scores, torch.zeros_like(first_scores)), "step-zero scorer is not zero")
    _require(torch.equal(second_scores, torch.zeros_like(second_scores)), "step-zero scorer depends on input")
    output = local_cell_deformation(first_scores, k=7, temperature=0.7, training=True)
    _require(
        torch.equal(output.selected_positions, output.anchor_positions.expand(2, -1)),
        "step-zero hard selection is not exact uniform",
    )
    _require(torch.allclose(output.soft_slot_assignment.sum(dim=2), torch.ones((2, 7), device=device)), "slot mass drift")
    for cell_index in range(7):
        start = int(output.cell_starts[cell_index].item())
        end = int(output.cell_ends[cell_index].item())
        _require(not bool(output.soft_slot_assignment[:, cell_index, :start].any().item()), "soft mass left cell")
        _require(not bool(output.soft_slot_assignment[:, cell_index, end:].any().item()), "soft mass right cell")
    return {
        "zero_initialized_output": True,
        "two_distinct_inputs_checked": True,
        "hard_soft_same_cell_family": True,
    }


def _counterfactual_contract(device: torch.device) -> dict[str, Any]:
    temporal_len, budget = 24, 8
    anchors, starts, ends = exact_uniform_cell_bounds(temporal_len, budget, device=device)
    scores = torch.linspace(-1.0, 1.0, temporal_len, device=device).reshape(1, -1).requires_grad_()
    request = build_local_cell_hard_flip_candidates(
        anchors.reshape(1, -1),
        scores,
        torch.ones_like(scores, dtype=torch.bool),
        starts.reshape(1, -1),
        ends.reshape(1, -1),
        anchors.reshape(1, -1),
        max_candidates=4,
    )
    valid = request["candidate_valid"]
    cells = request["candidate_cell_indices"][valid]
    _require(int(valid.sum().item()) == 4, "synthetic gate expected four local flips")
    _require(torch.unique(cells).numel() == cells.numel(), "candidate cells are not distinct")
    incidence = build_swap_incidence_matrix(
        scores,
        request["candidate_positions"],
        request["replaced_slots"],
        anchors.reshape(1, -1),
        valid,
    )[0, valid[0]]
    gram = incidence @ incidence.transpose(0, 1)
    _require(torch.equal(gram, 2.0 * torch.eye(gram.shape[0], device=device)), "different-cell AA^T is not 2I")
    utility = torch.tensor([[2.0, -3.0, 1.0, -0.5]], device=device)
    loss = local_cell_signed_logistic_loss(
        scores,
        request["candidate_positions"],
        request["replaced_slots"],
        anchors.reshape(1, -1),
        utility,
        valid,
        temperature=0.7,
    )
    gradient = torch.autograd.grad(loss, scores)[0]
    score_descent = incidence @ (-gradient[0])
    _require(torch.all(score_descent * utility[0, valid[0]] > 0).item(), "signed-logistic descent sign is wrong")
    zero_scores = torch.zeros((1, 4), device=device, requires_grad=True)
    zero_loss = local_cell_signed_logistic_loss(
        zero_scores,
        torch.tensor([[1]], device=device),
        torch.tensor([[0]], device=device),
        torch.tensor([[0]], device=device),
        torch.zeros((1, 1), device=device),
        torch.ones((1, 1), dtype=torch.bool, device=device),
    )
    zero_gradient = torch.autograd.grad(zero_loss, zero_scores)[0]
    _require(float(zero_loss.item()) == 0.0 and not bool(zero_gradient.any().item()), "zero utility is not a connected zero")
    return {
        "candidate_count": int(valid.sum().item()),
        "distinct_candidate_cells": True,
        "swap_incidence_gram_is_2I": True,
        "signed_gradient_direction": True,
        "zero_utility_connected_zero": True,
    }


def run_gate(*, device: str = "cpu", require_clean: bool = True) -> dict[str, Any]:
    git_commit = _git_output("rev-parse", "HEAD")
    status = _git_output("status", "--porcelain", "--untracked-files=normal")
    if require_clean:
        _require(not status, "CellCF synthetic gate requires a clean exact-commit checkout")
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA requested but unavailable")
    config_contracts = {name: validate_config(name) for name in sorted(VARIANTS)}
    return {
        "ok": True,
        "schema": "duca_cellcf_synthetic_gate_v1",
        "git_commit": git_commit,
        "git_tree_clean": not bool(status),
        "audited_file_sha256": {path: _sha256(ROOT / path) for path in AUDITED_PATHS},
        "input_provenance": "deterministic_synthetic_contract_probe",
        "real_dataset_loader_executed": False,
        "ddp_executed": False,
        "device": str(torch_device),
        "config_contracts": config_contracts,
        "geometry": _exhaustive_geometry(torch_device),
        "step_zero": _soft_family_and_step_zero(torch_device),
        "counterfactual": _counterfactual_contract(torch_device),
        "claims": {"c3_supported": False, "c4_supported": False, "paper_ready": False},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output-json")
    args = parser.parse_args(argv)
    try:
        summary = run_gate(device=args.device, require_clean=not args.allow_dirty)
    except Exception as exc:
        summary = {"ok": False, "schema": "duca_cellcf_synthetic_gate_v1", "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    else:
        code = 0
    payload = json.dumps(summary, indent=2, sort_keys=True)
    print(payload)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
