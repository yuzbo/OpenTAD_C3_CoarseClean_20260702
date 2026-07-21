from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from tools.bata.aggregate_duca_frontend_candidates import EXPECTED_VARIANTS
    from tools.bata.run_duca_frontend_p0_real_gate import (
        _coarse_subgroup,
        _group_parameter_change_evidence,
    )
    from tools.bata.validate_duca_frontend_p0_contract import validate_config
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"DUCA contract dependencies are unavailable: {exc}", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


@pytest.mark.parametrize(
    ("name", "expected_lrs"),
    (
        (
            "duca_frontend_pretrain_lr_control_c25_a50_s100.py",
            {"coarse_trunk": 2.5e-5, "action_head": 5.0e-5, "transition_scorer": 1.0e-4},
        ),
        (
            "duca_frontend_pretrain_lr_coarse50_action100_scorer25.py",
            {"coarse_trunk": 5.0e-5, "action_head": 1.0e-4, "transition_scorer": 2.5e-5},
        ),
        (
            "duca_frontend_pretrain_lr_coarse100_action200_scorer50.py",
            {"coarse_trunk": 1.0e-4, "action_head": 2.0e-4, "transition_scorer": 5.0e-5},
        ),
    ),
)
def test_frontend_variants_satisfy_strict_p0_contract(
    name: str,
    expected_lrs: dict[str, float],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DUCA_FRONTEND_TRAIN_BLOCK_LIST", "train_block.txt")

    payload = validate_config(CONFIG_ROOT / name)

    assert payload["ok"] is True
    assert payload["detector_executed"] is False
    assert payload["active_losses"] == [
        "actionness",
        "transition",
        "transition_boundary",
    ]
    assert all(
        value == 0.0
        for key, value in payload["loss_weights"].items()
        if key not in payload["active_losses"]
    )
    assert payload["spatial_norm"] == "groupnorm"
    assert payload["auxiliary_hidden_gradient_scale"] == 0.0
    assert payload["optimizer"]["global_gradient_clipping_enabled"] is False
    assert payload["component_lrs"] == expected_lrs
    assert payload["loss_weights"]["actionness"] == 1.0
    assert payload["loss_weights"]["transition"] == 0.10
    assert payload["loss_weights"]["transition_boundary"] == 16.0


def test_serial_curriculum_runs_one_real_gate_before_frontend_training() -> None:
    serial_launcher = (ROOT / "scripts" / "run_duca_two_stage_curriculum_serial_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert "run_duca_frontend_p0_real_gate.py" in serial_launcher
    assert "--standalone" in serial_launcher
    assert '"${DUCA_FRONTEND_ONLY:-0}" == "1"' in serial_launcher


def test_frontend_quality_tools_are_invoked_as_repo_modules() -> None:
    source = (
        ROOT / "scripts" / "run_duca_frontend_pretrain_variant_gpu1.sh"
    ).read_text(encoding="utf-8")
    assert "-m tools.bata.export_duca_selection_quality" in source
    assert "-m tools.bata.analyze_duca_selection_quality" in source
    assert "tools/bata/export_duca_selection_quality.py" not in source
    assert "tools/bata/analyze_duca_selection_quality.py" not in source


def test_serial_aggregators_are_invoked_as_repo_modules() -> None:
    source = (
        ROOT / "scripts" / "run_duca_two_stage_curriculum_serial_gpu1.sh"
    ).read_text(encoding="utf-8")
    assert "-m tools.bata.aggregate_duca_frontend_candidates" in source
    assert "-m tools.bata.aggregate_duca_two_stage_results" in source
    assert "tools/bata/aggregate_duca_frontend_candidates.py" not in source
    assert "tools/bata/aggregate_duca_two_stage_results.py" not in source


def test_real_gate_classifies_the_executed_spatial_stem_parameter_path() -> None:
    gate_source = (ROOT / "tools" / "bata" / "run_duca_frontend_p0_real_gate.py").read_text(
        encoding="utf-8"
    )
    assert 'if ".spatial_stem." in normalized:' in gate_source
    assert 'if ".spatial_encoder." in normalized:' not in gate_source


def test_real_gate_binds_declared_component_learning_rates_to_optimizer_groups() -> None:
    gate_source = (
        ROOT / "tools" / "bata" / "run_duca_frontend_p0_real_gate.py"
    ).read_text(encoding="utf-8")
    assert "def _expected_parameter_lr(name: str, selector)" in gate_source
    assert "declared_component_learning_rates_realized" in gate_source
    assert "_optimizer_partition(model, optimizer, selector)" in gate_source


def test_real_gate_measures_ema_updates_over_each_parameter_group() -> None:
    import torch

    coarse_first = "frame_selector.raw_actionness_source.probe_module.spatial_stem.0.weight"
    coarse_second = "frame_selector.raw_actionness_source.probe_module.spatial_stem.1.bias"
    scorer = "frame_selector.adapter.transition_scorer.0.weight"
    before = {
        coarse_first: torch.tensor([1.0]),
        coarse_second: torch.tensor([0.0]),
        scorer: torch.tensor([0.0]),
    }
    after = {
        coarse_first: before[coarse_first].clone(),
        coarse_second: torch.tensor([1.0e-8]),
        scorer: torch.tensor([2.0e-8]),
    }

    evidence = _group_parameter_change_evidence(after, before)

    assert evidence["coarse_probe"]["parameter_count"] == 2
    assert evidence["coarse_probe"]["changed_parameter_count"] == 1
    assert evidence["coarse_probe"]["max_abs_change"] > 0.0
    assert evidence["transition_scorer"]["changed_parameter_count"] == 1


@pytest.mark.parametrize(
    ("name", "expected"),
    (
        (
            "frame_selector.raw_actionness_source.probe_module.official_temporal.encoder.conv_out.weight",
            "action_head",
        ),
        (
            "frame_selector.raw_actionness_source.probe_module.official_temporal.decoders.0.conv_out.bias",
            "action_head",
        ),
        (
            "frame_selector.raw_actionness_source.probe_module.official_temporal.encoder.layers.0.att_layer.conv_out.weight",
            "temporal_trunk",
        ),
        (
            "frame_selector.raw_actionness_source.probe_module.official_temporal.decoders.0.layers.0.att_layer.conv_out.weight",
            "temporal_trunk",
        ),
    ),
)
def test_real_gate_matches_actionformer_asformer_head_partition(name: str, expected: str) -> None:
    assert _coarse_subgroup(name) == expected


def test_frontend_grid_varies_learning_speed_not_auxiliary_loss_definition() -> None:
    assert set(EXPECTED_VARIANTS) == {
        "lr_control_c25_a50_s100",
        "lr_coarse50_action100_scorer25",
        "lr_coarse100_action200_scorer50",
    }
    losses = {tuple(sorted(spec["loss_weights"].items())) for spec in EXPECTED_VARIANTS.values()}
    lrs = {tuple(sorted(spec["component_lrs"].items())) for spec in EXPECTED_VARIANTS.values()}
    assert losses == {
        (("actionness", 1.0), ("transition", 0.10), ("transition_boundary", 16.0))
    }
    assert len(lrs) == 3
    for name, spec in EXPECTED_VARIANTS.items():
        component_lrs = spec["component_lrs"]
        if name == "lr_control_c25_a50_s100":
            assert component_lrs["transition_scorer"] > component_lrs["coarse_trunk"]
        else:
            assert component_lrs["transition_scorer"] < component_lrs["coarse_trunk"]
