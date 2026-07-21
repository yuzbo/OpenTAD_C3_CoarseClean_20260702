from __future__ import annotations

import os
from pathlib import Path

import pytest

try:
    from tools.bata.validate_duca_frontend_p0_contract import validate_config
except Exception as exc:  # pragma: no cover - local Windows torch/c10.dll guard.
    pytest.skip(f"DUCA contract dependencies are unavailable: {exc}", allow_module_level=True)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"


@pytest.mark.parametrize(
    "name",
    (
        "duca_frontend_pretrain_a1_t005_b8.py",
        "duca_frontend_pretrain_a1_t010_b16.py",
        "duca_frontend_pretrain_a1_t020_b32.py",
    ),
)
def test_frontend_variants_satisfy_strict_p0_contract(
    name: str, monkeypatch: pytest.MonkeyPatch
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


def test_serial_curriculum_runs_one_real_gate_before_frontend_training() -> None:
    serial_launcher = (ROOT / "scripts" / "run_duca_two_stage_curriculum_serial_gpu1.sh").read_text(
        encoding="utf-8"
    )
    assert "run_duca_frontend_p0_real_gate.py" in serial_launcher
    assert "--standalone" in serial_launcher
    assert '"${DUCA_FRONTEND_ONLY:-0}" == "1"' in serial_launcher
