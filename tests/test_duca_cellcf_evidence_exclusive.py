from __future__ import annotations

import os
from pathlib import Path

import pytest

if os.name == "nt":
    pytest.skip(
        "real-loader gate imports the remote Linux Torch runtime",
        allow_module_level=True,
    )

from tools.bata import finalize_duca_cellcf_run as finalizer
from tools.bata import run_duca_cellcf_real_loader_cuda_gate as real_gate
from tools.bata import run_duca_cellcf_synthetic_gate as synthetic_gate
from tools.bata import validate_duca_cellcf_ddp_pilot as pilot_validator
from tools.bata import validate_duca_cellcf_fixed384 as variant_validator
from tools.bata import validate_duca_cellcf_suite as suite_validator


def _sealed(path: Path) -> bytes:
    content = b"sealed-evidence\n"
    path.write_bytes(content)
    return content


def _must_not_run(*_args, **_kwargs):
    raise AssertionError("expensive validation must not run")


def test_synthetic_gate_refuses_existing_output_before_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "synthetic.json"
    original = _sealed(output)
    monkeypatch.setattr(synthetic_gate, "run_gate", _must_not_run)

    assert synthetic_gate.main(["--output-json", str(output)]) == 1
    assert output.read_bytes() == original


def test_real_loader_gate_refuses_existing_output_before_gate(
    tmp_path: Path, monkeypatch,
) -> None:
    output = tmp_path / "real-loader.json"
    original = _sealed(output)
    monkeypatch.setattr(real_gate, "run_gate", _must_not_run)

    code = real_gate.main(
        [
            "--expected-commit",
            "a" * 40,
            "--synthetic-gate-json",
            "synthetic.json",
            "--videomae-checkpoint",
            "model.pth",
            "--expected-videomae-sha256",
            "b" * 64,
            "--official-repos-root",
            "official",
            "--output-json",
            str(output),
        ]
    )
    assert code == 1
    assert output.read_bytes() == original


def test_suite_and_finalizer_refuse_existing_outputs_before_validation(
    tmp_path: Path, monkeypatch,
) -> None:
    suite_output = tmp_path / "suite.json"
    final_output = tmp_path / "post-run.json"
    suite_original = _sealed(suite_output)
    final_original = _sealed(final_output)
    monkeypatch.setattr(suite_validator, "validate_suite", _must_not_run)
    monkeypatch.setattr(finalizer, "finalize_run", _must_not_run)

    assert (
        suite_validator.main(
            [
                "--gate-json",
                "gate.json",
                "--pilot-json",
                "pilot.json",
                "--output-json",
                str(suite_output),
            ]
        )
        == 1
    )
    assert (
        finalizer.main(
            [
                "--variant",
                "cellcf",
                "--run-manifest",
                "manifest.json",
                "--training-audit",
                "audit.json",
                "--checkpoint",
                "checkpoint.pth",
                "--checkpoint-sidecar",
                "checkpoint.metadata.json",
                "--evaluation-json",
                "evaluation.json",
                "--output-json",
                str(final_output),
            ]
        )
        == 1
    )
    assert suite_output.read_bytes() == suite_original
    assert final_output.read_bytes() == final_original


def test_variant_and_pilot_validators_refuse_existing_outputs(
    tmp_path: Path, monkeypatch,
) -> None:
    variant_output = tmp_path / "variant.json"
    pilot_output = tmp_path / "pilot.json"
    variant_original = _sealed(variant_output)
    pilot_original = _sealed(pilot_output)
    monkeypatch.setattr(variant_validator, "validate_config", _must_not_run)

    assert (
        variant_validator.main(
            ["--variant", "cellcf", "--output-json", str(variant_output)]
        )
        == 1
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pilot_validator.main(
            [
                "--repo-root",
                str(tmp_path),
                "--real-loader-gate-json",
                "gate.json",
                "--expected-real-loader-gate-sha256",
                "a" * 64,
                "--expected-commit",
                "b" * 40,
                "--precheck-only",
                "--output-json",
                str(pilot_output),
            ]
        )
    assert variant_output.read_bytes() == variant_original
    assert pilot_output.read_bytes() == pilot_original
