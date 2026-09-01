from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "scripts/duca_cellcf_path_contract.sh"


@pytest.mark.skipif(os.name == "nt", reason="validated by remote Linux gate")
def test_path_contract_accepts_only_external_paths_under_base(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    repo = base / "projects" / "repo"
    external = base / "runs" / "formal"
    repo.mkdir(parents=True)
    external.parent.mkdir(parents=True)

    command = (
        f"source {CONTRACT!s}; "
        "duca_cellcf_require_external_path RUN_ROOT \"$1\" \"$2\" \"$3\""
    )
    accepted = subprocess.run(
        ["bash", "-c", command, "bash", str(repo), str(base), str(external)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(accepted.stdout.strip()) == external.resolve()

    for rejected in (repo / "ignored" / "run", tmp_path / "outside"):
        result = subprocess.run(
            [
                "bash",
                "-c",
                command,
                "bash",
                str(repo),
                str(base),
                str(rejected),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    symlink = base / "runs" / "escape"
    symlink.symlink_to(tmp_path / "outside-target", target_is_directory=True)
    escaped = subprocess.run(
        [
            "bash",
            "-c",
            command,
            "bash",
            str(repo),
            str(base),
            str(symlink / "formal"),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert escaped.returncode != 0


def test_all_formal_launchers_reuse_the_path_contract() -> None:
    launchers = (
        "prepare_duca_cellcf_suite.sh",
        "submit_duca_cellcf_suite.sh",
        "prepare_duca_cellcf_ddp_pilot.sh",
        "run_duca_cellcf_ddp_pilot.sh",
        "run_duca_cellcf_variant.sh",
        "run_duca_cellcf_convergence_variant.sh",
        "summarize_duca_cellcf_convergence.sh",
        "summarize_duca_cellcf_training_cost.sh",
        "run_duca_cellcf_cost_pair.sh",
        "run_duca_cellcf_dense_full_stack_cost.sh",
        "run_duca_cellcf_official60_synthetic_gate.sh",
        "run_duca_cellcf_official60_real_loader_gate.sh",
    )
    for name in launchers:
        source = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "duca_cellcf_path_contract.sh" in source, name
        assert "duca_cellcf_require_external_path" in source, name
