from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "seal_duca_cellcf_postrun_evidence.sh"


def test_external_seal_is_terminal_only_and_commit_bound() -> None:
    source = SCRIPT.read_text(encoding="utf-8")

    assert "DUCA_CELLCF_POSTRUN_CONTROL_ROOT" in source
    assert "DUCA_EVIDENCE_EXPECTED_COMMIT" in source
    assert "DUCA_CELLCF_AGGREGATE_EVIDENCE_SHA256" in source
    assert "DUCA_CELLCF_COST_RECOVERY_MANIFEST" in source
    assert "--cost-recovery-manifest" in source
    assert "1642f265e48391418a7c8a4a087e33e2b7bf6899" in source
    assert "trained and evidence commits must be distinct" in source
    assert "observer outside the post-run DAG" in source
    assert "postrun_evidence_candidate.json" in source
    assert "postrun_evidence_complete.json" in source
    assert "finalize_duca_cellcf_postrun_evidence" in source
    assert "--candidate" not in source
    assert "sbatch " not in source
    assert " release " not in source
