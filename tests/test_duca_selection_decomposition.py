from __future__ import annotations

import json
from pathlib import Path

from tools.bata.analyze_duca_selection_quality import RECORD_SCHEMA_VERSION
from tools.bata import diagnose_duca_selection_decomposition as decomposition


def _row() -> dict:
    return {"schema_version": RECORD_SCHEMA_VERSION, "sample_id": "v|0", "video_id": "v", "valid_len": 8, "budget": 4, "gt_segments": [[2, 6]], "p_action": [.1, .1, .8, .9, .9, .8, .1, .1], "transition_policy_scores": [0, 0, 1, 0, 0, 0, 1, 0], "abs_delta_p_action": [0, 0, .7, .1, 0, .1, .7, 0], "raw_transition_scores": [0, 0, 1, 0, 0, 0, 1, 0], "selected_positions": [0, 2, 6, 7], "decode_diagnostics": {"repair_count": 1}}


def test_decomposition_contains_required_three_policies_and_metrics(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(_row()) + "\n", encoding="utf-8")
    result = decomposition.run(source, tmp_path / "out")
    assert set(result["selection_macro"]) == {"exact_uniform", "learned", "gt_informed_heuristic_evaluation_only"}
    assert result["selection_macro"]["gt_informed_heuristic_evaluation_only"]["boundary_recall_r0"] == 1.0
    assert result["contract"]["is_optimized_oracle"] is False
    assert result["selection_macro"]["learned"]["repair_ratio"] == 0.25
    assert set(result["coarse_macro"]) == {"auroc", "auprc", "ece"}
    assert (tmp_path / "out" / "selection_quality_decomposition_per_sample.csv").is_file()
