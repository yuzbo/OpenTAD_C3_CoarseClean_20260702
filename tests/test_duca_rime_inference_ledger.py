from __future__ import annotations

import json

import pytest

from tools.bata.finalize_duca_rime_inference_ledger import finalize_ledger


def _row(video, start=0):
    return {
        "schema_version": "duca_rime_inference_ledger_v1",
        "video_id": video,
        "window_start_frame": start,
        "arm": "rime_full",
        "candidate_budgets": [2, 4],
        "requested_k": 4,
        "effective_k": 4,
        "unique_k": 4,
        "backbone_input_k": 4,
        "padded_k": 4,
        "risk_fallback": False,
        "cost_unit": "heavy_rgb_frames",
        "selected_dense_indices": [0, 1, 2, 3],
        "budget_protocol_sha256": "a" * 64,
        "provenance": {
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "uses_test_batch_composition": False,
            "raw_predictions_stored": False,
        },
    }


def test_finalize_rime_inference_ledger_is_sorted_unique_and_no_padding(tmp_path):
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(
        json.dumps(_row("v1")) + "\n" + json.dumps(_row("v0")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "ledger.jsonl"
    result = finalize_ledger(
        shards=[shard],
        output_jsonl=output,
        expected_arm="rime_full",
        expected_protocol_sha256="a" * 64,
    )
    assert result["record_count"] == 2
    assert result["no_padding_ledger"] is True
    assert json.loads(output.read_text(encoding="utf-8").splitlines()[0])["video_id"] == "v0"


def test_finalize_rime_inference_ledger_rejects_padding(tmp_path):
    row = _row("v0")
    row["padded_k"] = 8
    shard = tmp_path / "rank0.jsonl"
    shard.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no-padding"):
        finalize_ledger(
            shards=[shard],
            output_jsonl=tmp_path / "ledger.jsonl",
            expected_arm="rime_full",
        )
