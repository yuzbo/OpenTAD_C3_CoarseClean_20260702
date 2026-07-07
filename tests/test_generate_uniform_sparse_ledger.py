from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.bata import generate_uniform_sparse_ledger as uniform_ledger


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_generate_uniform_sparse_ledger_exact_384_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    summary_json = tmp_path / "summary.json"
    _write_jsonl(
        source,
        [
            {"sample_id": "video_a|0", "video_name": "video_a", "dense_len": 768, "valid_len": 768},
            {"sample_id": "video_b|0", "video_name": "video_b", "dense_len": 400, "valid_len": 400},
        ],
    )

    summary = uniform_ledger.run_generation(
        source,
        ledger,
        summary_json=summary_json,
        target_len=384,
    )

    rows = _read_jsonl(ledger)
    assert len(rows) == 2
    assert summary["selected_count_histogram"] == {"384": 2}
    assert summary["min_selected_count"] == 384
    assert summary["max_selected_count"] == 384
    assert summary["short_valid_count"] == 0
    assert summary["source_sha256"] == uniform_ledger.sha256_file(source)
    assert summary["ledger_sha256"] == uniform_ledger.sha256_file(ledger)
    assert json.loads(summary_json.read_text(encoding="utf-8")) == summary

    for row in rows:
        assert row["selected_count"] == 384
        assert row["target_len"] == 384
        assert row["selection_family"] == "uniform_exact"
        assert row["uses_uniform_scaffold"] is True
        assert row["uses_gt"] is False
        assert row["uses_teacher"] is False
        assert row["uses_prediction_cache"] is False
        assert row["deploy_selection_ledger"] is True
        assert row["selected_positions"][0] == 0
        assert row["selected_positions"][-1] == row["valid_len"] - 1


def test_generate_uniform_sparse_ledger_short_valid_fails_by_default(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    _write_jsonl(source, [{"sample_id": "short|0", "dense_len": 128, "valid_len": 128}])

    with pytest.raises(ValueError, match="short valid_len"):
        uniform_ledger.run_generation(
            source,
            tmp_path / "ledger.jsonl",
            target_len=384,
        )


def test_generate_uniform_sparse_ledger_short_valid_opt_in_records_summary(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(
        source,
        [
            {"sample_id": "short|0", "dense_len": 5, "valid_len": 5},
            {"sample_id": "mask_short|0", "dense_len": 6, "valid_mask": [1, 1, 1, 0, 0, 0]},
        ],
    )

    summary = uniform_ledger.run_generation(
        source,
        ledger,
        target_len=4,
        allow_short_valid=True,
    )

    rows = _read_jsonl(ledger)
    assert [row["selected_positions"] for row in rows] == [[0, 1, 3, 4], [0, 1, 2]]
    assert [row["selected_count"] for row in rows] == [4, 3]
    assert summary["selected_count_histogram"] == {"3": 1, "4": 1}
    assert summary["short_valid_count"] == 1
    assert summary["allow_short_valid"] is True


def test_uniform_positions_are_sorted_unique_and_within_valid_range() -> None:
    positions = uniform_ledger.uniform_positions(valid_len=401, target_len=384)

    assert positions == sorted(positions)
    assert len(positions) == len(set(positions)) == 384
    assert min(positions) == 0
    assert max(positions) == 400


def test_generate_uniform_sparse_ledger_summary_and_no_leak_flags(tmp_path: Path) -> None:
    source = tmp_path / "source.jsonl"
    ledger = tmp_path / "ledger.jsonl"
    _write_jsonl(source, [{"sample_id": "video_a|0", "video_name": "video_a", "dense_len": 8, "valid_len": 8}])

    summary = uniform_ledger.run_generation(source, ledger, target_len=4)
    rows = _read_jsonl(ledger)

    assert summary["decision"] == "C3_UNIFORM_EXACT_LEDGER_READY"
    assert summary["selection_family"] == "uniform_exact"
    assert summary["uses_uniform_scaffold"] is True
    assert summary["uses_gt"] is False
    assert summary["uses_teacher"] is False
    assert summary["uses_prediction_cache"] is False
    assert summary["deploy_selection_ledger"] is True
    assert rows == [
        {
            "schema_version": "c3_uniform_sparse_ledger_v1",
            "sample_id": "video_a|0",
            "video_name": "video_a",
            "dense_len": 8,
            "valid_len": 8,
            "selected_positions_unit": "local_dense_index",
            "selected_positions": [0, 2, 5, 7],
            "selected_count": 4,
            "target_len": 4,
            "selection_family": "uniform_exact",
            "uses_uniform_scaffold": True,
            "uses_gt": False,
            "uses_teacher": False,
            "uses_prediction_cache": False,
            "deploy_selection_ledger": True,
        }
    ]
