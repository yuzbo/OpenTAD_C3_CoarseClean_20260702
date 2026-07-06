from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="torch training smoke tests run on the Linux/remote training environment",
)

from tools.bata import apply_paction_acquisition_policy as apply_policy
from tools.bata import train_paction_acquisition_policy as train_policy


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sample_row(sample_id: str, p_action: list[float], action_target: list[float], boundaries: list[int]) -> dict:
    return {
        "sample_id": sample_id,
        "split": "training",
        "dense_len": len(p_action),
        "valid_len": len(p_action),
        "frame_signals": {"p_action": p_action},
        "p_action": p_action,
        "action_target": action_target,
        "gt_boundaries": boundaries,
        "strategy_selected_positions": {"delta_p_action": [1, 4]},
    }


def test_paction_policy_training_writes_checkpoint_used_by_application(tmp_path: Path) -> None:
    train_jsonl = tmp_path / "train.samples.jsonl"
    checkpoint = tmp_path / "paction_policy.pth"
    summary_json = tmp_path / "train.summary.json"
    enriched_jsonl = tmp_path / "samples.policy.jsonl"
    apply_summary_json = tmp_path / "apply.summary.json"
    rows = [
        _sample_row("video_test_0001|0", [0.05, 0.70, 0.92, 0.20, 0.80, 0.10], [0, 1, 1, 0, 1, 0], [1, 4]),
        _sample_row("video_test_0002|0", [0.10, 0.25, 0.88, 0.91, 0.35, 0.05], [0, 0, 1, 1, 0, 0], [2, 3]),
    ]
    _write_jsonl(train_jsonl, rows)

    train_summary = train_policy.run_training(
        train_jsonl,
        out_dir=tmp_path / "train_out",
        checkpoint_path=checkpoint,
        summary_json=summary_json,
        epochs=2,
        batch_size=2,
        hidden_dim=8,
        num_layers=1,
        fixed_budget=3,
        dynamic_budget_buckets=[2, 3, 4],
        device="cpu",
        seed=0,
    )

    assert train_summary["decision"] == "C3_PACTION_POLICY_TRAIN_READY"
    assert checkpoint.is_file()
    assert summary_json.is_file()
    assert train_summary["uses_uniform_scaffold"] is False
    assert train_summary["uses_uniform_fill"] is False

    apply_summary = apply_policy.run_policy_application(
        train_jsonl,
        enriched_jsonl,
        summary_json=apply_summary_json,
        fixed_budget=3,
        dynamic_budget_buckets=[2, 3, 4],
        checkpoint_path=checkpoint,
        device="cpu",
    )
    enriched_rows = _read_jsonl(enriched_jsonl)

    assert apply_summary["source"] == "learned_paction_gap_loss_policy_checkpoint"
    assert apply_summary["checkpoint_path"] == str(checkpoint)
    assert enriched_rows[0]["paction_policy"]["source"] == "learned_paction_gap_loss_policy_checkpoint"
    assert enriched_rows[0]["paction_policy"]["uses_uniform_fill"] is False
    assert len(enriched_rows[0]["strategy_selected_positions"]["learned_paction_gap_loss_value"]) == 3
    assert len(enriched_rows[0]["strategy_selected_positions"]["learned_paction_gap_loss_dynamic_budget"]) in {2, 3, 4}
