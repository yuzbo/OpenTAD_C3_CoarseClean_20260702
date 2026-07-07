from __future__ import annotations

import json
from pathlib import Path

from tools.bata import summarize_sparse_selector_evidence as evidence


def test_parse_adatad_map_curve_accepts_epoch_tiou_and_average_map(tmp_path: Path) -> None:
    log_path = tmp_path / "train.out"
    log_path.write_text(
        "\n".join(
            [
                "Epoch [9] validation",
                "tIoU = 0.30: mAP = 69.50",
                "tIoU = 0.50: mAP = 50.26",
                "Average-mAP: 48.08",
                "Epoch: 19 validation",
                "mAP@0.30 = 76.79",
                "mAP@0.70 = 39.08",
                "Avg mAP: 60.41",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = evidence.parse_adatad_map_curve(log_path)

    assert records == [
        {"epoch": 9, "average_mAP": 48.08, "tIoU_mAP": {"0.30": 69.50, "0.50": 50.26}},
        {"epoch": 19, "average_mAP": 60.41, "tIoU_mAP": {"0.30": 76.79, "0.70": 39.08}},
    ]


def test_build_evidence_summary_keeps_claims_diagnostic_only(tmp_path: Path) -> None:
    log_path = tmp_path / "train.out"
    log_path.write_text("Epoch 9\nAverage-mAP: 44.90\n", encoding="utf-8")
    ledger_path = tmp_path / "validation.json"
    ledger_path.write_text(
        json.dumps(
            {
                "strategy": "gas_vt_fixed_384",
                "mean_selected_count": 366.14,
                "selected_count_histogram": {"384": 10},
                "p_action_topk_jaccard": 0.43,
                "uses_uniform_fill": False,
                "ignored_payload": "not copied",
            }
        ),
        encoding="utf-8",
    )

    summary = evidence.build_evidence_summary(
        map_logs=[("gas", log_path)],
        ledger_summaries=[("gas_val", ledger_path)],
    )

    assert summary["decision"] == "C3_SPARSE_SELECTOR_EVIDENCE_SUMMARY_READY"
    assert summary["claim_status"] == "diagnostic_only_no_causal_claim"
    assert summary["map_curves"]["gas"]["records"][0]["average_mAP"] == 44.90
    assert summary["ledger_summaries"]["gas_val"]["metrics"] == {
        "strategy": "gas_vt_fixed_384",
        "mean_selected_count": 366.14,
        "selected_count_histogram": {"384": 10},
        "p_action_topk_jaccard": 0.43,
        "uses_uniform_fill": False,
    }
