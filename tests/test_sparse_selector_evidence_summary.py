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


def test_parse_adatad_map_curve_accepts_open_tad_logged_tiou_format(tmp_path: Path) -> None:
    log_path = tmp_path / "train.out"
    log_path.write_text(
        "\n".join(
            [
                "2026-07-07 12:37:37 Train INFO: [Train]: Epoch 19 started",
                "2026-07-07 12:37:37 Train INFO: mAP at tIoU 0.30 is 74.41%",
                "2026-07-07 12:37:37 Train INFO: mAP at tIoU 0.50 is 54.12%",
                "2026-07-07 12:37:37 Train INFO: mAP at tIoU 0.70 is 24.22%",
                "2026-07-07 12:37:37 Train INFO: Average-mAP: 51.61 (%)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = evidence.parse_adatad_map_curve(log_path)

    assert records == [
        {
            "epoch": 19,
            "average_mAP": 51.61,
            "tIoU_mAP": {"0.30": 74.41, "0.50": 54.12, "0.70": 24.22},
        }
    ]


def test_curve_diagnostics_flags_plateau_and_high_iou_delta() -> None:
    records = [
        {"epoch": 9, "average_mAP": 44.90, "tIoU_mAP": {"0.70": 26.92}},
        {"epoch": 19, "average_mAP": 45.20, "tIoU_mAP": {"0.70": 27.10}},
        {"epoch": 29, "average_mAP": 45.70, "tIoU_mAP": {"0.70": 27.50}},
    ]

    diagnostics = evidence.curve_diagnostics(records, plateau_epsilon=1.0)

    assert diagnostics["eval_count"] == 3
    assert diagnostics["first_eval_epoch"] == 9
    assert diagnostics["last_eval_epoch"] == 29
    assert diagnostics["best_eval_epoch"] == 29
    assert diagnostics["first_to_last_average_mAP_delta"] == 0.8
    assert diagnostics["plateau_after_first_eval"] is True
    assert diagnostics["first_to_last_high_iou_delta"] == 0.58
    assert diagnostics["records_have_high_iou"] is True


def test_build_evidence_summary_keeps_claims_diagnostic_only(tmp_path: Path) -> None:
    log_path = tmp_path / "train.out"
    paction_log_path = tmp_path / "paction.train.out"
    log_path.write_text(
        "Epoch 9\nmAP at tIoU 0.70 is 26.92%\nAverage-mAP: 44.90\n"
        "Epoch 19\nmAP at tIoU 0.70 is 27.40%\nAverage-mAP: 45.20\n",
        encoding="utf-8",
    )
    paction_log_path.write_text(
        "Epoch 9\nmAP at tIoU 0.70 is 39.51%\nAverage-mAP: 59.10\n",
        encoding="utf-8",
    )
    ledger_path = tmp_path / "validation.json"
    paction_ledger_path = tmp_path / "paction.validation.json"
    ledger_path.write_text(
        json.dumps(
            {
                "strategy": "gas_vt_fixed_384",
                "mean_selected_count": 366.14,
                "max_unselected_hole": 96,
                "p95_unselected_hole": 96.0,
                "boundary_support_r1": 0.7,
                "action_positive_coverage": 0.8,
                "selected_count_histogram": {"384": 10},
                "p_action_topk_jaccard": 0.43,
                "uses_uniform_fill": False,
                "ignored_payload": "not copied",
            }
        ),
        encoding="utf-8",
    )
    paction_ledger_path.write_text(
        json.dumps(
            {
                "strategy": "learned_fixed_384",
                "mean_selected_count": 384.0,
                "max_unselected_hole": 16,
                "p95_unselected_hole": 8.0,
                "boundary_support_r1": 0.95,
                "action_positive_coverage": 0.96,
                "p_action_topk_overlap_ratio": 0.82,
            }
        ),
        encoding="utf-8",
    )

    summary = evidence.build_evidence_summary(
        map_logs=[("gas", log_path), ("paction", paction_log_path)],
        ledger_summaries=[("gas_val", ledger_path), ("paction_val", paction_ledger_path)],
    )

    assert summary["decision"] == "C3_SPARSE_SELECTOR_EVIDENCE_SUMMARY_READY"
    assert summary["claim_status"] == "diagnostic_only_no_causal_claim"
    assert summary["map_curves"]["gas"]["records"][0]["average_mAP"] == 44.90
    assert summary["map_curves"]["gas"]["diagnostics"]["plateau_after_first_eval"] is True
    assert (
        summary["map_curve_comparisons"]["gas__minus__paction"]["delta_by_epoch"]["9"][
            "gas_minus_paction_average_mAP"
        ]
        == -14.2
    )
    assert summary["map_curve_comparisons"]["gas__minus__paction"]["last_common_epoch"] == 9
    assert summary["ledger_metric_comparisons"]["gas_val__minus__paction_val"]["numeric_delta"]["max_unselected_hole"] == 80.0
    assert summary["evidence_gaps"] == []
    assert summary["ledger_summaries"]["gas_val"]["metrics"] == {
        "strategy": "gas_vt_fixed_384",
        "mean_selected_count": 366.14,
        "max_unselected_hole": 96,
        "p95_unselected_hole": 96.0,
        "boundary_support_r1": 0.7,
        "action_positive_coverage": 0.8,
        "selected_count_histogram": {"384": 10},
        "p_action_topk_jaccard": 0.43,
        "uses_uniform_fill": False,
    }
