import hashlib
import json

import pytest

from tools.bata.build_duca_rime_source_manifest import (
    build_o1_manifest,
    build_o2_manifest,
    build_phase0_manifest,
)


def _json(path, payload):
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def _jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(tmp_path, *, name, budget, seed, checkpoint_sha256):
    terminal = _json(
        tmp_path / f"{name}_terminal.json",
        {"checkpoint_sha256": checkpoint_sha256},
    )
    return _json(
        tmp_path / f"{name}_metrics.json",
        {
            "schema_version": "duca_rime_localization_metrics_v1",
            "phase": 2,
            "uses_official_final": False,
            "official_final_used_for_training_or_selection": False,
            "padded_to_kmax": False,
            "target_mean_cost": float(budget),
            "seed": int(seed),
            "split_role": "certification_development",
            "split_assignment_sha256": "a" * 64,
            "terminal_evaluation_path": str(terminal.resolve()),
            "terminal_evaluation_sha256": _sha(terminal),
        },
    )


def _ledger(tmp_path, *, family, budget):
    selected = list(range(budget))
    path = tmp_path / f"{family}_{budget}_ledger.jsonl"
    return _jsonl(
        path,
        [
            {
                "schema_version": "duca_rime_inference_ledger_v1",
                "video_id": "video_0001",
                "window_start_frame": 0,
                "requested_k": budget,
                "effective_k": budget,
                "unique_k": budget,
                "backbone_input_k": budget,
                "padded_k": budget,
                "selected_dense_indices": selected,
                "observed_max_gap_seconds": 0.25,
                "max_gap_seconds_cap": 0.25,
            }
        ],
    )


def _mixed_k_receipt(tmp_path, *, checkpoint_sha256):
    return _json(
        tmp_path / "mixed_k_training_receipt.json",
        {
            "schema_version": "duca_rime_phase2_mixed_k_training_receipt_v1",
            "status": "passed",
            "arm": "U-mixed-K",
            "detector_training_exposure": "mixed_k_registered_panel",
            "checkpoint_sha256": checkpoint_sha256,
            "successful_detector_updates": 6000,
            "uses_official_final": False,
        },
    )


def test_phase0_manifest_reopens_metrics_and_terminal_checkpoint(tmp_path):
    checkpoint = "1" * 64
    first = _metrics(
        tmp_path,
        name="first",
        budget=384,
        seed=3407,
        checkpoint_sha256=checkpoint,
    )
    second = _metrics(
        tmp_path,
        name="second",
        budget=384,
        seed=3408,
        checkpoint_sha256=checkpoint,
    )
    result = build_phase0_manifest(
        replicates=[
            ("reexecution_3407", "deterministic_reexecution", first, _sha(first)),
            ("reexecution_3408", "deterministic_reexecution", second, _sha(second)),
        ],
        output=tmp_path / "phase0_manifest.json",
    )
    assert result["payload"]["checkpoint_sha256"] == checkpoint
    assert result["payload"]["claim_scope"] == "deterministic_reproducibility_only"


def test_o1_manifest_binds_all_budgets_to_one_checkpoint(tmp_path):
    checkpoint = "2" * 64
    receipt = _mixed_k_receipt(tmp_path, checkpoint_sha256=checkpoint)
    low = _metrics(
        tmp_path,
        name="low",
        budget=192,
        seed=3407,
        checkpoint_sha256=checkpoint,
    )
    high = _metrics(
        tmp_path,
        name="high",
        budget=384,
        seed=3407,
        checkpoint_sha256=checkpoint,
    )
    result = build_o1_manifest(
        evaluations=[
            ("192", "192", low, _sha(low)),
            ("384", "384", high, _sha(high)),
        ],
        mixed_k_detector_identity_sha256=checkpoint,
        detector_training_exposure="mixed_k_registered_panel",
        training_receipt=receipt,
        training_receipt_sha256=_sha(receipt),
        output=tmp_path / "o1_manifest.json",
    )
    assert [
        row["budget"] for row in result["payload"]["budget_evaluations"]
    ] == [192, 384]

    terminal = json.loads(
        (tmp_path / "low_terminal.json").read_text(encoding="utf-8")
    )
    terminal["checkpoint_sha256"] = "3" * 64
    _json(tmp_path / "low_terminal.json", terminal)
    with pytest.raises(ValueError, match="terminal evaluation binding drifted"):
        build_o1_manifest(
            evaluations=[
                ("192", "192", low, _sha(low)),
                ("384", "384", high, _sha(high)),
            ],
            mixed_k_detector_identity_sha256=checkpoint,
            detector_training_exposure="mixed_k_registered_panel",
            training_receipt=receipt,
            training_receipt_sha256=_sha(receipt),
            output=tmp_path / "tampered_o1.json",
        )


def test_o1_manifest_labels_fixed_k_transfer_as_diagnostic(tmp_path):
    checkpoint = "5" * 64
    low = _metrics(
        tmp_path,
        name="diagnostic_low",
        budget=192,
        seed=3407,
        checkpoint_sha256=checkpoint,
    )
    high = _metrics(
        tmp_path,
        name="diagnostic_high",
        budget=384,
        seed=3407,
        checkpoint_sha256=checkpoint,
    )
    result = build_o1_manifest(
        evaluations=[
            ("192", "192", low, _sha(low)),
            ("384", "384", high, _sha(high)),
        ],
        mixed_k_detector_identity_sha256=checkpoint,
        detector_training_exposure="fixed_k_384_only",
        output=tmp_path / "diagnostic_o1_manifest.json",
    )
    assert result["payload"]["claim_scope"] == (
        "diagnostic_cross_budget_transfer_from_fixed_k_384_only"
    )


def test_o2_manifest_requires_a_rectangular_panel_with_independent(tmp_path):
    checkpoint = "4" * 64
    rows = []
    for family in ("independent", "weak_overlap"):
        for budget in (2, 3):
            metrics = _metrics(
                tmp_path,
                name=f"{family}_{budget}",
                budget=budget,
                seed=3407,
                checkpoint_sha256=checkpoint,
            )
            ledger = _ledger(tmp_path, family=family, budget=budget)
            rows.append(
                (
                    family,
                    str(budget),
                    metrics,
                    _sha(metrics),
                    ledger,
                    _sha(ledger),
                )
            )
    result = build_o2_manifest(
        evaluations=rows,
        output=tmp_path / "o2_manifest.json",
    )
    assert len(result["payload"]["decoder_evaluations"]) == 4

    with pytest.raises(ValueError, match="rectangular panel"):
        build_o2_manifest(
            evaluations=rows[:-1],
            output=tmp_path / "incomplete_o2.json",
        )
