from __future__ import annotations

import hashlib
import json

import pytest

from tools.bata.build_duca_rime_gate_records import (
    o1_records,
    o2_records,
    phase0_records,
)
from tools.bata.duca_rime_phase2 import analyze_o2


def _json(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metrics(path, *, values, seed=1):
    return _json(
        path,
        {
            "schema_version": "duca_rime_localization_metrics_v1",
            "phase": 2,
            "seed": seed,
            "split_role": "utility_risk_fit",
            "split_assignment_sha256": "a" * 64,
            "evaluation_video_ids": sorted(values),
            "video_metrics": {"avg_map": values},
            "uses_official_final": False,
            "official_final_used_for_training_or_selection": False,
            "padded_to_kmax": False,
        },
    )


def test_build_phase0_and_o1_records_from_hash_bound_metrics(tmp_path):
    videos = {f"v{index}": 0.2 + index * 0.1 for index in range(3)}
    first = _metrics(tmp_path / "rep1.json", values=videos, seed=1)
    second = _metrics(
        tmp_path / "rep2.json",
        values={key: value + 0.01 for key, value in videos.items()},
        seed=2,
    )
    phase0_manifest = _json(
        tmp_path / "phase0_manifest.json",
        {
            "schema_version": "duca_rime_phase0_source_manifest_v1",
            "uses_official_final": False,
            "replicates": [
                {"replicate_id": "s1", "path": str(first), "sha256": _sha(first)},
                {"replicate_id": "s2", "path": str(second), "sha256": _sha(second)},
            ],
        },
    )
    phase0 = phase0_records(
        source_manifest=phase0_manifest,
        output=tmp_path / "phase0.jsonl",
        primary_metric="avg_map",
    )
    assert phase0["output"]["record_count"] == 6

    low = _metrics(tmp_path / "k2.json", values=videos)
    high = _metrics(
        tmp_path / "k4.json",
        values={key: value + 0.1 for key, value in videos.items()},
    )
    o1_manifest = _json(
        tmp_path / "o1_manifest.json",
        {
            "schema_version": "duca_rime_o1_source_manifest_v1",
            "uses_official_final": False,
            "position_policy": "exact_uniform",
            "detector_training_exposure": "mixed_k_registered_panel",
            "mixed_k_detector_identity_sha256": "b" * 64,
            "budget_evaluations": [
                {
                    "budget": 2,
                    "measured_heavy_frame_cost": 2,
                    "path": str(low),
                    "sha256": _sha(low),
                },
                {
                    "budget": 4,
                    "measured_heavy_frame_cost": 4,
                    "path": str(high),
                    "sha256": _sha(high),
                },
            ],
        },
    )
    o1 = o1_records(
        source_manifest=o1_manifest,
        output=tmp_path / "o1.jsonl",
        score_metric="avg_map",
    )
    assert o1["budgets"] == [2, 4]
    assert o1["output"]["record_count"] == 6

    diagnostic_manifest = json.loads(o1_manifest.read_text(encoding="utf-8"))
    diagnostic_manifest["detector_training_exposure"] = "fixed_k_384_only"
    diagnostic_manifest = _json(
        tmp_path / "diagnostic_o1_manifest.json",
        diagnostic_manifest,
    )
    with pytest.raises(ValueError, match="mixed-K-trained"):
        o1_records(
            source_manifest=diagnostic_manifest,
            output=tmp_path / "must_not_be_formal_o1.jsonl",
            score_metric="avg_map",
        )


def test_build_o2_records_checks_every_window_exact_k(tmp_path):
    videos = {f"v{index}": 0.3 + index * 0.05 for index in range(3)}
    entries = []
    for family_index, family in enumerate(("independent", "strict_nested")):
        for budget in (2, 4):
            metrics = _metrics(
                tmp_path / f"{family}_{budget}.json",
                values={
                    video: score + family_index * 0.01
                    for video, score in videos.items()
                },
            )
            ledger = tmp_path / f"{family}_{budget}.jsonl"
            rows = []
            for video in videos:
                positions = list(range(budget))
                rows.append(
                    {
                        "schema_version": "duca_rime_inference_ledger_v1",
                        "video_id": video,
                        "window_start_frame": 0,
                        "requested_k": budget,
                        "effective_k": budget,
                        "unique_k": budget,
                        "backbone_input_k": budget,
                        "padded_k": budget,
                        "selected_dense_indices": positions,
                        "max_gap_seconds_cap": 2.0,
                        "observed_max_gap_seconds": 1.0,
                    }
                )
            ledger.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            entries.append(
                {
                    "family": family,
                    "budget": budget,
                    "metrics_path": str(metrics),
                    "metrics_sha256": _sha(metrics),
                    "ledger_path": str(ledger),
                    "ledger_sha256": _sha(ledger),
                }
            )
    manifest = _json(
        tmp_path / "o2_manifest.json",
        {
            "schema_version": "duca_rime_o2_source_manifest_v1",
            "uses_official_final": False,
            "mixed_k_detector_identity_sha256": "b" * 64,
            "decoder_evaluations": entries,
        },
    )
    output = tmp_path / "o2.jsonl"
    built = o2_records(
        source_manifest=manifest,
        output=output,
        score_metric="avg_map",
    )
    assert built["output"]["record_count"] == 12
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    summary = analyze_o2(
        rows,
        selected_family="strict_nested",
        max_regret=0.1,
        bootstrap_samples=100,
        seed=7,
    )
    assert summary["gate_pass"] is True
