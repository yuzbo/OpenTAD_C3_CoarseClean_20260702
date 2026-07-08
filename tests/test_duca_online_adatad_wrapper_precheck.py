from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRECHECK = ROOT / "tools" / "bata" / "run_duca_online_adatad_wrapper_precheck.py"


def test_duca_online_adatad_wrapper_precheck_runs_real_sparse_path(tmp_path: Path) -> None:
    output_json = tmp_path / "duca_online_adatad_precheck.json"

    completed = subprocess.run(
        [sys.executable, str(PRECHECK), "--output-json", str(output_json)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    stdout_payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert stdout_payload == payload
    assert payload["status"] == "ok"
    assert payload["implementation"] == "opentad.models.duca.acquisition"
    assert payload["wrapper_class"] == "DucaOnlineSparseDetectorWrapper"
    assert payload["config_path"] == "configs/adatad/thumos/duca_online_adatad_smoke.py"
    assert payload["detector_family"] == "AdaTAD"
    assert payload["uses_ledger_for_decision"] is False
    assert payload["teacher_free_inference"] is True
    assert payload["nested_forbidden_rejected"] is True
    assert payload["budget"] == 384
    assert payload["budget_unit"] == "detector_consumed_temporal_observation"
    assert payload["coordinate"] == "original_time"
    assert payload["selected_count"] == [384, 384]
    assert payload["detector_input_length"] == 384
    assert payload["detector_consumes_selected_positions"] is True
    assert payload["train_detector_batch_sanitized"] is True
    assert payload["train_detector_loss_present"] is True
    assert payload["pre_backbone_model"] == "ZeroShotActionnessSource(mode=motion)+DUCASelectorMLP"
    assert payload["estimated_flops"] > 0
    assert payload["estimated_macs"] > 0
    assert payload["compute_profile"]["parameters"]["trainable"] > 0
    assert payload["compute_profile"]["latency_ms"]["adapter_total_ms"] >= 0.0
    assert payload["precheck_pass"] is True
