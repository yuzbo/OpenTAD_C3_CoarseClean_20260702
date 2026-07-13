from pathlib import Path

from tools.bata.audit_phystime_g0_native_geometry import run_audit


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "selected_axis": ROOT / "configs/adatad/thumos/phystime_g1a_selected_axis_native_j192.py",
    "physical_metric": ROOT / "configs/adatad/thumos/phystime_g1a_physical_metric_native_j192.py",
}


def test_g0_static_audit_separates_k_j_q_and_preserves_atomic_gaps(tmp_path):
    output = tmp_path / "g0.json"
    report = run_audit(CONFIGS, build_models=False, output=output)

    assert report["gate_pass"] is False
    assert report["static_precheck_pass"] is True
    assert report["status"] == "static_precheck_only_not_real_gate"
    assert report["K_raw_observations"] == 384
    assert report["J_native_tubelet_tokens"] == 192
    assert report["Q0_base_candidates"] == 192
    assert report["Q_level_lengths"] == [192, 96, 48, 24, 12, 6]
    assert report["Q_total_candidates"] == 378
    assert report["feature_interpolation"] is False
    assert report["synthetic_selected_index_checksum_match"] is True
    assert report["synthetic_native_position_checksum_match"] is True
    assert report["real_pipeline_verified"] is False
    assert report["lineage_evidence_level"] == "structural_graph_upper_bound_not_jacobian"
    assert report["synthetic_geometry"]["physical_metric"]["disconnected_token_count"] > 0
    assert report["synthetic_geometry"]["physical_metric"]["envelope_inflation_sec_sum"] > 0
    assert output.is_file()
