from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_cuda_gates_require_config_declared_normalized_lf_hashes() -> None:
    for name in (
        "run_duca_frontend_p0_real_gate.py",
        "run_duca_protected_e2e_exact_full_model_gate.py",
    ):
        source = (ROOT / "tools" / "bata" / name).read_text(encoding="utf-8")
        assert "def _normalized_lf_sha256" in source
        assert "official_asformer_source_normalized_lf_sha256" in source
        assert "config lacks an official ASFormer normalized-LF SHA256 declaration" in source
        assert "config_declared_normalized_lf_sha256" in source
        assert "observed == expected" in source
        assert "official_asformer_binding = _official_asformer_binding(cfg, selector)" in source


def test_full_model_gate_preserves_and_traces_real_boundary_validity() -> None:
    source = (
        ROOT / "tools" / "bata" / "run_duca_protected_e2e_exact_full_model_gate.py"
    ).read_text(encoding="utf-8")
    assert '"gt_boundary_validity": batch["gt_boundary_validity"]' in source
    assert '"gt_boundary_validity" in batch' in source
    assert "_record_boundary_validity_consumption" in source
    assert "gt_boundary_validity=batch[\"gt_boundary_validity\"]" in source
    assert "production train_one_epoch did not consume gt_boundary_validity" in source
