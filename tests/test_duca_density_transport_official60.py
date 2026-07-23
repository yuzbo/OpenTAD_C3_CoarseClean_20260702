from __future__ import annotations

from pathlib import Path

from tools.bata.duca_selected_axis_training import VARIANT_CONFIGS
from tools.bata.export_duca_selection_quality import _selector_sampling_contract
from tools.bata.validate_duca_density_transport_official60 import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs" / "adatad" / "thumos"

VARIANTS = {
    "density_transport_nomax": "duca_density_transport_nomax_fixed384_official60.py",
    "density_transport_softmax14": "duca_density_transport_softmax14_fixed384_official60.py",
    "density_transport_hardmax14": "duca_density_transport_hardmax14_fixed384_official60.py",
    "mixture_density_transport_nomax": "duca_mixture_density_transport_nomax_fixed384_official60.py",
}


def test_density_transport_official60_configs_are_matched_and_registered(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DUCA_CELLCF_TRAINING_PROFILE", "official60")
    payloads = {
        variant: validate_config(CONFIG_ROOT / config)
        for variant, config in VARIANTS.items()
    }
    assert all(payload["ok"] for payload in payloads.values())
    assert all(VARIANT_CONFIGS[variant] == config for variant, config in VARIANTS.items())
    assert payloads["density_transport_nomax"]["hard_max_unselected_hole"] is None
    assert payloads["density_transport_nomax"]["soft_max_gap_enabled"] is False
    assert payloads["density_transport_softmax14"]["hard_max_unselected_hole"] is None
    assert payloads["density_transport_softmax14"]["soft_max_unselected_hole_target"] == 14
    assert payloads["density_transport_hardmax14"]["hard_max_unselected_hole"] == 14
    assert payloads["mixture_density_transport_nomax"]["density_model"] == (
        "boundary_uncertainty_context_mixture"
    )


def test_independent_runner_exposes_all_density_variants_without_a_new_launcher() -> None:
    runner = (ROOT / "scripts" / "run_duca_independent_official60_gpu1.sh").read_text(
        encoding="utf-8"
    )
    submitter = (
        ROOT / "scripts" / "submit_duca_independent_official60_suite.sh"
    ).read_text(encoding="utf-8")
    for variant, config in VARIANTS.items():
        assert f"{variant})" in runner
        assert config in runner
    assert "validate_duca_density_transport_official60.py" in runner
    assert "export_duca_selection_quality" in runner
    assert "analyze_duca_selection_quality" in runner
    assert "DUCA_INDEPENDENT_VARIANTS" in submitter


def test_selection_export_accepts_no_max_and_hard_max_density_contracts() -> None:
    class Selector:
        budget = 384
        max_unselected_hole = None

    assert _selector_sampling_contract(
        Selector(), {"budget": 384, "max_unselected_hole": None}
    ) == (384, None)
    Selector.max_unselected_hole = 14
    assert _selector_sampling_contract(
        Selector(), {"budget": 384, "max_unselected_hole": 14}
    ) == (384, 14)
