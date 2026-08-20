from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_paired_configs_preserve_route_contract():
    dn = (ROOT / "configs/adatad/thumos/georoute_p1_dn_seed3407_v001.py").read_text()
    q = (ROOT / "configs/adatad/thumos/georoute_p1_q_seed3407_v001.py").read_text()
    assert 'arm_surface="DN"' in dn and 'route_mode="dense"' in dn
    assert 'arm_surface="Q"' in q and 'route_mode="dynamic_scnr"' in q
    assert "seed=3407" in dn and "seed=3407" in q
    assert "official_test_open_allowed=False" in dn + q


def test_true_ragged_and_no_leakage_contract_is_present():
    source = (ROOT / "opentad/models/backbones/georoute_wrapper.py").read_text()
    adapter = (ROOT / "opentad/models/backbones/vit_adapter.py").read_text()
    assert "true_clip_ragged_no_padding" in source + adapter
    assert "masked-zero carrier" in source
    assert "forward_native_ragged" in source and "forward_native_ragged" in adapter
    for config in ("georoute_p1_dn_seed3407_v001.py", "georoute_p1_q_seed3407_v001.py"):
        text = (ROOT / "configs/adatad/thumos" / config).read_text()
        assert "gt_for_route_allowed=False" in text
        assert "raw_prediction_cache_allowed=False" in text


def test_checkpoint_policy_keeps_recovery_state_and_latest_three():
    checkpoint = (ROOT / "opentad/utils/checkpoint.py").read_text()
    assert "recovery" in checkpoint.lower()
    assert "latest" in checkpoint.lower()
