"""Static factorization checks for the CT-DP geometry campaign."""

from pathlib import Path

from mmengine.config import Config


CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "adatad" / "thumos"


def _load(arm: str) -> Config:
    return Config.fromfile(CONFIG_ROOT / f"duca_ctdp_geometry_{arm}.py")


def _flags(cfg: Config) -> tuple[bool, bool, bool, bool]:
    model = cfg.model
    selector = model["frame_selector"]
    backbone = model["backbone"]["backbone"]
    amod = backbone["amod_config"]
    physical = model["rpn_head"]["physical_grid_actionformer"]
    return (
        bool(selector.get("force_uniform", False)),
        bool(backbone["ct_tubelet"]),
        bool(amod["enabled"]),
        bool(physical.get("enabled", False)),
    )


def test_geometry_arms_are_orthogonal_and_match_contract():
    assert _flags(_load("g0")) == (False, True, False, False)
    assert _flags(_load("g1")) == (False, True, True, False)
    assert _flags(_load("g2")) == (False, True, False, True)
    assert _flags(_load("g3")) == (False, True, True, True)
