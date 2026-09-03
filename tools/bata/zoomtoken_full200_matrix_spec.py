from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Full200MatrixSpec:
    key: str
    protocol_id: str
    arms: tuple[str, str, str]
    candidate_arm: str
    reference_arm: str = "D160"
    low_cost_control_arm: str = "G96"


SPECS = {
    "s2_v3": Full200MatrixSpec(
        key="s2_v3",
        protocol_id="ZOOMTOKEN-CONTINUOUS-ROI-S2-V3-FULL200-COMPUTE-PARETO-3X3-v001",
        arms=("D160", "G96", "U128-A0"),
        candidate_arm="U128-A0",
    ),
    "d2s": Full200MatrixSpec(
        key="d2s",
        protocol_id="ZOOMTOKEN-D2S-TAD-FULL200-COMPUTE-PARETO-3X3-v001",
        arms=("D160", "G96", "D2S-U128-B128"),
        candidate_arm="D2S-U128-B128",
    ),
    "patad": Full200MatrixSpec(
        key="patad",
        protocol_id="ZOOMTOKEN-PATAD-FULL200-COMPUTE-PARETO-3X3-v001",
        arms=("D160", "G96", "PATAD-U128-B128"),
        candidate_arm="PATAD-U128-B128",
    ),
}


def get_matrix_spec(key: str | None = None) -> Full200MatrixSpec:
    resolved = str(key or os.environ.get("ZOOMTOKEN_MATRIX_KIND", "s2_v3"))
    if resolved not in SPECS:
        raise ValueError(
            f"unknown full-200 matrix kind {resolved!r}; expected one of {sorted(SPECS)}"
        )
    return SPECS[resolved]


def binding_from_config(cfg: Any, spec: Full200MatrixSpec) -> Any:
    names = {
        "s2_v3": "continuous_roi_s2_v3_full200_compute",
        "d2s": "continuous_roi_d2s_v3_full200_compute",
        "patad": "continuous_roi_patad_v3_full200_compute",
    }
    name = names[spec.key]
    if not hasattr(cfg, name):
        raise ValueError(f"config is missing matrix binding {name}")
    return getattr(cfg, name)


def validate_matrix_cell(
    path: str | Path, *, arm: str, seed: int, spec: Full200MatrixSpec
) -> dict[str, Any]:
    if spec.key == "s2_v3":
        from tools.bata.continuous_roi_s2_v3_full200_compute import validate_cell_config

        return validate_cell_config(path, arm=arm, seed=seed)
    if spec.key == "d2s":
        from tools.bata.d2s_tad_full200_compute import validate_d2s_cell_config

        return validate_d2s_cell_config(path, arm=arm, seed=seed)
    from tools.bata.patad_full200_compute import validate_patad_cell_config

    return validate_patad_cell_config(path, arm=arm, seed=seed)


__all__ = [
    "Full200MatrixSpec",
    "SPECS",
    "binding_from_config",
    "get_matrix_spec",
    "validate_matrix_cell",
]

