from __future__ import annotations

from typing import Any


def resolve_effective_seed(cfg: Any, cli_seed: int | None) -> int:
    cfg_seed = cfg.get("seed", None)
    if cli_seed is None:
        return int(cfg_seed) if cfg_seed is not None else 42
    if cfg_seed is not None and int(cli_seed) != int(cfg_seed):
        raise ValueError(
            f"CLI --seed={cli_seed} conflicts with config seed={int(cfg_seed)}"
        )
    return int(cli_seed)
