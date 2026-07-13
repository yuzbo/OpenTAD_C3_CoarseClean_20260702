from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from mmengine.config import Config

from tools.bata.validate_duca_transition_only_p0_variant import CONFIGS, validate_variant


REFERENCE_VARIANT = "transition_beta025"
VARIANT_ORDER = ("uniform", "direct", "transition_beta0", "transition_beta025")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def _shared_protocol(cfg: Config) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    return {
        "detector_type": cfg.model.type,
        "detector_head": _plain(cfg.model.rpn_head),
        "dataset": _plain(cfg.dataset),
        "optimizer": _plain(cfg.optimizer),
        "scheduler": _plain(cfg.scheduler),
        "workflow": _plain(cfg.workflow),
        "dense_window_size": int(cfg.dense_window_size),
        "budget": int(cfg.window_size),
        "backbone_frames": int(cfg.model.backbone.backbone.total_frames),
        "projection_length": int(cfg.model.projection.max_seq_len),
        "max_unselected_hole": int(selector.max_unselected_hole),
        "coordinate_space": str(selector.coordinate_space),
        "detector_output_coordinate_space": str(selector.detector_output_coordinate_space),
        "selected_axis_remap_required": bool(selector.selected_axis_remap_required),
        "component_learning_rates": {
            "coarse_trunk": float(selector.get("coarse_trunk_lr", 2.5e-5)),
            "action_head": float(selector.get("action_head_lr", 5.0e-5)),
            "transition_scorer": float(selector.get("transition_scorer_lr", 1.0e-4)),
        },
        "expected_optimizer_steps": 13200,
    }


def validate_suite(
    *,
    repo_root: str | Path = ".",
    seed: int = 0,
    expected_commit: str | None = None,
    require_clean: bool = False,
    core_gate_json: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    _require(seed >= 0, "seed must be non-negative")
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None:
        _require(commit == expected_commit, "current HEAD does not match expected commit")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if require_clean:
        _require(not dirty, "formal P0 suite preparation requires a clean git tree")

    core_gate: dict[str, Any] | None = None
    if core_gate_json is not None:
        gate_path = Path(core_gate_json).resolve()
        _require(gate_path.is_file(), f"formal core gate is missing: {gate_path}")
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        _require(gate_payload.get("ok") is True, "formal core gate must declare ok=true")
        _require(gate_payload.get("formal_proof_ok") is True, "formal core gate proof did not pass")
        _require(gate_payload.get("git_commit") == commit, "formal core gate commit is stale")
        _require(
            gate_payload.get("uniform_reference_definition") == "round_linspace_endpoints",
            "formal core gate did not use endpoint linspace uniform reference",
        )
        _require(gate_payload.get("uniform_reference_exact") is True, "formal core gate did not verify exact uniform")
        core_gate = {
            "path": str(gate_path),
            "sha256": _sha256(gate_path),
            "git_commit": gate_payload["git_commit"],
        }

    configs = {name: Config.fromfile(str(root / CONFIGS[name])) for name in VARIANT_ORDER}
    reference = configs[REFERENCE_VARIANT]
    reference_protocol = _shared_protocol(reference)
    variants: list[dict[str, Any]] = []
    for name in VARIANT_ORDER:
        path = root / CONFIGS[name]
        summary = validate_variant(name, str(path))
        protocol = _shared_protocol(configs[name])
        _require(protocol == reference_protocol, f"{name}: shared protocol differs from reference")
        variants.append(
            {
                "name": name,
                "config": CONFIGS[name].replace("\\", "/"),
                "config_sha256": _sha256(path),
                "variant_validation": summary,
            }
        )

    protocol_payload = json.dumps(reference_protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_payload.encode("utf-8")).hexdigest()
    return {
        "ok": True,
        "status": "deployable_not_submitted",
        "git_commit": commit,
        "git_tree_clean": not bool(dirty),
        "seed": int(seed),
        "variant_order": list(VARIANT_ORDER),
        "shared_protocol": reference_protocol,
        "shared_protocol_sha256": protocol_sha256,
        "variants": variants,
        "formal_core_gate": core_gate,
        "submission_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--core-gate-json")
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        payload = validate_suite(
            repo_root=args.repo_root,
            seed=args.seed,
            expected_commit=args.expected_commit,
            require_clean=args.require_clean,
            core_gate_json=args.core_gate_json,
        )
        code = 0
    except Exception as exc:
        payload = {"ok": False, "error_type": exc.__class__.__name__, "error": str(exc)}
        code = 1
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
