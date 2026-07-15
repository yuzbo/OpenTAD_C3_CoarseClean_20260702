from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from mmengine.config import Config

from tools.bata.validate_duca_transition_only_p0_suite import validate_suite


VARIANT_ORDER = ("uniform", "direct", "transition_beta0", "transition_counterfactual")
PILOT_CONFIGS = {
    "uniform": "configs/adatad/thumos/duca_exact_uniform_fixed384_p0_ddp_pilot.py",
    "direct": "configs/adatad/thumos/duca_direct_boundary_fixed384_p0_ddp_pilot.py",
    "transition_beta0": "configs/adatad/thumos/duca_transition_only_fixed384_beta0_p0_ddp_pilot.py",
    "transition_counterfactual": (
        "configs/adatad/thumos/duca_transition_only_fixed384_counterfactual_p0_ddp_pilot.py"
    ),
}
EXPECTED_STEPS = 10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=root, text=True, encoding="utf-8"
    ).strip()


def _budget_vectors(payload: dict[str, Any]) -> list[list[int]]:
    vectors = []
    for step in payload.get("selector_steps", []):
        values = step.get("effective_budget") if isinstance(step, dict) else None
        _require(isinstance(values, list) and values, "pilot step is missing effective_budget")
        vector = [int(value) for value in values]
        _require(all(0 < value <= 384 for value in vector), "pilot effective budget is invalid")
        vectors.append(vector)
    return vectors


def validate_probe(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    _require(variant in VARIANT_ORDER, f"unknown pilot variant: {variant}")
    _require(payload.get("schema_version") == "duca_training_probe_v1", f"{variant}: bad schema")
    for key in (
        "attempted_steps", "successful_optimizer_steps", "finite_loss_steps",
        "finite_gradient_steps",
    ):
        _require(int(payload.get(key, -1)) == EXPECTED_STEPS, f"{variant}: {key} != {EXPECTED_STEPS}")
    _require(int(payload.get("skipped_optimizer_steps", -1)) == 0, f"{variant}: optimizer step skipped")
    _require(payload.get("static_graph") is False, f"{variant}: static_graph must be false")
    _require(payload.get("find_unused_parameters") is True, f"{variant}: unused discovery must be enabled")
    _require(int(payload.get("world_size", 0)) == 1, f"{variant}: pilot must match one-GPU jobs")

    coverage = payload.get("parameter_group_coverage")
    _require(isinstance(coverage, dict), f"{variant}: parameter coverage is missing")
    for group in ("backbone", "coarse_probe", "selector", "detector_head"):
        counts = coverage.get(group)
        _require(isinstance(counts, dict), f"{variant}: missing {group} parameter group")
        _require(int(counts.get("trainable", 0)) > 0, f"{variant}: {group} has no trainable parameter")
        _require(int(counts.get("gradient_seen", 0)) > 0, f"{variant}: {group} never received gradient")

    steps = payload.get("selector_steps")
    _require(isinstance(steps, list) and len(steps) == EXPECTED_STEPS, f"{variant}: selector trace is incomplete")
    vectors = _budget_vectors(payload)
    has_full = any(all(value == 384 for value in vector) for vector in vectors)
    has_mixed = any(min(vector) < 384 and max(vector) == 384 for vector in vectors)
    has_all_short = any(max(vector) < 384 for vector in vectors)
    _require(has_full, f"{variant}: pilot did not cover a full-length batch")
    _require(has_mixed, f"{variant}: pilot did not cover a mixed-length batch")
    _require(has_all_short, f"{variant}: pilot did not cover an all-short batch")

    if variant == "direct":
        weights = [float(step.get("detector_gradient_weight", 0.0)) for step in steps]
        _require(min(weights) == 0.0 and max(weights) > 0.0, "direct: detector-gradient schedule did not cross phases")
    elif variant in ("transition_beta0", "transition_counterfactual"):
        alphas = [float(step.get("policy_mix_alpha", 0.0)) for step in steps]
        _require(min(alphas) == 0.0 and max(alphas) > 0.0, f"{variant}: policy schedule did not cross phases")

    candidate_counts = []
    if variant == "transition_counterfactual":
        for step in steps:
            counterfactual = step.get("counterfactual")
            if isinstance(counterfactual, dict):
                _require(counterfactual.get("finite") is True, "counterfactual utility became non-finite")
                candidate_counts.append(int(counterfactual.get("candidate_count", -1)))
        _require(len(candidate_counts) == EXPECTED_STEPS, "counterfactual trace is incomplete")
        _require(any(value == 0 for value in candidate_counts), "counterfactual pilot missed zero-candidate path")
        _require(any(value > 0 for value in candidate_counts), "counterfactual pilot missed valid-candidate path")

    max_memory = float(payload.get("max_cuda_memory_mb", math.nan))
    _require(math.isfinite(max_memory) and max_memory > 0.0, f"{variant}: CUDA memory evidence is invalid")
    return {
        "variant": variant,
        "attempted_steps": EXPECTED_STEPS,
        "successful_optimizer_steps": EXPECTED_STEPS,
        "parameter_group_coverage": coverage,
        "gradient_never_seen": payload.get("gradient_never_seen", []),
        "budget_coverage": {
            "full_batch": has_full,
            "mixed_batch": has_mixed,
            "all_short_batch": has_all_short,
        },
        "counterfactual_candidate_counts": candidate_counts,
        "max_cuda_memory_mb": max_memory,
    }


def validate_pilot_suite(
    *,
    repo_root: str | Path,
    probe_dir: str | Path,
    core_gate_json: str | Path,
    expected_commit: str | None = None,
    require_clean: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    probes = Path(probe_dir).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None:
        _require(commit == expected_commit, "pilot HEAD differs from expected commit")
    if require_clean:
        _require(not _git(root, "status", "--porcelain", "--untracked-files=normal"), "pilot tree is dirty")

    formal_suite = validate_suite(
        repo_root=root,
        seed=0,
        expected_commit=commit,
        require_clean=require_clean,
        core_gate_json=core_gate_json,
    )
    variants = []
    for variant in VARIANT_ORDER:
        pilot_config_path = root / PILOT_CONFIGS[variant]
        config = Config.fromfile(str(pilot_config_path))
        _require(config.model.backbone.backbone.with_cp is False, f"{variant}: pilot enabled checkpointing")
        _require(config.solver.static_graph is False, f"{variant}: pilot enabled static DDP")
        _require(config.solver.find_unused_parameters is True, f"{variant}: pilot disabled unused discovery")
        _require(int(config.workflow.max_train_iters) == EXPECTED_STEPS, f"{variant}: pilot step count drift")
        _require(config.workflow.disable_checkpoint is True, f"{variant}: pilot must not write model checkpoints")
        probe_path = probes / f"{variant}.training_probe.json"
        _require(probe_path.is_file(), f"{variant}: training probe is missing: {probe_path}")
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
        summary = validate_probe(payload, variant)
        summary.update(
            {
                "pilot_config": PILOT_CONFIGS[variant],
                "pilot_config_sha256": _sha256(pilot_config_path),
                "probe_json": str(probe_path),
                "probe_json_sha256": _sha256(probe_path),
            }
        )
        variants.append(summary)

    gate_path = Path(core_gate_json).resolve()
    return {
        "schema_version": "duca_p0_ddp_pilot_suite_v1",
        "ok": True,
        "git_commit": commit,
        "shared_protocol_sha256": formal_suite["shared_protocol_sha256"],
        "core_gate_json": str(gate_path),
        "core_gate_json_sha256": _sha256(gate_path),
        "expected_steps_per_variant": EXPECTED_STEPS,
        "variants": variants,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the four-arm DUCA P0 DDP pilot")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--core-gate-json", required=True)
    parser.add_argument("--expected-commit", default=None)
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--output-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = validate_pilot_suite(
        repo_root=args.repo_root,
        probe_dir=args.probe_dir,
        core_gate_json=args.core_gate_json,
        expected_commit=args.expected_commit,
        require_clean=args.require_clean,
    )
    output = Path(args.output_json).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
