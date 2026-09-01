from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must contain a JSON object")
    return resolved, payload


def _resolved_pilot_config_sha256(context: dict[str, Any]) -> str:
    probe_path = str(context["training_probe_json"])
    context_path = str(context["context_json"])
    previous_probe = os.environ.get("DUCA_TRAINING_PROBE_JSON")
    previous_context = os.environ.get("DUCA_TRAINING_PROBE_CONTEXT_JSON")
    os.environ["DUCA_TRAINING_PROBE_JSON"] = probe_path
    os.environ["DUCA_TRAINING_PROBE_CONTEXT_JSON"] = context_path
    try:
        config = Config.fromfile(str(context["source_config_path"]))
        config.merge_from_dict(
            {
                "work_dir": str(context["work_dir"]),
                "model.backbone.custom.pretrain": str(context["checkpoint_path"]),
            }
        )
        return _canonical_sha256(config.to_dict())
    finally:
        if previous_probe is None:
            os.environ.pop("DUCA_TRAINING_PROBE_JSON", None)
        else:
            os.environ["DUCA_TRAINING_PROBE_JSON"] = previous_probe
        if previous_context is None:
            os.environ.pop("DUCA_TRAINING_PROBE_CONTEXT_JSON", None)
        else:
            os.environ["DUCA_TRAINING_PROBE_CONTEXT_JSON"] = previous_context


def _validate_run_manifest_files(manifest: dict[str, Any]) -> None:
    for path_key, hash_key, label in (
        ("checkpoint_path", "checkpoint_sha256", "AdaTAD checkpoint"),
        ("official_asformer_source", "official_asformer_source_sha256", "official ASFormer source"),
        ("canonical_env_path", "canonical_env_sha256", "canonical environment"),
        ("reference_config_path", "reference_config_sha256", "formal reference config"),
    ):
        path = Path(str(manifest.get(path_key, ""))).resolve()
        _require(path.is_file(), f"{label} is missing: {path}")
        _require(manifest.get(hash_key) == _sha256(path), f"{label} hash mismatch")


def _budget_vectors(payload: dict[str, Any]) -> list[list[int]]:
    vectors = []
    for step in payload.get("selector_steps", []):
        if not bool(step.get("optimizer_step_ran", False)):
            continue
        values = step.get("effective_budget") if isinstance(step, dict) else None
        _require(isinstance(values, list) and values, "pilot step is missing effective_budget")
        vector = [int(value) for value in values]
        _require(all(0 < value <= 384 for value in vector), "pilot effective budget is invalid")
        vectors.append(vector)
    return vectors


def validate_probe(payload: dict[str, Any], variant: str) -> dict[str, Any]:
    _require(variant in VARIANT_ORDER, f"unknown pilot variant: {variant}")
    _require(payload.get("schema_version") == "duca_training_probe_v1", f"{variant}: bad schema")
    attempted = int(payload.get("attempted_steps", -1))
    successful = int(payload.get("successful_optimizer_steps", -1))
    skipped = int(payload.get("skipped_optimizer_steps", -1))
    _require(successful == EXPECTED_STEPS, f"{variant}: successful updates != {EXPECTED_STEPS}")
    _require(attempted >= successful, f"{variant}: attempted steps are inconsistent")
    _require(skipped == attempted - successful, f"{variant}: AMP skip accounting is inconsistent")
    _require(int(payload.get("finite_loss_steps", -1)) == attempted, f"{variant}: non-finite loss")
    _require(int(payload.get("finite_gradient_steps", -1)) >= successful, f"{variant}: too few finite-gradient attempts")
    _require(payload.get("static_graph") is False, f"{variant}: static_graph must be false")
    _require(payload.get("find_unused_parameters") is True, f"{variant}: unused discovery must be enabled")
    _require(int(payload.get("world_size", 0)) == 1, f"{variant}: pilot must match one-GPU jobs")

    coverage = payload.get("parameter_group_coverage")
    _require(isinstance(coverage, dict), f"{variant}: parameter coverage is missing")
    for group in ("backbone", "coarse_probe", "selector", "projection", "detector_head"):
        counts = coverage.get(group)
        _require(isinstance(counts, dict), f"{variant}: missing {group} parameter group")
        trainable = int(counts.get("trainable", 0))
        gradient_seen = int(counts.get("gradient_seen", 0))
        _require(trainable > 0, f"{variant}: {group} has no trainable parameter")
        _require(gradient_seen == trainable, f"{variant}: {group} has trainable parameters never receiving gradient")
    never_seen = payload.get("gradient_never_seen")
    _require(isinstance(never_seen, list), f"{variant}: gradient_never_seen is missing")
    _require(not never_seen, f"{variant}: trainable parameters never received gradient")
    _require(bool(payload.get("gradient_seen")), f"{variant}: no parameter received gradient")

    all_steps = payload.get("selector_steps")
    _require(isinstance(all_steps, list) and len(all_steps) == attempted, f"{variant}: selector trace is incomplete")
    steps = [step for step in all_steps if bool(step.get("optimizer_step_ran", False))]
    _require(len(steps) == EXPECTED_STEPS, f"{variant}: successful selector trace is incomplete")
    audit = payload.get("update_audit")
    _require(isinstance(audit, dict), f"{variant}: successful-update audit is missing")
    _require(int(audit.get("attempted_batches", -1)) == EXPECTED_STEPS, f"{variant}: batch count mismatch")
    _require(int(audit.get("successful_optimizer_updates", -1)) == EXPECTED_STEPS, f"{variant}: audit update count mismatch")
    _require(int(audit.get("optimizer_attempts", -1)) == attempted, f"{variant}: audit attempt count mismatch")
    _require(int(audit.get("amp_skipped_attempts", -1)) == skipped, f"{variant}: audit skip count mismatch")
    _require(int(audit.get("replay_exhaustions", -1)) == 0, f"{variant}: AMP replay exhausted")
    _require(int(audit.get("scheduler_updates", -1)) == EXPECTED_STEPS, f"{variant}: scheduler exposure mismatch")
    _require(int(audit.get("ema_updates", -1)) == EXPECTED_STEPS, f"{variant}: EMA exposure mismatch")
    _require(int(audit.get("duca_schedule_updates", -1)) == EXPECTED_STEPS, f"{variant}: selector schedule exposure mismatch")
    _require(int(audit.get("forced_amp_overflow_attempts", -1)) == 1, f"{variant}: forced AMP overflow was not exercised exactly once")
    _require(skipped >= 1, f"{variant}: forced AMP overflow did not skip an optimizer attempt")
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
        "attempted_steps": attempted,
        "successful_optimizer_steps": EXPECTED_STEPS,
        "amp_skipped_attempts": skipped,
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


def _validate_probe_bindings(
    payload: dict[str, Any],
    *,
    variant: str,
    probe_path: Path,
    context_path: Path,
    context: dict[str, Any],
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    _require(context.get("schema_version") == "duca_p0_ddp_pilot_context_v1", f"{variant}: bad context schema")
    expected_context = {
        "git_commit": manifest["git_commit"],
        "variant": variant,
        "seed": int(manifest["seed"]),
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "pilot_nonce": manifest["pilot_nonce"],
        "training_probe_json": str(probe_path),
        "context_json": str(context_path),
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "core_gate_json_sha256": manifest["core_gate_json_sha256"],
        "shared_protocol_sha256": manifest["shared_protocol_sha256"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
    }
    for key, expected in expected_context.items():
        _require(context.get(key) == expected, f"{variant}: context {key} mismatch")
    source_config = Path(str(context.get("source_config_path", ""))).resolve()
    _require(source_config.is_file(), f"{variant}: source config is missing")
    _require(context.get("source_config_sha256") == _sha256(source_config), f"{variant}: source config hash mismatch")

    bindings = payload.get("bindings")
    _require(isinstance(bindings, dict), f"{variant}: probe bindings are missing")
    expected_bindings = {
        "git_commit": manifest["git_commit"],
        "seed": int(manifest["seed"]),
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "source_config_path": str(source_config),
        "source_config_sha256": context["source_config_sha256"],
        "training_probe_json": str(probe_path),
        "context_json": str(context_path),
        "context_json_sha256": _sha256(context_path),
        "context": context,
    }
    for key, expected in expected_bindings.items():
        _require(bindings.get(key) == expected, f"{variant}: probe binding {key} mismatch")
    _require(
        bindings.get("resolved_config_sha256") == _resolved_pilot_config_sha256(context),
        f"{variant}: resolved pilot config hash mismatch",
    )


def validate_pilot_artifact(
    path: str | Path,
    *,
    repo_root: str | Path,
    expected_commit: str,
    expected_protocol_sha256: str,
    expected_core_gate_sha256: str,
    expected_checkpoint_sha256: str,
    expected_official_asformer_source_sha256: str,
    expected_reference_config_sha256: str,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    artifact_path, artifact = _load_json(path, "DDP pilot artifact")
    _require(artifact.get("schema_version") == "duca_p0_ddp_pilot_suite_v1", "DDP pilot schema mismatch")
    _require(artifact.get("ok") is True, "DDP pilot must declare ok=true")
    _require(artifact.get("git_commit") == expected_commit, "DDP pilot commit is stale")
    _require(
        artifact.get("shared_protocol_sha256") == expected_protocol_sha256,
        "DDP pilot shared protocol is stale",
    )
    _require(
        artifact.get("core_gate_json_sha256") == expected_core_gate_sha256,
        "DDP pilot core-gate binding is stale",
    )
    _require(artifact.get("checkpoint_sha256") == expected_checkpoint_sha256, "DDP pilot checkpoint binding is stale")
    _require(
        artifact.get("official_asformer_source_sha256") == expected_official_asformer_source_sha256,
        "DDP pilot ASFormer source binding is stale",
    )
    _require(
        artifact.get("reference_config_sha256") == expected_reference_config_sha256,
        "DDP pilot reference config binding is stale",
    )
    manifest_path, manifest = _load_json(artifact.get("run_manifest_path", ""), "DDP pilot run manifest")
    _require(artifact.get("run_manifest_sha256") == _sha256(manifest_path), "DDP pilot run-manifest hash mismatch")
    _require(manifest.get("schema_version") == "duca_p0_ddp_pilot_run_v1", "DDP pilot run-manifest schema mismatch")
    _require(manifest.get("git_commit") == expected_commit, "DDP pilot run-manifest commit mismatch")
    _require(manifest.get("shared_protocol_sha256") == expected_protocol_sha256, "DDP pilot run-manifest protocol mismatch")
    _require(manifest.get("core_gate_json_sha256") == expected_core_gate_sha256, "DDP pilot run-manifest core gate mismatch")
    _require(manifest.get("checkpoint_sha256") == expected_checkpoint_sha256, "DDP pilot checkpoint differs from core gate")
    _require(
        manifest.get("official_asformer_source_sha256") == expected_official_asformer_source_sha256,
        "DDP pilot ASFormer source differs from core gate",
    )
    _require(
        manifest.get("reference_config_sha256") == expected_reference_config_sha256,
        "DDP pilot reference config differs from core gate",
    )
    _require(str(manifest.get("slurm_job_id", "")).isdigit(), "DDP pilot lacks a Slurm job identity")
    _require(bool(manifest.get("pilot_nonce")), "DDP pilot nonce is missing")
    _validate_run_manifest_files(manifest)
    _require(int(artifact.get("seed", -1)) == int(manifest.get("seed", -2)), "DDP pilot seed mismatch")
    _require(artifact.get("slurm_job_id") == manifest.get("slurm_job_id"), "DDP pilot Slurm identity mismatch")
    _require(artifact.get("pilot_nonce") == manifest.get("pilot_nonce"), "DDP pilot nonce mismatch")

    variants = artifact.get("variants")
    _require(isinstance(variants, list), "DDP pilot variant evidence is missing")
    _require([item.get("variant") for item in variants] == list(VARIANT_ORDER), "DDP pilot arm order mismatch")
    validated = []
    run_root = manifest_path.parent
    for item, variant in zip(variants, VARIANT_ORDER):
        config_path = (root / PILOT_CONFIGS[variant]).resolve()
        _require(item.get("pilot_config") == PILOT_CONFIGS[variant], f"{variant}: pilot config path mismatch")
        _require(item.get("pilot_config_sha256") == _sha256(config_path), f"{variant}: pilot config hash mismatch")
        probe_path, probe = _load_json(item.get("probe_json", ""), f"{variant} training probe")
        context_path, context = _load_json(item.get("context_json", ""), f"{variant} pilot context")
        _require(
            Path(str(context.get("source_config_path", ""))).resolve() == config_path,
            f"{variant}: raw probe did not use the canonical pilot config",
        )
        _require(
            context.get("source_config_sha256") == item.get("pilot_config_sha256"),
            f"{variant}: raw probe pilot config hash mismatch",
        )
        _require(probe_path.parent == run_root / "probes", f"{variant}: probe escaped the run root")
        _require(context_path.parent == run_root / "probes", f"{variant}: context escaped the run root")
        _require(item.get("probe_json_sha256") == _sha256(probe_path), f"{variant}: probe hash mismatch")
        _require(item.get("context_json_sha256") == _sha256(context_path), f"{variant}: context hash mismatch")
        summary = validate_probe(probe, variant)
        _validate_probe_bindings(
            probe,
            variant=variant,
            probe_path=probe_path,
            context_path=context_path,
            context=context,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        _require(
            item.get("validated_probe_summary_sha256") == _canonical_sha256(summary),
            f"{variant}: validated probe summary mismatch",
        )
        validated.append(summary)
    return {
        "path": str(artifact_path),
        "sha256": _sha256(artifact_path),
        "git_commit": expected_commit,
        "seed": int(manifest["seed"]),
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "pilot_nonce": manifest["pilot_nonce"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
        "variants": validated,
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

    manifest_path, manifest = _load_json(probes.parent / "manifest.json", "DDP pilot run manifest")
    _require(manifest.get("schema_version") == "duca_p0_ddp_pilot_run_v1", "bad DDP pilot run-manifest schema")
    _require(manifest.get("git_commit") == commit, "DDP pilot run-manifest commit is stale")
    _require(str(manifest.get("slurm_job_id", "")).isdigit(), "DDP pilot lacks a Slurm job identity")
    seed = int(manifest.get("seed", -1))
    _require(seed >= 0, "DDP pilot seed is invalid")
    gate_path = Path(core_gate_json).resolve()
    _require(manifest.get("core_gate_json_sha256") == _sha256(gate_path), "run manifest core-gate hash mismatch")
    _validate_run_manifest_files(manifest)

    formal_suite = validate_suite(
        repo_root=root,
        seed=seed,
        expected_commit=commit,
        require_clean=require_clean,
        core_gate_json=core_gate_json,
    )
    _require(
        manifest.get("shared_protocol_sha256") == formal_suite["shared_protocol_sha256"],
        "run manifest shared-protocol hash mismatch",
    )
    formal_gate = formal_suite["formal_core_gate"]
    _require(manifest.get("checkpoint_sha256") == formal_gate["checkpoint_sha256"], "pilot checkpoint differs from core gate")
    _require(
        manifest.get("official_asformer_source_sha256") == formal_gate["official_asformer_source_sha256"],
        "pilot ASFormer source differs from core gate",
    )
    _require(
        manifest.get("reference_config_sha256") == formal_gate["reference_config_sha256"],
        "pilot reference config differs from core gate",
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
        context_path = probes / f"{variant}.context.json"
        _require(probe_path.is_file(), f"{variant}: training probe is missing: {probe_path}")
        _require(context_path.is_file(), f"{variant}: pilot context is missing: {context_path}")
        payload = json.loads(probe_path.read_text(encoding="utf-8"))
        context = json.loads(context_path.read_text(encoding="utf-8"))
        summary = validate_probe(payload, variant)
        _validate_probe_bindings(
            payload,
            variant=variant,
            probe_path=probe_path,
            context_path=context_path,
            context=context,
            manifest_path=manifest_path,
            manifest=manifest,
        )
        summary.update(
            {
                "pilot_config": PILOT_CONFIGS[variant],
                "pilot_config_sha256": _sha256(pilot_config_path),
                "probe_json": str(probe_path),
                "probe_json_sha256": _sha256(probe_path),
                "context_json": str(context_path),
                "context_json_sha256": _sha256(context_path),
                "validated_probe_summary_sha256": _canonical_sha256(
                    validate_probe(payload, variant)
                ),
            }
        )
        variants.append(summary)

    return {
        "schema_version": "duca_p0_ddp_pilot_suite_v1",
        "ok": True,
        "git_commit": commit,
        "seed": seed,
        "slurm_job_id": str(manifest["slurm_job_id"]),
        "pilot_nonce": manifest["pilot_nonce"],
        "shared_protocol_sha256": formal_suite["shared_protocol_sha256"],
        "core_gate_json": str(gate_path),
        "core_gate_json_sha256": _sha256(gate_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "official_asformer_source_sha256": manifest["official_asformer_source_sha256"],
        "reference_config_sha256": manifest["reference_config_sha256"],
        "run_manifest_path": str(manifest_path),
        "run_manifest_sha256": _sha256(manifest_path),
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
    validate_pilot_artifact(
        output,
        repo_root=args.repo_root,
        expected_commit=payload["git_commit"],
        expected_protocol_sha256=payload["shared_protocol_sha256"],
        expected_core_gate_sha256=payload["core_gate_json_sha256"],
        expected_checkpoint_sha256=payload["checkpoint_sha256"],
        expected_official_asformer_source_sha256=payload["official_asformer_source_sha256"],
        expected_reference_config_sha256=payload["reference_config_sha256"],
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
