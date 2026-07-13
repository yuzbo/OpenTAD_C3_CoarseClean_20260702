from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

from mmengine.config import Config

from tools.bata.validate_duca_transition_only_p0_variant import CONFIGS, validate_variant


REFERENCE_VARIANT = "transition_counterfactual"
VARIANT_ORDER = ("uniform", "direct", "transition_beta0", "transition_counterfactual")
EXPECTED_OPTIMIZER_UPDATES = 13200


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


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(_plain(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _variant_contract(cfg: Config) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    policy_alpha = selector.loss_weight_schedule.get("policy_alpha")
    return {
        "selector_variant": str(selector.selector_variant),
        "acquisition_policy": str(selector.acquisition_policy),
        "inference_policy_alpha": float(selector.get("inference_policy_alpha", 1.0)),
        "train_policy_alpha_end": (
            None if policy_alpha is None else float(policy_alpha.end)
        ),
        "counterfactual_utility_distillation_weight": float(
            selector.get("counterfactual_utility_distillation_weight", 0.0)
        ),
        "require_counterfactual_utility_teacher": bool(
            selector.get("require_counterfactual_utility_teacher", False)
        ),
        "structured_temperature": float(selector.structured_temperature),
        "scoring_weights": {
            key: float(selector.get(key))
            for key in (
                "actionness_weight", "transition_weight", "uncertainty_weight",
                "utility_weight", "boundary_weight",
            )
        },
        "loss_weights": _plain(selector.loss_weights),
        "loss_weight_schedule": _plain(selector.loss_weight_schedule),
        "actionness_source_contract": {
            key: _plain(selector.actionness_source_cfg.get(key))
            for key in (
                "source_name", "probe_model", "tcn_variant", "spatial_size",
                "tcn_hidden_dim", "checkpoint_path", "require_checkpoint",
                "frozen", "trainable", "official_action_seg_backend",
            )
        },
    }


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, encoding="utf-8"
    ).strip()


def _shared_protocol(cfg: Config) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    return {
        "detector_type": cfg.model.type,
        "backbone": _plain(cfg.model.backbone),
        "projection": _plain(cfg.model.projection),
        "neck": _plain(cfg.model.neck),
        "detector_head": _plain(cfg.model.rpn_head),
        "dataset": _plain(cfg.dataset),
        "solver": _plain(cfg.solver),
        "optimizer": _plain(cfg.optimizer),
        "scheduler": _plain(cfg.scheduler),
        "workflow": _plain(cfg.workflow),
        "evaluation": _plain(cfg.evaluation),
        "inference": _plain(cfg.inference),
        "post_processing": _plain(cfg.post_processing),
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
        "expected_optimizer_steps": EXPECTED_OPTIMIZER_UPDATES,
    }


def _post_run_contract(protocol_sha256: str, bindings: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "duca_p0_post_run_evidence_v1",
        "shared_protocol_sha256": protocol_sha256,
        "successful_optimizer_updates": EXPECTED_OPTIMIZER_UPDATES,
        "lr_scheduler_successful_update_exposure": EXPECTED_OPTIMIZER_UPDATES,
        "ema_successful_update_exposure": EXPECTED_OPTIMIZER_UPDATES,
        "evaluator_identity": "mAP:validation:0.3,0.4,0.5,0.6,0.7",
        "checkpoint_criterion": "same_workflow_best_or_final_declared",
        "non_finite_collapse": False,
        **bindings,
    }


def _validate_formal_optimizer_step(gate_payload: dict[str, Any]) -> None:
    _require(gate_payload.get("optimizer_step_ran") is True, "formal core gate did not run optimizer.step")
    _require(
        gate_payload.get("optimizer_parameter_change_verified") is True,
        "formal core gate did not verify a trainable parameter change",
    )
    changed_groups = gate_payload.get("optimizer_changed_parameter_groups")
    _require(isinstance(changed_groups, list) and bool(changed_groups), "formal core gate changed no parameter group")
    _require("detector_head" in changed_groups, "formal core gate did not update the detector head")
    changes = gate_payload.get("optimizer_parameter_max_abs_change")
    _require(isinstance(changes, dict), "formal optimizer parameter-change evidence is missing")
    for group in changed_groups:
        delta = changes.get(group)
        _require(
            isinstance(delta, (int, float)) and math.isfinite(delta) and delta > 0.0,
            f"formal optimizer parameter change is invalid for {group}",
        )
    _require(gate_payload.get("optimizer_step_loss_finite") is True, "formal optimizer-step loss is non-finite")
    _require(
        gate_payload.get("optimizer_step_gradients_finite") is True,
        "formal optimizer-step gradients are non-finite",
    )
    loss = gate_payload.get("optimizer_step_loss")
    _require(
        isinstance(loss, (int, float)) and math.isfinite(loss) and loss >= 0.0,
        "formal optimizer-step loss is invalid",
    )
    normalizer = gate_payload.get("loss_normalizer_contract")
    _require(isinstance(normalizer, dict), "formal loss-normalizer contract is missing")
    for key in ("finite", "positive", "updated_by_training_forward", "unchanged_by_optimizer_step"):
        _require(normalizer.get(key) is True, f"formal loss-normalizer contract failed: {key}")
    _require(
        normalizer.get("state_kind") == "ActionFormerHead.loss_normalizer_ema_buffer",
        "formal loss-normalizer state kind is invalid",
    )
    for key in ("before_forward", "after_forward", "after_optimizer_step"):
        value = normalizer.get(key)
        _require(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0.0,
            f"formal loss-normalizer value is invalid: {key}",
        )
    _require(
        normalizer["before_forward"] != normalizer["after_forward"],
        "formal loss-normalizer did not change during training forward",
    )
    _require(
        normalizer["after_forward"] == normalizer["after_optimizer_step"],
        "formal loss-normalizer changed during optimizer.step",
    )


def validate_post_run_evidence(
    path: str | Path, *, variant: str, protocol_sha256: str, bindings: dict[str, Any]
) -> dict[str, Any]:
    evidence_path = Path(path).resolve()
    _require(evidence_path.is_file(), f"{variant}: post-run evidence missing: {evidence_path}")
    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    expected = _post_run_contract(protocol_sha256, bindings)
    _require(payload.get("ok") is True, f"{variant}: post-run evidence must declare ok=true")
    _require(payload.get("variant") == variant, f"{variant}: post-run variant mismatch")
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{variant}: post-run {key} mismatch")
    run_manifest_path = Path(str(payload.get("run_manifest_path", ""))).resolve()
    _require(run_manifest_path.is_file(), f"{variant}: run manifest is missing")
    _require(payload.get("run_manifest_sha256") == _sha256(run_manifest_path), f"{variant}: run manifest hash mismatch")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest_fields = {
        "git_commit": "git_commit",
        "config_sha256": "config_sha256",
        "resolved_config_sha256": "resolved_config_sha256",
        "variant_contract_sha256": "variant_contract_sha256",
        "core_gate_sha256": "core_gate_json_sha256",
        "shared_protocol_sha256": "shared_protocol_sha256",
        "seed": "seed",
    }
    for evidence_key, manifest_key in manifest_fields.items():
        _require(
            run_manifest.get(manifest_key) == expected[evidence_key],
            f"{variant}: run manifest {manifest_key} mismatch",
        )
    _require(run_manifest.get("variant") == variant, f"{variant}: run manifest variant mismatch")
    return {"path": str(evidence_path), "sha256": _sha256(evidence_path), "validated": True}


def validate_suite(
    *,
    repo_root: str | Path = ".",
    seed: int = 0,
    expected_commit: str | None = None,
    require_clean: bool = False,
    core_gate_json: str | Path | None = None,
    post_run_evidence: dict[str, str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    _require(seed >= 0, "seed must be non-negative")
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None:
        _require(commit == expected_commit, "current HEAD does not match expected commit")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if require_clean:
        _require(not dirty, "formal P0 suite preparation requires a clean git tree")

    _require(core_gate_json is not None, "formal core gate is required")
    core_gate: dict[str, Any]
    if core_gate_json is not None:
        gate_path = Path(core_gate_json).resolve()
        _require(gate_path.is_file(), f"formal core gate is missing: {gate_path}")
        gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
        _require(gate_payload.get("ok") is True, "formal core gate must declare ok=true")
        _require(gate_payload.get("formal_proof_ok") is True, "formal core gate proof did not pass")
        _validate_formal_optimizer_step(gate_payload)
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
    _require(
        reference_protocol["solver"].get("static_graph") is True,
        "P0 suite requires static_graph=true with a scorer-connected no-candidate loss",
    )
    _require(
        reference_protocol["solver"].get("find_unused_parameters") is False,
        "P0 suite requires find_unused_parameters=false with reentrant backbone checkpointing",
    )
    variants: list[dict[str, Any]] = []
    variant_bindings: dict[str, dict[str, Any]] = {}
    for name in VARIANT_ORDER:
        path = root / CONFIGS[name]
        summary = validate_variant(name, str(path))
        protocol = _shared_protocol(configs[name])
        _require(protocol == reference_protocol, f"{name}: shared protocol differs from reference")
        variant_contract = _variant_contract(configs[name])
        resolved_config_sha256 = _canonical_sha256(configs[name].to_dict())
        variant_contract_sha256 = _canonical_sha256(variant_contract)
        variant_bindings[name] = {
            "git_commit": commit,
            "seed": int(seed),
            "config_sha256": _sha256(path),
            "resolved_config_sha256": resolved_config_sha256,
            "variant_contract_sha256": variant_contract_sha256,
            "core_gate_sha256": core_gate["sha256"],
        }
        variants.append(
            {
                "name": name,
                "config": CONFIGS[name].replace("\\", "/"),
                "config_sha256": _sha256(path),
                "resolved_config_sha256": resolved_config_sha256,
                "variant_contract": variant_contract,
                "variant_contract_sha256": variant_contract_sha256,
                "variant_validation": summary,
            }
        )

    protocol_payload = json.dumps(reference_protocol, sort_keys=True, separators=(",", ":"))
    protocol_sha256 = hashlib.sha256(protocol_payload.encode("utf-8")).hexdigest()
    post_run_contract = {
        name: _post_run_contract(protocol_sha256, variant_bindings[name])
        for name in VARIANT_ORDER
    }
    validated_post_runs: dict[str, Any] = {}
    if post_run_evidence is not None:
        _require(set(post_run_evidence) == set(VARIANT_ORDER), "post-run evidence must cover exactly four variants")
        validated_post_runs = {
            name: validate_post_run_evidence(
                post_run_evidence[name],
                variant=name,
                protocol_sha256=protocol_sha256,
                bindings=variant_bindings[name],
            )
            for name in VARIANT_ORDER
        }
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
        "post_run_contract": post_run_contract,
        "validated_post_runs": validated_post_runs,
        "submission_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--core-gate-json")
    parser.add_argument(
        "--post-run-evidence", action="append", default=[], metavar="VARIANT=JSON",
        help="repeat for all four variants to validate completed-run evidence",
    )
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        evidence: dict[str, str] | None = None
        if args.post_run_evidence:
            evidence = {}
            for item in args.post_run_evidence:
                name, separator, path = item.partition("=")
                _require(bool(separator and name and path), "--post-run-evidence must be VARIANT=JSON")
                _require(name not in evidence, f"duplicate post-run evidence for {name}")
                evidence[name] = path
        payload = validate_suite(
            repo_root=args.repo_root,
            seed=args.seed,
            expected_commit=args.expected_commit,
            require_clean=args.require_clean,
            core_gate_json=args.core_gate_json,
            post_run_evidence=evidence,
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
