from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from mmengine.config import Config

from tools.bata.duca_p0_evaluation import evaluation_config_sha256
from tools.bata.validate_duca_cellcf_real_loader_gate import (
    validate_real_loader_gate_artifact,
)
from tools.bata.validate_duca_cellcf_ddp_pilot import validate_pilot_artifact
from tools.bata.validate_duca_cellcf_fixed384 import VARIANTS, validate_config


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "duca_cellcf_suite_manifest_v1"
POST_RUN_SCHEMA = "duca_cellcf_post_run_evidence_v1"
VARIANT_ORDER = ("uniform", "transition_beta0", "cellcf")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


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


def _load_json(path: str | Path, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    _require(resolved.is_file(), f"{label} is missing: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{label} must be a JSON object")
    return resolved, payload


def _shared_protocol(cfg: Config) -> dict[str, Any]:
    model = _plain(cfg.model)
    selector = dict(model["frame_selector"])
    for key in (
        "local_cell_force_exact_uniform",
        "counterfactual_utility_distillation_weight",
        "require_counterfactual_utility_teacher",
    ):
        selector.pop(key, None)
    model["frame_selector"] = selector
    return {
        "task": "offline_temporal_action_detection",
        "dense_window_size": int(cfg.dense_window_size),
        "budget": int(cfg.window_size),
        "model": model,
        "dataset": _plain(cfg.dataset),
        "solver": _plain(cfg.solver),
        "optimizer": _plain(cfg.optimizer),
        "scheduler": _plain(cfg.scheduler),
        "workflow": _plain(cfg.workflow),
        "evaluation": _plain(cfg.evaluation),
        "inference": _plain(cfg.inference),
        "post_processing": _plain(cfg.post_processing),
    }


def _variant_contract(cfg: Config, variant: str) -> dict[str, Any]:
    selector = cfg.model.frame_selector
    return {
        "variant": variant,
        "force_exact_uniform": bool(selector.local_cell_force_exact_uniform),
        "counterfactual_weight": float(
            selector.counterfactual_utility_distillation_weight
        ),
        "requires_counterfactual_teacher": bool(
            selector.require_counterfactual_utility_teacher
        ),
        "acquisition_policy": str(selector.acquisition_policy),
        "counterfactual_objective": str(selector.counterfactual_objective),
        "detector_gradient_mode": str(selector.detector_gradient_mode),
    }


def _reference_data(cfg: Config) -> dict[str, Any]:
    annotation = Path(cfg.evaluation.ground_truth_filename).expanduser().resolve()
    class_map = Path(cfg.dataset.test.class_map).expanduser().resolve()
    _require(annotation.is_file(), f"evaluation annotation is missing: {annotation}")
    _require(class_map.is_file(), f"evaluation class map is missing: {class_map}")
    return {
        "evaluation_annotation_path": str(annotation),
        "evaluation_annotation_sha256": _sha256(annotation),
        "evaluation_class_map_path": str(class_map),
        "evaluation_class_map_sha256": _sha256(class_map),
        "evaluation_config_sha256": evaluation_config_sha256(cfg.evaluation),
    }


def _validate_gate(path: str | Path, commit: str) -> dict[str, Any]:
    resolved, payload = _load_json(path, "CellCF real-loader CUDA gate")
    validated = validate_real_loader_gate_artifact(
        resolved,
        expected_commit=commit,
        expected_sha256=_sha256(resolved),
        require_clean=False,
    )
    return {
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "synthetic_gate_sha256": validated["synthetic_gate_sha256"],
        "payload": payload,
    }


def _validate_pilot(
    path: str | Path,
    *,
    commit: str,
    gate_sha256: str,
    protocol_sha256: str,
    order_sha256: str,
) -> dict[str, Any]:
    resolved, payload = _load_json(path, "CellCF DDP pilot")
    validate_pilot_artifact(
        resolved,
        repo_root=ROOT,
        expected_commit=commit,
        expected_real_loader_gate_sha256=gate_sha256,
        require_clean=False,
    )
    _require(tuple(payload.get("variant_order", ())) == VARIANT_ORDER, "CellCF pilot arm order drift")
    return {"path": str(resolved), "sha256": _sha256(resolved), "payload": payload}


def _validate_post_run(
    path: str | Path,
    *,
    variant: str,
    commit: str,
    protocol_sha256: str,
    order_sha256: str,
    gate_sha256: str,
    pilot_sha256: str,
) -> dict[str, Any]:
    resolved, payload = _load_json(path, f"{variant} post-run evidence")
    _require(payload.get("schema") == POST_RUN_SCHEMA, f"{variant}: bad post-run schema")
    _require(payload.get("ok") is True, f"{variant}: post-run evidence did not pass")
    expected = {
        "variant": variant,
        "git_commit": commit,
        "protocol_sha256": protocol_sha256,
        "ordered_exposure_sha256": order_sha256,
        "real_loader_gate_sha256": gate_sha256,
        "ddp_pilot_sha256": pilot_sha256,
        "checkpoint_epoch": 131,
        "checkpoint_state_key": "state_dict_ema",
        "successful_optimizer_updates": 13200,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"{variant}: post-run {key} mismatch")
    metrics = payload.get("metrics")
    _require(isinstance(metrics, Mapping), f"{variant}: post-run metrics are missing")
    for key in ("average_mAP", "mAP@0.3", "mAP@0.4", "mAP@0.5", "mAP@0.6", "mAP@0.7"):
        value = metrics.get(key)
        _require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{variant}: invalid {key}")
    artifact_hash = payload.get("artifact_chain_sha256")
    unsigned = dict(payload)
    unsigned.pop("artifact_chain_sha256", None)
    _require(artifact_hash == _canonical_sha256(unsigned), f"{variant}: post-run artifact hash mismatch")
    return {"path": str(resolved), "sha256": _sha256(resolved), "metrics": dict(metrics)}


def validate_suite(
    *,
    repo_root: str | Path,
    seed: int,
    expected_commit: str | None,
    require_clean: bool,
    gate_json: str | Path,
    pilot_json: str | Path,
    post_run_evidence: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    commit = _git(root, "rev-parse", "HEAD")
    if expected_commit is not None:
        _require(commit == expected_commit, "HEAD differs from expected CellCF commit")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal")
    if require_clean:
        _require(not dirty, "CellCF suite requires a clean exact-commit tree")
    _require(seed >= 0, "seed must be non-negative")

    configs = {
        variant: Config.fromfile(str(root / VARIANTS[variant]))
        for variant in VARIANT_ORDER
    }
    reference_protocol = _shared_protocol(configs[VARIANT_ORDER[0]])
    protocol_sha256 = _canonical_sha256(reference_protocol)
    order_sha256 = _canonical_sha256(list(VARIANT_ORDER))
    variants = []
    for variant in VARIANT_ORDER:
        cfg = configs[variant]
        validation = validate_config(variant, VARIANTS[variant])
        _require(_shared_protocol(cfg) == reference_protocol, f"{variant}: shared protocol drift")
        config_path = root / VARIANTS[variant]
        contract = _variant_contract(cfg, variant)
        variants.append(
            {
                "name": variant,
                "config": VARIANTS[variant].replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "resolved_config_sha256": _canonical_sha256(cfg.to_dict()),
                "variant_contract": contract,
                "variant_contract_sha256": _canonical_sha256(contract),
                "validation": validation,
            }
        )

    data = _reference_data(configs[VARIANT_ORDER[0]])
    gate = _validate_gate(gate_json, commit)
    gate_dataset = gate["payload"].get("dataset", {})
    _require(
        gate_dataset.get("annotation_sha256") == data["evaluation_annotation_sha256"],
        "CellCF gate annotation differs from the suite",
    )
    _require(
        gate_dataset.get("class_map_sha256") == data["evaluation_class_map_sha256"],
        "CellCF gate class map differs from the suite",
    )
    pilot = _validate_pilot(
        pilot_json,
        commit=commit,
        gate_sha256=gate["sha256"],
        protocol_sha256=protocol_sha256,
        order_sha256=order_sha256,
    )

    completed = {}
    if post_run_evidence is not None:
        _require(set(post_run_evidence) == set(VARIANT_ORDER), "post-run evidence must cover exactly three CellCF arms")
        completed = {
            variant: _validate_post_run(
                post_run_evidence[variant],
                variant=variant,
                commit=commit,
                protocol_sha256=protocol_sha256,
                order_sha256=order_sha256,
                gate_sha256=gate["sha256"],
                pilot_sha256=pilot["sha256"],
            )
            for variant in VARIANT_ORDER
        }

    return {
        "schema": SCHEMA,
        "ok": True,
        "status": "complete" if completed else "deployable_not_submitted",
        "task": "offline_temporal_action_detection",
        "git_commit": commit,
        "git_tree_clean": not bool(dirty),
        "seed": int(seed),
        "variant_order": list(VARIANT_ORDER),
        "ordered_exposure_sha256": order_sha256,
        "shared_protocol": reference_protocol,
        "shared_protocol_sha256": protocol_sha256,
        "variants": variants,
        "reference_data_artifacts": data,
        "real_loader_gate": {key: value for key, value in gate.items() if key != "payload"},
        "ddp_pilot": {key: value for key, value in pilot.items() if key != "payload"},
        "completed_runs": completed,
        "submission_performed": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--expected-commit")
    parser.add_argument("--require-clean", action="store_true")
    parser.add_argument("--gate-json", required=True)
    parser.add_argument("--pilot-json", required=True)
    parser.add_argument("--post-run-evidence", action="append", default=[])
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args(argv)
    try:
        post_runs = None
        if args.post_run_evidence:
            post_runs = {}
            for item in args.post_run_evidence:
                name, separator, path = item.partition("=")
                _require(bool(separator and name and path), "post-run evidence must be VARIANT=JSON")
                _require(name not in post_runs, f"duplicate post-run evidence for {name}")
                post_runs[name] = path
        payload = validate_suite(
            repo_root=args.repo_root,
            seed=args.seed,
            expected_commit=args.expected_commit,
            require_clean=args.require_clean,
            gate_json=args.gate_json,
            pilot_json=args.pilot_json,
            post_run_evidence=post_runs,
        )
        code = 0
    except Exception as exc:
        payload = {"schema": SCHEMA, "ok": False, "error_type": type(exc).__name__, "error": str(exc)}
        code = 1
    output = json.dumps(payload, indent=2, sort_keys=True)
    print(output)
    Path(args.output_json).write_text(output + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
