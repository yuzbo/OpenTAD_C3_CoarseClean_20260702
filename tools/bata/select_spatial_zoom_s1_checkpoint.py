from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.analyze_spatial_zoom_s1_results import (  # noqa: E402
    DetectionCorpus,
    evaluate_corpus,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_CHECKPOINT_RULE,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_evidence import (  # noqa: E402
    validate_s1_gate_evidence,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    validate_bound_s1_training_config,
)

S1_CHECKPOINT_SELECTION_SCHEMA = "spatial_zoom_s1_checkpoint_selection_v4"


def _candidate_from_evidence(
    *,
    evidence_path: Path,
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
    annotation_path: Path,
) -> dict[str, Any]:
    quartiles = manifest["duration_quartiles_seconds"]
    corpus = DetectionCorpus.from_files(
        ground_truth_path=annotation_path,
        prediction_path=evidence["prediction_path"],
        subset=manifest["annotation_subsets"]["development"],
        video_ids=manifest["splits"]["gate"],
    )
    metrics = evaluate_corpus(
        corpus,
        video_sample=corpus.video_ids,
        duration_quartiles=(quartiles["q1"], quartiles["q2"], quartiles["q3"]),
    )
    return {
        "epoch": int(evidence["epoch"]),
        "checkpoint_path": evidence["checkpoint_path"],
        "checkpoint_sha256": evidence["checkpoint_sha256"],
        "checkpoint_sidecar_path": evidence["checkpoint_sidecar_path"],
        "checkpoint_sidecar_sha256": evidence["checkpoint_sidecar_sha256"],
        "prediction_path": evidence["prediction_path"],
        "prediction_sha256": evidence["prediction_sha256"],
        "gate_evidence_path": str(evidence_path),
        "gate_evidence_file_sha256": sha256_file(evidence_path),
        "gate_evidence_sha256": evidence["evidence_sha256"],
        "state_key": evidence["state_key"],
        "successful_updates": int(evidence["successful_updates"]),
        "gate_average_map": metrics["average_map"],
        "gate_map_at": metrics["map_at"],
        "gate_high_tiou_headroom": metrics["high_tiou_headroom"],
    }


def select_s1_checkpoint(
    *,
    config_path: str | Path,
    seed: int,
    evidence_paths: Sequence[str | Path],
) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = Config.fromfile(str(config_path))
    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    manifest_path = Path(binding["manifest_path"])
    annotation_path = Path(binding["annotation_path"])
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )
    evidence_rows = []
    seen_epochs = set()
    for path_value in evidence_paths:
        path = Path(path_value).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        evidence = validate_s1_gate_evidence(
            json.loads(path.read_text(encoding="utf-8")),
            cfg=cfg,
            binding=binding,
        )
        epoch = int(evidence["epoch"])
        if epoch in seen_epochs:
            raise ValueError("duplicate S1 gate evidence epoch")
        seen_epochs.add(epoch)
        evidence_rows.append((path, evidence))
    expected_epochs = set(map(int, binding["eligible_checkpoint_epochs"]))
    if seen_epochs != expected_epochs:
        missing = sorted(expected_epochs - seen_epochs)
        extra = sorted(seen_epochs - expected_epochs)
        raise ValueError(
            f"S1 gate evidence must cover every eligible checkpoint; missing={missing}, extra={extra}"
        )

    rows = []
    for evidence_path, evidence in evidence_rows:
        rows.append(
            _candidate_from_evidence(
                evidence_path=evidence_path,
                evidence=evidence,
                manifest=manifest,
                annotation_path=annotation_path,
            )
        )
    rows.sort(key=lambda row: (-row["gate_high_tiou_headroom"], row["epoch"]))
    selected = dict(rows[0])
    report: dict[str, Any] = {
        "schema_version": S1_CHECKPOINT_SELECTION_SCHEMA,
        "selection_rule": S1_CHECKPOINT_RULE,
        "selection_metric": "(mAP@0.6 + mAP@0.7) / 2 on frozen gate videos",
        "tie_break": "earliest_epoch",
        "resolution": int(binding["resolution"]),
        "seed": int(binding["seed"]),
        "bound_config_path": str(config_path),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "source_config_sha256": binding["source_config_sha256"],
        "protocol_fingerprint": binding["protocol_fingerprint"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "ground_truth_path": str(annotation_path.resolve()),
        "ground_truth_sha256": sha256_file(annotation_path),
        "gate_split_hash": manifest["split_hashes"]["gate"],
        "eligible_checkpoint_epochs": sorted(expected_epochs),
        "candidates": sorted(rows, key=lambda row: row["epoch"]),
        "selected": selected,
        "official_test_read": False,
        "paper_claim_allowed": False,
    }
    report["selection_sha256"] = canonical_sha256(report)
    return report


def validate_checkpoint_selection(
    selection: Mapping[str, Any],
    *,
    config: Config,
    seed: int,
    manifest: Mapping[str, Any],
    checkpoint_path: str | Path,
    protocol_fingerprint: str,
) -> dict[str, Any]:
    binding = validate_bound_s1_training_config(config, seed=int(seed))
    checked = json.loads(json.dumps(dict(selection)))
    expected_hash = checked.pop("selection_sha256", None)
    if not expected_hash or canonical_sha256(checked) != expected_hash:
        raise ValueError("S1 checkpoint selection hash mismatch")
    checked["selection_sha256"] = expected_hash
    if checked.get("schema_version") != S1_CHECKPOINT_SELECTION_SCHEMA:
        raise ValueError("unsupported S1 checkpoint selection schema")
    if checked.get("selection_rule") != S1_CHECKPOINT_RULE:
        raise ValueError("S1 checkpoint selection rule changed")
    expected = {
        "resolution": int(binding["resolution"]),
        "seed": int(seed),
        "bound_config_sha256": canonical_sha256(config.to_dict()),
        "source_config_sha256": binding["source_config_sha256"],
        "protocol_fingerprint": str(protocol_fingerprint),
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "manifest_sha256": manifest["manifest_sha256"],
        "gate_split_hash": manifest["split_hashes"]["gate"],
        "ground_truth_sha256": manifest["annotation_sha256"],
        "official_test_read": False,
        "eligible_checkpoint_epochs": sorted(
            map(int, binding["eligible_checkpoint_epochs"])
        ),
    }
    for key, value in expected.items():
        if checked.get(key) != value:
            raise ValueError(f"S1 checkpoint selection {key} does not match the run")
    candidates = checked.get("candidates")
    if not isinstance(candidates, list) or {
        int(row["epoch"]) for row in candidates
    } != set(binding["eligible_checkpoint_epochs"]):
        raise ValueError("S1 checkpoint selection candidate set is incomplete")
    annotation_path = Path(binding["annotation_path"])
    for row in candidates:
        evidence_path = Path(row["gate_evidence_path"])
        if sha256_file(evidence_path) != row["gate_evidence_file_sha256"]:
            raise ValueError("S1 gate evidence file changed after checkpoint selection")
        evidence = validate_s1_gate_evidence(
            json.loads(evidence_path.read_text(encoding="utf-8")),
            cfg=config,
            binding=binding,
        )
        if evidence["evidence_sha256"] != row["gate_evidence_sha256"]:
            raise ValueError("S1 gate evidence identity mismatch")
        recomputed = _candidate_from_evidence(
            evidence_path=evidence_path.resolve(),
            evidence=evidence,
            manifest=manifest,
            annotation_path=annotation_path,
        )
        if dict(row) != recomputed:
            raise ValueError(
                "S1 checkpoint candidate does not match recomputed gate evidence"
            )
    selected = checked.get("selected")
    if not isinstance(selected, Mapping):
        raise ValueError("S1 checkpoint selection has no selected candidate")
    checkpoint_path = Path(checkpoint_path)
    if Path(selected.get("checkpoint_path", "")).resolve() != checkpoint_path.resolve():
        raise ValueError("S1 descriptor checkpoint is not the gate-selected checkpoint")
    if selected.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("S1 selected checkpoint hash mismatch")
    ordered = sorted(
        candidates,
        key=lambda row: (-float(row["gate_high_tiou_headroom"]), int(row["epoch"])),
    )
    if dict(ordered[0]) != dict(selected):
        raise ValueError("S1 selected checkpoint does not satisfy the frozen rule")
    return checked


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Select an S1 checkpoint from the complete frozen gate evidence set"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--evidence", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an S1 checkpoint selection")
        report = select_s1_checkpoint(
            config_path=args.config,
            seed=args.seed,
            evidence_paths=args.evidence,
        )
        from tools.bata.spatial_zoom_s1_training import require_clean_git_checkout

        require_clean_git_checkout(
            expected_commit=Config.fromfile(
                str(args.config)
            ).spatial_zoom_s1_runtime_binding.code_commit
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

__all__ = [
    "S1_CHECKPOINT_SELECTION_SCHEMA",
    "select_s1_checkpoint",
    "validate_checkpoint_selection",
]
