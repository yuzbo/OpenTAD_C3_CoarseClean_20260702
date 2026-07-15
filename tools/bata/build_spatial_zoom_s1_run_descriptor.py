from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    S1_CHECKPOINT_RULE,
    S1_PROFILE_ORDER_SEED,
    S1_TRAINING_SEEDS,
    build_s1_profile_order,
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_evidence import (  # noqa: E402
    validate_s1_test_evidence,
)
from tools.bata.spatial_zoom_s1_cost import (  # noqa: E402
    S1_PROFILE_SCHEMA,
    validate_profile_summary,
)
from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
    validate_profile_attempt_marker,
)
from tools.bata.select_spatial_zoom_s1_checkpoint import (  # noqa: E402
    validate_checkpoint_selection,
)
from tools.bata.validate_spatial_zoom_s1 import validate_config_matrix  # noqa: E402
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_clean_git_checkout,
    validate_bound_s1_training_config,
)


def build_descriptor(args: argparse.Namespace) -> dict:
    matrix = validate_config_matrix()
    cfg = Config.fromfile(str(args.config))
    binding = validate_bound_s1_training_config(cfg, seed=int(args.seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("formal S1 descriptor requires the bound full precheck")
    resolution = int(cfg.spatial_zoom_s1_contract.runtime_resolution)
    profile_order = build_s1_profile_order()
    profile_order_entry = next(
        row
        for row in profile_order
        if int(row["resolution"]) == resolution and int(row["seed"]) == int(args.seed)
    )
    profile_order_sha256 = canonical_sha256(profile_order)
    if int(args.seed) not in S1_TRAINING_SEEDS:
        raise ValueError("S1 run seed is outside the frozen seed schema")
    manifest = validate_s1_manifest(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        annotation_path=args.annotation,
    )
    for path in (
        args.checkpoint,
        args.checkpoint_selection,
        args.test_evidence,
        args.profile,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    checkpoint_sha = sha256_file(args.checkpoint)
    selection = validate_checkpoint_selection(
        json.loads(args.checkpoint_selection.read_text(encoding="utf-8")),
        config=cfg,
        seed=int(args.seed),
        manifest=manifest,
        checkpoint_path=args.checkpoint,
        protocol_fingerprint=matrix["protocol_fingerprint"],
    )
    checkpoint_epoch = int(selection["selected"]["epoch"])
    test_evidence = validate_s1_test_evidence(
        json.loads(args.test_evidence.read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(args.seed),
    )
    if Path(test_evidence["checkpoint_path"]).resolve() != args.checkpoint.resolve():
        raise ValueError("S1 test evidence does not use the selected checkpoint")
    if int(test_evidence["checkpoint_epoch"]) != checkpoint_epoch:
        raise ValueError("S1 test evidence checkpoint epoch mismatch")
    profile = validate_profile_summary(
        json.loads(args.profile.read_text(encoding="utf-8"))
    )
    if profile.get("schema_version") != S1_PROFILE_SCHEMA:
        raise ValueError("S1 run descriptor requires an S1 full-stack profile")
    marker_path = Path(profile["profile_attempt_marker_path"]).resolve()
    if not marker_path.is_file():
        raise FileNotFoundError(marker_path)
    marker_file_sha = sha256_file(marker_path)
    marker = validate_profile_attempt_marker(marker_path)
    if marker_file_sha != profile["profile_attempt_marker_file_sha256"]:
        raise ValueError("S1 profile-attempt marker file hash mismatch")
    if marker["marker_sha256"] != profile["profile_attempt_marker_sha256"]:
        raise ValueError("S1 profile-attempt marker identity mismatch")
    profile_name = args.profile.resolve().name
    if not profile_name.endswith(".summary.json"):
        raise ValueError("formal S1 profile summary must end in .summary.json")
    canonical_output_prefix = args.profile.resolve().with_name(
        profile_name[: -len(".summary.json")]
    )
    expected_marker = {
        "resolution": resolution,
        "seed": int(args.seed),
        "bound_config_sha256": canonical_sha256(cfg.to_dict()),
        "code_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "test_open_certificate_sha256": test_evidence["test_open_certificate_sha256"],
        "test_evidence_sha256": test_evidence["evidence_sha256"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "software_fingerprint": profile["software_fingerprint"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
        "canonical_output_prefix": str(canonical_output_prefix),
    }
    for key, expected in expected_marker.items():
        if marker.get(key) != expected:
            raise ValueError(f"S1 profile-attempt marker {key} mismatch")
    expected_profile = {
        "resolution": resolution,
        "manifest_sha256": manifest["manifest_sha256"],
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint_epoch,
        "seed": int(args.seed),
        "split": "test",
        "trained_checkpoint": True,
        "test_open_certificate_sha256": test_evidence["test_open_certificate_sha256"],
        "test_evidence_sha256": test_evidence["evidence_sha256"],
        "test_open_marker_sha256": test_evidence["test_open_marker_sha256"],
        "config_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "profile_attempt_marker_path": str(marker_path),
        "profile_attempt_marker_file_sha256": marker_file_sha,
        "profile_attempt_marker_sha256": marker["marker_sha256"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
    }
    for key, expected in expected_profile.items():
        if profile.get(key) != expected:
            raise ValueError(
                f"S1 profile {key}={profile.get(key)!r} does not match {expected!r}"
            )
    prediction_path = Path(test_evidence["prediction_path"])
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    results = prediction.get("results")
    if not isinstance(results, dict):
        raise ValueError("S1 prediction artifact must use result_detection.json format")
    unexpected = sorted(set(results) - set(manifest["splits"]["test"]))
    if unexpected:
        raise ValueError(
            f"S1 prediction artifact contains videos outside sealed test: {unexpected}"
        )
    descriptor = {
        "schema_version": "spatial_zoom_s1_run_v4",
        "resolution": resolution,
        "seed": int(args.seed),
        "config_path": str(args.config.resolve()),
        "resolved_config_sha256": canonical_sha256(cfg.to_dict()),
        "code_commit": binding["code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_path": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_epoch": checkpoint_epoch,
        "checkpoint_selection_rule": S1_CHECKPOINT_RULE,
        "checkpoint_selection_path": str(args.checkpoint_selection.resolve()),
        "checkpoint_selection_sha256": sha256_file(args.checkpoint_selection),
        "checkpoint_selection_internal_sha256": selection["selection_sha256"],
        "test_evidence_path": str(args.test_evidence.resolve()),
        "test_evidence_file_sha256": sha256_file(args.test_evidence),
        "test_evidence_sha256": test_evidence["evidence_sha256"],
        "test_open_certificate_path": test_evidence["test_open_certificate_path"],
        "test_open_certificate_file_sha256": test_evidence[
            "test_open_certificate_file_sha256"
        ],
        "test_open_certificate_sha256": test_evidence["test_open_certificate_sha256"],
        "test_open_marker_path": test_evidence["test_open_marker_path"],
        "test_open_marker_file_sha256": test_evidence["test_open_marker_file_sha256"],
        "test_open_marker_sha256": test_evidence["test_open_marker_sha256"],
        "prediction_path": str(prediction_path.resolve()),
        "prediction_sha256": sha256_file(prediction_path),
        "profile_summary_path": str(args.profile.resolve()),
        "profile_summary_sha256": sha256_file(args.profile),
        "profile_summary_internal_sha256": profile["profile_sha256"],
        "profile_attempt_marker_path": str(marker_path),
        "profile_attempt_marker_file_sha256": marker_file_sha,
        "profile_attempt_marker_sha256": marker["marker_sha256"],
        "ground_truth_path": str(args.annotation.resolve()),
        "ground_truth_sha256": sha256_file(args.annotation),
        "evaluation_split": manifest["annotation_subsets"]["sealed_test"],
        "official_test_opened_after_protocol_freeze": bool(
            test_evidence["official_test_read"]
        ),
        "paper_claim_allowed": False,
    }
    descriptor["descriptor_sha256"] = canonical_sha256(descriptor)
    return descriptor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind one trained S1 run to immutable evidence artifacts"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-selection", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an S1 run descriptor")
        descriptor = build_descriptor(args)
        require_clean_git_checkout(expected_commit=descriptor["code_commit"])
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
        json.dump(descriptor, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
