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
    atomic_publish_json,
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
from tools.bata.spatial_zoom_s1_profile_recovery import (  # noqa: E402
    load_profile_recovery_certificate,
    profile_campaign_prefix,
)
from tools.bata.spatial_zoom_s1_matrix import (  # noqa: E402
    canonical_test_matrix_binding_path,
    validate_profile_matrix_start_receipt,
    validate_test_matrix_binding,
)
from tools.bata.spatial_zoom_s1_sidecar_gate import (  # noqa: E402
    load_sidecar_gate_evidence,
    sidecar_gate_path,
    validate_sidecar_gate_runtime_identity,
)
from tools.bata.spatial_zoom_s1_power import (  # noqa: E402
    validate_nvml_sidecar_attempt,
)
from tools.bata.select_spatial_zoom_s1_checkpoint import (  # noqa: E402
    validate_checkpoint_selection,
)
from tools.bata.validate_spatial_zoom_s1 import validate_config_matrix  # noqa: E402
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    validate_bound_s1_training_config,
)


def build_descriptor(args: argparse.Namespace) -> dict:
    matrix = validate_config_matrix()
    cfg = Config.fromfile(str(args.config))
    binding = validate_bound_s1_training_config(cfg, seed=int(args.seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("formal S1 descriptor requires the bound full precheck")
    recovery_path = args.profile_recovery_certificate.resolve()
    recovery = load_profile_recovery_certificate(
        recovery_path,
        binding=binding,
        verify_checkout=True,
    )
    sidecar_gate = load_sidecar_gate_evidence(
        sidecar_gate_path(recovery),
        recovery=recovery,
    )
    resolution = int(cfg.spatial_zoom_s1_contract.runtime_resolution)
    canonical_profile_prefix = profile_campaign_prefix(
        recovery, resolution=resolution, seed=int(args.seed)
    )
    expected_profile_path = canonical_profile_prefix.with_suffix(".summary.json")
    if args.profile.resolve() != expected_profile_path:
        raise ValueError("S1 descriptor profile is outside the recovery campaign")
    expected_descriptor_path = (
        Path(recovery["campaign_root"])
        / "descriptors"
        / f"dense{resolution}_seed{int(args.seed)}.run.json"
    ).resolve()
    if args.output.resolve() != expected_descriptor_path:
        raise ValueError("S1 run descriptor output is outside the recovery campaign")
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
    matrix_start = validate_profile_matrix_start_receipt(
        args.matrix_start_receipt,
        recovery=recovery,
    )
    legacy_unbound_test_evidence = (
        resolution == int(recovery["legacy_unbound_test_resolution"])
        and int(args.seed) == int(recovery["legacy_unbound_test_seed"])
        and args.test_evidence.resolve()
        == Path(recovery["legacy_unbound_test_evidence_path"]).resolve()
        and sha256_file(args.test_evidence)
        == recovery["legacy_unbound_test_evidence_file_sha256"]
        and test_evidence["evidence_sha256"]
        == recovery["legacy_unbound_test_evidence_sha256"]
    )
    test_matrix_binding_path = canonical_test_matrix_binding_path(
        args.test_evidence
    )
    if legacy_unbound_test_evidence:
        if test_matrix_binding_path.exists():
            raise RuntimeError(
                "legacy S1 test evidence unexpectedly has a matrix binding"
            )
        test_matrix_binding = None
    else:
        test_matrix_binding = validate_test_matrix_binding(
            test_matrix_binding_path,
            test_evidence_path=args.test_evidence,
            start_receipt_path=args.matrix_start_receipt,
            recovery=recovery,
            resolution=resolution,
            seed=int(args.seed),
        )
    if profile.get("schema_version") != S1_PROFILE_SCHEMA:
        raise ValueError("S1 run descriptor requires an S1 full-stack profile")
    profile_samples_path = canonical_profile_prefix.with_suffix(".samples.jsonl")
    profile_power_path = canonical_profile_prefix.with_suffix(".power.jsonl")
    profile_power_attempt_path = Path(
        f"{canonical_profile_prefix}.power_attempt.json"
    )
    profile_power_attempt_trace_path = Path(
        f"{canonical_profile_prefix}.power_attempt.jsonl"
    )
    for artifact_path, expected_hash in (
        (profile_samples_path, profile["sample_trace_file_sha256"]),
        (profile_power_path, profile["power_trace_file_sha256"]),
        (
            profile_power_attempt_path,
            profile["power_attempt_report_file_sha256"],
        ),
        (
            profile_power_attempt_trace_path,
            profile["power_attempt_trace_file_sha256"],
        ),
    ):
        if not artifact_path.is_file() or sha256_file(artifact_path) != expected_hash:
            raise ValueError(f"S1 profile trace mismatch: {artifact_path}")
    power_attempt = validate_nvml_sidecar_attempt(
        profile_power_attempt_path,
        profile_power_attempt_trace_path,
        expected_uuid=profile["hardware_identity"]["nvidia_smi"]["uuid"],
        require_pass=True,
    )
    if (
        power_attempt["attempt_sha256"] != profile["power_attempt_sha256"]
        or power_attempt["trace_file_sha256"]
        != profile["power_attempt_trace_file_sha256"]
    ):
        raise ValueError("S1 sidecar attempt report identity mismatch")
    validate_sidecar_gate_runtime_identity(
        sidecar_gate,
        hardware_identity=profile["hardware_identity"],
        software_fingerprint=profile["software_fingerprint"],
    )
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
        "profile_code_commit": recovery["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "protocol_fingerprint": matrix["protocol_fingerprint"],
        "manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "test_open_certificate_sha256": test_evidence["test_open_certificate_sha256"],
        "test_evidence_sha256": test_evidence["evidence_sha256"],
        "legacy_unbound_test_evidence": legacy_unbound_test_evidence,
        "test_matrix_binding_sha256": (
            None
            if test_matrix_binding is None
            else test_matrix_binding["binding_sha256"]
        ),
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "hardware_fingerprint": profile["hardware_fingerprint"],
        "software_fingerprint": profile["software_fingerprint"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
        "canonical_output_prefix": str(canonical_output_prefix),
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "gate_only": False,
        "power_sampler_backend": recovery["power_sampler_backend"],
        "allocated_cpu_ids": profile["allocated_cpu_ids"],
        "detector_cpu_ids": profile["detector_cpu_ids"],
        "sidecar_cpu_id": profile["sidecar_cpu_id"],
        "sidecar_gate_evidence_path": str(sidecar_gate_path(recovery)),
        "sidecar_gate_sha256": sidecar_gate["gate_sha256"],
        "matrix_start_receipt_path": str(args.matrix_start_receipt.resolve()),
        "matrix_start_receipt_file_sha256": sha256_file(
            args.matrix_start_receipt
        ),
        "matrix_sha256": matrix_start["matrix_sha256"],
        "slurm_job_id": matrix_start["slurm_job_id"],
        "slurm_step_id": matrix_start["slurm_step_id"],
        "step_gpu_uuid": matrix_start["step_gpu_uuid"],
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
        "legacy_unbound_test_evidence": legacy_unbound_test_evidence,
        "test_matrix_binding_path": (
            None
            if test_matrix_binding is None
            else str(test_matrix_binding_path)
        ),
        "test_matrix_binding_file_sha256": (
            None
            if test_matrix_binding is None
            else sha256_file(test_matrix_binding_path)
        ),
        "test_matrix_binding_sha256": (
            None
            if test_matrix_binding is None
            else test_matrix_binding["binding_sha256"]
        ),
        "test_open_marker_sha256": test_evidence["test_open_marker_sha256"],
        "config_commit": binding["code_commit"],
        "profile_code_commit": recovery["profile_code_commit"],
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
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "power_sampler_backend": recovery["power_sampler_backend"],
        "sidecar_gate_evidence_path": str(sidecar_gate_path(recovery)),
        "sidecar_gate_evidence_file_sha256": sha256_file(
            sidecar_gate_path(recovery)
        ),
        "sidecar_gate_sha256": sidecar_gate["gate_sha256"],
        "matrix_start_receipt_path": str(args.matrix_start_receipt.resolve()),
        "matrix_start_receipt_file_sha256": sha256_file(
            args.matrix_start_receipt
        ),
        "matrix_sha256": matrix_start["matrix_sha256"],
        "slurm_job_id": matrix_start["slurm_job_id"],
        "slurm_step_id": matrix_start["slurm_step_id"],
        "step_gpu_uuid": matrix_start["step_gpu_uuid"],
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
        "schema_version": "spatial_zoom_s1_run_v8",
        "resolution": resolution,
        "seed": int(args.seed),
        "config_path": str(args.config.resolve()),
        "resolved_config_sha256": canonical_sha256(cfg.to_dict()),
        "code_commit": binding["code_commit"],
        "profile_code_commit": recovery["profile_code_commit"],
        "experiment_namespace": binding["experiment_namespace"],
        "canonical_experiment_root": binding["canonical_experiment_root"],
        "precheck_file_sha256": binding["precheck_file_sha256"],
        "precheck_sha256": binding["precheck_sha256"],
        "pretrained_checkpoint_sha256": binding["pretrained_checkpoint_sha256"],
        "profile_order_seed": S1_PROFILE_ORDER_SEED,
        "profile_order_sha256": profile_order_sha256,
        "profile_order_ordinal": int(profile_order_entry["ordinal"]),
        "profile_recovery_certificate_path": str(recovery_path),
        "profile_recovery_certificate_file_sha256": sha256_file(recovery_path),
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
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
        "legacy_unbound_test_evidence": legacy_unbound_test_evidence,
        "test_matrix_binding_path": (
            None
            if test_matrix_binding is None
            else str(test_matrix_binding_path)
        ),
        "test_matrix_binding_file_sha256": (
            None
            if test_matrix_binding is None
            else sha256_file(test_matrix_binding_path)
        ),
        "test_matrix_binding_sha256": (
            None
            if test_matrix_binding is None
            else test_matrix_binding["binding_sha256"]
        ),
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
        "profile_samples_path": str(profile_samples_path),
        "profile_samples_sha256": sha256_file(profile_samples_path),
        "profile_power_path": str(profile_power_path),
        "profile_power_sha256": sha256_file(profile_power_path),
        "profile_power_attempt_path": str(profile_power_attempt_path),
        "profile_power_attempt_file_sha256": sha256_file(
            profile_power_attempt_path
        ),
        "profile_power_attempt_sha256": profile["power_attempt_sha256"],
        "profile_power_attempt_trace_path": str(
            profile_power_attempt_trace_path
        ),
        "profile_power_attempt_trace_sha256": sha256_file(
            profile_power_attempt_trace_path
        ),
        "sidecar_gate_evidence_path": str(sidecar_gate_path(recovery)),
        "sidecar_gate_evidence_file_sha256": sha256_file(
            sidecar_gate_path(recovery)
        ),
        "sidecar_gate_sha256": sidecar_gate["gate_sha256"],
        "profile_attempt_marker_path": str(marker_path),
        "profile_attempt_marker_file_sha256": marker_file_sha,
        "profile_attempt_marker_sha256": marker["marker_sha256"],
        "matrix_start_receipt_path": str(args.matrix_start_receipt.resolve()),
        "matrix_start_receipt_file_sha256": sha256_file(
            args.matrix_start_receipt
        ),
        "matrix_sha256": matrix_start["matrix_sha256"],
        "slurm_job_id": matrix_start["slurm_job_id"],
        "slurm_step_id": matrix_start["slurm_step_id"],
        "step_gpu_uuid": matrix_start["step_gpu_uuid"],
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
    parser.add_argument("--profile-recovery-certificate", type=Path, required=True)
    parser.add_argument("--matrix-start-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite an S1 run descriptor")
        descriptor = build_descriptor(args)
    except Exception as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)},
                indent=2,
            )
        )
        return 1
    atomic_publish_json(args.output, descriptor)
    print(json.dumps({"status": "PASS", "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
