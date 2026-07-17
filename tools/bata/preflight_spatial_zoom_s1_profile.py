from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
    _dataset_exposure_topology,
    _dataset_video_ids,
    _hardware_identity,
    _cpu_ids,
    _software_identity,
    validate_profile_order_ready,
)
from tools.bata.spatial_zoom_s1_evidence import (  # noqa: E402
    validate_s1_test_evidence,
)
from tools.bata.spatial_zoom_s1_profile_recovery import (  # noqa: E402
    load_profile_recovery_certificate,
)
from tools.bata.spatial_zoom_s1_matrix import (  # noqa: E402
    canonical_test_matrix_binding_path,
    validate_profile_matrix_start_receipt,
    validate_test_matrix_binding,
)
from tools.bata.spatial_zoom_s1_sidecar_gate import (  # noqa: E402
    load_sidecar_gate_evidence,
    validate_sidecar_gate_runtime_identity,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    canonical_sha256,
    sha256_file,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_test_open import (  # noqa: E402
    validate_test_open_certificate,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_slurm_memory_limit_mb,
    require_slurm_single_gpu_allocation,
    validate_bound_s1_training_config,
    validate_s1_checkpoint_sidecar,
)


def run_profile_preflight(
    *,
    config_path: str | Path,
    seed: int,
    manifest_path: str | Path,
    annotation_path: str | Path,
    checkpoint_path: str | Path,
    certificate_path: str | Path,
    profile_recovery_certificate_path: str | Path,
    sidecar_gate_evidence_path: str | Path,
    test_evidence_path: str | Path,
    allocated_cpus: str,
    detector_cpus: str,
    sidecar_cpu: int,
    matrix_start_receipt_path: str | Path | None = None,
    matrix_dry_run: bool = False,
) -> dict[str, object]:
    cfg = Config.fromfile(str(Path(config_path).resolve()))
    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("S1 profile preflight requires a full-precheck binding")
    physical_gpu_id = require_slurm_single_gpu_allocation()
    memory_limit_mb = require_slurm_memory_limit_mb(minimum_mb=90000)
    recovery_path = Path(profile_recovery_certificate_path).resolve()
    recovery = load_profile_recovery_certificate(
        recovery_path,
        binding=binding,
        verify_checkout=True,
    )
    sidecar_gate = load_sidecar_gate_evidence(
        sidecar_gate_evidence_path,
        recovery=recovery,
    )
    allocated_cpu_ids = _cpu_ids(allocated_cpus)
    detector_cpu_ids = _cpu_ids(detector_cpus)
    sidecar_cpu_id = int(sidecar_cpu)
    if (
        len(allocated_cpu_ids) != int(recovery["allocated_cpu_count"])
        or len(detector_cpu_ids) != int(recovery["detector_cpu_count"])
        or sidecar_cpu_id in detector_cpu_ids
        or set(detector_cpu_ids) | {sidecar_cpu_id} != set(allocated_cpu_ids)
    ):
        raise ValueError("S1 profile preflight CPU partition violates the recovery")
    if (
        int(os.environ.get("SLURM_CPUS_PER_TASK", -1))
        != int(recovery["allocated_cpu_count"])
        or not hasattr(os, "sched_getaffinity")
        or tuple(sorted(os.sched_getaffinity(0))) != detector_cpu_ids
    ):
        raise RuntimeError("S1 profile preflight lacks its four-CPU affinity")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("S1 profile preflight requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    hardware_identity = _hardware_identity(
        torch,
        device,
        physical_gpu_id=physical_gpu_id,
        allocated_cpu_ids=allocated_cpu_ids,
        detector_cpu_ids=detector_cpu_ids,
        sidecar_cpu_id=sidecar_cpu_id,
        memory_limit_mb=memory_limit_mb,
    )
    software_identity = _software_identity(torch)
    hardware_fingerprint = canonical_sha256(hardware_identity)
    software_fingerprint = canonical_sha256(software_identity)
    validate_sidecar_gate_runtime_identity(
        sidecar_gate,
        hardware_identity=hardware_identity,
        software_fingerprint=software_fingerprint,
    )
    if matrix_dry_run:
        if matrix_start_receipt_path is not None:
            raise ValueError("S1 matrix dry-run cannot consume a start receipt")
        matrix_start = None
    else:
        if matrix_start_receipt_path is None:
            raise ValueError("S1 profile preflight requires a matrix start receipt")
        matrix_start = validate_profile_matrix_start_receipt(
            matrix_start_receipt_path,
            recovery=recovery,
            verify_runtime=True,
            hardware_identity=hardware_identity,
            software_fingerprint=software_fingerprint,
            effective_memory_limit_mb=memory_limit_mb,
        )

    manifest_path = Path(manifest_path).resolve()
    annotation_path = Path(annotation_path).resolve()
    if str(manifest_path) != binding["manifest_path"]:
        raise ValueError("S1 profile preflight manifest differs from the binding")
    if str(annotation_path) != binding["annotation_path"]:
        raise ValueError("S1 profile preflight annotation differs from the binding")
    manifest = validate_s1_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        annotation_path=annotation_path,
    )

    checkpoint_path = Path(checkpoint_path).resolve()
    validate_s1_checkpoint_sidecar(checkpoint_path)
    certificate_path = Path(certificate_path).resolve()
    certificate = validate_test_open_certificate(
        json.loads(certificate_path.read_text(encoding="utf-8")),
        cfg=cfg,
        seed=int(seed),
        checkpoint_path=checkpoint_path,
    )
    test_evidence_path = Path(test_evidence_path).resolve()
    canonical_test_evidence = (
        Path(binding["work_dir"]) / "gpu1_id0" / "test_evidence" / "test.evidence.json"
    ).resolve()
    if test_evidence_path != canonical_test_evidence:
        raise ValueError("S1 profile preflight test-evidence path is non-canonical")
    reuse_test_evidence = test_evidence_path.is_file()
    test_matrix_binding_path = canonical_test_matrix_binding_path(
        test_evidence_path
    )
    legacy_unbound_test_evidence = False
    test_matrix_binding = None
    if reuse_test_evidence:
        evidence = validate_s1_test_evidence(
            json.loads(test_evidence_path.read_text(encoding="utf-8")),
            cfg=cfg,
            seed=int(seed),
        )
        if (
            Path(evidence["checkpoint_path"]).resolve() != checkpoint_path
            or evidence["test_open_certificate_sha256"]
            != certificate["certificate_sha256"]
        ):
            raise ValueError("reused S1 test evidence differs from the frozen cell")
        legacy_unbound_test_evidence = (
            int(binding["resolution"])
            == int(recovery["legacy_unbound_test_resolution"])
            and int(seed) == int(recovery["legacy_unbound_test_seed"])
            and test_evidence_path
            == Path(recovery["legacy_unbound_test_evidence_path"]).resolve()
            and sha256_file(test_evidence_path)
            == recovery["legacy_unbound_test_evidence_file_sha256"]
            and evidence["evidence_sha256"]
            == recovery["legacy_unbound_test_evidence_sha256"]
        )
        if legacy_unbound_test_evidence:
            if test_matrix_binding_path.exists():
                raise RuntimeError(
                    "legacy S1 test evidence unexpectedly has a matrix binding"
                )
        elif matrix_dry_run:
            raise RuntimeError(
                "S1 matrix dry-run found non-legacy test evidence from another matrix"
            )
        else:
            test_matrix_binding = validate_test_matrix_binding(
                test_matrix_binding_path,
                test_evidence_path=test_evidence_path,
                start_receipt_path=matrix_start_receipt_path,
                recovery=recovery,
                resolution=int(binding["resolution"]),
                seed=int(seed),
            )
    else:
        test_root = canonical_test_evidence.parents[1]
        partial_paths = (
            test_root / "test_open_started.json",
            canonical_test_evidence.parent / "result_detection.json",
            test_matrix_binding_path,
        )
        if any(path.exists() for path in partial_paths):
            raise RuntimeError(
                "S1 sealed test was opened without complete evidence; refusing a rerun"
            )

    from opentad.datasets import build_dataset

    dataset_cfg = copy.deepcopy(cfg.dataset.test)
    dataset_cfg.test_mode = True
    dataset_cfg.subset_name = manifest["annotation_subsets"]["sealed_test"]
    dataset_cfg.block_list = None
    dataset = build_dataset(dataset_cfg)
    if _dataset_video_ids(dataset) != set(manifest["splits"]["test"]):
        raise ValueError("S1 profile preflight dataset split mismatch")
    topology = _dataset_exposure_topology(dataset)
    topology_expected = {
        "loader_exposure_count": recovery["expected_loader_exposure_count"],
        "physical_window_count": recovery["expected_physical_window_count"],
        "duplicate_physical_window_ids": recovery[
            "expected_duplicate_physical_window_ids"
        ],
    }
    for key, expected in topology_expected.items():
        if topology[key] != expected:
            raise ValueError(f"S1 profile preflight measured unexpected {key}")
    resolution = int(binding["resolution"])
    cell, order_sha256 = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=resolution,
        seed=int(seed),
        hardware_fingerprint=hardware_fingerprint,
        software_fingerprint=software_fingerprint,
        campaign_root=recovery["campaign_root"],
        profile_code_commit=recovery["profile_code_commit"],
        profile_recovery_certificate_sha256=recovery["certificate_sha256"],
        profile_recovery_campaign_id=recovery["campaign_id"],
        matrix_dry_run=bool(matrix_dry_run),
    )
    return {
        "status": "PASS",
        "resolution": resolution,
        "seed": int(seed),
        "profile_order_ordinal": int(cell["ordinal"]),
        "profile_order_sha256": order_sha256,
        "hardware_fingerprint": hardware_fingerprint,
        "software_fingerprint": software_fingerprint,
        "experiment_namespace": binding["experiment_namespace"],
        "test_open_certificate_sha256": certificate["certificate_sha256"],
        "profile_recovery_certificate_sha256": recovery["certificate_sha256"],
        "profile_recovery_campaign_id": recovery["campaign_id"],
        "sidecar_gate_sha256": sidecar_gate["gate_sha256"],
        "reuse_test_evidence": reuse_test_evidence,
        "legacy_unbound_test_evidence": legacy_unbound_test_evidence,
        "test_matrix_binding_sha256": (
            None
            if test_matrix_binding is None
            else test_matrix_binding["binding_sha256"]
        ),
        "matrix_sha256": (
            None if matrix_start is None else matrix_start["matrix_sha256"]
        ),
        "loader_exposure_count": topology["loader_exposure_count"],
        "physical_window_count": topology["physical_window_count"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed on S1 profile order before opening this cell's test"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--test-open-certificate", type=Path, required=True)
    parser.add_argument("--profile-recovery-certificate", type=Path, required=True)
    parser.add_argument("--sidecar-gate-evidence", type=Path, required=True)
    parser.add_argument("--test-evidence", type=Path, required=True)
    parser.add_argument("--allocated-cpus", required=True)
    parser.add_argument("--detector-cpus", required=True)
    parser.add_argument("--sidecar-cpu", type=int, required=True)
    parser.add_argument("--matrix-start-receipt", type=Path)
    parser.add_argument("--matrix-dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_profile_preflight(
            config_path=args.config,
            seed=args.seed,
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            checkpoint_path=args.checkpoint,
            certificate_path=args.test_open_certificate,
            profile_recovery_certificate_path=args.profile_recovery_certificate,
            sidecar_gate_evidence_path=args.sidecar_gate_evidence,
            test_evidence_path=args.test_evidence,
            allocated_cpus=args.allocated_cpus,
            detector_cpus=args.detector_cpus,
            sidecar_cpu=args.sidecar_cpu,
            matrix_start_receipt_path=args.matrix_start_receipt,
            matrix_dry_run=args.matrix_dry_run,
        )
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "FAIL",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
            )
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
