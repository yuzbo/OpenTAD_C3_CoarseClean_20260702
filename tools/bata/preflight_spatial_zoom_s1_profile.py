from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from mmengine.config import Config

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.profile_spatial_zoom_s1 import (  # noqa: E402
    _hardware_identity,
    _software_identity,
    validate_profile_order_ready,
)
from tools.bata.spatial_zoom_s1_contract import (  # noqa: E402
    canonical_sha256,
    validate_s1_manifest,
)
from tools.bata.spatial_zoom_s1_test_open import (  # noqa: E402
    validate_test_open_certificate,
)
from tools.bata.spatial_zoom_s1_training import (  # noqa: E402
    require_clean_git_checkout,
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
) -> dict[str, object]:
    cfg = Config.fromfile(str(Path(config_path).resolve()))
    binding = validate_bound_s1_training_config(cfg, seed=int(seed))
    if not binding["formal_precheck_verified"]:
        raise RuntimeError("S1 profile preflight requires a full-precheck binding")
    physical_gpu_id = require_slurm_single_gpu_allocation()
    require_clean_git_checkout(expected_commit=binding["code_commit"])

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("S1 profile preflight requires CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    hardware_fingerprint = canonical_sha256(
        _hardware_identity(torch, device, physical_gpu_id=physical_gpu_id)
    )
    software_fingerprint = canonical_sha256(_software_identity(torch))

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
    resolution = int(binding["resolution"])
    cell, order_sha256 = validate_profile_order_ready(
        manifest=manifest,
        binding=binding,
        resolution=resolution,
        seed=int(seed),
        hardware_fingerprint=hardware_fingerprint,
        software_fingerprint=software_fingerprint,
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
    args = parser.parse_args(argv)
    try:
        result = run_profile_preflight(
            config_path=args.config,
            seed=args.seed,
            manifest_path=args.manifest,
            annotation_path=args.annotation,
            checkpoint_path=args.checkpoint,
            certificate_path=args.test_open_certificate,
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
