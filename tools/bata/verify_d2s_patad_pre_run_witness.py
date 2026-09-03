from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[2]
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import torch
from mmengine.config import Config

import opentad.datasets  # noqa: F401 - registers config-driven data transforms
from opentad.models import build_detector
from tools.bata.continuous_roi_s2_v3_full200_compute import canonical_sha256
from tools.bata.zoomtoken_full200_matrix_spec import (
    binding_from_config,
    get_matrix_spec,
    validate_matrix_cell,
)
from tools.bata.zoomtoken_batch_device import prepare_zoomtoken_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Runtime witness for D2S/PA-TAD")
    parser.add_argument("config", type=Path)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--matrix-kind", choices=("d2s", "patad"), required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if "SLURM_JOB_ID" not in os.environ or not torch.cuda.is_available():
        raise RuntimeError("D2S/PA-TAD runtime witness requires a Slurm GPU allocation")
    spec = get_matrix_spec()
    if spec.key != args.matrix_kind:
        raise ValueError("matrix-kind differs from ZOOMTOKEN_MATRIX_KIND")
    if not args.pretrained.is_file():
        raise FileNotFoundError(args.pretrained)

    cfg = Config.fromfile(args.config)
    binding = binding_from_config(cfg, spec)
    validate_matrix_cell(
        args.config, arm=str(binding.arm), seed=int(binding.seed), spec=spec
    )
    model_cfg = copy.deepcopy(cfg.model)
    model_cfg.backbone.custom.pretrain = str(args.pretrained.resolve())
    model = build_detector(model_cfg).to("cuda:0").eval()

    global_view = torch.zeros(
        1, 1, 3, 768, 96, 96, dtype=torch.uint8, device="cuda:0"
    )
    source_view = torch.zeros(
        1, 1, 3, 768, 180, 320, dtype=torch.uint8, device="cuda:0"
    )
    prepared = prepare_zoomtoken_batch(
        {
            "inputs": {"global": global_view, "source": source_view},
            "masks": torch.ones(1, 768, dtype=torch.bool),
        },
        torch.device("cuda", 0),
    )
    if prepared["inputs"]["source"].device.type != "cpu":
        raise RuntimeError("runtime witness moved source-native video off CPU")
    if prepared["inputs"]["global"].device.type != "cuda":
        raise RuntimeError("runtime witness did not move the global view to CUDA")
    masks = prepared["masks"]
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        backbone_output = model.backbone(
            prepared["inputs"], masks=masks
        )
        model._assert_feature_mask_temporal_match(
            backbone_output, masks, "runtime witness"
        )
        padded, padded_masks = model.pad_data(backbone_output, masks)
        projected, projected_masks = model.projection(padded, padded_masks)

    wrapper = model.backbone
    audit = dict(wrapper.latest_d2s_audit or {})
    if audit.get("local_chunks_executed") != 16 or audit.get("local_chunks_skipped") != 32:
        raise RuntimeError("runtime physical-skip counters differ from 16/32")
    if len(projected) != 6 or len(projected_masks) != 6:
        raise RuntimeError("runtime projection did not produce six pyramid levels")
    if spec.key == "d2s" and not torch.is_tensor(backbone_output):
        raise RuntimeError("plain D2S must return a fused feature tensor")
    if spec.key == "patad" and not isinstance(backbone_output, dict):
        raise RuntimeError("PA-TAD must receive the explicit G/R feature bundle")

    payload = {
        "schema_version": "d2s_patad_runtime_witness_v1",
        "protocol_id": spec.protocol_id,
        "matrix_kind": spec.key,
        "arm": str(binding.arm),
        "seed": int(binding.seed),
        "backbone_class": type(wrapper).__name__,
        "projection_class": type(model.projection).__name__,
        "pyramid_shapes": [list(value.shape) for value in projected],
        "physical_skip_audit": audit,
        "status": "PASS",
    }
    payload["receipt_sha256"] = canonical_sha256(payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
