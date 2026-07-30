#!/usr/bin/env python
"""Real-device G0: official dense versus identity-scaffold DCSR routing."""

import argparse
import hashlib
import json
import os
import subprocess

import torch

from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.utils import fix_random_seed


def _git(*args):
    return subprocess.check_output(
        ["git"] + list(args), text=True
    ).strip()


def _tensor_tuple_equal(first, second):
    return len(first) == len(second) and all(
        torch.equal(left, right) for left, right in zip(first, second)
    )


def _capture_forward(model, video):
    captured = {}
    original_inference = model.inference

    def capture_inference(
        video_list,
        points,
        fpn_masks,
        out_cls_logits,
        out_offsets,
    ):
        captured["points"] = tuple(tensor.detach().clone() for tensor in points)
        captured["fpn_masks"] = tuple(
            tensor.detach().clone() for tensor in fpn_masks
        )
        captured["out_cls_logits"] = tuple(
            tensor.detach().clone() for tensor in out_cls_logits
        )
        captured["out_offsets"] = tuple(
            tensor.detach().clone() for tensor in out_offsets
        )
        return original_inference(
            video_list,
            points,
            fpn_masks,
            out_cls_logits,
            out_offsets,
        )

    model.inference = capture_inference
    with torch.no_grad():
        final = model([video])
    model.inference = original_inference
    return captured, final


def _final_equal(first, second):
    if len(first) != len(second):
        return False
    for left, right in zip(first, second):
        if left["video_id"] != right["video_id"]:
            return False
        for key in ("segments", "scores", "labels"):
            if not torch.equal(left[key], right[key]):
                return False
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-config",
        default="configs/thumos_i3d.yaml",
    )
    parser.add_argument(
        "--g0-config",
        default="configs/thumos_i3d_dcsr_g0_identity.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234567891)
    parser.add_argument("--feature-length", type=int, default=257)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite G0 receipt")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("G0 formal gate requires real CUDA")

    fix_random_seed(args.seed, include_cuda=True)
    official_cfg = load_config(args.official_config)
    g0_cfg = load_config(args.g0_config)
    official = make_meta_arch(
        official_cfg["model_name"], **official_cfg["model"]
    )
    g0 = make_meta_arch(g0_cfg["model_name"], **g0_cfg["model"])
    state_keys_equal = (
        tuple(official.state_dict()) == tuple(g0.state_dict())
    )
    if not state_keys_equal:
        raise RuntimeError("G0 state-dict key identity failed")
    g0.load_state_dict(official.state_dict(), strict=True)
    official = official.to(args.device).eval()
    g0 = g0.to(args.device).eval()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    features = torch.randn(
        official_cfg["dataset"]["input_dim"],
        args.feature_length,
        generator=generator,
    )
    video = {
        "video_id": "dcsr_g0_synthetic_validation_only",
        "feats": features,
        "segments": None,
        "labels": None,
        "fps": 30.0,
        "duration": 120.0,
        "feat_stride": official_cfg["dataset"]["feat_stride"],
        "feat_num_frames": official_cfg["dataset"]["num_frames"],
    }
    official_capture, official_final = _capture_forward(official, video)
    g0_capture, g0_final = _capture_forward(g0, video)

    checks = {
        "state_dict_keys_exact": state_keys_equal,
        "points_exact": _tensor_tuple_equal(
            official_capture["points"], g0_capture["points"]
        ),
        "fpn_masks_exact": _tensor_tuple_equal(
            official_capture["fpn_masks"], g0_capture["fpn_masks"]
        ),
        "predecode_cls_logits_exact": _tensor_tuple_equal(
            official_capture["out_cls_logits"],
            g0_capture["out_cls_logits"],
        ),
        "predecode_offsets_exact": _tensor_tuple_equal(
            official_capture["out_offsets"],
            g0_capture["out_offsets"],
        ),
        "decoded_soft_nms_timestamp_outputs_exact": _final_equal(
            official_final, g0_final
        ),
    }
    gate_pass = all(checks.values())
    receipt = {
        "schema_version": "actionformer_dcsr_g0_equivalence_v1",
        "gate_pass": gate_pass,
        "checks": checks,
        "device": args.device,
        "cuda_device_name": torch.cuda.get_device_name(
            torch.device(args.device)
        ),
        "seed": args.seed,
        "feature_length": args.feature_length,
        "official_config": os.path.realpath(args.official_config),
        "g0_config": os.path.realpath(args.g0_config),
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_clean": _git("status", "--porcelain=v1") == "",
        "test_gt_used": False,
        "test_predictions_used": False,
        "model_selection_performed": False,
        "metric_claim_allowed": False,
        "efficiency_claim_allowed": False,
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    temporary_path = args.output + ".tmp"
    with open(temporary_path, "x", encoding="utf-8") as fid:
        json.dump(receipt, fid, indent=2, sort_keys=True)
        fid.write("\n")
    os.replace(temporary_path, args.output)
    with open(args.output, "rb") as fid:
        receipt_sha256 = hashlib.sha256(fid.read()).hexdigest()
    print(
        json.dumps(
            {
                "gate_pass": gate_pass,
                "receipt": os.path.realpath(args.output),
                "receipt_sha256": receipt_sha256,
            },
            sort_keys=True,
        )
    )
    if not gate_pass:
        raise RuntimeError("DCSR G0 exact-equivalence gate failed")


if __name__ == "__main__":
    main()
