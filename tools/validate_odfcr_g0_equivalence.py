#!/usr/bin/env python
"""Real-CUDA zero-tolerance implementation gate for the ODF-CR factorial."""

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
    return subprocess.check_output(["git"] + list(args), text=True).strip()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fid:
        for chunk in iter(lambda: fid.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_tensor(tensor):
    digest = hashlib.sha256()
    contiguous = tensor.detach().cpu().contiguous()
    digest.update(str(contiguous.dtype).encode("utf-8"))
    digest.update(str(tuple(contiguous.shape)).encode("utf-8"))
    digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def _tensor_tuple_equal(first, second):
    return len(first) == len(second) and all(
        torch.equal(left, right) for left, right in zip(first, second)
    )


def _state_equal(first, second):
    return tuple(first) == tuple(second) and all(
        torch.equal(first[key], second[key]) for key in first
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
        captured["points"] = tuple(
            tensor.detach().clone() for tensor in points
        )
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


def _build(config_path, seed):
    fix_random_seed(seed, include_cuda=True)
    cfg = load_config(config_path)
    return cfg, make_meta_arch(cfg["model_name"], **cfg["model"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--official-config",
        default="configs/thumos_i3d_odfcr_dev_dense_reference.yaml",
    )
    parser.add_argument(
        "--d1-off-config",
        default="configs/thumos_i3d_odfcr_dev_d1_off.yaml",
    )
    parser.add_argument(
        "--d1-all-config",
        default="configs/thumos_i3d_odfcr_dev_d1_all.yaml",
    )
    parser.add_argument(
        "--d3-off-config",
        default="configs/thumos_i3d_odfcr_dev_d3_off.yaml",
    )
    parser.add_argument(
        "--d3-all-config",
        default="configs/thumos_i3d_odfcr_dev_d3_all.yaml",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026073101)
    parser.add_argument("--feature-length", type=int, default=257)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if os.path.exists(args.output):
        raise FileExistsError("refusing to overwrite ODF-CR G0 receipt")
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("ODF-CR G0 formal gate requires real CUDA")
    git_clean = _git("status", "--porcelain=v1") == ""
    if not git_clean:
        raise RuntimeError("ODF-CR G0 requires a clean exact source tree")

    official_cfg, official = _build(args.official_config, args.seed)
    _, d3_off = _build(args.d3_off_config, args.seed)
    _, d3_all = _build(args.d3_all_config, args.seed)
    _, d1_off = _build(args.d1_off_config, args.seed)
    _, d1_all = _build(args.d1_all_config, args.seed)

    official_state = official.state_dict()
    d3_off_state = d3_off.state_dict()
    d3_off_state_exact_before_load = _state_equal(
        official_state, d3_off_state
    )
    if not d3_off_state_exact_before_load:
        raise RuntimeError("d3_off state tensors are not official-exact")
    d3_off.load_state_dict(official_state, strict=True)
    d3_all_floor_state_exact = (
        _state_equal(
            official.cls_head.state_dict(),
            d3_all.cls_head.state_dict(),
        )
        and _state_equal(
            official.reg_head.state_dict(),
            d3_all.reg_head.state_dict(),
        )
    )
    d1_scaffold_state_exact = (
        _state_equal(
            d1_off.odfcr_scaffold_cls_head.state_dict(),
            d1_all.odfcr_scaffold_cls_head.state_dict(),
        )
        and _state_equal(
            d1_off.odfcr_scaffold_reg_head.state_dict(),
            d1_all.odfcr_scaffold_reg_head.state_dict(),
        )
    )
    residual_final_zero = all(
        torch.count_nonzero(tensor).item() == 0
        for tensor in (
            d3_all.odfcr_residual_cls_head.cls_head.conv.weight,
            d3_all.odfcr_residual_cls_head.cls_head.conv.bias,
            d3_all.odfcr_residual_reg_head.offset_head.conv.weight,
            d3_all.odfcr_residual_reg_head.offset_head.conv.bias,
            d1_all.odfcr_residual_cls_head.cls_head.conv.weight,
            d1_all.odfcr_residual_cls_head.cls_head.conv.bias,
            d1_all.odfcr_residual_reg_head.offset_head.conv.weight,
            d1_all.odfcr_residual_reg_head.offset_head.conv.bias,
        )
    )

    official = official.to(args.device).eval()
    d3_off = d3_off.to(args.device).eval()
    d3_all = d3_all.to(args.device).eval()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    features = torch.randn(
        official_cfg["dataset"]["input_dim"],
        args.feature_length,
        generator=generator,
        dtype=torch.float32,
    )
    video = {
        "video_id": "odfcr_g0_synthetic_validation_only",
        "feats": features,
        "segments": None,
        "labels": None,
        "fps": 30.0,
        "duration": 120.0,
        "feat_stride": official_cfg["dataset"]["feat_stride"],
        "feat_num_frames": official_cfg["dataset"]["num_frames"],
    }
    official_capture, official_final = _capture_forward(official, video)
    d3_off_capture, d3_off_final = _capture_forward(d3_off, video)
    d3_all_capture, d3_all_final = _capture_forward(d3_all, video)

    checks = {
        "d3_off_state_keys_and_tensors_exact": (
            d3_off_state_exact_before_load
        ),
        "d3_all_floor_tensors_exact": d3_all_floor_state_exact,
        "d1_off_all_scaffold_tensors_exact": d1_scaffold_state_exact,
        "residual_final_projections_exact_zero": residual_final_zero,
        "d3_off_points_exact": _tensor_tuple_equal(
            official_capture["points"], d3_off_capture["points"]
        ),
        "d3_off_fpn_masks_exact": _tensor_tuple_equal(
            official_capture["fpn_masks"], d3_off_capture["fpn_masks"]
        ),
        "d3_off_predecode_cls_exact": _tensor_tuple_equal(
            official_capture["out_cls_logits"],
            d3_off_capture["out_cls_logits"],
        ),
        "d3_off_predecode_offsets_exact": _tensor_tuple_equal(
            official_capture["out_offsets"],
            d3_off_capture["out_offsets"],
        ),
        "d3_off_final_outputs_exact": _final_equal(
            official_final, d3_off_final
        ),
        "d3_all_initial_points_exact": _tensor_tuple_equal(
            official_capture["points"], d3_all_capture["points"]
        ),
        "d3_all_initial_masks_exact": _tensor_tuple_equal(
            official_capture["fpn_masks"], d3_all_capture["fpn_masks"]
        ),
        "d3_all_initial_predecode_cls_exact": _tensor_tuple_equal(
            official_capture["out_cls_logits"],
            d3_all_capture["out_cls_logits"],
        ),
        "d3_all_initial_predecode_offsets_exact": _tensor_tuple_equal(
            official_capture["out_offsets"],
            d3_all_capture["out_offsets"],
        ),
        "d3_all_initial_final_outputs_exact": _final_equal(
            official_final, d3_all_final
        ),
    }
    gate_pass = all(checks.values())
    config_paths = {
        "official": args.official_config,
        "d1_off": args.d1_off_config,
        "d1_all": args.d1_all_config,
        "d3_off": args.d3_off_config,
        "d3_all": args.d3_all_config,
    }
    receipt = {
        "schema_version": "actionformer_odfcr_g0_equivalence_v1",
        "gate_pass": gate_pass,
        "checks": checks,
        "device": args.device,
        "dtype": str(features.dtype),
        "cuda_device_name": torch.cuda.get_device_name(
            torch.device(args.device)
        ),
        "seed": args.seed,
        "feature_length": args.feature_length,
        "input_tensor_sha256": _sha256_tensor(features),
        "ordered_input_video_ids": [video["video_id"]],
        "model_eval": True,
        "torch_no_grad": True,
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "config_paths": {
            arm: os.path.realpath(path)
            for arm, path in config_paths.items()
        },
        "config_sha256": {
            arm: _sha256_file(path) for arm, path in config_paths.items()
        },
        "git_branch": _git("branch", "--show-current"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_tree": _git("rev-parse", "HEAD^{tree}"),
        "git_clean": git_clean,
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
    print(
        json.dumps(
            {
                "gate_pass": gate_pass,
                "receipt": os.path.realpath(args.output),
                "receipt_sha256": _sha256_file(args.output),
            },
            sort_keys=True,
        )
    )
    if not gate_pass:
        raise RuntimeError("ODF-CR G0 exact-equivalence gate failed")


if __name__ == "__main__":
    main()
