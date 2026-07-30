#!/usr/bin/env python3
"""Run the no-performance CUDA/DDP KAT for disabling FP16 communication."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_ddp_fp16_cast_repair import (  # noqa: E402
    KAT_FAIL_STATUS,
    KAT_LOSS_SCALE,
    KAT_PASS_STATUS,
    KAT_SCALED_GRADIENT,
    KAT_SCHEMA,
    validate_kat_receipt,
)
from tools.bata.georoute_amp_diagnostic import (  # noqa: E402
    AMP_REPAIR_INTERVENTION,
    AMP_REPAIR_REGISTERED_CLASS,
    AMP_REPAIR_STUDY_ID,
)
from tools.bata.georoute_experiment_contract import canonical_sha256  # noqa: E402


BOUNDARY = Path("/data/run01/sczc063/yuzibo")


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return path != root


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip() or f"git {' '.join(arguments)} failed"
        )
    return completed.stdout.strip()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    finite = torch.isfinite(detached)
    finite_values = detached.float().masked_select(finite)
    return {
        "dtype": str(detached.dtype),
        "shape": list(detached.shape),
        "finite": bool(finite.all().item()),
        "finite_count": int(finite.sum().item()),
        "nonfinite_count": int(detached.numel() - finite.sum().item()),
        "max_abs": (
            float(finite_values.abs().max().item())
            if int(finite.sum().item()) > 0
            else None
        ),
    }


class _ScaledBucketProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        return (
            self.weight.sum() * (KAT_SCALED_GRADIENT / KAT_LOSS_SCALE)
            + dummy.to(torch.float32).sum() * 0.0
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("DDP FP16-cast repair KAT root leaves write boundary")
    output = run_root / "kat_receipt.json"
    if output.exists():
        raise FileExistsError("DDP FP16-cast repair KAT receipt already exists")
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("DDP FP16-cast repair KAT source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("DDP FP16-cast repair KAT requires a clean source")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not slurm_job_id.isdigit() or not visible or "," in visible:
        raise RuntimeError("DDP FP16-cast repair KAT requires one Slurm GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("DDP FP16-cast repair KAT requires CUDA")

    device = torch.device("cuda:0")
    rendezvous_file = run_root / ".kat_nccl_rendezvous"
    if rendezvous_file.exists():
        raise FileExistsError("DDP FP16-cast repair KAT rendezvous exists")
    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{rendezvous_file}",
        rank=0,
        world_size=1,
    )
    try:
        torch.cuda.set_device(device)
        model = _ScaledBucketProbe().to(device)
        ddp = DistributedDataParallel(model, device_ids=[0])
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=KAT_LOSS_SCALE)
        optimizer.zero_grad(set_to_none=True)
        before = model.weight.detach().clone()

        # Intentionally do not call register_comm_hook. The default DDP reducer
        # therefore communicates this FP32 parameter bucket in FP32.
        scaler.scale(ddp(torch.zeros(1, device=device))).backward()
        torch.cuda.synchronize()
        scaled = _tensor_stats(model.weight.grad)
        shadow = _tensor_stats(model.weight.grad.detach().to(torch.float16))
        if scaled["finite"] is not True or shadow["finite"] is not False:
            raise RuntimeError(
                "default FP32 DDP reduction did not isolate the removed FP16 cast"
            )

        scaler.unscale_(optimizer)
        unscaled = _tensor_stats(model.weight.grad)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        after = model.weight.detach()
        if not bool(torch.isfinite(after).all().item()) or bool(
            torch.equal(before, after)
        ):
            raise RuntimeError("DDP FP16-cast repair KAT optimizer update failed")

        payload: dict[str, Any] = {
            "schema_version": KAT_SCHEMA,
            "status": KAT_PASS_STATUS,
            "study_id": AMP_REPAIR_STUDY_ID,
            "runtime_commit": expected_commit,
            "slurm_job_id": slurm_job_id,
            "device": str(device),
            "world_size": dist.get_world_size(),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "nccl_version": list(torch.cuda.nccl.version()),
            "registered_repair_class": AMP_REPAIR_REGISTERED_CLASS,
            "registered_single_variable_intervention": AMP_REPAIR_INTERVENTION,
            "loss_scale": float(KAT_LOSS_SCALE),
            "comm_hook_registration_invoked": False,
            "ddp_default_fp32_reduction_completed": True,
            "scaled_fp32_gradient": scaled,
            "detached_fp16_cast_shadow": shadow,
            "unscaled_fp32_gradient": unscaled,
            "optimizer_update_completed": True,
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        payload["kat_sha256"] = canonical_sha256(payload)
        return validate_kat_receipt(
            payload,
            expected_commit=expected_commit,
            expected_slurm_job_id=slurm_job_id,
        )
    finally:
        dist.destroy_process_group()
        rendezvous_file.unlink(missing_ok=True)


def main() -> int:
    args = _parse_args()
    output = args.run_root.resolve() / "kat_receipt.json"
    try:
        payload = _execute(args)
    except BaseException as error:
        payload = {
            "schema_version": KAT_SCHEMA,
            "status": KAT_FAIL_STATUS,
            "study_id": AMP_REPAIR_STUDY_ID,
            "runtime_commit": str(args.expected_commit).lower(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "failure": {
                "exception_type": type(error).__name__,
                "exception_message": str(error)[:2000],
                "traceback_sha256": hashlib.sha256(
                    traceback.format_exc().encode("utf-8", errors="replace")
                ).hexdigest(),
            },
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        payload["kat_sha256"] = canonical_sha256(payload)
        _atomic_write_json(output, payload)
        raise
    _atomic_write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
