#!/usr/bin/env python3
"""Run the two-rank, no-performance FP32 default-DDP reduction KAT."""

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

from tools.bata.georoute_experiment_contract import canonical_sha256  # noqa: E402
from tools.bata.georoute_official_comparable_contract import (  # noqa: E402
    FORMAL_WORLD_SIZE,
    OFFICIAL_DDP_WORLD2_KAT_FAIL,
    OFFICIAL_DDP_WORLD2_KAT_PASS,
    OFFICIAL_DDP_WORLD2_KAT_SCHEMA,
    validate_world2_kat_receipt,
)


BOUNDARY = Path("/data/run01/sczc063/yuzibo")
LOSS_SCALE = 65536.0
RANK_SCALED_GRADIENT_TARGETS = (70000.0, 90000.0)


def _inside(path: Path, boundary: Path) -> bool:
    try:
        path.relative_to(boundary)
    except ValueError:
        return False
    return path != boundary


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


class _World2ScaledBucketProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, dummy: torch.Tensor, coefficient: float) -> torch.Tensor:
        return (
            self.weight.sum() * float(coefficient)
            + dummy.to(torch.float32).sum() * 0.0
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> dict[str, Any] | None:
    expected_commit = str(args.expected_commit).lower()
    run_root = args.run_root.resolve()
    output = run_root / "control" / "world2_fp32_ddp_kat.json"
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("world2 DDP KAT root leaves the write boundary")
    if output.exists():
        raise FileExistsError("world2 DDP KAT receipt already exists")
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("world2 DDP KAT source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("world2 DDP KAT requires a clean source")

    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if (
        not slurm_job_id.isdigit()
        or not visible
        or len([item for item in visible.split(",") if item]) != FORMAL_WORLD_SIZE
    ):
        raise RuntimeError(
            "world2 DDP KAT requires exactly two Slurm-visible GPUs"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("world2 DDP KAT requires CUDA")

    dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != FORMAL_WORLD_SIZE or rank not in (0, 1):
        raise RuntimeError("world2 DDP KAT requires exactly two ranks")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    try:
        model = _World2ScaledBucketProbe().to(device)
        ddp = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )
        optimizer = torch.optim.SGD(ddp.parameters(), lr=1e-3)
        scaler = torch.cuda.amp.GradScaler(init_scale=LOSS_SCALE)
        optimizer.zero_grad(set_to_none=True)
        before = model.weight.detach().clone()

        target = RANK_SCALED_GRADIENT_TARGETS[rank]
        coefficient = target / LOSS_SCALE
        loss = ddp(torch.zeros(1, device=device), coefficient)
        scaler.scale(loss).backward()
        torch.cuda.synchronize(device)
        reduced = _tensor_stats(model.weight.grad)
        shadow = _tensor_stats(model.weight.grad.detach().to(torch.float16))
        expected_reduced = sum(RANK_SCALED_GRADIENT_TARGETS) / world_size
        if (
            reduced["dtype"] != "torch.float32"
            or reduced["finite"] is not True
            or abs(float(reduced["max_abs"]) - expected_reduced) > 1e-3
            or shadow["finite"] is not False
        ):
            raise RuntimeError(
                "default two-rank FP32 DDP reduction did not preserve the "
                "registered overflow-shadow gradient"
            )

        scaler.unscale_(optimizer)
        unscaled = _tensor_stats(model.weight.grad)
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize(device)
        updated = bool(
            torch.isfinite(model.weight.detach()).all().item()
            and not torch.equal(before, model.weight.detach())
        )
        update_tensor = torch.tensor(
            [1 if updated else 0], device=device, dtype=torch.int32
        )
        dist.all_reduce(update_tensor, op=dist.ReduceOp.MIN)
        all_updated = bool(update_tensor.item() == 1)
        if not all_updated:
            raise RuntimeError("world2 DDP KAT optimizer update failed")

        rank_record = {
            "rank": rank,
            "local_rank": local_rank,
            "scaled_gradient_target": float(target),
            "reduced_scaled_gradient": reduced,
            "unscaled_gradient": unscaled,
            "optimizer_update_completed": updated,
            "device_name": torch.cuda.get_device_name(device),
            "device_capability": list(torch.cuda.get_device_capability(device)),
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_record)
        if rank != 0:
            return None

        payload: dict[str, Any] = {
            "schema_version": OFFICIAL_DDP_WORLD2_KAT_SCHEMA,
            "status": OFFICIAL_DDP_WORLD2_KAT_PASS,
            "runtime_commit": expected_commit,
            "slurm_job_id": slurm_job_id,
            "backend": "nccl",
            "world_size": world_size,
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "nccl_version": list(torch.cuda.nccl.version()),
            "loss_scale": LOSS_SCALE,
            "rank_local_scaled_gradient_targets": list(
                RANK_SCALED_GRADIENT_TARGETS
            ),
            "expected_reduced_scaled_gradient": expected_reduced,
            "comm_hook_registration_invoked": False,
            "default_fp32_ddp_reduction_completed": True,
            "reduced_scaled_gradient": reduced,
            "detached_fp16_cast_shadow": shadow,
            "unscaled_gradient": unscaled,
            "optimizer_update_completed_on_all_ranks": all_updated,
            "rank_records": gathered,
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        payload["kat_sha256"] = canonical_sha256(payload)
        return validate_world2_kat_receipt(
            payload,
            expected_commit=expected_commit,
            expected_slurm_job_id=slurm_job_id,
        )
    finally:
        dist.destroy_process_group()


def main() -> int:
    args = _parse_args()
    output = (
        args.run_root.resolve()
        / "control"
        / "world2_fp32_ddp_kat.json"
    )
    rank = int(os.environ.get("RANK", "-1"))
    try:
        payload = _execute(args)
    except BaseException as error:
        if rank in {-1, 0} and not output.exists():
            failure: dict[str, Any] = {
                "schema_version": OFFICIAL_DDP_WORLD2_KAT_SCHEMA,
                "status": OFFICIAL_DDP_WORLD2_KAT_FAIL,
                "runtime_commit": str(args.expected_commit).lower(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "world_size": int(os.environ.get("WORLD_SIZE", "-1")),
                "failure": {
                    "exception_type": type(error).__name__,
                    "exception_message": str(error)[:2000],
                    "traceback_sha256": hashlib.sha256(
                        traceback.format_exc().encode(
                            "utf-8", errors="replace"
                        )
                    ).hexdigest(),
                },
                "checkpoint_emitted": False,
                "prediction_emitted": False,
                "evaluator_invoked": False,
                "official_test_opened": False,
                "performance_inference_allowed": False,
                "paper_claim_allowed": False,
            }
            failure["kat_sha256"] = canonical_sha256(failure)
            _atomic_write_json(output, failure)
        raise
    if rank == 0:
        assert payload is not None
        _atomic_write_json(output, payload)
        print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
