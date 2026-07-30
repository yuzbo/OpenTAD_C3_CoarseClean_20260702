#!/usr/bin/env python3
"""Run the no-performance CUDA/DDP KAT for gradient-decomposition telemetry."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.distributed as dist
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as comm_hooks
from torch.nn.parallel import DistributedDataParallel


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.bata.georoute_experiment_contract import canonical_sha256  # noqa: E402
from tools.bata.georoute_gradient_decomposition import (  # noqa: E402
    KAT_FAIL_STATUS,
    KAT_PASS_STATUS,
    KAT_SCHEMA,
    ObservedFp16HookState,
    bucket_cast_telemetry,
    expected_scaled_logit_gradient,
    observed_fp16_compress_hook,
    ordered_pl_log_prob_and_score,
    tensor_stats,
)


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


class _BucketObserver:
    def __init__(self, *, loss_scale: float) -> None:
        self.loss_scale = float(loss_scale)
        self.records: list[dict[str, Any]] = []
        self.original_bucket_unchanged = True

    def record_ddp_bucket(
        self,
        *,
        bucket_index: int,
        parameter_names: list[str],
        bucket_buffer: torch.Tensor,
    ) -> None:
        before = bucket_buffer.detach().clone()
        telemetry = bucket_cast_telemetry(
            bucket_buffer=bucket_buffer,
            loss_scale=self.loss_scale,
            world_size=dist.get_world_size(),
        )
        unchanged = bool(torch.equal(bucket_buffer.detach(), before))
        self.original_bucket_unchanged &= unchanged
        self.records.append(
            {
                "bucket_index": int(bucket_index),
                "parameter_names": list(parameter_names),
                "telemetry": telemetry,
                "observer_left_original_bucket_bitwise_unchanged": unchanged,
            }
        )


class _ScaledBucketProbe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))

    def forward(self, dummy: torch.Tensor) -> torch.Tensor:
        # With scale 65536 this yields a finite FP32 gradient of 70000, which is
        # outside the finite FP16 range and therefore exercises the exact cast.
        return (
            self.weight.sum() * (70000.0 / 65536.0)
            + dummy.to(torch.float32).sum() * 0.0
        )


def _analytic_direction_kat(device: torch.device) -> dict[str, Any]:
    logits = torch.tensor(
        [[[0.2, -0.1, 0.7, 0.0], [0.5, -0.4, 0.1, 0.3]]],
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    order = torch.tensor(
        [[[2, 0], [0, 3]]], dtype=torch.long, device=device
    )
    valid = torch.ones_like(logits, dtype=torch.bool)
    temperature = 0.7
    analytic_logp, score, _ = ordered_pl_log_prob_and_score(
        logits=logits.detach(),
        ordered_indices=order,
        valid_mask=valid,
        temperature=temperature,
    )
    available = valid.clone()
    production_logp = torch.zeros(
        logits.shape[:2], dtype=logits.dtype, device=device
    )
    for slot in range(order.shape[-1]):
        choice = order[..., slot]
        masked = (logits / temperature).masked_fill(~available, float("-inf"))
        production_logp = production_logp + torch.log_softmax(
            masked, dim=-1
        ).gather(-1, choice.unsqueeze(-1)).squeeze(-1)
        available.scatter_(-1, choice.unsqueeze(-1), False)
    if not torch.allclose(
        analytic_logp, production_logp.detach(), rtol=1e-6, atol=1e-7
    ):
        raise RuntimeError("CUDA KAT analytic log probability mismatch")
    advantage = torch.tensor(3.25, dtype=torch.float32, device=device)
    loss_scale = 4096.0
    (advantage * production_logp.mean() * loss_scale).backward()
    expected = expected_scaled_logit_gradient(
        score=score,
        advantage=advantage,
        weight=1.0,
        temporal_reduction="mean",
        loss_scale=loss_scale,
    )
    dot = torch.sum(logits.grad.to(torch.float64) * expected.to(torch.float64))
    close = bool(
        torch.allclose(
            logits.grad.to(torch.float32),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
    )
    if not close or float(dot.item()) <= 0.0:
        raise RuntimeError("CUDA KAT analytic/actual PL gradient mismatch")
    return {
        "actual_scaled_gradient": tensor_stats(logits.grad),
        "expected_scaled_gradient": tensor_stats(expected),
        "close": close,
        "direction_positive": float(dot.item()) > 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    return parser.parse_args()


def _execute(args: argparse.Namespace) -> dict[str, Any]:
    expected_commit = str(args.expected_commit).lower()
    run_root = args.run_root.resolve()
    if not _inside(run_root, BOUNDARY.resolve()):
        raise ValueError("gradient-decomposition KAT root leaves write boundary")
    output = run_root / "kat_receipt.json"
    if output.exists():
        raise FileExistsError("gradient-decomposition KAT receipt already exists")
    if _git_output("rev-parse", "HEAD").lower() != expected_commit:
        raise RuntimeError("gradient-decomposition KAT source mismatch")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("gradient-decomposition KAT requires a clean source")
    slurm_job_id = str(os.environ.get("SLURM_JOB_ID", ""))
    visible = str(os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    if not slurm_job_id.isdigit() or not visible or "," in visible:
        raise RuntimeError("gradient-decomposition KAT requires one Slurm GPU")
    if not torch.cuda.is_available():
        raise RuntimeError("gradient-decomposition KAT requires CUDA")
    device = torch.device("cuda:0")
    rendezvous_file = run_root / ".kat_nccl_rendezvous"
    if rendezvous_file.exists():
        raise FileExistsError("gradient-decomposition KAT rendezvous exists")

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
        scaler = torch.cuda.amp.GradScaler(init_scale=65536.0)
        observer = _BucketObserver(loss_scale=float(scaler.get_scale()))
        state = ObservedFp16HookState(
            observer=observer,
            parameter_names={
                id(parameter): name
                for name, parameter in ddp.named_parameters()
            },
        )
        ddp.register_comm_hook(state=state, hook=observed_fp16_compress_hook)
        scaler.scale(ddp(torch.zeros(1, device=device))).backward()
        torch.cuda.synchronize()
        if (
            not observer.records
            or observer.original_bucket_unchanged is not True
            or not any(
                record["telemetry"]["cast_introduced_nonfinite"] is True
                for record in observer.records
            )
        ):
            raise RuntimeError("gradient-decomposition DDP observer KAT failed")
        # Successful DDP backward proves the wrapper returned the Future expected
        # by DDP; the observed post-hook gradient is nonfinite by construction.
        post_backward = tensor_stats(model.weight.grad)
        if post_backward["finite"] is not False:
            raise RuntimeError("authoritative FP16 hook did not exercise overflow")
        analytic = _analytic_direction_kat(device)
        payload: dict[str, Any] = {
            "schema_version": KAT_SCHEMA,
            "status": KAT_PASS_STATUS,
            "runtime_commit": expected_commit,
            "slurm_job_id": slurm_job_id,
            "device": str(device),
            "world_size": dist.get_world_size(),
            "torch_version": str(torch.__version__),
            "cuda_version": str(torch.version.cuda),
            "nccl_version": list(torch.cuda.nccl.version()),
            "fp16_compress_hook_source_sha256": hashlib.sha256(
                inspect.getsource(comm_hooks.fp16_compress_hook).encode("utf-8")
            ).hexdigest(),
            "standard_hook_future_completed": True,
            "observer_left_original_bucket_bitwise_unchanged": (
                observer.original_bucket_unchanged
            ),
            "bucket_records": observer.records,
            "post_authoritative_hook_gradient": post_backward,
            "analytic_gradient_direction": analytic,
            "checkpoint_emitted": False,
            "prediction_emitted": False,
            "evaluator_invoked": False,
            "official_test_opened": False,
            "performance_inference_allowed": False,
            "paper_claim_allowed": False,
        }
        payload["kat_sha256"] = canonical_sha256(payload)
        return payload
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
