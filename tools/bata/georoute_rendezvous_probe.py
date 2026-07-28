#!/usr/bin/env python3
"""One-rank worker used by the GeoRoute concurrent rendezvous isolation gate."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch.distributed as dist


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--post-release-seconds", type=float, required=True)
    args = parser.parse_args()
    if args.post_release_seconds <= 0:
        raise ValueError("post-release duration must be positive")
    if args.ready_file.exists():
        raise FileExistsError(args.ready_file)

    dist.init_process_group(backend="gloo")
    try:
        if dist.get_world_size() != 1 or dist.get_rank() != 0:
            raise RuntimeError("GeoRoute rendezvous probe must be an isolated one-rank group")
        ready = {
            "event": "GEOROUTE_RDZV_READY",
            "label": args.label,
            "rank": dist.get_rank(),
            "world_size": dist.get_world_size(),
            "torchelastic_run_id": os.environ.get("TORCHELASTIC_RUN_ID"),
            "master_addr": os.environ.get("MASTER_ADDR"),
            "master_port": os.environ.get("MASTER_PORT"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        args.ready_file.write_text(
            json.dumps(ready, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        deadline = time.monotonic() + 30.0
        while not args.release_file.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("GeoRoute rendezvous probe release marker timed out")
            time.sleep(0.05)
        time.sleep(args.post_release_seconds)
        print(
            json.dumps(
                {
                    "event": "GEOROUTE_RDZV_DONE",
                    "label": args.label,
                    "rank": dist.get_rank(),
                    "world_size": dist.get_world_size(),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
