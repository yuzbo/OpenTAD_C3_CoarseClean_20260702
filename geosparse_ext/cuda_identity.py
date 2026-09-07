"""Resolve CUDA's visible device to its UUID without assuming NVML index order."""
import ctypes
from functools import lru_cache
import os
import subprocess
import uuid


@lru_cache(maxsize=1)
def allocated_cuda_device():
    import torch
    driver = ctypes.CDLL("libcuda.so.1")
    def check(status):
        if status != 0:
            raise RuntimeError(f"CUDA driver device identity query failed: {status}")
    check(driver.cuInit(0))
    device = ctypes.c_int()
    logical = torch.cuda.current_device()
    check(driver.cuDeviceGet(ctypes.byref(device), logical))
    raw = (ctypes.c_ubyte * 16)()
    check(driver.cuDeviceGetUuid(ctypes.byref(raw), device))
    cuda_uuid = "GPU-" + str(uuid.UUID(bytes=bytes(raw)))
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,index,pci.bus_id,name",
                                    "--format=csv,noheader,nounits"], text=True)
    matches = [line.split(",", 3) for line in rows.splitlines() if line.split(",")[0].strip() == cuda_uuid]
    if len(matches) != 1:
        raise RuntimeError("CUDA device UUID has no unique NVML device mapping")
    _, index, bus, name = [value.strip() for value in matches[0]]
    return dict(cuda_uuid=cuda_uuid, cuda_logical_index=logical, nvml_index=index, pci_bus_id=bus,
                name=name, slurm_job_gpus=os.environ.get("SLURM_JOB_GPUS"),
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
