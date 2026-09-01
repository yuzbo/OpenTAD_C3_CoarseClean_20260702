import sys
import os

from mmengine.config import Config
import opentad.datasets
from opentad.models import build_detector

cfgs = [
    "configs/adatad/thumos/duca_ct_dual_phase_bamod_thumos.py",
    "configs/adatad/thumos/duca_dual_phase_bamod_thumos.py",
    "configs/adatad/thumos/duca_ct_dual_phase_densevit_thumos.py",
]

for cfg_path in cfgs:
    print(f"--> Building detector from: {cfg_path}")
    cfg = Config.fromfile(cfg_path)
    model = build_detector(cfg.model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    [SUCCESS] Total params: {total_params:,} | Trainable: {trainable_params:,}")

print("--> All 3 experiment models built successfully with zero errors!")
